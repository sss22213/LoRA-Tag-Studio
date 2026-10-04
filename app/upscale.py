"""用 waifu2x 放大低解析圖片（nunif 的 swin_unet ONNX 模型，作者 nagadomi，MIT 授權）。

LoRA 訓練會把短邊太小的圖用一般演算法放大，結果偏糊，LoRA 會連模糊一起學進去。
先用 waifu2x 放大（JPEG 一併降噪）可以改善。原圖備份在專案的 originals/，可隨時還原；
重新放大時一律從備份的原圖開始。

切塊、邊緣填補與接縫混合的做法移植自 nunif（waifu2x/unlimited_waifu2x、nunif/utils/seam_blending.py）。
"""
from __future__ import annotations

import gc
import hashlib
import io
import logging
import math
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from PIL import Image

from . import db, storage
from .config import settings
from .i18n import t
from .tagging.blocks import detect_white_blocks

log = logging.getLogger(__name__)

STYLES = ("art", "art_scan", "photo")  # 插畫 / 掃描圖（網點、紙紋）/ 照片
NOISES = ("auto", "none", "0", "1", "2", "3")  # auto：JPEG 與有損 WebP 用 1 級，其他不降噪
SCALES = ("auto", "1", "2", "4")  # auto：2x，還不到目標短邊才用 4x；1 = 只降噪（尺寸不變）
DEFAULT_MIN_SIDE = 1024
TOO_SMALL = 384  # 短邊低於這個值，放大也救不回細節，建議直接刪除
MAX_OUTPUT_PIXELS = 40_000_000

OFFSET = {1: 8, 2: 16, 4: 32}  # 模型輸出比輸入少一圈（以輸出像素計）
BLEND = 16  # 相鄰切塊重疊混合的寬度（輸出像素）
TILE = 256  # swin_unet 的限制：(TILE - 16) 必須同時是 12 與 16 的倍數
# 一次只送一個切塊、記憶體池按需要增加：每個模型約 0.6 GB VRAM（一次 4 塊快約 1/3，但要 2 GB 以上）
CUDA_OPTIONS = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}
PADDING = {"art": "edge", "art_scan": "edge", "photo": "reflect"}
COLOR_STABILITY = {"art"}  # 單色的切塊直接填色，不經過模型（白色色塊保持純白）


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ 模型下載
class _HTTPRange(io.RawIOBase):
    """用 HTTP Range 讀遠端檔案：只下載 zip 目錄與需要的模型，不必下載整包（約 700 MB）。"""

    def __init__(self, client: httpx.Client, url: str) -> None:
        with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as r:
            r.raise_for_status()
            if r.status_code != 206:
                raise OSError(f"{url} 不支援分段下載")
            self.url = str(r.url)  # GitHub 會轉址到實際存放位置
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.client, self.pos = client, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        r = self.client.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        data = r.content
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def model_name(style: str, noise: int, scale: int) -> str:
    if scale == 1:
        method = f"noise{noise}"
    else:
        method = f"noise{noise}_scale{scale}x" if noise >= 0 else f"scale{scale}x"
    return f"swin_unet/{style}/{method}.onnx"


def downloaded_models() -> list[str]:
    root = settings.waifu2x_dir
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.onnx")) if root.is_dir() else []


def ensure_models(names: set[str] | list[str]) -> None:
    """下載還沒有的模型檔（每個約 17–19 MB）。"""
    missing = [n for n in sorted(set(names)) if not (settings.waifu2x_dir / n).exists()]
    if not missing:
        return
    log.info("下載 waifu2x 模型：%s", ", ".join(missing))
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "lora-tag-studio/1.0"}) as client:
        remote = io.BufferedReader(_HTTPRange(client, settings.waifu2x_models_url), buffer_size=4 << 20)
        with zipfile.ZipFile(remote) as zf:
            for name in missing:
                dest = settings.waifu2x_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                with zf.open(f"onnx_models/{name}") as src, open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)  # 讀完會檢查 CRC
                tmp.replace(dest)


# ------------------------------------------------------------------ 推論
_lock = threading.Lock()
_sessions: dict[str, Any] = {}


def _session(name: str) -> Any:
    import onnxruntime as ort

    from .tagging.wd14 import ort_providers

    with _lock:
        sess = _sessions.get(name)
        if sess is None:
            ensure_models([name])
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3  # swin_unet 每次推論都會印 ScatterND 警告（無害）
            providers = [(p, CUDA_OPTIONS) if p == "CUDAExecutionProvider" else p for p in ort_providers()]
            sess = ort.InferenceSession(str(settings.waifu2x_dir / name), sess_options=opts, providers=providers)
            _sessions[name] = sess
        return sess


def unload_all() -> list[str]:
    """釋放已載入的 waifu2x 模型（GPU 版會釋放 VRAM）。放大工作結束時會自動呼叫。"""
    with _lock:
        names = list(_sessions)
        _sessions.clear()
    gc.collect()
    if names:
        log.info("已釋放 waifu2x 模型：%s", ", ".join(names))
    return names


def loaded_models() -> list[dict[str, Any]]:
    return [{"name": k, "providers": v.get_providers()} for k, v in list(_sessions.items())]


def _blend_filter(scale: int, offset: int) -> np.ndarray:
    size = TILE * scale - offset * 2 - BLEND * 2
    f = np.ones((size, size), np.float32)
    for i in range(BLEND):
        f = np.pad(f, 1, constant_values=1 - (i + 1) / (BLEND + 1))
    return f


def _render(sess: Any, x: np.ndarray, scale: int, pad_mode: str, stable: bool,
            tick: Callable[[], None]) -> np.ndarray:
    """切塊推論再以權重混合接縫。x：(3, H, W)、0–1 → (3, H×scale, W×scale)。"""
    _, h, w = x.shape
    offset = OFFSET[scale]
    in_off = math.ceil(offset / scale)
    step = TILE - (in_off * 2 + math.ceil(BLEND / scale))

    def blocks(n: int) -> tuple[int, int]:
        count = size = 0
        while size < n + in_off * 2:
            size = count * step + TILE
            count += 1
        return count, size

    hb, ih = blocks(h)
    wb, iw = blocks(w)
    xp = np.pad(x, ((0, 0), (in_off, ih - h - in_off), (in_off, iw - w - in_off)), mode=pad_mode)
    out = TILE * scale - offset * 2
    filt = _blend_filter(scale, offset)
    pixels = np.zeros((x.shape[0], ih * scale, iw * scale), np.float32)
    weights = np.zeros((ih * scale, iw * scale), np.float32)
    ostep = step * scale

    def put(y: np.ndarray, hi: int, wi: int) -> None:
        ys, xs = slice(hi * ostep, hi * ostep + out), slice(wi * ostep, wi * ostep + out)
        nxt = weights[ys, xs] + filt
        keep = weights[ys, xs] / nxt
        pixels[:, ys, xs] = pixels[:, ys, xs] * keep + y * (1 - keep)
        weights[ys, xs] = nxt

    for hi in range(hb):
        for wi in range(wb):
            tile = xp[:, hi * step:hi * step + TILE, wi * step:wi * step + TILE]
            if stable and (tile == tile[:, :1, :1]).all():
                y = np.broadcast_to(tile[:, :1, :1], (x.shape[0], out, out))
            else:
                y = sess.run(None, {"x": np.ascontiguousarray(tile[None])})[0][0]
            put(y, hi, wi)
            tick()
    return np.clip(pixels[:, :h * scale, :w * scale], 0, 1)


def _tiles(h: int, w: int, scale: int) -> int:
    in_off = math.ceil(OFFSET[scale] / scale)
    step = TILE - (in_off * 2 + math.ceil(BLEND / scale))
    count = lambda n: max(1, math.ceil((n + in_off * 2 - TILE) / step) + 1)  # noqa: E731
    return count(h) * count(w)


def _box3(a: np.ndarray) -> np.ndarray:
    """3×3 鄰域加總（外圍補 0）。"""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)])
    h, w = a.shape[-2:]
    return sum(p[..., dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3))


def _alpha_border_padding(rgb: np.ndarray, alpha: np.ndarray, n: int) -> np.ndarray:
    """透明區域的顏色用旁邊不透明像素的平均往外延伸，避免放大後邊緣出現黑邊或雜色。"""
    rgb = rgb.copy()
    mask = alpha > 0
    rgb[:, ~mask] = 0
    for _ in range(n):
        if mask.all():
            break
        weight = _box3(mask.astype(np.float32))
        border = _box3(rgb) / (weight + 1e-7)
        rgb[:, ~mask] = border[:, ~mask]
        mask = weight > 0
    return np.clip(rgb, 0, 1)


def upscale_pil(im: Image.Image, style: str = "art", noise: int = -1, scale: int = 2,
                progress: Callable[[int, int], None] | None = None) -> Image.Image:
    """放大一張圖。noise：-1 = 不降噪、0–3；scale：1（只降噪）/ 2 / 4。progress(完成切塊, 總切塊)。"""
    if scale == 1 and noise < 0:
        raise ValueError(t("msg.upscale_nothing_to_do"))
    mode = im.mode
    has_alpha = mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in im.info)
    arr = np.asarray(im.convert("RGBA" if has_alpha else "RGB"), dtype=np.float32) / 255
    rgb = np.ascontiguousarray(arr[..., :3].transpose(2, 0, 1))
    alpha = arr[..., 3] if has_alpha else None
    if alpha is not None and (alpha >= 1).all():
        alpha = None  # 其實完全不透明
    h, w = rgb.shape[1:]
    alpha_pass = alpha is not None and scale > 1
    total = _tiles(h, w, scale) * (2 if alpha_pass else 1)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total)

    if alpha is not None:
        rgb = _alpha_border_padding(rgb, alpha, OFFSET[scale])
    y = _render(_session(model_name(style, noise, scale)), rgb, scale, PADDING[style], style in COLOR_STABILITY,
                tick)
    out = np.rint(y.transpose(1, 2, 0) * 255).astype(np.uint8)
    if alpha is None:
        result = Image.fromarray(out, "RGB")
    else:
        if alpha_pass:  # 透明度用不降噪的放大模型（三個通道都放透明度，再取平均）
            a = _render(_session(model_name(style, -1, scale)), np.repeat(alpha[None], 3, axis=0), scale,
                        PADDING[style], True, tick).mean(axis=0)
        else:
            a = alpha
        result = Image.fromarray(np.dstack([out, np.rint(a * 255).astype(np.uint8)]), "RGBA")
    if mode in ("L", "LA"):
        result = result.convert(mode)
    return result


# ------------------------------------------------------------------ 專案圖片
def is_lossy(path: Path) -> bool:
    """JPEG 與有損 WebP 有壓縮雜訊，自動降噪時才降噪。"""
    try:
        f = open(path, "rb")
    except OSError:
        return False
    with f:
        head = f.read(12)
        if head[:3] == b"\xff\xd8\xff":
            return True
        if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
            return False
        while True:  # WebP 的區塊：VP8 = 有損、VP8L = 無損（前面可能有 VP8X / ICCP / ALPH 等）
            chunk = f.read(8)
            if len(chunk) < 8:
                return True
            if chunk[:4] in (b"VP8 ", b"VP8L"):
                return chunk[:4] == b"VP8 "
            size = int.from_bytes(chunk[4:], "little")
            f.seek(size + (size & 1), 1)


def original_size(img: dict[str, Any]) -> tuple[int, int]:
    o = (img.get("upscale") or {}).get("original") or img
    return int(o.get("width") or 0), int(o.get("height") or 0)


def source_path(img: dict[str, Any]) -> Path:
    """放大用的來源：已放大過的圖從備份的原圖開始。"""
    rec = img.get("upscale")
    return storage.originals_dir(img["project_id"]) / rec["original"]["filename"] if rec else storage.image_path(img)


def plan(img: dict[str, Any], noise: str, scale: str, min_side: int) -> tuple[int, int]:
    """依設定決定這張圖的 (降噪等級, 倍率)。降噪 -1 = 不降噪。"""
    if noise == "auto":
        level = 1 if is_lossy(source_path(img)) else -1
    else:
        level = -1 if noise == "none" else int(noise)
    if scale == "auto":
        factor = 2 if min(original_size(img)) * 2 >= min_side else 4
    else:
        factor = int(scale)
    return level, factor


def _blocks_for(im: Image.Image, img: dict[str, Any]) -> dict[str, Any]:
    """重新偵測白色色塊，保留手動標記。"""
    found = detect_white_blocks(im)
    override = (img.get("blocks") or {}).get("override")
    if override is not None:
        found["override"] = override
    return found


def upscale_image(img: dict[str, Any], style: str, noise: int, scale: int,
                  progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """放大專案裡的一張圖，結果存成 PNG。第一次放大時把原圖備份到 originals/。"""
    pid, iid = img["project_id"], img["id"]
    rec = img.get("upscale")
    with Image.open(source_path(img)) as im:
        im.load()
        if im.width * im.height * scale * scale > MAX_OUTPUT_PIXELS:
            raise ValueError(t("msg.upscale_too_large", w=im.width * scale, h=im.height * scale))
        out = upscale_pil(im, style, noise, scale, progress)

    idir = storage.images_dir(pid)
    final, tmp = idir / f"{iid}.png", idir / f".{iid}.upscale.png"
    out.save(tmp, format="PNG")
    data = tmp.read_bytes()
    current = storage.image_path(img)
    if rec:
        original = rec["original"]
    else:
        original = {k: img[k] for k in ("filename", "width", "height", "size", "sha1")}
        backup = storage.originals_dir(pid) / img["filename"]
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, backup)
    try:
        db.update_image(iid, filename=final.name, width=out.width, height=out.height, size=len(data),
                        sha1=hashlib.sha1(data).hexdigest(), blocks=_blocks_for(out, img),
                        upscale={"original": original, "style": style, "noise": noise, "scale": scale,
                                 "at": time.time()})
    except Exception:
        tmp.unlink(missing_ok=True)
        if not rec:
            (storage.originals_dir(pid) / img["filename"]).unlink(missing_ok=True)
        raise
    tmp.replace(final)
    if current != final:
        current.unlink(missing_ok=True)  # 原圖已備份
    storage.make_thumb(out, storage.thumb_path(img))
    return db.get_image(iid)  # type: ignore[return-value]


def restore_image(img: dict[str, Any]) -> None:
    """換回備份的原圖。"""
    pid, iid = img["project_id"], img["id"]
    original = img["upscale"]["original"]
    backup = storage.originals_dir(pid) / original["filename"]
    if not backup.exists():
        raise ValueError(t("msg.upscale_backup_missing", name=img["original_name"]))
    with Image.open(backup) as im:
        im.load()
        blocks = _blocks_for(im, img)
        storage.make_thumb(im, storage.thumb_path(img))
    current, dest = storage.image_path(img), storage.images_dir(pid) / original["filename"]
    db.update_image(iid, filename=original["filename"], width=original["width"], height=original["height"],
                    size=original["size"], sha1=original["sha1"], blocks=blocks, upscale=None)
    backup.replace(dest)
    if current != dest:
        current.unlink(missing_ok=True)


def run_job(job: Any) -> None:
    """背景工作：依序放大 / 降噪 job.image_ids（由 jobs 的 worker 呼叫，與標註工作輪流使用 GPU）。"""
    p = job.params
    imgs = [i for i in db.list_images(job.project_id, ids=job.image_ids)]
    plans = {i["id"]: plan(i, p["noise"], p["scale"], p["min_side"]) for i in imgs}
    names = {model_name(p["style"], noise, scale) for noise, scale in plans.values() if scale > 1 or noise >= 0}
    names |= {model_name(p["style"], -1, scale) for _, scale in plans.values() if scale > 1}  # 透明圖的透明度
    job.say("msg.upscale_loading_model")
    try:
        ensure_models(names)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(t("msg.upscale_download_failed", error=e)) from e
    for img in imgs:
        if job.cancel_event.is_set():
            return
        noise, scale = plans[img["id"]]
        db.set_status([img["id"]], "processing")
        key = "msg.denoise_progress" if scale == 1 else "msg.upscale_progress"
        job.say(key, pct=0)

        def progress(done: int, tiles: int, key: str = key) -> None:
            if job.cancel_event.is_set():
                raise Cancelled
            job.say(key, pct=done * 100 // tiles)

        try:
            upscale_image(img, p["style"], noise, scale, progress)
        except Cancelled:
            return
        except Exception as e:  # noqa: BLE001
            log.exception("放大失敗 %s", img["id"])
            job.failed += 1
            job.errors.append({"image_id": img["id"], "file": img["original_name"], "error": str(e)})
        finally:
            db.restore_status([img["id"]])
        job.done += 1
