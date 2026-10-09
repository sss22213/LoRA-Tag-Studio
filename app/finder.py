"""角色篩選：用 CCIP（deepghs/ccip_onnx，OpenRAIL 授權）從一堆圖片挑出目標角色。

和專案分開。每個「篩選」有三組圖片：
- ref：目標角色的參考圖（1 張就能用，3–10 張較穩）
- neg：長得像但不是目標的角色（可不加）。圖片比較像這些時就不算符合，用來分開髮色相近的角色
- pool：要篩選的一堆圖片

上傳的圖片存在 data/finder/<篩選 id>/；從專案加入的用硬連結（同一個磁碟不多佔空間）；
伺服器匯入資料夾的只記路徑、不複製。特徵與分數存在資料庫，調整門檻不必重算。

分數 = 和參考圖差異的中位數；差異 ≤ 門檻（預設是模型公布的最佳 F1 門檻），且比任何一張排除參考圖都更像目標，
才算符合。CCIP 對整張圖算一個特徵：多人同框、人物很小時可能被別人蓋過。

其他：
- 找重複：讀圖時順便算指紋（64 位元感知雜湊 + 32×32 小圖）。先用雜湊挑出候選，再比小圖的平均差異，門檻由使用者調整
  （小 = 幾乎一樣；大 = 同一個鏡頭表情、動作不同也算）。每組保留畫質最好的一張（目前符合的優先，不會留下別人那張、
  丟掉目標角色那張），其他的要使用者確認後才移除。可以只在目前的結果裡找（重複的標成不符合，可以復原）。
- tag 篩選：辨識時可以順便用 WD14 產生 tag，只用在 CCIP 挑出的結果上（必須有 / 不能有某些 tag）。
  有設定篩選時，還沒有 tag 的圖片先不算符合。手動 ✓（是目標）也會套用 tag 篩選；WD14 標錯時可以「忽略 tag 保留」。
"""
from __future__ import annotations

import functools
import gc
import hashlib
import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageOps

from . import db, storage
from .config import settings
from .i18n import t
from .services import BadRequest, NotFound

log = logging.getLogger(__name__)

REPO = "deepghs/ccip_onnx"
MODELS = {"default": "ccip-caformer-24-randaug-pruned", "large": "ccip-caformer_b36-24"}  # 150 MB / 384 MB
# 各模型公布的門檻（metrics.json，F1 最高的差異值）
THRESHOLDS = {"default": 0.17847511429108218, "large": 0.21323118981474148}
ROLES = ("ref", "neg", "pool")
SIZE = 384
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)[:, None, None]
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)[:, None, None]
BATCH = 8
CHUNK = 512  # 比對時一次放進差異模型的圖片數（輸出是 N×N）
# 讀圖、前處理的執行緒。圖片在 NAS 上時主要在等 SMB 回應，多開幾個同時讀；本機解碼 1080p PNG 約 40 ms
WORKERS = min(12, os.cpu_count() or 4)
PREFETCH = 4  # GPU 算這一批時，先讀好後面幾批（同時有 PREFETCH × BATCH 張在讀）
THUMB_SIZE = storage.THUMB_SIZE
# 找重複：32×32 小圖每個值（0–255）的平均差異不超過門檻就算重複。用 4 萬張動畫截圖看過：
# ≤ 2 幾乎一模一樣（尺寸、壓縮、字幕）；3–6 同一個鏡頭，嘴型、眨眼不同；7–11 表情、手勢也不同；更大時鏡頭移動也會算進來
DUP_DEFAULT, DUP_MAX = 4.0, 16.0
DUP_PREFILTER = 16  # 感知雜湊距離在這以內才比小圖（平均差異 ≤ 10 的配對 98% 在這以內）
_DCT = np.sqrt(2 / 32) * np.cos(np.pi * (2 * np.arange(32)[None, :] + 1) * np.arange(32)[:, None] / 64)
_DCT[0] /= np.sqrt(2)
KEEP = 2  # 手動判定「忽略 tag 保留」（manual 欄位；1 = 是目標、-1 = 不是目標）
# WD14 tag 門檻（和專案的預設相同）
TAG_GENERAL = 0.35
TAG_CHARACTER = 0.85


# ------------------------------------------------------------------ 路徑
def session_dir(sid: str) -> Path:
    return settings.finder_dir / sid


def thumb_path(img: dict[str, Any]) -> Path:
    return session_dir(img["session_id"]) / "thumbs" / f"{img['id']}.webp"


def _owned(img: dict[str, Any]) -> bool:
    """檔案是不是這個篩選自己的（刪除時才可以刪檔；伺服器匯入資料夾的原檔絕不刪）。"""
    try:
        return Path(img["path"]).resolve().is_relative_to(session_dir(img["session_id"]).resolve())
    except OSError:
        return False


def is_remote(img: dict[str, Any]) -> bool:
    return str(img["path"]).startswith("smb://")


def read_bytes(path: str) -> bytes:
    """圖片內容：本機檔案或 SMB（smb://<連線 id>/<路徑>，用到時才讀）。"""
    from . import smb

    remote = smb.parse_ref(path)
    return smb.read_bytes(*remote) if remote else Path(path).read_bytes()


def _open(path: str | Path) -> Image.Image:
    im = Image.open(io.BytesIO(read_bytes(str(path)))) if str(path).startswith("smb://") else Image.open(path)
    if getattr(im, "is_animated", False):
        im.seek(0)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGB", im.size, (255, 255, 255))  # 透明背景補白，與 CCIP 原本的前處理相同
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").getchannel("A"))
        return bg
    return im.convert("RGB")


def ensure_thumb(img: dict[str, Any]) -> Path:
    dest = thumb_path(img)
    if not dest.exists():
        im = _open(img["path"])
        db.finder_image_update(img["id"], width=im.width, height=im.height)
        storage.make_thumb(im, dest)
    return dest


# ------------------------------------------------------------------ 篩選
def require_session(sid: str) -> dict[str, Any]:
    s = db.finder_session_get(sid)
    if s is None:
        raise NotFound(t("msg.finder_not_found", sid=sid))
    return s


def default_threshold(model: str) -> float:
    return THRESHOLDS[model]


def get_image(fid: str) -> dict[str, Any]:
    img = db.finder_image_get(fid)
    if img is None:
        raise NotFound(t("msg.image_not_found", iid=fid))
    return img


def session_out(s: dict[str, Any], with_images: bool = False) -> dict[str, Any]:
    from . import jobs

    out = {k: s[k] for k in ("id", "name", "model", "threshold", "created_at", "updated_at")}
    out["default_threshold"] = default_threshold(s["model"])
    out["tag_filter"] = tag_filter(s)
    for k in ("ref_count", "pool_count", "cover_id"):
        if k in s:
            out[k] = s[k]
    job = next((j for j in jobs.list_for_project(s["id"]) if j.kind in ("finder", "smb_import")
                and j.status in ("queued", "running")), None)
    out["active_job"] = job.to_dict() if job else None
    if with_images:
        model = MODELS[s["model"]]
        out["images"] = [image_out(i, model) for i in db.finder_images(s["id"])]
    return out


def image_out(img: dict[str, Any], model: str) -> dict[str, Any]:
    current = img["feature_model"] == model
    return {
        "id": img["id"], "role": img["role"], "original_name": img["original_name"], "rel_path": img["rel_path"],
        "width": img["width"], "height": img["height"], "error": img["error"],
        "score": img["score"] if current else None,  # 和參考圖差異的中位數（參考圖：和其他參考圖的差異）
        "neg_score": img["neg_score"] if current else None,  # 和最像的排除參考圖的差異
        "scored": current and img["score"] is not None,
        "has_feature": current and not img["error"],
        "manual": img["manual"],  # 1 = 手動加入、-1 = 手動排除
        "tags": _tag_list(img),  # WD14 tag（角色 tag 在前），None = 還沒產生
        "tags_char": len((_tag_data(img) or {}).get("character", [])),  # 前幾個是角色 tag
        "thumb_url": f"/api/finder/images/{img['id']}/thumb",
        "image_url": f"/api/finder/images/{img['id']}/file",
    }


def list_sessions() -> list[dict[str, Any]]:
    return [session_out(s) for s in db.finder_sessions()]


def create_session(name: str, model: str = "default") -> dict[str, Any]:
    if model not in MODELS:
        raise BadRequest(t("msg.finder_bad_model", models=", ".join(MODELS)))
    name = (name or "").strip() or t("msg.untitled_project")
    return session_out(db.finder_session_create(name, model))


def norm_tag(tag: str) -> str:
    """tag 統一成 WD14 的寫法：小寫、空白換成底線。"""
    return "_".join(str(tag).strip().lower().split())


def tag_filter(s: dict[str, Any]) -> dict[str, list[str]]:
    try:
        f = json.loads(s.get("tag_filter") or "{}")
    except ValueError:
        f = {}
    return {"include": list(f.get("include") or []), "exclude": list(f.get("exclude") or [])}


def _tag_data(img: dict[str, Any]) -> dict[str, list[str]] | None:
    if not img.get("tags"):
        return None
    try:
        return json.loads(img["tags"])
    except ValueError:
        return None


def _tag_list(img: dict[str, Any]) -> list[str] | None:
    d = _tag_data(img)
    return None if d is None else d.get("character", []) + d.get("general", [])


def update_session(sid: str, name: str | None = None, model: str | None = None,
                   threshold: float | None = None, reset_threshold: bool = False,
                   tags: dict[str, list[str]] | None = None) -> dict[str, Any]:
    s = require_session(sid)
    fields: dict[str, Any] = {}
    if name is not None and name.strip():
        fields["name"] = name.strip()
    if model is not None and model != s["model"]:
        if model not in MODELS:
            raise BadRequest(t("msg.finder_bad_model", models=", ".join(MODELS)))
        fields["model"], fields["threshold"] = model, None  # 各模型的門檻不同，換模型就回到預設
    if threshold is not None:
        fields["threshold"] = float(threshold)
    if reset_threshold:
        fields["threshold"] = None
    if tags is not None:
        clean = {k: list(dict.fromkeys(t for t in map(norm_tag, tags.get(k) or []) if t)) for k in ("include", "exclude")}
        clean["include"] = [t for t in clean["include"] if t not in clean["exclude"]]
        fields["tag_filter"] = json.dumps(clean, ensure_ascii=False)
    if fields:
        db.finder_session_update(sid, **fields)
    # 圖片清單只有換模型（分數全部失效）時才回傳：5 萬張的篩選整包有幾十 MB，調門檻、改 tag 篩選不必重抓
    return session_out(require_session(sid), with_images="model" in fields)


def delete_session(sid: str) -> None:
    from . import jobs

    require_session(sid)
    for j in jobs.list_for_project(sid):
        jobs.cancel(j.id)
    db.finder_session_delete(sid)
    shutil.rmtree(session_dir(sid), ignore_errors=True)


def remove_images(sid: str, ids: list[str] | None = None, role: str | None = None) -> int:
    require_session(sid)
    removed = db.finder_images_delete(sid, ids=ids, role=role)
    if any(i["role"] != "pool" for i in removed):
        db.finder_clear_scores(sid)
    for img in removed:
        if _owned(img):
            Path(img["path"]).unlink(missing_ok=True)
        thumb_path(img).unlink(missing_ok=True)
    db.finder_session_touch(sid)
    return len(removed)


# ------------------------------------------------------------------ 加入圖片
def _check_role(role: str) -> None:
    if role not in ROLES:
        raise BadRequest(t("msg.finder_bad_role", roles=", ".join(ROLES)))


def after_add(sid: str, role: str, added: int) -> None:
    """加入圖片後：參考圖或排除參考圖變了，之前的分數就不能用了。"""
    if added and role != "pool":
        db.finder_clear_scores(sid)
    db.finder_session_touch(sid)


def _is_image(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return p.suffix.lower() in storage.IMAGE_EXTS and not any(
        part.startswith(".") or part == "__MACOSX" for part in p.parts)


def add_bytes(sid: str, role: str, name: str, data: bytes, rel_path: str = "") -> str | None:
    """存一張上傳的圖片；同一組裡已經有一樣的圖片就略過（回傳 None）。"""
    sha1 = hashlib.sha1(data).hexdigest()
    if db.finder_has(sid, role, sha1=sha1):
        return None
    iid = db.new_id()
    dest = session_dir(sid) / f"{iid}{PurePosixPath(name).suffix.lower()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return db.finder_image_add(sid, role, PurePosixPath(name.replace("\\", "/")).name, str(dest),
                               rel_path=rel_path or name, sha1=sha1, iid=iid)


def add_uploads(sid: str, role: str, entries: Iterable[tuple[str, Callable[[], bytes]]],
                zips: Iterable[tuple[str, Any]] = ()) -> dict[str, Any]:
    """上傳的檔案與 zip（zip 從暫存檔逐一讀取）。"""
    require_session(sid)
    _check_role(role)
    added, skipped = 0, []

    def one(name: str, read: Callable[[], bytes]) -> None:
        nonlocal added
        if not _is_image(name):
            return
        try:
            if add_bytes(sid, role, name, read(), rel_path=name):
                added += 1
            else:
                skipped.append({"file": name, "reason": t("msg.duplicate_image")})
        except OSError as e:
            skipped.append({"file": name, "reason": t("msg.read_failed", error=e)})

    for name, read in entries:
        one(name, read)
    for zname, fileobj in zips:
        try:
            with zipfile.ZipFile(fileobj) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        one(info.filename, functools.partial(zf.read, info))
        except zipfile.BadZipFile:
            skipped.append({"file": zname, "reason": t("msg.zip_corrupt")})
    after_add(sid, role, added)
    return {"added": added, "skipped": skipped}


def add_server_folder(sid: str, role: str, subpath: str = "", recursive: bool = True) -> dict[str, Any]:
    """伺服器匯入資料夾：只記路徑，不複製（大量圖片不多佔空間）。"""
    require_session(sid)
    _check_role(role)
    try:
        root = storage.safe_import_path(subpath)
    except ValueError as e:
        raise BadRequest(str(e)) from e
    files = root.rglob("*") if recursive else root.iterdir()
    added, skipped = 0, []
    for p in sorted(files):
        rel = str(p.relative_to(root))
        if not p.is_file() or not _is_image(rel):
            continue
        if db.finder_has(sid, role, path=str(p)):
            skipped.append({"file": rel, "reason": t("msg.duplicate_image")})
            continue
        db.finder_image_add(sid, role, p.name, str(p), rel_path=str(Path(subpath) / rel) if subpath else rel)
        added += 1
    after_add(sid, role, added)
    return {"added": added, "skipped": skipped}


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)  # 同一個磁碟：不多佔空間，專案之後刪掉圖片也不影響
    except OSError:
        shutil.copy2(src, dest)


def add_from_project(sid: str, role: str, project_id: str, ids: list[str] | None = None) -> dict[str, Any]:
    require_session(sid)
    _check_role(role)
    if db.get_project(project_id) is None:
        raise NotFound(t("msg.project_not_found", pid=project_id))
    added, skipped = 0, []
    for img in db.list_images(project_id, ids=ids):
        if img["sha1"] and db.finder_has(sid, role, sha1=img["sha1"]):
            skipped.append({"file": img["original_name"], "reason": t("msg.duplicate_image")})
            continue
        src = storage.image_path(img)
        if not src.exists():
            skipped.append({"file": img["original_name"], "reason": t("msg.read_failed", error="missing")})
            continue
        iid = db.new_id()
        dest = session_dir(sid) / f"{iid}{src.suffix}"
        _link_or_copy(src, dest)
        db.finder_image_add(sid, role, img["original_name"], str(dest), rel_path=img["rel_path"] or img["original_name"],
                            sha1=img["sha1"], iid=iid)
        if storage.thumb_path(img).exists():  # 專案已有縮圖，直接沿用
            _link_or_copy(storage.thumb_path(img), session_dir(sid) / "thumbs" / f"{iid}.webp")
            db.finder_image_update(iid, width=img["width"], height=img["height"])
        added += 1
    after_add(sid, role, added)
    return {"added": added, "skipped": skipped}


# ------------------------------------------------------------------ CCIP
_lock = threading.Lock()
_sessions: dict[str, Any] = {}


def _ort(name: str) -> Any:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    from .tagging.wd14 import ort_providers

    with _lock:
        sess = _sessions.get(name)
        if sess is None:
            path = hf_hub_download(REPO, name)
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3
            providers = [(p, {"arena_extend_strategy": "kSameAsRequested"}) if p == "CUDAExecutionProvider" else p
                         for p in ort_providers()]
            sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
            _sessions[name] = sess
        return sess


def unload_all() -> list[str]:
    """釋放 CCIP 模型（GPU 版會釋放 VRAM）。辨識工作結束時會自動呼叫。"""
    with _lock:
        names = list(_sessions)
        _sessions.clear()
    gc.collect()
    if names:
        log.info("已釋放 CCIP 模型：%s", ", ".join(names))
    return names


def loaded_models() -> list[dict[str, Any]]:
    return [{"name": k, "providers": v.get_providers()} for k, v in list(_sessions.items())]


def _signature(im: Image.Image) -> tuple[str, bytes]:
    """找重複用的指紋：64 位元感知雜湊（DCT）與 32×32 的小圖。"""
    g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    c = (_DCT @ g @ _DCT.T)[:8, :8].ravel()
    bits = c > np.median(c)
    phash = int("".join("1" if b else "0" for b in bits), 2)
    return f"{phash:016x}", np.asarray(im.resize((32, 32), Image.BOX), dtype=np.uint8).tobytes()


def _load(img: dict[str, Any]) -> tuple[Image.Image | None, str | None]:
    """讀圖（順便補縮圖、尺寸與找重複用的指紋）。"""
    try:
        im = _open(img["path"])
        if not thumb_path(img).exists():
            storage.make_thumb(im, thumb_path(img))
        fields: dict[str, Any] = {}
        if not img["width"]:
            fields.update(width=im.width, height=im.height)
        if not img.get("phash"):
            fields["phash"], fields["sig"] = _signature(im)
        if fields:
            db.finder_image_update(img["id"], **fields)
        return im, None
    except Exception as e:  # noqa: BLE001
        return None, t("msg.cannot_read_image", error=e)


def _ccip_input(im: Image.Image) -> np.ndarray:
    x = np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - MEAN) / STD


def _differences(features: np.ndarray, model: str) -> np.ndarray:
    return _ort(f"{model}/model_metrics.onnx").run(["output"], {"input": features.astype(np.float32)})[0]


def run_job(job: Any) -> tuple[str, dict[str, Any]] | None:
    """背景工作：算還沒算過的特徵（可順便產生 WD14 tag），再和參考圖比對（由 jobs 的 worker 呼叫，與標註、放大輪流使用 GPU）。"""
    if job.params.get("task") == "hash":
        return _run_hash(job)
    from .tagging import wd14

    sid = job.project_id
    s = require_session(sid)
    model = MODELS[s["model"]]
    imgs = db.finder_images(sid)
    need_feat = {i["id"] for i in imgs if i["feature_model"] != model}
    need_tags = {i["id"] for i in imgs if job.params.get("tags") and i["role"] == "pool" and i["tags"] is None
                 and not (i["error"] and i["feature_model"] == model)}
    todo = [i for i in imgs if i["id"] in need_feat or i["id"] in need_tags]
    job.image_ids = [i["id"] for i in todo]
    if job.params.get("free_vram"):
        job.say("msg.finder_freeing_vram")
        log.info("辨識前釋放 VRAM：%s", ", ".join(free_other_vram()) or "（沒有載入的模型）")
    job.say("msg.finder_loading_model")
    try:
        feat = _ort(f"{model}/model_feat.onnx")
        _ort(f"{model}/model_metrics.onnx")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(t("msg.finder_download_failed", error=e)) from e
    tagger, own_tagger = None, False
    if need_tags:
        repo = settings.wd14_default_model
        own_tagger = repo not in {m["repo_id"] for m in wd14.loaded_models()}  # 原本沒載入：用完就釋放
        try:
            tagger = wd14.get_tagger(repo)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(t("msg.finder_tagger_failed", error=e)) from e

    def work(img: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        """在執行緒裡讀圖並做好 CCIP / WD14 的前處理，主執行緒只負責送進 GPU。"""
        im, err = _load(img)
        if im is None:
            return None, None, err
        return (_ccip_input(im) if img["id"] in need_feat else None,
                tagger.prepare_fast(im) if img["id"] in need_tags else None, None)

    batches = deque(todo[k:k + BATCH] for k in range(0, len(todo), BATCH))
    pending: deque[tuple[list[dict[str, Any]], list[Future]]] = deque()
    try:
        job.say("msg.finder_extracting_tags" if need_tags and need_feat else
                "msg.finder_tagging" if need_tags else "msg.finder_extracting")
        with ThreadPoolExecutor(WORKERS) as pool:
            def refill() -> None:
                while batches and len(pending) < PREFETCH:
                    b = batches.popleft()
                    pending.append((b, [pool.submit(work, i) for i in b]))

            refill()
            while pending:
                if job.cancel_event.is_set():
                    for _, futs in pending:
                        for f in futs:
                            f.cancel()
                    return None
                batch, futs = pending.popleft()
                refill()  # 這一批送進 GPU 的同時，後面的繼續讀
                results = [f.result() for f in futs]
                for img, (_, _, err) in zip(batch, results):
                    if err:
                        db.finder_image_update(img["id"], error=err, feature=None, feature_model=model, score=None,
                                               neg_score=None)
                        job.failed += 1
                        job.errors.append({"image_id": img["id"], "file": img["original_name"], "error": err})
                fe = [(img, x) for img, (x, _, _) in zip(batch, results) if x is not None]
                if fe:
                    out = feat.run(["output"], {"input": np.stack([x for _, x in fe])})[0]
                    for (img, _), f in zip(fe, out):
                        db.finder_image_update(img["id"], feature=f.astype(np.float32).tobytes(), feature_model=model,
                                               error=None)
                tg = [(img, w) for img, (_, w, _) in zip(batch, results) if w is not None]
                if tg:
                    for (img, _), r in zip(tg, tagger.predict_arrays([w for _, w in tg])):
                        tags = {"character": [n for n, p in r["character"] if p >= TAG_CHARACTER],
                                "general": [n for n, p in r["general"] if p >= TAG_GENERAL]}
                        db.finder_image_update(img["id"], tags=json.dumps(tags, ensure_ascii=False))
                job.done += len(batch)
    finally:
        if own_tagger:
            wd14.unload(settings.wd14_default_model)
    job.say("msg.finder_comparing")
    _score(sid, model)
    db.finder_session_touch(sid)
    return "msg.finder_done", {"n": len(matches(require_session(sid)))}


def _run_hash(job: Any) -> tuple[str, dict[str, Any]] | None:
    """背景工作：算找重複用的指紋（只用 CPU，不載入模型）。辨識過的圖片已經有指紋。"""
    sid, role = job.project_id, job.params.get("role", "pool")
    todo = [i for i in db.finder_images(sid, role=role) if not i["phash"]]
    job.image_ids = [i["id"] for i in todo]
    job.say("msg.finder_hashing")
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(todo), WORKERS * 8):
            if job.cancel_event.is_set():
                return None
            batch = todo[start:start + WORKERS * 8]
            for img, err in zip(batch, pool.map(lambda i: _load(i)[1], batch)):  # 只要指紋，不留解碼後的圖
                if err:
                    db.finder_image_update(img["id"], error=err)
                    job.failed += 1
                    job.errors.append({"image_id": img["id"], "file": img["original_name"], "error": err})
            job.done += len(batch)
    return "msg.finder_hash_done", {"n": len(todo) - job.failed}


def _score(sid: str, model: str) -> None:
    """分數：和參考圖差異的中位數；參考圖本身算和其他參考圖的差異（看參考圖有沒有放錯）。"""
    imgs = [i for i in db.finder_images(sid, with_feature=True) if i["feature_model"] == model and i["feature"]]
    vec = lambda i: np.frombuffer(i["feature"], dtype=np.float32)  # noqa: E731
    refs = [i for i in imgs if i["role"] == "ref"]
    negs = [i for i in imgs if i["role"] == "neg"]
    pool = [i for i in imgs if i["role"] == "pool"]
    if not refs:
        for i in pool:
            db.finder_image_update(i["id"], score=None, neg_score=None)
        return
    head = np.stack([vec(i) for i in refs + negs])
    k, n = len(refs), len(negs)
    d = _differences(head, model)
    for j, r in enumerate(refs):
        others = [d[j, m] for m in range(k) if m != j]
        db.finder_image_update(r["id"], score=float(np.median(others)) if others else None, neg_score=None)
    for start in range(0, len(pool), CHUNK):
        chunk = pool[start:start + CHUNK]
        d = _differences(np.concatenate([head, np.stack([vec(i) for i in chunk])]), model)[k + n:]
        for row, img in zip(d, chunk):
            db.finder_image_update(img["id"], score=float(np.median(row[:k])),
                                   neg_score=float(row[k:k + n].min()) if n else None)


def matches(s: dict[str, Any], imgs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """目前門檻與 tag 篩選下符合的圖片（依差異由小到大）。
    手動 ✓（是目標）只蓋過分數，tag 篩選仍然套用；✕ 一定不符合；「忽略 tag 保留」（WD14 標錯時）一定符合。"""
    thr = s["threshold"] if s["threshold"] is not None else default_threshold(s["model"])
    model = MODELS[s["model"]]
    imgs = imgs if imgs is not None else db.finder_images(s["id"], role="pool")

    f = tag_filter(s)
    inc, exc = set(f["include"]), set(f["exclude"])

    def tag_ok(i: dict[str, Any]) -> bool:
        if not (inc or exc):
            return True
        tags = _tag_list(i)
        if tags is None:  # 有設定篩選時，還沒產生 tag 的圖片無法確認，先不算符合（手動 ✓ 仍然算）
            return False
        have = set(tags)
        return inc <= have and not (exc & have)

    def ok(i: dict[str, Any]) -> bool:
        if i["role"] != "pool" or i["manual"] == -1:
            return False
        if i["manual"] == KEEP:
            return True
        scored = i["feature_model"] == model and i["score"] is not None
        target = i["manual"] == 1 or (scored and i["score"] <= thr
                                      and (i["neg_score"] is None or i["score"] < i["neg_score"]))
        return target and tag_ok(i)

    return sorted((i for i in imgs if ok(i)), key=lambda i: (i["score"] is None, i["score"] or 0))


MANUAL = {"include": 1, "exclude": -1, "keep": KEEP, "clear": None}


def set_manual(sid: str, ids: list[str], action: str) -> int:
    """手動判定：include = 是目標（模型漏掉；tag 篩選仍然套用）、exclude = 不是目標（模型挑錯）、
    keep = 被 tag 篩掉但要保留（WD14 標錯）、clear = 回到依分數。"""
    require_session(sid)
    if action not in MANUAL:
        raise BadRequest(t("msg.finder_bad_manual", actions=", ".join(MANUAL)))
    n = db.finder_set_manual(sid, ids, MANUAL[action])
    db.finder_session_touch(sid)
    return n


def free_other_vram() -> list[str]:
    """把 VRAM 讓給 CCIP：釋放 WD14、waifu2x，讓 vLLM（JoyCaption）休眠。做不到的略過。"""
    from . import upscale
    from .tagging import vlm
    from .tagging.wd14 import unload_all

    done = [f"WD14 {n}" for n in unload_all()] + [f"waifu2x {n}" for n in upscale.unload_all()]
    try:
        st = vlm.sleep_status()
        if st.get("supported") and st.get("online") and not st.get("sleeping"):
            vlm.sleep()
            done.append("VLM sleep")
    except Exception as e:  # noqa: BLE001  VLM 沒開、不支援休眠：不影響辨識
        log.warning("辨識前讓 VLM 休眠失敗：%s", e)
    return done


def start(sid: str, free_vram: bool = False, tags: bool = False) -> dict[str, Any]:
    """開始辨識；tags = 順便用 WD14 幫要篩選的圖片產生 tag（給結果的 tag 篩選用）。"""
    from . import jobs

    require_session(sid)
    if not db.finder_images(sid, role="ref"):
        raise BadRequest(t("msg.finder_need_refs"))
    if not db.finder_images(sid, role="pool"):
        raise BadRequest(t("msg.finder_need_pool"))
    model = MODELS[require_session(sid)["model"]]
    todo = [i["id"] for i in db.finder_images(sid)
            if i["feature_model"] != model or (tags and i["role"] == "pool" and i["tags"] is None)]
    return jobs.submit_task(sid, "finder", todo, params={"free_vram": free_vram, "tags": tags}).to_dict()


# ------------------------------------------------------------------ 找重複
def start_hash(sid: str, role: str = "pool") -> dict[str, Any] | None:
    """還沒有指紋的圖片先讀一次（背景工作）；都有了回傳 None。"""
    from . import jobs

    require_session(sid)
    if role not in ROLES:
        raise BadRequest(t("msg.finder_bad_role", roles=", ".join(ROLES)))
    todo = [i["id"] for i in db.finder_images(sid, role=role) if not i["phash"]]
    if not todo:
        return None
    return jobs.submit_task(sid, "finder", todo, params={"task": "hash", "role": role}).to_dict()


def _keep_rank(img: dict[str, Any], matched: set[str]) -> tuple:
    """每組保留哪一張：手動加入的 > 目前符合的 > 解析度高 > 無損格式（PNG）> 先加入的。"""
    lossless = PurePosixPath(img["original_name"]).suffix.lower() in (".png", ".bmp", ".tif", ".tiff")
    return (img["manual"] not in (1, KEEP), img["id"] not in matched, -((img["width"] or 0) * (img["height"] or 0)),
            not lossless, img["created_at"])


_dup_cache: dict[tuple[str, str], dict[str, Any]] = {}
_dup_lock = threading.Lock()


def _dup_pairs(sid: str, role: str, ids: list[str]) -> dict[str, Any]:
    """可能重複的配對與平均差異（只留 ≤ DUP_MAX 的），調整門檻時不必重算。
    暫存起來：只是少了圖片（移除重複之後）時沿用，有新的圖片才重算（5 萬張約幾秒）。"""
    with _dup_lock:
        c = _dup_cache.get((sid, role))
        if c is not None and c["index"].keys() >= set(ids):
            return c
        rows = db.finder_dup_sigs(sid, role)
        n = len(rows)
        hashes = np.array([int(r["phash"], 16) for r in rows], dtype=np.uint64)
        small = np.frombuffer(b"".join(r["sig"] for r in rows), dtype=np.uint8).reshape(n, -1)
        index = {r["id"]: k for k, r in enumerate(rows)}
        del rows
        pairs_a, pairs_b, pairs_d = [], [], []
        step = max(16, 8_000_000 // max(n, 1))  # 每次比 step×n 個雜湊，暫存陣列不超過約 64 MB
        for start in range(0, n, step):
            d = np.bitwise_count(hashes[start:start + step, None] ^ hashes[None, start:])  # 只比右上三角
            a, b = np.nonzero(d <= DUP_PREFILTER)
            a, b = a + start, b + start
            a, b = a[b > a], b[b > a]
            for k in range(0, len(a), 4096):  # 候選配對再比小圖
                pa, pb = a[k:k + 4096], b[k:k + 4096]
                sa, sb = small[pa], small[pb]
                diff = np.maximum(sa, sb)
                diff -= np.minimum(sa, sb)  # uint8 的 |a − b|
                mean = diff.sum(axis=1, dtype=np.uint32) / small.shape[1]
                ok = mean <= DUP_MAX
                pairs_a.append(pa[ok].astype(np.int32))
                pairs_b.append(pb[ok].astype(np.int32))
                pairs_d.append(mean[ok].astype(np.float32))
        cat = lambda x, t: np.concatenate(x) if x else np.empty(0, t)  # noqa: E731
        c = {"index": index, "a": cat(pairs_a, np.int32), "b": cat(pairs_b, np.int32), "d": cat(pairs_d, np.float32)}
        _dup_cache.pop((sid, role), None)
        _dup_cache[(sid, role)] = c
        while len(_dup_cache) > 4:  # 只留最近用的幾個篩選
            _dup_cache.pop(next(iter(_dup_cache)))
        return c


def duplicates(sid: str, role: str = "pool", threshold: float = DUP_DEFAULT, scope: str = "all") -> dict[str, Any]:
    """找出幾乎相同的圖片分組。每組第一張是建議保留的，其他的都和它在門檻以內（平均差異，0–255）。
    scope = matches：只在目前符合的結果裡找。"""
    s = require_session(sid)
    if role not in ROLES:
        raise BadRequest(t("msg.finder_bad_role", roles=", ".join(ROLES)))
    if not 0 < threshold <= DUP_MAX:
        raise BadRequest(t("msg.finder_bad_dup_threshold", max=DUP_MAX))
    if scope not in ("all", "matches"):
        raise BadRequest(t("msg.finder_bad_dup_scope"))
    from . import jobs

    imgs = db.finder_dup_rows(sid, role)
    matched = {i["id"] for i in matches(s)} if role == "pool" else set()
    if scope == "matches":
        imgs = [i for i in imgs if i["id"] in matched]
    missing = sum(1 for i in imgs if not i["phash"] and not i["error"])
    have = [i for i in imgs if i["phash"] and i["has_sig"]]
    busy = any(j.kind == "finder" and j.status in ("queued", "running") for j in jobs.list_for_project(sid))
    out = {"role": role, "scope": scope, "threshold": threshold, "total": len(imgs), "missing": missing, "busy": busy,
           "groups": [], "removable": 0, "items": {}}
    if len(have) < 2:
        return out
    c = _dup_pairs(sid, role, [i["id"] for i in have])
    by_k = {c["index"][i["id"]]: i for i in have}
    present = np.zeros(len(c["index"]), dtype=bool)
    present[list(by_k)] = True
    sel = (c["d"] <= threshold) & present[c["a"]] & present[c["b"]]
    near: dict[int, list[int]] = {}
    for x, y in zip(c["a"][sel].tolist(), c["b"][sel].tolist()):
        near.setdefault(x, []).append(y)
        near.setdefault(y, []).append(x)
    # 依保留順序分組：最該留的當代表，只收和代表本身夠像的（不會一路串成整段鏡頭）
    taken: set[int] = set()
    for img in sorted(have, key=lambda i: _keep_rank(i, matched)):
        k = c["index"][img["id"]]
        if k in taken or k not in near:
            continue
        members = sorted(j for j in near[k] if j not in taken)  # 依加入順序（通常就是畫面順序）
        if not members:
            continue
        taken.update([k, *members])
        out["groups"].append({"keep": img["id"], "ids": [img["id"], *(by_k[j]["id"] for j in members)]})
        out["removable"] += len(members)
        for j in (k, *members):  # 視窗要顯示的檔名與尺寸（不必再抓整個篩選）
            i = by_k[j]
            out["items"][i["id"]] = {"name": i["original_name"], "rel_path": i["rel_path"], "width": i["width"],
                                     "height": i["height"], "match": i["id"] in matched}
    return out


# ------------------------------------------------------------------ 結果
def _selected(sid: str, ids: list[str] | None) -> list[dict[str, Any]]:
    s = require_session(sid)
    imgs = db.finder_images(sid, role="pool", ids=ids) if ids is not None else matches(s)
    if not imgs:
        raise BadRequest(t("msg.finder_nothing_selected"))
    return imgs


def _unique_names(imgs: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for img in imgs:
        name = img["original_name"]
        n = seen.get(name.lower(), 0)
        seen[name.lower()] = n + 1
        if n:
            stem, ext = os.path.splitext(name)
            name = f"{stem} ({n + 1}){ext}"
        names.append(name)
    return names


def download(sid: str, ids: list[str] | None = None) -> tuple[Path, int]:
    """把選取的圖片（沒指定 = 目前符合的）打包成 zip，放在匯出資料夾。"""
    s = require_session(sid)
    imgs = _selected(sid, ids)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in s["name"]).strip("_") or "finder"
    out = settings.exports_dir / f"{safe}_finder_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:  # 圖片本來就壓縮過
        for img, name in zip(imgs, _unique_names(imgs)):
            try:
                zf.writestr(name, read_bytes(img["path"]))
                count += 1
            except (OSError, BadRequest) as e:  # 檔案被移走、SMB 連不上：略過這張
                log.warning("打包時讀不到 %s：%s", img["path"], e)
    return out, count


def to_project(sid: str, ids: list[str] | None, project_id: str) -> dict[str, Any]:
    """把選取的圖片（沒指定 = 目前符合的）匯入專案。"""
    p = db.get_project(project_id)
    if p is None:
        raise NotFound(t("msg.project_not_found", pid=project_id))
    imgs = _selected(sid, ids)
    def reader(path: str) -> Callable[[], bytes]:
        def read() -> bytes:
            try:
                return read_bytes(path)
            except BadRequest as e:  # SMB 連不上：這張略過，不中斷整批
                raise OSError(str(e)) from e
        return read

    entries = [(name, reader(img["path"])) for img, name in zip(imgs, _unique_names(imgs))]
    result = storage.import_entries(project_id, entries, trigger=p["settings"].get("trigger", ""))
    result.pop("added_ids", None)
    return result
