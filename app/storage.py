"""圖片檔案處理：上傳、資料夾 / zip 匯入、縮圖、讀取。"""
from __future__ import annotations

import functools
import hashlib
import io
import logging
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator

from PIL import Image, ImageOps, UnidentifiedImageError

from . import db
from .config import settings
from .i18n import t
from .tagging.blocks import detect_white_blocks
from .tagging.postprocess import canon, split_tags

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 200_000_000

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".jfif"}
KEEP_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
THUMB_SIZE = 384


def project_dir(pid: str) -> Path:
    return settings.projects_dir / pid


def images_dir(pid: str) -> Path:
    return project_dir(pid) / "images"


def thumbs_dir(pid: str) -> Path:
    return project_dir(pid) / "thumbs"


def originals_dir(pid: str) -> Path:
    """放大（waifu2x）前的原圖備份。"""
    return project_dir(pid) / "originals"


def image_path(img: dict[str, Any]) -> Path:
    return images_dir(img["project_id"]) / img["filename"]


def thumb_path(img: dict[str, Any]) -> Path:
    return thumbs_dir(img["project_id"]) / f"{img['id']}.webp"


def load_image(img: dict[str, Any]) -> Image.Image:
    im = Image.open(image_path(img))
    im.load()
    return im


def parse_caption_file(text: str, trigger: str = "") -> tuple[list[str], str]:
    """把既有 .txt caption 拆成（標籤, 自然語句）。長片段（>5 個字）視為自然語句。"""
    tags, sentences = [], []
    for seg in text.replace("\n", ",").split(","):
        seg = seg.strip()
        if not seg:
            continue
        if len(seg.split()) > 5 or seg.endswith("."):
            sentences.append(seg)
        else:
            tags.append(seg)
    trig = canon(trigger).lower()
    tags = [t for t in split_tags(tags) if t.lower() != trig]
    return tags, ", ".join(sentences)


def make_thumb(im: Image.Image, dest: Path) -> None:
    t = im.copy()
    if t.mode not in ("RGB", "RGBA"):
        t = t.convert("RGBA")
    t.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    t.save(dest, format="WEBP", quality=82)


def save_image_bytes(
    pid: str, data: bytes, original_name: str, rel_path: str = "", caption_text: str | None = None,
    trigger: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """儲存一張圖片。回傳 (圖片紀錄, 略過原因)。"""
    sha1 = hashlib.sha1(data).hexdigest()
    if db.find_by_sha(pid, sha1):
        return None, t("msg.duplicate_image")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError) as e:
        return None, t("msg.cannot_read_image", error=e)

    iid = db.new_id()
    idir = images_dir(pid)
    idir.mkdir(parents=True, exist_ok=True)

    fmt = im.format
    rotated = im.getexif().get(0x0112, 1) not in (1, None)
    if fmt in KEEP_FORMATS and not rotated and not getattr(im, "is_animated", False):
        ext = KEEP_FORMATS[fmt]
        (idir / f"{iid}{ext}").write_bytes(data)
        final = im
    else:
        # 其他格式 / 動畫 / EXIF 旋轉 → 轉存為 PNG（避免訓練時方向錯誤）
        if getattr(im, "is_animated", False):
            im.seek(0)
        final = ImageOps.exif_transpose(im) if rotated else im
        if final.mode not in ("RGB", "RGBA", "L", "LA"):
            final = final.convert("RGBA" if "A" in final.getbands() or "transparency" in final.info else "RGB")
        ext = ".png"
        final.save(idir / f"{iid}{ext}", format="PNG")

    make_thumb(final, thumbs_dir(pid) / f"{iid}.webp")
    try:
        blocks = detect_white_blocks(final)
    except Exception:  # noqa: BLE001 — 偵測失敗不影響匯入，之後可再手動偵測
        log.exception("白色色塊偵測失敗：%s", original_name)
        blocks = None

    tags, nl = parse_caption_file(caption_text, trigger) if caption_text else ([], "")
    rec = db.add_image(
        pid, id=iid, filename=f"{iid}{ext}", original_name=Path(original_name).name, rel_path=rel_path,
        width=final.width, height=final.height, size=len(data), sha1=sha1, tags=tags, nl_caption=nl, blocks=blocks,
    )
    return rec, None


def delete_image_files(img: dict[str, Any]) -> None:
    paths = [image_path(img), thumb_path(img)]
    if img.get("upscale"):
        paths.append(originals_dir(img["project_id"]) / img["upscale"]["original"]["filename"])
    for p in paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def delete_project_files(pid: str) -> None:
    shutil.rmtree(project_dir(pid), ignore_errors=True)


# ------------------------------------------------------------------ 批次匯入
# 匯入項目 = (相對路徑, 讀取函式)。圖片在儲存時才讀取，避免大量圖片同時佔用記憶體。
Entry = tuple[str, Callable[[], bytes]]


def stem_key(path: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    return str(p.with_suffix("")).lower()


def _is_wanted(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/"))
    if any(part.startswith(".") or part == "__MACOSX" for part in name.parts):
        return False
    return name.suffix.lower() in IMAGE_EXTS or name.suffix.lower() in (".txt", ".caption")


def pair_files(entries: Iterable[Entry]) -> Iterator[tuple[str, Callable[[], bytes], str | None]]:
    """把圖片與同名 .txt 配對（caption 很小，先讀進來；圖片保持延遲讀取）。"""
    images: list[Entry] = []
    captions: dict[str, str] = {}
    for path, read in entries:
        ext = PurePosixPath(path).suffix.lower()
        if ext in IMAGE_EXTS:
            images.append((path, read))
        elif ext in (".txt", ".caption"):
            captions[stem_key(path)] = read().decode("utf-8", errors="replace")
    for path, read in images:
        yield path, read, captions.get(stem_key(path))


def import_entries(pid: str, entries: Iterable[Entry], trigger: str = "") -> dict[str, Any]:
    added, skipped = [], []
    for path, read, caption in pair_files(entries):
        try:
            data = read()
        except OSError as e:
            skipped.append({"file": path, "reason": t("msg.read_failed", error=e)})
            continue
        rec, reason = save_image_bytes(pid, data, PurePosixPath(path).name, rel_path=path,
                                       caption_text=caption, trigger=trigger)
        if rec:
            added.append(rec["id"])
        else:
            skipped.append({"file": path, "reason": reason})
    if added:
        db.touch_project(pid)
    return {"added": len(added), "added_ids": added, "skipped": skipped}


def bytes_entry(path: str, data: bytes) -> Entry:
    return path, lambda: data


def import_zip(pid: str, fileobj: Any, trigger: str = "") -> dict[str, Any]:
    with zipfile.ZipFile(fileobj) as zf:
        entries = [(i.filename, functools.partial(zf.read, i)) for i in zf.infolist()
                   if not i.is_dir() and _is_wanted(i.filename)]
        return import_entries(pid, entries, trigger=trigger)


def safe_import_path(subpath: str) -> Path:
    base = settings.import_dir.resolve()
    target = (base / subpath.lstrip("/")).resolve()
    if target != base and base not in target.parents:
        raise ValueError(t("msg.path_outside_import"))
    if not target.is_dir():
        raise ValueError(t("msg.folder_not_found", path=subpath))
    return target


def server_dir_entries(subpath: str, recursive: bool = True) -> list[Entry]:
    root = safe_import_path(subpath)
    files = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (str(p.relative_to(root)), p.read_bytes)
        for p in files
        if p.is_file() and _is_wanted(str(p.relative_to(root)))
    )


def list_import_dirs(max_depth: int = 2) -> list[dict[str, Any]]:
    base = settings.import_dir
    if not base.is_dir():
        return []
    out = []
    for dirpath, dirs, files in os.walk(base):
        rel = Path(dirpath).relative_to(base)
        depth = 0 if str(rel) == "." else len(rel.parts)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if depth >= max_depth:
            dirs[:] = []
        n = sum(1 for f in files if Path(f).suffix.lower() in IMAGE_EXTS)
        out.append({"path": "" if str(rel) == "." else str(rel), "images": n})
    return out
