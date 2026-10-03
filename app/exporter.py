"""匯出訓練資料集：Civitai (圖片 + 同名 .txt)、kohya 資料夾、metadata.jsonl。"""
from __future__ import annotations

import io
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from . import db, storage
from .config import settings
from .pipeline import caption_for
from .i18n import t, tr
from .profiles import lora_type_name, profile_view, normalize_settings
from .tagging.postprocess import policy_flag, training_hints

EXPORT_FORMATS = ("civitai", "kohya", "jsonl")


def export_formats_view() -> dict[str, str]:
    return {f: tr(f"export_formats.{f}", default=f) for f in EXPORT_FORMATS}
IMAGE_FORMATS = {"original": None, "png": ("PNG", ".png"), "jpg": ("JPEG", ".jpg"), "webp": ("WEBP", ".webp")}


def slugify(text: str, fallback: str = "") -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9_-]+", "_", t).strip("_")
    return t[:80] or fallback


def convert_image(img: dict[str, Any], image_format: str, max_side: int) -> tuple[bytes, str]:
    src = storage.image_path(img)
    target = IMAGE_FORMATS.get(image_format)
    if target is None and not max_side:
        return src.read_bytes(), src.suffix.lower()
    im = Image.open(src)
    im.load()
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    fmt, ext = target or ({".png": "PNG", ".jpg": "JPEG", ".webp": "WEBP"}[src.suffix.lower()], src.suffix.lower())
    if fmt == "JPEG" and im.mode != "RGB":
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        im = bg
    buf = io.BytesIO()
    im.save(buf, format=fmt, **({"quality": 95} if fmt in ("JPEG", "WEBP") else {}))
    return buf.getvalue(), ext


def _training_readme(project: dict[str, Any], s: dict[str, Any], n: int, fmt: str, repeats: int) -> str:
    p = profile_view(s["profile"])
    g = p["guide"]
    r = lambda k, **kw: t(f"readme.{k}", **kw)  # noqa: E731
    hints = training_hints(s)
    on_off = lambda v: r("on") if v else r("off")  # noqa: E731
    mode_name = tr(f"caption_modes.{s['caption_mode']}", default=s["caption_mode"])
    lines = [
        f"# {r('title', name=project['name'])}",
        "",
        f"- {r('images', n=n)}",
        f"- {r('base_model', name=p['name'], civitai=p['civitai_base'])}",
        f"- {r('lora_type', value=lora_type_name(s['lora_type']))}",
        f"- {r('trigger', value=s['trigger'] or r('trigger_unset'))}",
        f"- {r('caption_mode', value=mode_name)}",
        "",
        f"## {r('training')}",
    ]
    lines += [f"- {k}: {v}" for k, v in g["training"].items()]
    lines.append(f"- {r('hints', shuffle=on_off(hints['shuffle_caption']), keep=hints['keep_tokens'])}")
    if fmt == "kohya":
        lines += ["", f"## {r('kohya_title')}", f"- {r('kohya_repeats', repeats=repeats)}"]
        if hints["shuffle_caption"]:  # 不打亂時 keep_tokens 沒有作用，就不提示
            args = "--shuffle_caption --keep_tokens={}".format(hints["keep_tokens"])
            lines.append(f"- {r('kohya_args', args=args)}")
        lines.append(f"- {r('kohya_no_shuffle')}")
    negative = g["a1111"]["negative"] or g["a1111"]["negative_note"]
    lines += [
        "",
        f"## {r('a1111_title')}",
        f"- Positive: {g['a1111']['positive']}",
        f"- Negative: {negative}",
        f"- {r('settings')}: {g['a1111']['settings']}",
    ]
    return "\n".join(lines) + "\n"


def cleanup_exports(max_age_s: int = 6 * 3600) -> None:
    now = time.time()
    for f in settings.exports_dir.glob("*.zip"):
        if now - f.stat().st_mtime > max_age_s:
            f.unlink(missing_ok=True)


def dataset_images(project_id: str, only_done: bool = False) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]],
                                                                     list[dict[str, str]]]:
    """要放進訓練資料集的圖片：回傳 (專案, 設定, 圖片, 被排除的圖片)。

    依 Civitai 政策，「未成年特徵 + 性內容」的圖片一律排除（匯出與雲端訓練共用這個規則）。
    """
    project = db.get_project(project_id)
    if project is None:
        raise KeyError(t("msg.project_missing"))
    s = normalize_settings(project["settings"])
    images = db.list_images(project_id)
    if only_done:
        images = [i for i in images if i["status"] == "done"]
    excluded = []
    for img in list(images):
        reason = policy_flag(img["tags"], img.get("rating"), img.get("nl_caption") or "")
        if reason:
            excluded.append({"image_id": img["id"], "file": img["original_name"], "reason": reason})
            images.remove(img)
    if not images:
        raise ValueError(t("msg.nothing_to_export"))
    return project, s, images, excluded


def export_dataset(
    project_id: str,
    fmt: str = "civitai",
    image_format: str = "original",
    max_side: int = 0,
    repeats: int = 10,
    naming: str = "original",
    only_done: bool = False,
) -> tuple[Path, list[dict[str, str]]]:
    """建立 zip，回傳 (zip 路徑, 被排除的圖片清單)。"""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(t("msg.unknown_export_format", fmt=fmt))
    if image_format not in IMAGE_FORMATS:
        raise ValueError(t("msg.unknown_image_format", fmt=image_format))
    project, s, images, excluded = dataset_images(project_id, only_done)

    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    cleanup_exports()
    zip_name = f"{slugify(project['name'], project_id)}_{fmt}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    out = settings.exports_dir / zip_name

    trigger = s["trigger"] or slugify(project["name"], "lora")
    class_word = s["class_word"] or {"character": "person", "style": "style", "concept": "concept"}[s["lora_type"]]
    folder = {"civitai": "", "kohya": f"img/{max(1, repeats)}_{trigger} {class_word}/", "jsonl": "train/"}[fmt]

    used: set[str] = set()
    meta_lines = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for n, img in enumerate(images, 1):
            if naming == "sequential":
                stem = f"{slugify(trigger, 'img')}_{n:04d}"
            else:
                stem = slugify(Path(img["original_name"]).stem, f"img_{n:04d}")
            base, k = stem, 2
            while stem.lower() in used:
                stem, k = f"{base}_{k}", k + 1
            used.add(stem.lower())

            data, ext = convert_image(img, image_format, max_side)
            caption = caption_for(img, s)
            # 圖片已壓縮過，直接存放較快
            zf.writestr(zipfile.ZipInfo(f"{folder}{stem}{ext}", date_time=time.localtime()[:6]), data,
                        compress_type=zipfile.ZIP_STORED)
            if fmt == "jsonl":
                meta_lines.append(json.dumps({"file_name": f"{stem}{ext}", "text": caption}, ensure_ascii=False))
            else:
                zf.writestr(f"{folder}{stem}.txt", caption + "\n")
        if fmt == "jsonl":
            zf.writestr(f"{folder}metadata.jsonl", "\n".join(meta_lines) + "\n")
        if fmt != "civitai":
            # Civitai 訓練器會把 zip 內的 .txt 當成 caption，因此只在其他格式附上說明
            zf.writestr("README_training.md", _training_readme(project, s, len(images), fmt, repeats))
    return out, excluded
