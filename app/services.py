"""REST API 與 MCP 共用的業務邏輯。"""
from __future__ import annotations

import fnmatch
import io
import ipaddress
import socket
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from PIL import Image

from . import db, jobs, storage
from concurrent.futures import ThreadPoolExecutor

from .config import settings
from .pipeline import block_tags, caption_for, config_problem
from .i18n import t, tr
from .profiles import PROFILES, default_settings, normalize_settings, profile_name
from .tagging.blocks import detect_white_blocks, has_blocks
from .tagging.postprocess import (
    apply_filters,
    count_tokens_estimate,
    matches_blacklist,
    policy_flag,
    select_tags,
    split_tags,
    training_hints,
)


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


# ------------------------------------------------------------------ 序列化
def image_out(img: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    caption = caption_for(img, s)
    return {
        "id": img["id"],
        "project_id": img["project_id"],
        "original_name": img["original_name"],
        "rel_path": img["rel_path"],
        "width": img["width"],
        "height": img["height"],
        "status": img["status"],
        "error": img["error"],
        "tags": img["tags"],
        "nl_caption": img["nl_caption"],
        "rating": img["rating"],
        "caption": caption,
        "token_estimate": count_tokens_estimate(caption),
        "blocks": img.get("blocks"),  # 白色色塊偵測結果（null = 尚未偵測）
        "has_blocks": has_blocks(img.get("blocks")),
        "block_tag_applied": bool(block_tags(img, s)),  # caption 已加上白色色塊關鍵字
        "flag": policy_flag(img["tags"], img.get("rating"), img.get("nl_caption") or ""),
        "image_url": f"/api/images/{img['id']}/file",
        "thumb_url": f"/api/images/{img['id']}/thumb?v={int(img['updated_at'])}",
    }


def project_out(p: dict[str, Any]) -> dict[str, Any]:
    s = normalize_settings(p["settings"])
    out = {
        "id": p["id"],
        "name": p["name"],
        "settings": s,
        "profile_name": profile_name(s["profile"]),
        "training_hints": training_hints(s),
        "created_at": p["created_at"],
        "updated_at": p["updated_at"],
    }
    for k in ("image_count", "done_count", "cover_id"):
        if k in p:
            out[k] = p[k]
    job = jobs.active_for_project(p["id"])
    out["active_job"] = job.to_dict() if job else None
    return out


def absolute_url(path: str) -> str:
    return f"{settings.public_base_url}{path}" if settings.public_base_url else path


# ------------------------------------------------------------------ 專案
def require_project(pid: str) -> dict[str, Any]:
    p = db.get_project(pid)
    if p is None:
        raise NotFound(t("msg.project_not_found", pid=pid))
    return p


def create_project(name: str, profile: str = "illustrious", lora_type: str = "character", trigger: str = "",
                   class_word: str = "", **overrides: Any) -> dict[str, Any]:
    if profile not in PROFILES:
        raise BadRequest(t("msg.unknown_profile", profile=profile, available=", ".join(PROFILES)))
    name = (name or "").strip() or t("msg.untitled_project")
    s = default_settings(profile, lora_type, trigger=trigger.strip(), class_word=class_word.strip(), **overrides)
    return project_out(db.create_project(name, normalize_settings(s)))


def update_project(pid: str, name: str | None = None, patch: dict[str, Any] | None = None,
                   reset_to_profile: bool = False) -> dict[str, Any]:
    p = require_project(pid)
    s = normalize_settings(p["settings"])
    if patch:
        if reset_to_profile and patch.get("profile"):
            # 切換底模：套用新底模的預設值，但保留 trigger 等專案資訊
            keep = {k: s[k] for k in ("trigger", "class_word", "blacklist")}
            keep.update({k: v for k, v in patch.items() if k in keep})
            s = default_settings(patch["profile"], patch.get("lora_type", s["lora_type"]), **keep)
        else:
            s.update(patch)
    db.update_project(pid, name=name.strip() if name else None, project_settings=normalize_settings(s))
    return project_out(require_project(pid))


def delete_project(pid: str) -> None:
    require_project(pid)
    jobs.cancel_project(pid)
    db.delete_project(pid)
    storage.delete_project_files(pid)


def list_images(pid: str, status: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
    p = require_project(pid)
    s = normalize_settings(p["settings"])
    imgs = db.list_images(pid, status=status)
    if search:
        def hit(pat: str, text: str) -> bool:
            return fnmatch.fnmatchcase(text, pat) if "*" in pat else pat in text

        pats = [x.lower() for x in split_tags(search)]
        imgs = [i for i in imgs if all(any(hit(pat, t.lower()) for t in i["tags"])
                                       or hit(pat, (i["nl_caption"] or "").lower())
                                       or hit(pat, i["original_name"].lower()) for pat in pats)]
    return [image_out(i, s) for i in imgs]


def require_image(iid: str, pid: str | None = None) -> dict[str, Any]:
    img = db.get_image(iid)
    if img is None or (pid and img["project_id"] != pid):
        raise NotFound(t("msg.image_not_found", iid=iid))
    return img


def update_image(pid: str, iid: str, tags: Iterable[str] | str | None = None,
                 nl_caption: str | None = None, has_blocks: bool | None = None,
                 clear_blocks_override: bool = False) -> dict[str, Any]:
    p = require_project(pid)
    img = require_image(iid, pid)
    fields: dict[str, Any] = {}
    if has_blocks is not None or clear_blocks_override:  # 手動標記有沒有白色色塊（覆蓋偵測結果）
        fields["blocks"] = {**(img.get("blocks") or {"rects": [], "area": 0}),
                            "override": None if clear_blocks_override else bool(has_blocks)}
    if tags is not None:
        fields["tags"] = split_tags(tags)
    if nl_caption is not None:
        fields["nl_caption"] = nl_caption.strip()
    if fields:
        if (tags is not None or nl_caption is not None) and img["status"] in ("pending", "error"):
            fields["status"], fields["error"] = "done", None
        db.update_image(iid, **fields)
    return image_out(require_image(iid), normalize_settings(p["settings"]))


def scan_blocks(pid: str, force: bool = False) -> dict[str, Any]:
    """偵測白色色塊。預設只偵測還沒偵測過的圖片（新匯入的圖片會自動偵測）；force 重新偵測全部，保留手動標記。"""
    require_project(pid)
    imgs = db.list_images(pid)
    todo = [i for i in imgs if force or i.get("blocks") is None]

    def scan(img: dict[str, Any]) -> None:
        try:
            with Image.open(storage.image_path(img)) as im:
                found = detect_white_blocks(im)
        except OSError:
            return
        if (img.get("blocks") or {}).get("override") is not None:
            found["override"] = img["blocks"]["override"]
        db.update_image(img["id"], blocks=found)
        img["blocks"] = found

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(scan, todo))
    with_blocks = [i["id"] for i in imgs if has_blocks(i.get("blocks"))]
    return {"scanned": len(todo), "total": len(imgs), "with_blocks": len(with_blocks), "ids": with_blocks,
            "unscanned": sum(1 for i in imgs if i.get("blocks") is None)}


def delete_images(pid: str, ids: list[str]) -> int:
    require_project(pid)
    removed = db.delete_images(pid, ids)
    for img in removed:
        storage.delete_image_files(img)
    return len(removed)


# ------------------------------------------------------------------ 匯入
def _check_host(request: httpx.Request) -> None:
    """避免 SSRF：預設禁止下載內網 / 本機位址（ALLOW_PRIVATE_URLS=1 可解除）。"""
    if settings.allow_private_urls:
        return
    host = request.url.host
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise BadRequest(t("msg.cannot_resolve_host", host=host)) from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise BadRequest(t("msg.private_url_blocked", host=host))


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "lora-tag-studio/1.0"},
                        event_hooks={"request": [_check_host]})


def import_urls(pid: str, urls: list[str]) -> dict[str, Any]:
    p = require_project(pid)
    trigger = normalize_settings(p["settings"])["trigger"]
    entries, skipped = [], []
    with _http_client() as client:
        for url in urls[:500]:
            if urlparse(url).scheme not in ("http", "https"):
                skipped.append({"file": url, "reason": t("msg.http_only")})
                continue
            try:
                r = client.get(url)
                r.raise_for_status()
                if len(r.content) > 50 * 1024 * 1024:
                    raise ValueError(t("msg.file_too_large"))
                name = PurePosixPath(urlparse(url).path).name or "image"
                if not PurePosixPath(name).suffix:
                    name += ".png"
                entries.append(storage.bytes_entry(name, r.content))
            except Exception as e:  # noqa: BLE001
                skipped.append({"file": url, "reason": str(e)})
    result = storage.import_entries(pid, entries, trigger=trigger)
    result["skipped"] = skipped + result["skipped"]
    return result


def decode_image(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception as e:  # noqa: BLE001
        raise BadRequest(t("msg.cannot_read_image", error=e)) from e


def fetch_image(url: str) -> Image.Image:
    if urlparse(url).scheme not in ("http", "https"):
        raise BadRequest(t("msg.http_only"))
    try:
        with _http_client() as client:
            r = client.get(url)
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise BadRequest(t("msg.download_failed", error=e)) from e
    return decode_image(r.content)


# ------------------------------------------------------------------ 標註
def start_tagging(pid: str, ids: list[str] | None = None, only_untagged: bool = False) -> dict[str, Any]:
    project = require_project(pid)
    problem = config_problem(normalize_settings(project["settings"]))
    if problem:
        raise BadRequest(problem)
    imgs = db.list_images(pid, ids=ids)
    if only_untagged:
        imgs = [i for i in imgs if i["status"] in ("pending", "error")]
    return jobs.submit(pid, [i["id"] for i in imgs]).to_dict()


def tag_stats(pid: str) -> dict[str, Any]:
    require_project(pid)
    imgs = db.list_images(pid)
    counter: Counter[str] = Counter()
    for i in imgs:
        counter.update({t for t in i["tags"]})
    ratings = Counter(i["rating"] or "unknown" for i in imgs)
    status = Counter(i["status"] for i in imgs)
    return {
        "images": len(imgs),
        "unique_tags": len(counter),
        "tags": [{"tag": t, "count": c} for t, c in counter.most_common()],
        "ratings": dict(ratings),
        "status": dict(status),
    }


BULK_ACTIONS = ("add", "remove", "replace", "reapply", "filter", "clear_nl")


def bulk_actions_view() -> dict[str, str]:
    return {a: tr(f"bulk_actions.{a}", default=a) for a in BULK_ACTIONS}


def bulk_edit(pid: str, action: str, ids: list[str] | None = None, tags: Iterable[str] | str | None = None,
              find: str = "", replace: str = "", position: str = "back") -> dict[str, Any]:
    if action not in BULK_ACTIONS:
        raise BadRequest(t("msg.unknown_action", action=action, available=", ".join(BULK_ACTIONS)))
    p = require_project(pid)
    s = normalize_settings(p["settings"])
    tag_list = split_tags(tags)
    patterns = [t.lower() for t in tag_list]
    find_c, repl = (split_tags(find) or [""])[0].lower(), split_tags(replace)
    changed = 0
    for img in db.list_images(pid, ids=ids):
        old, nl = list(img["tags"]), img["nl_caption"]
        new = list(old)
        if action == "add":
            existing = {t.lower() for t in old}
            extra = [t for t in tag_list if t.lower() not in existing]
            new = extra + old if position == "front" else old + extra
        elif action == "remove":
            new = [t for t in old if not matches_blacklist(t, patterns)]
        elif action == "replace":
            new = []
            for t in old:
                new.extend(repl if t.lower() == find_c else [t])
            new = split_tags(new)
        elif action == "reapply":
            if img.get("raw"):
                new = select_tags(img["raw"], s)
        elif action == "filter":
            new = apply_filters(old, s)
        elif action == "clear_nl":
            nl = ""
        if new != old or nl != img["nl_caption"]:
            db.update_image(img["id"], tags=new, nl_caption=nl)
            changed += 1
    db.touch_project(pid)
    return {"action": action, "changed": changed}


def captions(pid: str, limit: int = 1000, offset: int = 0) -> dict[str, Any]:
    p = require_project(pid)
    s = normalize_settings(p["settings"])
    imgs = db.list_images(pid)
    items = [
        {"image_id": i["id"], "file": i["original_name"], "status": i["status"], "caption": caption_for(i, s)}
        for i in imgs[offset: offset + limit]
    ]
    return {"total": len(imgs), "offset": offset, "items": items}
