"""SMB 伺服器（NAS / Windows 分享資料夾）來源：在 app 裡連線、像檔案瀏覽器一樣挑資料夾或圖片，匯入到專案或角色篩選。

用 smbprotocol（MIT）。連線設定（含密碼）存在 data/studio.db，API 與網頁都不回傳密碼。
- 專案：圖片下載到專案裡（訓練、匯出都要本機檔案），同名 .txt 一起當成既有 caption
- 角色篩選：只記路徑（smb://<連線 id>/<分享裡的路徑>），用到時才讀，大量圖片不多佔空間
一次匯入可以包含多個資料夾與圖片，也可以來自不同連線。匯入在背景工作裡跑（列出檔案、下載），網頁不會卡住。
瀏覽時的縮圖存在 data/cache/smb/<連線 id>/，刪除或修改連線時清掉。
"""
from __future__ import annotations

import hashlib
import io
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from PIL import Image, ImageOps

from . import db, storage
from .config import settings
from .i18n import t
from .services import BadRequest, NotFound

log = logging.getLogger(__name__)

SCHEME = "smb://"
TIMEOUT = 10
TARGETS = ("project", "finder")
MAX_ITEMS = 20000  # 一次匯入最多勾選的項目

_caches: dict[str, dict] = {}  # 每個連線自己的連線快取，不同帳號不會混用
_lock = threading.Lock()


# ------------------------------------------------------------------ 連線設定
def conn_out(c: dict[str, Any]) -> dict[str, Any]:
    out = {k: c[k] for k in ("id", "name", "host", "port", "share", "username", "domain", "created_at", "updated_at")}
    out["has_password"] = bool(c["password"])
    out["finder_refs"] = db.smb_finder_refs(c["id"])
    return out


def require(cid: str) -> dict[str, Any]:
    c = db.smb_get(cid)
    if c is None:
        raise NotFound(t("msg.smb_not_found", cid=cid))
    return c


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    out = {k: (v.strip() if isinstance(v, str) and k != "password" else v) for k, v in fields.items() if v is not None}
    host = out.get("host", "")
    if host.startswith("\\\\") or host.startswith("//") or host.lower().startswith("smb://"):
        # 貼上 \\nas\share\folder 或 smb://nas/share 時，自動拆出主機與分享
        parts = [p for p in host.replace("smb://", "").replace("\\", "/").split("/") if p]
        out["host"] = parts[0] if parts else ""
        if len(parts) > 1 and not out.get("share"):
            out["share"] = parts[1]
    if "share" in out:
        out["share"] = out["share"].strip("/\\")
    if "port" in out:
        out["port"] = int(out["port"] or 445)
    return out


def _check(c: dict[str, Any]) -> None:
    if not c.get("host") or not c.get("share"):
        raise BadRequest(t("msg.smb_need_host_share"))


def list_connections() -> list[dict[str, Any]]:
    return [conn_out(c) for c in db.smb_list()]


def create(name: str = "", host: str = "", share: str = "", username: str = "", password: str = "",
           domain: str = "", port: int = 445) -> dict[str, Any]:
    c = _clean(dict(name=name, host=host, share=share, username=username, password=password, domain=domain, port=port))
    _check(c)
    c["name"] = c.get("name") or f"{c['host']}/{c['share']}"
    test(c)  # 連得上才存
    return conn_out(db.smb_create(**c))


def update(cid: str, **fields: Any) -> dict[str, Any]:
    old = require(cid)
    patch = _clean(fields)
    if not patch.get("password"):
        patch.pop("password", None)  # 沒填密碼 = 不變
    merged = {**old, **patch}
    _check(merged)
    test({**merged, "id": None})  # 用新的設定、新的連線測試，不沿用舊帳號的連線
    db.smb_update(cid, **patch)
    _reset(cid)
    return conn_out(require(cid))


def delete(cid: str) -> None:
    require(cid)
    db.smb_delete(cid)
    _reset(cid)


def _thumb_dir(cid: str) -> Path:
    return settings.data_dir / "cache" / "smb" / cid


def _reset(cid: str) -> None:
    """關掉這個連線的 SMB 連線快取，清掉瀏覽縮圖（主機或分享可能換了）。"""
    import smbclient

    shutil.rmtree(_thumb_dir(cid), ignore_errors=True)
    with _lock:
        cache = _caches.pop(cid, None)
    if cache:
        try:
            smbclient.reset_connection_cache(connection_cache=cache)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ 讀取
def _kw(c: dict[str, Any]) -> dict[str, Any]:
    user = c["username"]
    if user and c.get("domain"):
        user = f"{c['domain']}\\{user}"
    with _lock:
        cache = _caches.setdefault(c["id"], {}) if c.get("id") else {}
    return {"username": user or None, "password": c["password"] or None, "port": int(c["port"] or 445),
            "connection_cache": cache, "connection_timeout": TIMEOUT}


def _rel(path: str) -> str:
    """分享裡的路徑：統一用 /，不允許 .. 跳出分享。"""
    parts = [p for p in str(path or "").replace("\\", "/").split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise BadRequest(t("msg.smb_bad_path"))
    return "/".join(parts)


def _unc(c: dict[str, Any], rel: str = "") -> str:
    rel = _rel(rel)
    return f"\\\\{c['host']}\\{c['share']}" + ("\\" + rel.replace("/", "\\") if rel else "")


def _error(e: Exception) -> BadRequest:
    """把 SMB 錯誤轉成看得懂的訊息。"""
    text = str(e)
    if "LOGON_FAILURE" in text or "ACCESS_DENIED" in text or "0xc000006d" in text or "0xc0000022" in text:
        return BadRequest(t("msg.smb_login_failed"))
    if "BAD_NETWORK_NAME" in text or "0xc00000cc" in text:
        return BadRequest(t("msg.smb_share_not_found"))
    if any(k in text for k in ("No such file", "NO_SUCH_FILE", "OBJECT_NAME_NOT_FOUND", "OBJECT_PATH_NOT_FOUND")):
        return BadRequest(t("msg.smb_path_not_found"))
    if "Failed to connect" in text or "timed out" in text.lower() or "refused" in text:
        return BadRequest(t("msg.smb_unreachable", error=text))
    return BadRequest(t("msg.smb_failed", error=text))


def _scandir(c: dict[str, Any], rel: str = "") -> list[Any]:
    return _scandir_kw(c, rel, _kw(c))


def _scandir_kw(c: dict[str, Any], rel: str, kw: dict[str, Any]) -> list[Any]:
    import smbclient

    try:
        return list(smbclient.scandir(_unc(c, rel), **kw))
    except BadRequest:
        raise
    except Exception as e:  # noqa: BLE001
        raise _error(e) from e


def test(c: dict[str, Any]) -> dict[str, Any]:
    """連得上、帳密正確、分享存在。沒有 id 的設定（新增 / 修改前測試）用完就關掉連線。"""
    import smbclient

    kw = _kw(c)
    try:
        entries = _scandir_kw(c, "", kw)
    except BadRequest as e:
        if str(e) == t("msg.smb_path_not_found"):  # 分享根目錄找不到 = 分享名稱錯了
            raise BadRequest(t("msg.smb_share_not_found")) from e
        raise
    finally:
        if not c.get("id"):
            try:
                smbclient.reset_connection_cache(connection_cache=kw["connection_cache"])
            except Exception:  # noqa: BLE001
                pass
    return {"ok": True, "entries": len(entries)}


def test_settings(fields: dict[str, Any], cid: str | None = None) -> dict[str, Any]:
    """測試還沒存的設定；cid = 修改既有連線，沒填的欄位（含密碼）沿用已存的。"""
    if cid:
        saved = require(cid)
        fields = {**{k: saved[k] for k in ("host", "share", "username", "domain", "port", "password")},
                  **{k: v for k, v in fields.items() if v not in ("", None)}}
    c = _clean(fields)
    _check(c)
    return test({**c, "id": None})


def _is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in storage.IMAGE_EXTS and not name.startswith(".")


def browse(cid: str, path: str = "") -> dict[str, Any]:
    """一個資料夾的子資料夾與圖片（名稱、大小、修改時間），像檔案瀏覽器一樣列出來給使用者挑。"""
    c = require(cid)
    rel = _rel(path)
    entries = _scandir(c, rel)
    dirs = sorted((e.name for e in entries if e.is_dir() and not e.name.startswith(".") and e.name != "__MACOSX"),
                  key=str.lower)
    images = sorted(({"name": e.name, "size": e.smb_info.end_of_file,  # DirEntry.stat() 不帶連接埠，用列目錄的資訊
                      "mtime": int(e.smb_info.last_write_time.timestamp())}
                     for e in entries if e.is_file() and _is_image(e.name)), key=lambda x: x["name"].lower())
    return {"path": rel, "share": c["share"], "dirs": dirs, "images": images,
            "others": sum(1 for e in entries if e.is_file()) - len(images)}


def walk(c: dict[str, Any], rel: str, recursive: bool = True) -> Iterator[str]:
    """分享裡的檔案路徑（圖片與 .txt），依路徑排序。"""
    stack = [_rel(rel)]
    while stack:
        cur = stack.pop()
        entries = sorted(_scandir(c, cur), key=lambda e: e.name.lower())
        subdirs = []
        for e in entries:
            if e.name.startswith(".") or e.name == "__MACOSX":
                continue
            child = f"{cur}/{e.name}" if cur else e.name
            if e.is_dir():
                subdirs.append(child)
            elif _is_image(e.name) or e.name.lower().endswith((".txt", ".caption")):
                yield child
        if recursive:
            stack.extend(reversed(subdirs))


def read_bytes(cid: str, rel: str) -> bytes:
    import smbclient

    c = require(cid)
    try:
        with smbclient.open_file(_unc(c, rel), mode="rb", **_kw(c)) as f:
            return f.read()
    except BadRequest:
        raise
    except Exception as e:  # noqa: BLE001
        raise _error(e) from e


def thumb(cid: str, rel: str, version: str = "") -> Path:
    """瀏覽時的縮圖；version（大小-修改時間）變了就重做。"""
    rel = _rel(rel)
    if not _is_image(PurePosixPath(rel).name):
        raise BadRequest(t("msg.smb_not_image", name=PurePosixPath(rel).name))
    dest = _thumb_dir(cid) / f"{hashlib.sha1(f'{rel}|{version}'.encode()).hexdigest()}.webp"
    if not dest.exists():
        data = read_bytes(cid, rel)
        try:
            im = Image.open(io.BytesIO(data))
            im.draft("RGB", (storage.THUMB_SIZE * 2, storage.THUMB_SIZE * 2))  # JPEG 直接用縮小解碼，大圖快很多
            if getattr(im, "is_animated", False):
                im.seek(0)
            im = ImageOps.exif_transpose(im)
        except Exception as e:  # noqa: BLE001
            raise BadRequest(t("msg.smb_bad_image", name=PurePosixPath(rel).name)) from e
        tmp = dest.with_name(f"{dest.stem}.{threading.get_ident()}.tmp")  # 同一張同時被要兩次也不會互相蓋掉
        storage.make_thumb(im, tmp)
        tmp.replace(dest)
    return dest


def ref(cid: str, rel: str) -> str:
    return f"{SCHEME}{cid}/{_rel(rel)}"


def parse_ref(path: str) -> tuple[str, str] | None:
    """smb://<連線 id>/<路徑> → (連線 id, 路徑)；不是 SMB 路徑回傳 None。"""
    if not path.startswith(SCHEME):
        return None
    cid, _, rel = path[len(SCHEME):].partition("/")
    return cid, rel


# ------------------------------------------------------------------ 匯入
def _parent(rel: str) -> str:
    parent = str(PurePosixPath(rel).parent)
    return "" if parent == "." else parent


def start_import(items: list[dict[str, Any]], recursive: bool = True, target: str = "project",
                 project_id: str | None = None, session_id: str | None = None, role: str = "pool") -> dict[str, Any]:
    """items：[{conn_id, path, dir}]，勾選的資料夾與圖片，可以來自不同連線。"""
    from . import finder, jobs

    if not items:
        raise BadRequest(t("msg.smb_nothing_selected"))
    if len(items) > MAX_ITEMS:
        raise BadRequest(t("msg.smb_too_many", n=MAX_ITEMS))
    clean, seen = [], set()
    for it in items:
        cid, rel, is_dir = str(it.get("conn_id") or ""), _rel(it.get("path", "")), bool(it.get("dir"))
        require(cid)
        if not is_dir and not _is_image(PurePosixPath(rel).name):
            raise BadRequest(t("msg.smb_not_image", name=rel))
        if (cid, rel, is_dir) not in seen:
            seen.add((cid, rel, is_dir))
            clean.append({"conn_id": cid, "path": rel, "dir": is_dir})
    for cid in dict.fromkeys(it["conn_id"] for it in clean):
        # 每個連線先確認一次連得上（第一個項目的資料夾），錯誤直接回給使用者
        first = next(it for it in clean if it["conn_id"] == cid)
        _scandir(require(cid), first["path"] if first["dir"] else _parent(first["path"]))
    if target == "project":
        if not project_id or db.get_project(project_id) is None:
            raise NotFound(t("msg.project_not_found", pid=project_id or ""))
        owner = project_id
    elif target == "finder":
        finder.require_session(session_id or "")
        if role not in finder.ROLES:
            raise BadRequest(t("msg.finder_bad_role", roles=", ".join(finder.ROLES)))
        owner = session_id
    else:
        raise BadRequest(t("msg.smb_bad_target", targets=", ".join(TARGETS)))
    params = {"items": clean, "recursive": recursive, "target": target, "role": role}
    return jobs.submit_task(owner, "smb_import", params=params).to_dict()


def _collect(job: Any, conns: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    """展開勾選的項目：資料夾裡的圖片（依設定含子資料夾）加上單獨勾選的圖片，去掉重複。
    同時找出同名的 .txt（專案匯入時當成既有 caption）。"""
    p = job.params
    want_captions = p["target"] == "project"
    images: dict[tuple[str, str], None] = {}  # 保持順序的集合
    captions: dict[tuple[str, str], str] = {}
    listed: set[tuple[str, str]] = set()

    def take(cid: str, files: Iterator[str], with_images: bool) -> None:
        for f in files:
            if _is_image(PurePosixPath(f).name):
                if with_images:
                    images.setdefault((cid, f))
            else:
                captions.setdefault((cid, storage.stem_key(f)), f)

    for it in p["items"]:
        if job.cancel_event.is_set():
            break
        cid, rel, c = it["conn_id"], it["path"], conns[it["conn_id"]]
        try:
            if it["dir"]:
                take(cid, walk(c, rel, p["recursive"]), True)
                listed.add((cid, rel))
            else:
                images.setdefault((cid, rel))
                if want_captions and (cid, _parent(rel)) not in listed:
                    listed.add((cid, _parent(rel)))
                    take(cid, walk(c, _parent(rel), False), False)
        except BadRequest as e:  # 某個項目讀不到（被刪掉、權限）：記下來，其他照常匯入
            job.errors.append({"image_id": "", "file": f"{c['share']}/{rel}", "error": str(e)})
    return list(images), captions


def run_import(job: Any) -> tuple[str, dict[str, Any]]:
    """背景工作：展開勾選的資料夾與圖片，匯入專案（下載）或角色篩選（只記路徑）。"""
    from . import finder

    p = job.params
    conns = {cid: require(cid) for cid in dict.fromkeys(it["conn_id"] for it in p["items"])}
    job.say("msg.smb_listing")
    images, captions = _collect(job, conns)
    job.image_ids = [ref(cid, rel) for cid, rel in images]
    job.say("msg.smb_importing")
    added, skipped = 0, 0
    display = lambda cid, rel: f"{conns[cid]['share']}/{rel}"  # noqa: E731

    if p["target"] == "finder":
        sid, role = job.project_id, p["role"]
        for cid, rel in images:
            if job.cancel_event.is_set():
                break
            path = ref(cid, rel)
            if db.finder_has(sid, role, path=path):
                skipped += 1
            else:
                db.finder_image_add(sid, role, PurePosixPath(rel).name, path, rel_path=display(cid, rel))
                added += 1
            job.done += 1
        finder.after_add(sid, role, added)
    else:
        pid = job.project_id
        project = db.get_project(pid)
        trigger = (project or {}).get("settings", {}).get("trigger", "")
        lock = threading.Lock()

        def one(item: tuple[str, str]) -> None:
            nonlocal added, skipped
            cid, rel = item
            if job.cancel_event.is_set():
                return
            try:
                data = read_bytes(cid, rel)
                cap_path = captions.get((cid, storage.stem_key(rel)))
                caption = read_bytes(cid, cap_path).decode("utf-8", errors="replace") if cap_path else None
                rec, reason = storage.save_image_bytes(pid, data, PurePosixPath(rel).name, rel_path=display(cid, rel),
                                                       caption_text=caption, trigger=trigger)
            except BadRequest as e:
                rec, reason = None, str(e)
            with lock:
                if rec:
                    added += 1
                else:
                    skipped += 1
                    if reason and reason != t("msg.duplicate_image"):
                        job.errors.append({"image_id": "", "file": display(cid, rel), "error": reason})
                job.done += 1

        with ThreadPoolExecutor(4) as pool:  # 下載與縮圖 / 白色色塊偵測並行
            list(pool.map(one, images))
        if added:
            db.touch_project(pid)
    return "msg.smb_done", {"added": added, "skipped": skipped}
