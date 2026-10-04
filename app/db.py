"""SQLite 儲存：專案與圖片標註資料。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    settings TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    original_name TEXT NOT NULL,
    rel_path TEXT NOT NULL DEFAULT '',
    width INTEGER,
    height INTEGER,
    size INTEGER,
    sha1 TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    nl_caption TEXT NOT NULL DEFAULT '',
    rating TEXT,
    raw TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_images_sha ON images(project_id, sha1);
-- 已上傳到 Civitai 的圖片（key = sha1:格式:長邊），重送訓練時不必再上傳
CREATE TABLE IF NOT EXISTS civitai_blobs (
    key TEXT PRIMARY KEY,
    blob_id TEXT NOT NULL,
    created_at REAL NOT NULL
);
-- 送到 Civitai 的訓練任務
CREATE TABLE IF NOT EXISTS civitai_runs (
    workflow_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    ecosystem TEXT NOT NULL,
    model TEXT,
    image_count INTEGER NOT NULL,
    cost INTEGER,
    status TEXT NOT NULL,
    summary TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_civitai_runs_project ON civitai_runs(project_id, created_at);
"""

_IMAGE_FIELDS = {"tags", "nl_caption", "rating", "raw", "status", "error", "width", "height", "filename", "size", "blocks",
                 "sha1", "upscale"}
_JSON_FIELDS = ("raw", "blocks", "upscale")
# 新增欄位（舊資料庫自動補上）
# blocks：白色色塊偵測結果，NULL = 尚未偵測；upscale：waifu2x 放大紀錄（含原圖資訊），NULL = 沒放大過
_ADDED_COLUMNS = {"images": [("blocks", "TEXT"), ("upscale", "TEXT")]}

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            settings.ensure_dirs()
            _conn = sqlite3.connect(settings.db_path, check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA foreign_keys=ON")
            _conn.executescript(_SCHEMA)
            for table, cols in _ADDED_COLUMNS.items():
                have = {r[1] for r in _conn.execute(f"PRAGMA table_info({table})")}
                for name, kind in cols:
                    if name not in have:
                        _conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
            # 服務重啟時，未完成的工作已不存在
            _conn.execute(f"UPDATE images SET status={_RESTORED_STATUS} WHERE status IN ('queued','processing')")
        return _conn


_RESTORED_STATUS = "CASE WHEN tags!='[]' OR nl_caption!='' THEN 'done' ELSE 'pending' END"


def _q(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with _lock:
        return connect().execute(sql, tuple(params)).fetchall()


def _x(sql: str, params: Iterable[Any] = ()) -> int:
    with _lock:
        return connect().execute(sql, tuple(params)).rowcount


# ------------------------------------------------------------------ projects
def _project(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["settings"] = json.loads(d["settings"])
    return d


def create_project(name: str, project_settings: dict[str, Any]) -> dict[str, Any]:
    pid, now = new_id(), time.time()
    _x("INSERT INTO projects (id, name, settings, created_at, updated_at) VALUES (?,?,?,?,?)",
       (pid, name, json.dumps(project_settings, ensure_ascii=False), now, now))
    return get_project(pid)  # type: ignore[return-value]


def get_project(pid: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM projects WHERE id=?", (pid,))
    return _project(rows[0]) if rows else None


def list_projects() -> list[dict[str, Any]]:
    rows = _q(
        """SELECT p.*,
                  (SELECT COUNT(*) FROM images i WHERE i.project_id=p.id) AS image_count,
                  (SELECT COUNT(*) FROM images i WHERE i.project_id=p.id AND i.status='done') AS done_count,
                  (SELECT id FROM images i WHERE i.project_id=p.id ORDER BY created_at LIMIT 1) AS cover_id
           FROM projects p ORDER BY p.updated_at DESC"""
    )
    return [_project(r) for r in rows]


def update_project(pid: str, name: str | None = None, project_settings: dict[str, Any] | None = None) -> None:
    now = time.time()
    if name is not None:
        _x("UPDATE projects SET name=?, updated_at=? WHERE id=?", (name, now, pid))
    if project_settings is not None:
        _x("UPDATE projects SET settings=?, updated_at=? WHERE id=?",
           (json.dumps(project_settings, ensure_ascii=False), now, pid))


def touch_project(pid: str) -> None:
    _x("UPDATE projects SET updated_at=? WHERE id=?", (time.time(), pid))


def delete_project(pid: str) -> None:
    _x("DELETE FROM images WHERE project_id=?", (pid,))
    _x("DELETE FROM projects WHERE id=?", (pid,))


# ------------------------------------------------------------------ images
def _image(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["tags"] = json.loads(d["tags"] or "[]")
    for k in _JSON_FIELDS:
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


def add_image(project_id: str, **fields: Any) -> dict[str, Any]:
    iid, now = fields.pop("id", None) or new_id(), time.time()
    tags = json.dumps(fields.get("tags") or [], ensure_ascii=False)
    status = "done" if fields.get("tags") or fields.get("nl_caption") else "pending"
    _x(
        """INSERT INTO images (id, project_id, filename, original_name, rel_path, width, height, size, sha1,
                               tags, nl_caption, status, blocks, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (iid, project_id, fields["filename"], fields["original_name"], fields.get("rel_path", ""),
         fields.get("width"), fields.get("height"), fields.get("size"), fields.get("sha1"),
         tags, fields.get("nl_caption") or "", status,
         json.dumps(fields["blocks"]) if fields.get("blocks") is not None else None, now, now),
    )
    return get_image(iid)  # type: ignore[return-value]


def find_by_sha(project_id: str, sha1: str) -> dict[str, Any] | None:
    """同一張圖（放大過的圖也比對放大前的原圖）。"""
    rows = _q("SELECT * FROM images WHERE project_id=? AND (sha1=? OR json_extract(upscale, '$.original.sha1')=?)",
              (project_id, sha1, sha1))
    return _image(rows[0]) if rows else None


def get_image(iid: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM images WHERE id=?", (iid,))
    return _image(rows[0]) if rows else None


def list_images(project_id: str, ids: Iterable[str] | None = None, status: str | None = None) -> list[dict[str, Any]]:
    sql, params = "SELECT * FROM images WHERE project_id=?", [project_id]
    if ids is not None:
        ids = list(ids)
        if not ids:
            return []
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        params += ids
    if status:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY original_name COLLATE NOCASE, created_at"
    return [_image(r) for r in _q(sql, params)]


def update_image(iid: str, **fields: Any) -> None:
    sets, params = [], []
    for k, v in fields.items():
        if k not in _IMAGE_FIELDS:
            continue
        if k == "tags":
            v = json.dumps(v or [], ensure_ascii=False)
        elif k in _JSON_FIELDS:
            v = json.dumps(v, ensure_ascii=False) if v is not None else None
        sets.append(f"{k}=?")
        params.append(v)
    if not sets:
        return
    sets.append("updated_at=?")
    params += [time.time(), iid]
    _x(f"UPDATE images SET {', '.join(sets)} WHERE id=?", params)


def set_status(ids: Iterable[str], status: str, error: str | None = None) -> None:
    ids = list(ids)
    if ids:
        _x(f"UPDATE images SET status=?, error=?, updated_at=? WHERE id IN ({','.join('?' * len(ids))})",
           [status, error, time.time(), *ids])


def restore_status(ids: Iterable[str]) -> None:
    """把還在排隊/處理中的圖片還原為 done 或 pending（用於取消工作）。"""
    ids = list(ids)
    if ids:
        _x(f"UPDATE images SET status={_RESTORED_STATUS} "
           f"WHERE status IN ('queued','processing') AND id IN ({','.join('?' * len(ids))})", ids)


def delete_images(project_id: str, ids: Iterable[str]) -> list[dict[str, Any]]:
    imgs = list_images(project_id, ids=ids)
    if imgs:
        _x(f"DELETE FROM images WHERE project_id=? AND id IN ({','.join('?' * len(imgs))})",
           [project_id, *[i["id"] for i in imgs]])
    return imgs


# ------------------------------------------------------------------ Civitai
# Civitai 保留上傳的 blob 約 30 天；保守一點，超過 25 天就重新上傳
BLOB_MAX_AGE_S = 25 * 86400


def civitai_blob_get(key: str) -> str | None:
    rows = _q("SELECT blob_id FROM civitai_blobs WHERE key=? AND created_at>?", (key, time.time() - BLOB_MAX_AGE_S))
    return rows[0]["blob_id"] if rows else None


def civitai_blob_put(key: str, blob_id: str) -> None:
    _x("INSERT OR REPLACE INTO civitai_blobs (key, blob_id, created_at) VALUES (?,?,?)", (key, blob_id, time.time()))


def _run(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["summary"] = json.loads(d["summary"]) if d["summary"] else None
    return d


def civitai_run_add(workflow_id: str, project_id: str, ecosystem: str, model: str | None, image_count: int,
                    cost: int | None, status: str, summary: dict[str, Any] | None) -> None:
    now = time.time()
    _x("INSERT OR REPLACE INTO civitai_runs (workflow_id, project_id, ecosystem, model, image_count, cost, status, "
       "summary, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
       (workflow_id, project_id, ecosystem, model, image_count, cost, status,
        json.dumps(summary, ensure_ascii=False) if summary else None, now, now))


def civitai_run_get(workflow_id: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM civitai_runs WHERE workflow_id=?", (workflow_id,))
    return _run(rows[0]) if rows else None


def civitai_runs(project_id: str) -> list[dict[str, Any]]:
    return [_run(r) for r in _q("SELECT * FROM civitai_runs WHERE project_id=? ORDER BY created_at DESC", (project_id,))]


def civitai_runs_recent(since: float) -> list[dict[str, Any]]:
    """所有專案中還沒結束、或在 since 之後建立的訓練任務（附專案名稱），給首頁顯示。"""
    rows = _q("SELECT r.*, p.name AS project_name FROM civitai_runs r JOIN projects p ON p.id = r.project_id "
              "WHERE r.created_at > ? OR r.status NOT IN ('succeeded', 'failed', 'expired', 'canceled') "
              "ORDER BY r.created_at DESC", (since,))
    return [_run(r) for r in rows]


def civitai_run_update(workflow_id: str, status: str, summary: dict[str, Any]) -> None:
    _x("UPDATE civitai_runs SET status=?, summary=?, updated_at=? WHERE workflow_id=?",
       (status, json.dumps(summary, ensure_ascii=False), time.time(), workflow_id))
