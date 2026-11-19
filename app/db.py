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
-- 不設唯一：匯入時可以選擇把專案裡已經有的同一張圖再加一份（加重那張圖）。以前的唯一索引 idx_images_sha 啟動時移除
CREATE INDEX IF NOT EXISTS idx_images_sha1 ON images(project_id, sha1);
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
-- 匯入 A1111 / Forge 的紀錄（每次訓練的每個 epoch 一筆）
CREATE TABLE IF NOT EXISTS a1111_imports (
    workflow_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    name TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    prompt TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (workflow_id, epoch)
);
-- 本機訓練（AI Toolkit）：訓練專案 = 一份資料來源（標註專案或伺服器資料夾）+ 上次用的設定
CREATE TABLE IF NOT EXISTS train_projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    folder TEXT,
    settings TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
-- 每次訓練。job_id = AI Toolkit 的工作，NULL 表示還在準備資料
CREATE TABLE IF NOT EXISTS train_runs (
    id TEXT PRIMARY KEY,
    train_project_id TEXT NOT NULL REFERENCES train_projects(id) ON DELETE CASCADE,
    job_id TEXT,
    job_name TEXT NOT NULL UNIQUE,
    model TEXT NOT NULL,
    image_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_train_runs_project ON train_runs(train_project_id, created_at);
-- 角色篩選（CCIP）：和專案分開，每個篩選有參考圖、排除參考圖與要篩選的圖片
CREATE TABLE IF NOT EXISTS finder_sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    threshold REAL,
    tag_filter TEXT,
    people_mode TEXT,
    people_min_side INTEGER,
    people_recover REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS finder_images (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES finder_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    original_name TEXT NOT NULL,
    rel_path TEXT NOT NULL DEFAULT '',
    path TEXT NOT NULL,
    sha1 TEXT,
    width INTEGER,
    height INTEGER,
    feature BLOB,
    feature_model TEXT,
    score REAL,
    neg_score REAL,
    manual INTEGER,
    error TEXT,
    phash TEXT,
    sig BLOB,
    tags TEXT,
    people TEXT,
    people_tags TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_finder_images ON finder_images(session_id, role, created_at);
-- SMB 伺服器（NAS / Windows 分享資料夾）連線；密碼只在伺服器端使用，API 不回傳
CREATE TABLE IF NOT EXISTS smb_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL DEFAULT 445,
    share TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    password TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
-- 整個 app 的設定（不屬於某個專案的，例如 VLM 的休眠等級），value 是 JSON
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_IMAGE_FIELDS = {"tags", "nl_caption", "rating", "raw", "status", "error", "width", "height", "filename", "size", "blocks",
                 "sha1", "upscale"}
_JSON_FIELDS = ("raw", "blocks", "upscale")
# 新增欄位（舊資料庫自動補上）
# blocks：白色色塊偵測結果，NULL = 尚未偵測；upscale：waifu2x 放大紀錄（含原圖資訊），NULL = 沒放大過
# 角色篩選：manual = 手動判定；phash / sig = 找重複圖片用的指紋；tags = WD14 標籤；tag_filter = 結果的 tag 篩選；
# people = 多人圖偵測結果（每個人和每個頭的框、每個人和參考圖的差異）；people_tags = 處理後的圖的 tag（依模式）；
# people_mode = 多人圖的處理方式（crop / mask，NULL = 不處理）；people_min_side = 裁切後短邊的下限（NULL = 預設）；
# people_recover = 找回的搜尋範圍（門檻再加多少，NULL = 不找）
_ADDED_COLUMNS = {"images": [("blocks", "TEXT"), ("upscale", "TEXT")],
                  "finder_sessions": [("tag_filter", "TEXT"), ("people_mode", "TEXT"),
                                      ("people_min_side", "INTEGER"), ("people_recover", "REAL")],
                  "finder_images": [("manual", "INTEGER"), ("phash", "TEXT"), ("sig", "BLOB"), ("tags", "TEXT"),
                                    ("people", "TEXT"), ("people_tags", "TEXT")]}

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
            _conn.execute("DROP INDEX IF EXISTS idx_images_sha")  # 舊版的唯一索引（同一個專案不能有一樣的圖）
            # 服務重啟時，未完成的工作已不存在（訓練是 AI Toolkit 在跑，只有還在準備資料的會中斷）
            _conn.execute("UPDATE train_runs SET status='failed', error='interrupted' "
                          "WHERE status='preparing' AND (job_id IS NULL OR job_id='')")
            # 已經送出給 AI Toolkit 的（中斷在 /start 前後）交給查詢更新：沒開始的會顯示為已停止，排進佇列的照常追蹤
            _conn.execute("UPDATE train_runs SET status='queued' WHERE status='preparing' AND job_id IS NOT NULL AND job_id<>''")
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


def a1111_import_put(workflow_id: str, epoch: int, name: str, relative_path: str, prompt: str) -> None:
    _x("INSERT OR REPLACE INTO a1111_imports (workflow_id, epoch, name, relative_path, prompt, created_at) "
       "VALUES (?,?,?,?,?,?)", (workflow_id, epoch, name, relative_path, prompt, time.time()))


def a1111_imports(workflow_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in _q("SELECT epoch, name, relative_path, prompt, created_at FROM a1111_imports "
                                "WHERE workflow_id=? ORDER BY epoch", (workflow_id,))]


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


# ------------------------------------------------------------------ 角色篩選
# role：ref = 目標角色參考圖、neg = 排除的相似角色、pool = 要篩選的圖片
# path：圖片檔的絕對路徑（上傳 / 從專案加入的放在 data/finder/<id>/，伺服器匯入資料夾的只記原位置）
# manual：手動判定，1 = 是目標、-1 = 不是目標（模型挑錯）、NULL = 依分數；換門檻或重新辨識都保留
_FINDER_FIELDS = {"width", "height", "feature", "feature_model", "score", "neg_score", "error", "sha1", "manual",
                  "phash", "sig", "tags", "people", "people_tags"}


_IN_CHUNK = 900  # 一個 IN (...) 最多放幾個參數（舊版 SQLite 上限 999）


def finder_session_create(name: str, model: str) -> dict[str, Any]:
    sid, now = new_id(), time.time()
    _x("INSERT INTO finder_sessions (id, name, model, threshold, created_at, updated_at) VALUES (?,?,?,?,?,?)",
       (sid, name, model, None, now, now))
    return finder_session_get(sid)  # type: ignore[return-value]


def finder_session_get(sid: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM finder_sessions WHERE id=?", (sid,))
    return dict(rows[0]) if rows else None


def finder_sessions() -> list[dict[str, Any]]:
    rows = _q(
        """SELECT s.*,
                  (SELECT COUNT(*) FROM finder_images i WHERE i.session_id=s.id AND i.role='ref') AS ref_count,
                  (SELECT COUNT(*) FROM finder_images i WHERE i.session_id=s.id AND i.role='pool') AS pool_count,
                  (SELECT id FROM finder_images i WHERE i.session_id=s.id AND i.role='ref' ORDER BY created_at LIMIT 1)
                      AS cover_id
           FROM finder_sessions s ORDER BY s.updated_at DESC""")
    return [dict(r) for r in rows]


def finder_session_update(sid: str, **fields: Any) -> None:
    keys = [k for k in fields if k in ("name", "model", "threshold", "tag_filter", "people_mode",
                                         "people_min_side", "people_recover")]
    if keys:
        _x(f"UPDATE finder_sessions SET {', '.join(f'{k}=?' for k in keys)}, updated_at=? WHERE id=?",
           [fields[k] for k in keys] + [time.time(), sid])


def finder_session_touch(sid: str) -> None:
    _x("UPDATE finder_sessions SET updated_at=? WHERE id=?", (time.time(), sid))


def finder_session_delete(sid: str) -> None:
    _x("DELETE FROM finder_sessions WHERE id=?", (sid,))


def finder_image_add(session_id: str, role: str, original_name: str, path: str, rel_path: str = "",
                     sha1: str | None = None, iid: str | None = None) -> str:
    iid = iid or new_id()
    _x("""INSERT INTO finder_images (id, session_id, role, original_name, rel_path, path, sha1, created_at)
          VALUES (?,?,?,?,?,?,?,?)""", (iid, session_id, role, original_name, rel_path, path, sha1, time.time()))
    return iid


def finder_images(session_id: str, role: str | None = None, ids: Iterable[str] | None = None,
                  with_feature: bool = False) -> list[dict[str, Any]]:
    cols = "*" if with_feature else ("id, session_id, role, original_name, rel_path, path, sha1, width, height, "
                                     "feature_model, score, neg_score, manual, error, phash, tags, people, "
                                     "people_tags, created_at")
    sql, params = f"SELECT {cols} FROM finder_images WHERE session_id=?", [session_id]
    if role:
        sql += " AND role=?"
        params.append(role)
    if ids is not None:
        ids = list(ids)
        if len(ids) > _IN_CHUNK:  # SQLite 一次能帶的參數有限：分批查
            out = [r for k in range(0, len(ids), _IN_CHUNK)
                   for r in finder_images(session_id, role, ids[k:k + _IN_CHUNK], with_feature)]
            return sorted(out, key=lambda r: (r["created_at"], r["original_name"]))
        if not ids:
            return []
        sql += f" AND id IN ({','.join('?' * len(ids))})"
        params += ids
    return [dict(r) for r in _q(sql + " ORDER BY created_at, original_name", params)]


def finder_dup_rows(session_id: str, role: str) -> list[dict[str, Any]]:
    """找重複需要的欄位（不讀 CCIP 特徵與小圖，5 萬張也不會吃太多記憶體）。"""
    return [dict(r) for r in _q("SELECT id, phash, sig IS NOT NULL AS has_sig, width, height, original_name, rel_path, "
                                "manual, error, created_at FROM finder_images WHERE session_id=? AND role=? "
                                "ORDER BY created_at, original_name", (session_id, role))]


def finder_dup_sigs(session_id: str, role: str) -> list[dict[str, Any]]:
    """有指紋的圖片的雜湊與小圖（每張 3 KB）。"""
    return [dict(r) for r in _q("SELECT id, phash, sig FROM finder_images WHERE session_id=? AND role=? "
                                "AND phash IS NOT NULL AND sig IS NOT NULL ORDER BY created_at, original_name",
                                (session_id, role))]


def finder_image_get(iid: str) -> dict[str, Any] | None:
    rows = _q("SELECT id, session_id, role, original_name, rel_path, path, width, height FROM finder_images WHERE id=?",
              (iid,))
    return dict(rows[0]) if rows else None


def finder_image_update(iid: str, **fields: Any) -> None:
    keys = [k for k in fields if k in _FINDER_FIELDS]
    if keys:
        _x(f"UPDATE finder_images SET {', '.join(f'{k}=?' for k in keys)} WHERE id=?",
           [fields[k] for k in keys] + [iid])


def finder_images_copy(src_session: str, dest_session: str, role: str, rows: list[tuple[str, str, str, float]]) -> int:
    """從另一個篩選複製圖片。rows = [(來源 id, 新 id, 新路徑, created_at)]。
    一起複製只跟圖片本身有關的結果（CCIP 特徵、WD14 tag、找重複的指紋、尺寸），不複製和參考圖有關的
    分數、多人圖的逐人比對，也不複製手動判定。來源讀取失敗的不帶特徵（新的篩選辨識時會再試）。
    幾萬張也只要一個交易：特徵和指紋直接在 SQLite 裡複製，不經過 Python。"""
    with _lock:
        c = connect()
        c.execute("BEGIN")
        try:
            c.execute("CREATE TEMP TABLE IF NOT EXISTS finder_copy_map "
                      "(src TEXT PRIMARY KEY, id TEXT NOT NULL, path TEXT NOT NULL, created_at REAL NOT NULL)")
            c.execute("DELETE FROM finder_copy_map")
            c.executemany("INSERT INTO finder_copy_map VALUES (?,?,?,?)", rows)
            n = c.execute(
                """INSERT INTO finder_images (id, session_id, role, original_name, rel_path, path, sha1, width, height,
                                              feature, feature_model, phash, sig, tags, created_at)
                   SELECT m.id, ?, ?, s.original_name, s.rel_path, m.path, s.sha1, s.width, s.height,
                          CASE WHEN s.error IS NULL THEN s.feature END, CASE WHEN s.error IS NULL THEN s.feature_model END,
                          s.phash, s.sig, s.tags, m.created_at
                   FROM finder_copy_map m JOIN finder_images s ON s.id = m.src AND s.session_id = ?""",
                (dest_session, role, src_session)).rowcount
            c.execute("DELETE FROM finder_copy_map")
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
    return n


def finder_has(session_id: str, role: str, sha1: str | None = None, path: str | None = None) -> bool:
    """同一個篩選、同一個角色裡是否已有這張圖（比對內容雜湊或檔案路徑）。"""
    col, val = ("sha1", sha1) if sha1 else ("path", path)
    return bool(_q(f"SELECT 1 FROM finder_images WHERE session_id=? AND role=? AND {col}=? LIMIT 1",
                   (session_id, role, val)))


def finder_set_manual(session_id: str, ids: Iterable[str], value: int | None) -> int:
    ids = list(ids)
    if not ids:
        return 0
    return _x(f"UPDATE finder_images SET manual=? WHERE session_id=? AND role='pool' AND id IN ({','.join('?' * len(ids))})",
              [value, session_id, *ids])


def finder_clear_scores(session_id: str) -> None:
    """參考圖或排除參考圖變了：之前算的分數和多人圖裡每個人的差異都不能用了（特徵還能用）。"""
    _x("UPDATE finder_images SET score=NULL, neg_score=NULL, people=NULL, people_tags=NULL WHERE session_id=?",
       (session_id,))


def finder_images_delete(session_id: str, ids: Iterable[str] | None = None, role: str | None = None) -> list[dict[str, Any]]:
    imgs = finder_images(session_id, role=role, ids=ids)
    for k in range(0, len(imgs), _IN_CHUNK):
        part = imgs[k:k + _IN_CHUNK]
        _x(f"DELETE FROM finder_images WHERE session_id=? AND id IN ({','.join('?' * len(part))})",
           [session_id, *[i["id"] for i in part]])
    return imgs


# ------------------------------------------------------------------ SMB 連線
_SMB_FIELDS = ("name", "host", "port", "share", "username", "password", "domain")


def smb_list() -> list[dict[str, Any]]:
    return [dict(r) for r in _q("SELECT * FROM smb_connections ORDER BY name COLLATE NOCASE")]


def smb_get(cid: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM smb_connections WHERE id=?", (cid,))
    return dict(rows[0]) if rows else None


def smb_create(**fields: Any) -> dict[str, Any]:
    cid, now = new_id(), time.time()
    _x(f"INSERT INTO smb_connections (id, {', '.join(_SMB_FIELDS)}, created_at, updated_at) "
       f"VALUES (?, {', '.join('?' * len(_SMB_FIELDS))}, ?, ?)", [cid, *[fields[k] for k in _SMB_FIELDS], now, now])
    return smb_get(cid)  # type: ignore[return-value]


def smb_update(cid: str, **fields: Any) -> None:
    keys = [k for k in _SMB_FIELDS if k in fields]
    if keys:
        _x(f"UPDATE smb_connections SET {', '.join(f'{k}=?' for k in keys)}, updated_at=? WHERE id=?",
           [fields[k] for k in keys] + [time.time(), cid])


def smb_delete(cid: str) -> None:
    _x("DELETE FROM smb_connections WHERE id=?", (cid,))


def smb_finder_refs(cid: str) -> int:
    """角色篩選裡有幾張圖片是從這個連線加入的（只記路徑，刪掉連線就讀不到）。"""
    return _q("SELECT COUNT(*) FROM finder_images WHERE path LIKE ?", (f"smb://{cid}/%",))[0][0]


# ------------------------------------------------------------------ 本機訓練
def _train_project(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["settings"] = json.loads(d["settings"])
    return d


def train_project_add(pid: str, name: str, source: str, project_id: str | None, folder: str | None,
                      settings_: dict[str, Any]) -> None:
    now = time.time()
    _x("INSERT INTO train_projects (id, name, source, project_id, folder, settings, created_at, updated_at) "
       "VALUES (?,?,?,?,?,?,?,?)", (pid, name, source, project_id, folder, json.dumps(settings_, ensure_ascii=False), now, now))


def train_project_get(pid: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM train_projects WHERE id=?", (pid,))
    return _train_project(rows[0]) if rows else None


def train_project_for(project_id: str) -> dict[str, Any] | None:
    """這個標註專案最近用的訓練專案。"""
    rows = _q("SELECT * FROM train_projects WHERE project_id=? ORDER BY updated_at DESC LIMIT 1", (project_id,))
    return _train_project(rows[0]) if rows else None


def train_projects() -> list[dict[str, Any]]:
    return [_train_project(r) for r in _q("SELECT * FROM train_projects ORDER BY updated_at DESC")]


def train_project_update(pid: str, name: str | None = None, settings_: dict[str, Any] | None = None) -> None:
    if name is not None:
        _x("UPDATE train_projects SET name=?, updated_at=? WHERE id=?", (name, time.time(), pid))
    if settings_ is not None:
        _x("UPDATE train_projects SET settings=?, updated_at=? WHERE id=?",
           (json.dumps(settings_, ensure_ascii=False), time.time(), pid))


def train_project_delete(pid: str) -> None:
    _x("DELETE FROM train_projects WHERE id=?", (pid,))


def _train_run(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["params"] = json.loads(d["params"])
    d["summary"] = json.loads(d["summary"])
    return d


def train_run_add(rid: str, train_project_id: str, job_name: str, model: str, params: dict[str, Any]) -> None:
    now = time.time()
    _x("INSERT INTO train_runs (id, train_project_id, job_name, model, status, params, created_at, updated_at) "
       "VALUES (?,?,?,?,'preparing',?,?,?)", (rid, train_project_id, job_name, model, json.dumps(params, ensure_ascii=False), now, now))


def train_run_get(rid: str) -> dict[str, Any] | None:
    rows = _q("SELECT * FROM train_runs WHERE id=?", (rid,))
    return _train_run(rows[0]) if rows else None


def train_runs(train_project_id: str) -> list[dict[str, Any]]:
    return [_train_run(r) for r in _q("SELECT * FROM train_runs WHERE train_project_id=? ORDER BY created_at DESC",
                                      (train_project_id,))]


def train_runs_with_status(statuses: Iterable[str]) -> list[dict[str, Any]]:
    marks = ",".join("?" * len(list(statuses)))
    return [_train_run(r) for r in _q(f"SELECT * FROM train_runs WHERE status IN ({marks}) ORDER BY created_at", tuple(statuses))]


def train_run_update(rid: str, when: Iterable[str] | None = None, **fields: Any) -> bool:
    """fields：job_id、image_count、status、summary、error。when：只在目前的狀態是其中之一時才更新
    （例如準備資料時被按了停止，不能再改回 queued）。回傳有沒有更新。"""
    if "summary" in fields:
        fields["summary"] = json.dumps(fields["summary"], ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in fields)
    statuses = list(when or ())
    cond = f" AND status IN ({','.join('?' * len(statuses))})" if statuses else ""
    return _x(f"UPDATE train_runs SET {cols}, updated_at=? WHERE id=?{cond}",
              (*fields.values(), time.time(), rid, *statuses)) > 0


def train_run_delete(rid: str) -> None:
    _x("DELETE FROM train_runs WHERE id=?", (rid,))
    _x("DELETE FROM a1111_imports WHERE workflow_id=?", (rid,))


# ------------------------------------------------------------------ app 設定
def setting_get(key: str, default: Any = None) -> Any:
    rows = _q("SELECT value FROM app_settings WHERE key=?", (key,))
    return json.loads(rows[0][0]) if rows else default


def setting_set(key: str, value: Any) -> None:
    """value = None 時刪除（回到預設）。"""
    if value is None:
        _x("DELETE FROM app_settings WHERE key=?", (key,))
    else:
        _x("INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
           (key, json.dumps(value)))
