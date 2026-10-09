"""背景工作佇列：標註與 waifu2x 放大（單一 worker 依序處理，標註工作內可平行處理多張圖）。"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import db, storage
from .config import settings
from .i18n import get_lang, t, use_lang
from .pipeline import tag_pil
from .profiles import normalize_settings

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    project_id: str
    image_ids: list[str]
    kind: str = "tag"  # tag：標註 | upscale：waifu2x 放大
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | cancelled | error
    done: int = 0
    failed: int = 0
    lang: str = "en"  # 提交者的語言：背景執行緒產生的錯誤訊息使用此語言
    message_key: str = ""  # 進度訊息存成語系鍵，查詢時以查詢者的語言顯示
    message_params: dict[str, Any] = field(default_factory=dict)
    message_raw: str = ""
    errors: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def say(self, key: str, **params: Any) -> None:
        self.message_key, self.message_params, self.message_raw = key, params, ""

    @property
    def message(self) -> str:
        return t(self.message_key, **self.message_params) if self.message_key else self.message_raw

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "kind": self.kind,
            "status": self.status,
            "total": len(self.image_ids),
            "done": self.done,
            "failed": self.failed,
            "message": self.message,
            "errors": self.errors[-20:],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


_jobs: dict[str, Job] = {}
_queue: "queue.Queue[Job]" = queue.Queue()
_lock = threading.Lock()
_worker: threading.Thread | None = None


def start_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, name="tag-worker", daemon=True)
        _worker.start()


def submit(project_id: str, image_ids: list[str], kind: str = "tag", params: dict[str, Any] | None = None) -> Job:
    # 已在佇列中的圖片不重複排入
    imgs = [i for i in db.list_images(project_id, ids=image_ids) if i["status"] not in ("queued", "processing")]
    job = Job(id=uuid.uuid4().hex[:12], project_id=project_id, image_ids=[i["id"] for i in imgs], kind=kind,
              params=params or {}, lang=get_lang())
    if not job.image_ids:
        job.status, job.finished_at = "done", time.time()
        job.say("msg.job_nothing")
    else:
        db.set_status(job.image_ids, "queued")
        _queue.put(job)
    with _lock:
        _jobs[job.id] = job
        _prune_old()
    start_worker()
    return job


def submit_task(owner_id: str, kind: str, image_ids: list[str] | None = None,
                params: dict[str, Any] | None = None) -> Job:
    """不屬於專案圖片的工作（例如角色篩選）；owner_id 放在 project_id，用來查詢進行中的工作。"""
    job = Job(id=uuid.uuid4().hex[:12], project_id=owner_id, image_ids=list(image_ids or []), kind=kind,
              params=params or {}, lang=get_lang())
    _queue.put(job)
    with _lock:
        _jobs[job.id] = job
        _prune_old()
    start_worker()
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_for_project(project_id: str) -> list[Job]:
    return sorted((j for j in _jobs.values() if j.project_id == project_id), key=lambda j: -j.created_at)


def active_for_project(project_id: str) -> Job | None:
    for j in list_for_project(project_id):
        if j.status in ("queued", "running"):
            return j
    return None


def any_active() -> bool:
    """有沒有排隊中或進行中的工作（標註或放大，任何專案）。"""
    return any(j.status in ("queued", "running") for j in list(_jobs.values()))


def cancel(job_id: str) -> Job | None:
    job = _jobs.get(job_id)
    if job and job.status in ("queued", "running"):
        job.cancel_event.set()
        if job.status == "queued":
            job.status, job.finished_at = "cancelled", time.time()
            job.say("msg.job_cancelled")
            db.restore_status(job.image_ids)
    return job


def cancel_project(project_id: str) -> None:
    for j in list_for_project(project_id):
        cancel(j.id)


def _prune_old(keep: int = 200) -> None:
    finished = sorted((j for j in _jobs.values() if j.finished_at), key=lambda j: j.finished_at or 0)
    for j in finished[:-keep] if len(finished) > keep else []:
        _jobs.pop(j.id, None)


def _process_one(job: Job, project_settings: dict[str, Any], iid: str) -> None:
    if job.cancel_event.is_set():
        return
    img = db.get_image(iid)
    if img is None:
        return
    db.set_status([iid], "processing")
    try:
        im = storage.load_image(img)
        with use_lang(job.lang):  # 執行緒池不繼承 contextvars，明確設定
            fields = tag_pil(im, project_settings, prev=img)
        db.update_image(iid, **fields)
        error = fields["error"] if fields["status"] == "error" else None
    except Exception as e:  # noqa: BLE001
        log.exception("標註失敗 %s", iid)
        db.update_image(iid, status="error", error=str(e))
        error = str(e)
    with _lock:
        job.done += 1
        if error is not None:
            job.failed += 1
            job.errors.append({"image_id": iid, "file": img["original_name"],
                               "error": error or t("msg.unknown_error", lang=job.lang)})


def _run_task(job: Job) -> None:
    """不屬於專案標註的工作：角色篩選、SMB 匯入。執行函式回傳完成訊息（語系鍵, 參數）。"""
    from . import finder, smb

    runner = {"finder": finder.run_job, "smb_import": smb.run_import}[job.kind]
    job.status, job.started_at = "running", time.time()
    try:
        with use_lang(job.lang):
            done = runner(job)
        if job.cancel_event.is_set() or done is None:
            job.status = "cancelled"
            job.say("msg.job_cancelled")
        else:
            job.status = "done"
            job.say(done[0], **done[1])
    except Exception as e:  # noqa: BLE001
        log.exception("工作失敗：%s", job.kind)
        job.status, job.message_key, job.message_raw = "error", "", str(e)
    finally:
        if job.kind == "finder":
            finder.unload_all()  # 偶爾才用，用完就釋放 VRAM
        job.finished_at = time.time()


def _run(job: Job) -> None:
    if job.kind in ("finder", "smb_import"):
        _run_task(job)
        return
    project = db.get_project(job.project_id)
    if project is None:
        job.status = "error"
        job.say("msg.project_missing")
        return
    s = normalize_settings(project["settings"])
    job.status, job.started_at = "running", time.time()
    job.say("msg.job_processing")
    done_key = {"tag": "msg.job_done", "upscale": "msg.upscale_done"}[job.kind]
    if job.kind == "upscale" and job.params.get("scale") == "1":
        done_key = "msg.denoise_done"
    try:
        if job.kind == "upscale":
            from . import upscale

            with use_lang(job.lang):
                upscale.run_job(job)
        else:
            _run_tagging(job, s)
        if job.cancel_event.is_set():
            job.status = "cancelled"
            job.say("msg.job_cancelled")
        else:
            job.status = "done"
            if job.failed:
                job.say(f"{done_key}_failed", ok=job.done - job.failed, failed=job.failed)
            else:
                job.say(done_key, ok=job.done)
    except Exception as e:  # noqa: BLE001
        log.exception("工作失敗")
        job.status, job.message_key, job.message_raw = "error", "", str(e)
    finally:
        if job.kind == "upscale":
            from . import upscale

            upscale.unload_all()  # 偶爾才用，用完就釋放 VRAM
        db.restore_status(job.image_ids)
        job.finished_at = time.time()
        db.touch_project(job.project_id)


def _run_tagging(job: Job, s: dict[str, Any]) -> None:
    # 第一次使用時會下載模型，先在主執行緒載入以取得清楚的錯誤訊息
    if s.get("use_wd14", True):
        job.say("msg.job_loading_model")
        from .tagging.wd14 import get_tagger

        get_tagger(s.get("wd14_model") or None)
        job.say("msg.job_processing")
    with ThreadPoolExecutor(max_workers=settings.tag_concurrency) as pool:
        list(pool.map(lambda iid: _process_one(job, s, iid), job.image_ids))


def _loop() -> None:
    while True:
        job = _queue.get()
        try:
            if job.status == "queued" and not job.cancel_event.is_set():
                _run(job)
        finally:
            _queue.task_done()
