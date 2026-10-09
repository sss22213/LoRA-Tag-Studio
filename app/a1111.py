"""把 Civitai 訓練好的 LoRA 匯入 A1111 / Forge。

用 Forge Neo Chino 的 LoRA 匯入 API（POST /sdapi/v1/lora/import）：把 Civitai 的簽名下載網址交給 Forge，
由 Forge 自己下載（不經過這裡），並寫好卡片資訊（觸發詞、底模類型）與預覽圖，不必重開 Forge。
A1111_URL 沒設定時不顯示這個功能。
"""
from __future__ import annotations

import re
import time
from typing import Any

import httpx

from . import civitai, db
from .config import settings
from .i18n import t
from .services import BadRequest, NotFound

# Civitai 訓練的 ecosystem → Forge 的底模類型（卡片的 "sd version"，Forge 用來依預設篩選 LoRA）
FORGE_ARCH = {"sd1": "sd", "sdxl": "xl", "flux1": "flux", "flux2klein": "klein", "qwen": "qwen", "qwen21": "qwen",
              "zimageturbo": "zit", "zimagebase": "zit", "wan": "wan", "anima": "anima", "ernie": "ernie",
              "krea2": "krea"}
_VIDEO = re.compile(r"\.(mp4|webm)(\?|$)", re.IGNORECASE)
_status_cache: dict[str, Any] = {"at": 0.0, "value": None}


class A1111Error(BadRequest):
    pass


class A1111Conflict(A1111Error):
    """Forge 已經有同名、內容不同的檔案（可以用 overwrite 覆蓋）。"""


def configured() -> bool:
    return bool(settings.a1111_url)


def _client(read_timeout: float = 30) -> httpx.Client:
    auth = tuple(settings.a1111_api_auth.split(":", 1)) if ":" in settings.a1111_api_auth else None
    return httpx.Client(base_url=settings.a1111_url, timeout=httpx.Timeout(15, read=read_timeout), auth=auth)


def status(max_age: float = 30) -> dict[str, Any]:
    """Forge 連得上、而且有匯入 API 嗎（結果暫存 max_age 秒）。"""
    if not configured():
        return {"configured": False}
    if _status_cache["value"] is not None and time.time() - _status_cache["at"] < max_age:
        return _status_cache["value"]
    out: dict[str, Any] = {"configured": True, "url": settings.a1111_url, "subfolder": settings.a1111_lora_subfolder}
    try:
        with _client(10) as c:
            r = c.get("/sdapi/v1/lora/import")
        if r.status_code == 200:
            out.update(available=True, lora_dir=r.json().get("lora_dir"))
        else:
            out.update(available=False, error=t("msg.a1111_no_api") if r.status_code == 404
                       else t("msg.a1111_failed", error=f"HTTP {r.status_code}"))
    except httpx.HTTPError as e:
        out.update(available=False, error=t("msg.a1111_unreachable", url=settings.a1111_url, error=e))
    _status_cache.update(at=time.time(), value=out)
    return out


def _slug(text: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", text.strip(), flags=re.UNICODE).strip("._")
    return re.sub(r"_+", "_", s)[:60] or "lora"


def _run_tag(workflow_id: str) -> str:
    """檔名裡辨識這次訓練的部分。Civitai 的 ID 像 3627068-20261004095650956-3rmo（帳號-時間-亂數），
    開頭的帳號每次都一樣，所以用日期加結尾的亂數：20261004_3rmo。"""
    m = re.match(r"^\d+-(\d{8})\d*-(\w+)$", workflow_id)
    return f"{m.group(1)}_{m.group(2)}" if m else re.sub(r"\W+", "", workflow_id)[-8:]


def import_epoch(workflow_id: str, epoch: int, overwrite: bool = False) -> dict[str, Any]:
    """把一次訓練的某個 epoch 匯入 Forge 的 Lora 資料夾（A1111_LORA_SUBFOLDER 子資料夾）。"""
    if not configured():
        raise A1111Error(t("msg.a1111_not_configured"))
    run = db.civitai_run_get(workflow_id)
    if run is None:
        raise NotFound(t("msg.civitai_run_not_found", id=workflow_id))
    summary = civitai.refresh_run(workflow_id, max_age=60)  # 下載網址是有時效的簽名網址：用新的
    epochs = summary.get("epochs") or []
    ep = next((e for e in epochs if e.get("epoch") == epoch and e.get("available") and e.get("url")), None)
    if ep is None:
        raise A1111Error(t("msg.a1111_epoch_missing", epoch=epoch))
    project = db.get_project(run["project_id"]) or {"name": "", "settings": {}}
    trigger = (project.get("settings") or {}).get("trigger", "")
    total = max((e.get("epoch") or 0 for e in epochs), default=epoch)
    params = summary.get("params") or {}
    notes = ", ".join(f"{k}={params[k]}" for k in ("steps", "epochs", "lr", "networkDim", "networkAlpha", "optimizerType")
                      if params.get(k) is not None)
    preview = next((u for u in ep.get("samples") or [] if not _VIDEO.search(u)), None) or next(iter(ep.get("samples") or []), None)
    body = {
        "url": ep["url"],
        "filename": f"{_slug(trigger or project['name'])}_{_run_tag(workflow_id)}_e{epoch:02d}",
        "subfolder": settings.a1111_lora_subfolder,
        "activation_text": trigger,
        "sd_version": FORGE_ARCH.get(run["ecosystem"], "Unknown"),
        "description": t("msg.a1111_description", project=project["name"], epoch=epoch, total=total, id=workflow_id),
        "notes": notes,
        "preview_url": preview,
        "overwrite": overwrite,
    }
    try:
        with _client(900) as c:  # Forge 要先把整個 LoRA 從 Civitai 下載下來（SDXL 約 50–200 MB）
            r = c.post("/sdapi/v1/lora/import", json=body)
    except httpx.HTTPError as e:
        raise A1111Error(t("msg.a1111_unreachable", url=settings.a1111_url, error=e)) from e
    detail = ""
    try:
        detail = r.json().get("detail") or ""
    except ValueError:
        detail = r.text[:300]
    if r.status_code == 404:
        raise A1111Error(t("msg.a1111_no_api"))
    if r.status_code == 409:
        raise A1111Conflict(t("msg.a1111_conflict", name=body["filename"]))
    if r.status_code != 200:
        raise A1111Error(t("msg.a1111_failed", error=f"HTTP {r.status_code} {detail}".strip()))
    result = r.json()
    db.a1111_import_put(workflow_id, epoch, result.get("name") or body["filename"], result.get("relative_path") or "",
                        result.get("prompt") or "")
    _status_cache["value"] = None
    return {**result, "epoch": epoch, "workflow_id": workflow_id}
