"""把訓練好的 LoRA 匯入 A1111 / Forge，以及手動卸載 Forge 的模型（把 VRAM 讓給本機訓練）。

本機訓練的 LoRA 用 POST /sdapi/v1/lora/upload 把檔案傳過去。Civitai 的
用 Forge Neo Chino 的 LoRA 匯入 API（POST /sdapi/v1/lora/import）：把 Civitai 的簽名下載網址交給 Forge，
由 Forge 自己下載（不經過這裡），並寫好卡片資訊（觸發詞、底模類型）與預覽圖，不必重開 Forge。
A1111_URL 沒設定時不顯示這個功能。
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from . import civitai, db
from .config import settings
from .i18n import get_lang, t, use_lang
from .services import BadRequest, NotFound

# Civitai 訓練的 ecosystem → Forge 的底模類型（卡片的 "sd version"，Forge 用來依預設篩選 LoRA）
FORGE_ARCH = {"sd1": "sd", "sdxl": "xl", "flux1": "flux", "flux2klein": "klein", "qwen": "qwen", "qwen21": "qwen",
              "zimageturbo": "zit", "zimagebase": "zit", "wan": "wan", "anima": "anima", "ernie": "ernie",
              "krea2": "krea"}
_VIDEO = re.compile(r"\.(mp4|webm)(\?|$)", re.IGNORECASE)
log = logging.getLogger(__name__)
_status_cache: dict[str, Any] = {"at": 0.0, "value": None}
_status_lock = threading.Lock()
_status_refreshing = False
STATUS_CONNECT_S = 3  # 檢查 Forge 狀態時的連線逾時：Forge 沒開、封包被丟掉時不要等 15 秒


class A1111Error(BadRequest):
    pass


class A1111Conflict(A1111Error):
    """Forge 已經有同名、內容不同的檔案（可以用 overwrite 覆蓋）。"""


def configured() -> bool:
    return bool(settings.a1111_url)


def _client(read_timeout: float = 30, connect_timeout: float = 15) -> httpx.Client:
    auth = tuple(settings.a1111_api_auth.split(":", 1)) if ":" in settings.a1111_api_auth else None
    return httpx.Client(base_url=settings.a1111_url, timeout=httpx.Timeout(15, connect=connect_timeout, read=read_timeout),
                        auth=auth)


def status(max_age: float = 30, wait: bool = True) -> dict[str, Any]:
    """Forge 連得上、而且有匯入 API 嗎（結果暫存 max_age 秒）。
    wait=False（VRAM 選單）：暫存過期時直接回上一次的結果，在背景重新檢查，選單不用等 Forge 回應或逾時。"""
    if not configured():
        return {"configured": False}
    cached = _status_cache["value"]
    if cached is not None and time.time() - _status_cache["at"] < max_age:
        return cached
    if cached is not None and not wait:
        _refresh_status_in_background()
        return cached
    return _check_status()


def _refresh_status_in_background() -> None:
    """同一時間只跑一個背景檢查。"""
    global _status_refreshing
    with _status_lock:
        if _status_refreshing:
            return
        _status_refreshing = True
    lang = get_lang()  # 錯誤訊息用觸發檢查的請求的語言（和原本在請求裡檢查時一樣）

    def run() -> None:
        global _status_refreshing
        try:
            with use_lang(lang):
                _check_status()
        except Exception:  # noqa: BLE001  背景檢查失敗不影響畫面，下次再查
            log.exception("檢查 Forge 狀態失敗")
        finally:
            _status_refreshing = False

    threading.Thread(target=run, name="forge-status", daemon=True).start()


def _check_status() -> dict[str, Any]:
    out: dict[str, Any] = {"configured": True, "url": settings.a1111_url, "subfolder": settings.a1111_lora_subfolder}
    try:
        with _client(10, connect_timeout=STATUS_CONNECT_S) as c:
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
    # civitai_runs.ecosystem 存的是訓練類型 id（flux1-dev、wan-2.2…），FORGE_ARCH 以 ecosystem（flux1、wan…）為鍵
    ecosystem = civitai.TRAINING_TYPES.get(run["ecosystem"], {}).get("ecosystem", run["ecosystem"])
    body = {
        "url": ep["url"],
        "filename": f"{_slug(trigger or project['name'])}_{_run_tag(workflow_id)}_e{epoch:02d}",
        "subfolder": settings.a1111_lora_subfolder,
        "activation_text": trigger,
        "sd_version": FORGE_ARCH.get(ecosystem, "Unknown"),
        "description": t("msg.a1111_description", project=project["name"], epoch=epoch, total=total, id=workflow_id),
        "notes": notes,
        "preview_url": preview,
        "overwrite": overwrite,
    }
    result = _post_lora(lambda c: c.post("/sdapi/v1/lora/import", json=body), body["filename"])
    db.a1111_import_put(workflow_id, epoch, result.get("name") or body["filename"], result.get("relative_path") or "",
                        result.get("prompt") or "")
    _status_cache["value"] = None
    return {**result, "epoch": epoch, "workflow_id": workflow_id}


def _post_lora(send: Any, filename: str) -> dict[str, Any]:
    """呼叫匯入 / 上傳 API，錯誤轉成看得懂的訊息。"""
    try:
        with _client(900) as c:  # Forge 要收完整個 LoRA（SDXL 約 50–200 MB）
            r = send(c)
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
        raise A1111Conflict(t("msg.a1111_conflict", name=filename))
    if r.status_code != 200:
        raise A1111Error(t("msg.a1111_failed", error=f"HTTP {r.status_code} {detail}".strip()))
    return r.json()


def import_local(run: dict[str, Any], path: Path, epoch: int, trigger: str, name: str, sd_version: str,
                 overwrite: bool = False) -> dict[str, Any]:
    """把本機訓練的一個 epoch 傳給 Forge（Forge 不一定連得到 LTS，所以是傳檔案，不是給網址）。"""
    if not configured():
        raise A1111Error(t("msg.a1111_not_configured"))
    filename = f"{_slug(trigger or name)}_{run['id']}_e{epoch:02d}"
    p = run["params"]
    notes = ", ".join(f"{k}={p[k]}" for k in ("steps", "epochs", "lr", "network_dim", "network_alpha", "optimizer") if p.get(k) is not None)
    data = {"filename": filename, "subfolder": settings.a1111_lora_subfolder, "activation_text": trigger,
            "sd_version": sd_version, "notes": notes, "overwrite": str(overwrite).lower(),
            "description": t("msg.a1111_description_local", project=name, epoch=epoch, total=p.get("epochs") or epoch)}
    with path.open("rb") as f:
        result = _post_lora(lambda c: c.post("/sdapi/v1/lora/upload", data=data,
                                             files={"file": (path.name, f, "application/octet-stream")}), filename)
    db.a1111_import_put(run["id"], epoch, result.get("name") or filename, result.get("relative_path") or "", result.get("prompt") or "")
    _status_cache["value"] = None
    return {**result, "epoch": epoch, "run_id": run["id"]}


def unload() -> None:
    """把 Forge 目前的模型移出 VRAM（下次生圖時它會自己載回來）。"""
    if not configured():
        raise A1111Error(t("msg.a1111_not_configured"))
    try:
        with _client(120) as c:
            r = c.post("/sdapi/v1/unload-checkpoint")
    except httpx.HTTPError as e:
        raise A1111Error(t("msg.a1111_unreachable", url=settings.a1111_url, error=e)) from e
    if r.status_code != 200:
        raise A1111Error(t("msg.a1111_failed", error=f"HTTP {r.status_code}"))
