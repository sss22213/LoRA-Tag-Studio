"""本機訓練：把資料集交給 AI Toolkit（ostris/ai-toolkit）訓練 LoRA，表單和流程比照 Civitai 訓練。

AI Toolkit 是獨立的 Docker 服務（compose 的 aitoolkit profile），它的網頁不對外開放，全部操作都在 LTS：
1. 送出時把資料集複製成快照（data/aitoolkit/datasets/<工作名>，圖片 + 同名 .txt），之後改標籤不影響進行中的訓練
2. 用 HTTP API 建立工作、排進佇列（一張 GPU 一次跑一個）
3. 進度和 loss 從 API 讀，LoRA 和範例圖從共用資料夾 data/aitoolkit/output/<工作名> 讀

Epochs：AI Toolkit 只有「每 N 步存檔」，這裡換算成 Civitai 表單的 epochs，並保留每個 epoch。
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from . import db, finder, jobs, storage
from .civitai import LR_SCHEDULERS, OPTIMIZERS, default_negative, default_samples
from .config import settings
from .exporter import convert_image, dataset_images, slugify
from .i18n import get_lang, t, use_lang
from .pipeline import caption_for
from .profiles import normalize_settings
from .services import BadRequest, NotFound
from .tagging.postprocess import policy_flag, split_tags, training_hints

log = logging.getLogger(__name__)


class AIToolkitError(BadRequest):
    pass


# 參數預設值比照 Civitai 的同類型（civitai.TRAINING_TYPES）；底模是本機檔案時 file 是檔名，repo 是 Hugging Face 的 repo
_SD_FIELDS = ("base_model", "min_snr_gamma", "train_text_encoder", "text_encoder_lr")


def _model(label: str, family: str, arch: str, repo: str | None = None, file: str | None = None,
           steps: int = 2000, optimizer: str = "adamw8bit", noise: float = 0.0, cfg: float = 7,
           resolutions: tuple[int, ...] = (512, 768, 1024), size: int = 1024) -> dict[str, Any]:
    sd = family != "anima"
    defaults: dict[str, Any] = {
        "steps": steps, "epochs": 10, "batch_size": 1, "lr": 0.0001, "lr_scheduler": "cosine", "optimizer": optimizer,
        "network_dim": 32, "network_alpha": 32, "noise_offset": noise, "flip_augmentation": False,
        "sample_cfg": cfg, "sample_strength": 1.0, "resolutions": list(resolutions),
    }
    if sd:
        defaults.update(min_snr_gamma=5, train_text_encoder=True, text_encoder_lr=0.00005)
    return {"label": label, "family": family, "arch": arch, "repo": repo, "file": file, "defaults": defaults,
            "fields": _SD_FIELDS if sd else (), "sd": sd, "size": size, "max_batch": 4}


MODELS: dict[str, dict[str, Any]] = {
    "sd15": _model("SD 1.5", "sd15", "sd1", repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
                   file="v1-5-pruned-emaonly.safetensors", resolutions=(512,), size=512),
    "sdxl": _model("SDXL", "sdxl", "sdxl", repo="stabilityai/stable-diffusion-xl-base-1.0", optimizer="adafactor", noise=0.1),
    "illustrious": _model("Illustrious XL", "sdxl", "sdxl", file="illustriousXL_v01.safetensors", optimizer="adafactor", noise=0.1),
    "noobai": _model("NoobAI-XL", "sdxl", "sdxl", file="noobaiXLNAIXL_earlyAccessVersion.safetensors",
                     optimizer="adafactor", noise=0.1),
    # 步數 1500 和 Civitai 的 Anima 預設相同
    "anima": _model("Anima", "anima", "anima", repo="circlestone-labs/Anima-Base-v1.0-Diffusers", steps=1500, cfg=4),
}
# 標註專案的底模 → 預設的訓練底模（其他底模沒有對應，預設 SDXL，自己選）
PROFILE_MODELS = {"sd15_anime": "sd15", "sd15_realistic": "sd15", "sdxl_base": "sdxl", "pony_v6": "sdxl",
                  "illustrious": "illustrious", "noobai_xl": "noobai", "animagine_xl": "sdxl", "anima": "anima"}
FORGE_VERSIONS = {"sd15": "sd", "sdxl": "xl", "anima": "anima"}  # Forge LoRA 卡片的底模類型
RESOLUTIONS = (512, 768, 1024, 1280, 1536)
SOURCES = ("project", "folder")
TRAIN_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")  # AI Toolkit 只讀這些，其他格式在快照裡轉成 JPEG
STOP_RETRY_S = 30  # 停止後這麼久還沒結束，再按一次就直接標記為停止
CANCEL_RECHECK_S = 1.5  # 取消剛排進佇列的工作後，隔多久確認 AI Toolkit 沒有剛好啟動它（它的 worker 每秒看一次佇列）
CANCEL_CHECKS = 3
ACTIVE = ("preparing", "queued", "running", "stopping")
# AI Toolkit 的工作狀態 → LTS 的狀態
JOB_STATUS = {"queued": "queued", "running": "running", "stopping": "stopping", "completed": "succeeded", "error": "failed",
              "stopped": "canceled"}
_JOB_ACTIVE = ("queued", "running", "stopping")  # AI Toolkit 這邊還會（或正在）跑的工作狀態
_STATUS_TTL = 5.0
_BASE_TTL = 60.0
_status_cache: dict[str, Any] = {"at": 0.0, "value": None}
_base_cache: dict[str, Any] = {"at": 0.0, "value": None}


# ------------------------------------------------------------------ HTTP
def configured() -> bool:
    return bool(settings.aitoolkit_url)


def _client(read_timeout: float = 30) -> httpx.Client:
    headers = {"Authorization": f"Bearer {settings.aitoolkit_auth}"} if settings.aitoolkit_auth else {}
    return httpx.Client(base_url=settings.aitoolkit_url, headers=headers, timeout=httpx.Timeout(10, read=read_timeout))


def _request(method: str, path: str, **kw: Any) -> httpx.Response:
    if not configured():
        raise AIToolkitError(t("msg.aitoolkit_not_configured"))
    try:
        with _client() as c:
            r = c.request(method, path, **kw)
    except httpx.HTTPError as e:
        raise AIToolkitError(t("msg.aitoolkit_unreachable", error=e)) from e
    if r.status_code == 401:
        raise AIToolkitError(t("msg.aitoolkit_auth"))
    return r


def status(force: bool = False) -> dict[str, Any]:
    """服務連得上嗎，以及它看到的資料夾（結果暫存幾秒）。"""
    if not configured():
        return {"configured": False, "online": False}
    if not force and _status_cache["value"] and time.time() - _status_cache["at"] < _STATUS_TTL:
        return _status_cache["value"]
    out: dict[str, Any] = {"configured": True, "online": False}
    try:
        r = _request("GET", "/api/settings")
        if r.status_code == 200:
            d = r.json()
            out.update(online=True, datasets=d.get("DATASETS_FOLDER"), output=d.get("TRAINING_FOLDER"))
        else:
            out["error"] = t("msg.aitoolkit_failed", error=f"HTTP {r.status_code}")
    except AIToolkitError as e:
        out["error"] = str(e)
    _status_cache.update(at=time.time(), value=out)
    return out


def _require_online() -> dict[str, Any]:
    st = status(force=True)
    if not st["configured"]:
        raise AIToolkitError(t("msg.aitoolkit_not_configured"))
    if not st["online"]:
        raise AIToolkitError(st.get("error") or t("msg.aitoolkit_offline"))
    return st


# ------------------------------------------------------------------ 底模
def _family_of(path: Path) -> str | None:
    """本機檔案的底模家族。Forge 的 .civitai.info 有記錄就用它（Anima 這類大小很接近 SD 1.5 的不會混進來），沒有就看檔案大小。"""
    try:
        base = str(json.loads(path.with_suffix(".civitai.info").read_text()).get("baseModel") or "")
    except (OSError, ValueError):
        base = ""
    if base:
        if base.startswith("SD 1"):
            return "sd15"
        return "sdxl" if re.search(r"SDXL|Illustrious|NoobAI|Pony|Animagine", base) else None
    return "sdxl" if path.stat().st_size > 5e9 else "sd15"


def base_models() -> dict[str, list[dict[str, Any]]]:
    """底模資料夾裡可以訓練的檔案，依家族分開（結果暫存一分鐘）。"""
    if _base_cache["value"] is not None and time.time() - _base_cache["at"] < _BASE_TTL:
        return _base_cache["value"]
    out: dict[str, list[dict[str, Any]]] = {"sd15": [], "sdxl": []}
    root = settings.aitoolkit_base_models
    if root.is_dir():
        for p in sorted(root.glob("*.safetensors"), key=lambda x: x.name.lower()):
            try:
                family = _family_of(p)
                if family:
                    out[family].append({"name": p.name, "size": p.stat().st_size})
            except OSError:
                continue
    _base_cache.update(at=time.time(), value=out)
    return out


def _resolve_base(tp: dict[str, Any], name: str) -> str:
    """要交給 AI Toolkit 的底模：指定的本機檔案 → 這個底模預設的檔案 → Hugging Face 的 repo。"""
    root = settings.aitoolkit_base_models
    if name:
        if not tp["sd"] or Path(name).name != name or not (root / name).is_file():
            raise AIToolkitError(t("msg.train_base_missing", name=name))
        return str(root / name)
    if tp["file"] and (root / tp["file"]).is_file():
        return str(root / tp["file"])
    if tp["repo"]:
        return tp["repo"]
    raise AIToolkitError(t("msg.train_base_missing", name=tp["file"]))


def info() -> dict[str, Any]:
    st = status()
    bases = base_models()
    models = []
    for key, tp in MODELS.items():
        root = settings.aitoolkit_base_models
        models.append({"id": key, "label": tp["label"], "family": tp["family"], "fields": list(tp["fields"]), "defaults": tp["defaults"],
                       "max_batch": tp["max_batch"], "default_file": tp["file"], "repo": tp["repo"],
                       "available": bool(tp["repo"] or (tp["file"] and (root / tp["file"]).is_file()))})
    return {**st, "models": models, "base_models": bases, "optimizers": list(OPTIMIZERS), "lr_schedulers": list(LR_SCHEDULERS),
            "resolutions": list(RESOLUTIONS), "profiles": PROFILE_MODELS}


# ------------------------------------------------------------------ 訓練專案
def _require_project(pid: str) -> dict[str, Any]:
    p = db.train_project_get(pid)
    if p is None:
        raise NotFound(t("msg.train_project_missing"))
    return p


def _folder_images(folder: str) -> list[Path]:
    try:
        root = storage.safe_import_path(folder)
    except ValueError as e:
        raise BadRequest(str(e)) from e
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in storage.IMAGE_EXTS)


def _dataset_view(p: dict[str, Any]) -> dict[str, Any]:
    """表單要用的資料：張數、觸發詞、caption 結構的建議、範例圖的負面提示。"""
    if p["source"] == "folder":
        try:
            files = _folder_images(p["folder"] or "")
        except BadRequest as e:
            return {"available": False, "error": str(e), "images": 0}
        return {"available": True, "images": len(files), "done": len(files), "pending": 0, "flagged": 0,
                "captions": sum(1 for f in files if f.with_suffix(".txt").is_file()), "trigger": "", "negative": "",
                "hints": {"shuffle_caption": False, "keep_tokens": 0}}
    project = db.get_project(p["project_id"]) if p["project_id"] else None
    if project is None:
        return {"available": False, "error": t("msg.train_source_gone"), "images": 0}
    s = normalize_settings(project["settings"])
    images = db.list_images(project["id"])
    done = sum(1 for i in images if i["status"] == "done")
    return {"available": True, "images": len(images), "done": done, "pending": len(images) - done, "project_name": project["name"],
            "trigger": s.get("trigger") or "", "negative": default_negative(s), "hints": training_hints(s), "profile": s["profile"],
            "cover_id": images[0]["id"] if images else None}


def project_view(p: dict[str, Any], runs: bool = False) -> dict[str, Any]:
    out = {**p, "dataset": _dataset_view(p)}
    all_runs = db.train_runs(p["id"])
    out["run_count"] = len(all_runs)
    if runs:
        out["runs"] = [run_view(_refresh(r)) for r in all_runs]
    elif all_runs:
        last = _refresh(all_runs[0])
        out["last_run"] = {"id": last["id"], "status": last["status"], "model": last["model"], "created_at": last["created_at"],
                           "progress": _progress(last)}
    return out


def list_projects() -> list[dict[str, Any]]:
    return [project_view(p) for p in db.train_projects()]


def get_project(pid: str) -> dict[str, Any]:
    return project_view(_require_project(pid), runs=True)


def create_project(name: str, source: str, project_id: str | None = None, folder: str | None = None,
                   model: str | None = None) -> dict[str, Any]:
    if source not in SOURCES:
        raise BadRequest(t("msg.train_bad_source", source=source))
    if source == "project":
        project = db.get_project(project_id or "")
        if project is None:
            raise NotFound(t("msg.project_missing"))
        folder = None
        name = name.strip() or project["name"]
        model = model or PROFILE_MODELS.get(normalize_settings(project["settings"])["profile"], "sdxl")
    else:
        project_id = None
        if not _folder_images(folder or ""):
            raise BadRequest(t("msg.train_folder_empty"))
        name = name.strip() or Path(folder or "").name or "training"
    model = model or "sdxl"
    if model not in MODELS:
        raise BadRequest(t("msg.train_bad_model", model=model, available=", ".join(MODELS)))
    pid = db.new_id()
    db.train_project_add(pid, name[:120], source, project_id, folder, {"model": model})
    return project_view(_require_project(pid))


def project_for(project_id: str) -> dict[str, Any]:
    """標註專案送本機訓練時用的訓練專案：有就沿用，沒有就建立。"""
    p = db.train_project_for(project_id)
    return project_view(p, runs=False) if p else create_project("", "project", project_id=project_id)


def rename_project(pid: str, name: str) -> dict[str, Any]:
    _require_project(pid)
    if not name.strip():
        raise BadRequest(t("msg.train_name_required"))
    db.train_project_update(pid, name=name.strip()[:120])
    return project_view(_require_project(pid))


def delete_project(pid: str) -> None:
    """連同訓練紀錄和 LoRA 一起刪掉。還在跑的不能刪。"""
    _require_project(pid)
    runs = db.train_runs(pid)
    if any(_refresh(r)["status"] in ACTIVE for r in runs):
        raise BadRequest(t("msg.train_project_busy"))
    for r in runs:
        delete_run(r["id"])
    db.train_project_delete(pid)


# ------------------------------------------------------------------ 送出訓練
def _save_every(steps: int, epochs: int) -> int:
    return max(1, math.ceil(steps / max(1, epochs)))


def build_job(name: str, folder: str, output: str, tp: dict[str, Any], base: str, p: dict[str, Any], prompts: list[str],
              negative: str, trigger: str, pretrained: str | None = None) -> dict[str, Any]:
    """AI Toolkit 的工作設定（和它網頁送出的 job_config 相同）。"""
    sd = tp["sd"]
    every = _save_every(p["steps"], p["epochs"])
    cache_text = bool(p.get("cache_text")) and not p.get("train_text_encoder") and not p.get("shuffle_tokens")
    network: dict[str, Any] = {"type": "lora", "linear": p["network_dim"], "linear_alpha": p["network_alpha"],
                               "network_kwargs": {"ignore_if_contains": []}}
    if pretrained:
        network["pretrained_lora_path"] = pretrained
    train: dict[str, Any] = {
        "batch_size": p["batch_size"], "steps": p["steps"], "gradient_accumulation": 1, "train_unet": True,
        "train_text_encoder": bool(p.get("train_text_encoder")), "gradient_checkpointing": True,
        "noise_scheduler": "ddpm" if sd else "flowmatch", "optimizer": p["optimizer"], "lr": p["lr"],
        "lr_scheduler": p["lr_scheduler"], "optimizer_params": {"weight_decay": 1e-4}, "noise_offset": p["noise_offset"],
        "timestep_type": "sigmoid" if sd else "weighted", "content_or_style": "balanced", "loss_type": "mse", "dtype": "bf16",
        "unload_text_encoder": False, "cache_text_embeddings": cache_text, "skip_first_sample": True, "force_first_sample": False,
        "disable_sampling": False, "diff_output_preservation": False, "ema_config": {"use_ema": False, "ema_decay": 0.99},
    }
    if p.get("train_text_encoder") and p.get("text_encoder_lr"):
        train["text_encoder_lr"] = p["text_encoder_lr"]
    if p.get("min_snr_gamma"):
        train["min_snr_gamma"] = p["min_snr_gamma"]
    return {"job": "extension", "meta": {"name": "[name]", "version": "1.0"}, "config": {"name": name, "process": [{
        "type": "diffusion_trainer", "training_folder": output, "sqlite_db_path": "./aitk_db.db", "device": "cuda",
        "trigger_word": trigger or None, "performance_log_every": 10, "network": network,
        "save": {"dtype": "float16" if sd else "bf16", "save_every": every, "max_step_saves_to_keep": p["epochs"] + 1,
                 "save_format": "diffusers", "push_to_hub": False},
        "datasets": [{
            "folder_path": folder, "caption_ext": "txt", "default_caption": "", "caption_dropout_rate": 0.0 if cache_text else 0.05,
            "cache_latents_to_disk": bool(p.get("cache_latents", True)), "resolution": p["resolutions"], "flip_x": bool(p["flip_augmentation"]),
            "flip_y": False, "num_repeats": 1, "is_reg": False, "network_weight": 1, "controls": [],
            "shuffle_tokens": bool(p.get("shuffle_tokens")), "keep_tokens": p.get("keep_tokens") or 0,
        }],
        "train": train, "logging": {"log_every": 1, "use_ui_logger": True},
        "model": {"name_or_path": base, "arch": tp["arch"], "quantize": False, "qtype": "", "quantize_te": False, "qtype_te": "",
                  "low_vram": False, "model_kwargs": {}, "compile": False},
        "sample": {"sampler": "ddpm" if sd else "flowmatch", "sample_every": every, "sample_start_step": 0, "width": tp["size"],
                   "height": tp["size"], "samples": [{"prompt": x} for x in prompts], "neg": negative, "seed": 42, "walk_seed": True,
                   "guidance_scale": p["sample_cfg"], "sample_steps": 25, "network_multiplier": p["sample_strength"]},
    }]}}


def _clean_params(tp: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """表單沒填的用這個底模的預設值；沒有這個參數的底模不留。"""
    d = tp["defaults"]
    p = {k: (v if v is not None else d.get(k)) for k, v in params.items()}
    for k, v in d.items():
        if p.get(k) is None:
            p[k] = v
    p["batch_size"] = max(1, min(int(p["batch_size"]), tp["max_batch"]))
    p["resolutions"] = sorted({int(x) for x in p["resolutions"]}) or d["resolutions"]
    if not tp["sd"]:
        for k in ("min_snr_gamma", "train_text_encoder", "text_encoder_lr", "base_model"):
            p.pop(k, None)
    return p


def _pretrained(train_project_id: str, ref: str, st: dict[str, Any]) -> str | None:
    """接續訓練：<run id>:<epoch> → AI Toolkit 看到的 LoRA 路徑。"""
    if not ref:
        return None
    rid, _, epoch = ref.partition(":")
    run = db.train_run_get(rid)
    if run is None or run["train_project_id"] != train_project_id:
        raise BadRequest(t("msg.train_run_missing"))
    ep = next((e for e in _epochs(run) if str(e["epoch"]) == epoch and e.get("file")), None)
    if ep is None:
        raise BadRequest(t("msg.train_epoch_missing", epoch=epoch))
    return f"{st['output']}/{run['job_name']}/{ep['file']}"


def _write_snapshot(p: dict[str, Any], params: dict[str, Any], dest: Path) -> dict[str, Any]:
    """把資料集寫成圖片 + 同名 .txt，回傳張數、排除的圖片、預設範例提示和負面提示。"""
    dest.mkdir(parents=True, exist_ok=True)
    max_side = int(params.get("max_side") or 0)
    if p["source"] == "folder":
        return _snapshot_folder(p, dest, max_side)
    project, s, images, excluded = dataset_images(p["project_id"], bool(params.get("only_done", True)))
    for k, img in enumerate(images):
        # 不需要縮小的圖直接複製原檔，不重新壓縮
        big = max_side and max(img["width"] or 0, img["height"] or 0) > max_side
        data, ext = convert_image(img, "jpg" if big else "original", max_side if big else 0)
        (dest / f"{k:05d}{ext}").write_bytes(data)
        (dest / f"{k:05d}.txt").write_text(caption_for(img, s) + "\n", encoding="utf-8")
    return {"count": len(images), "excluded": excluded, "samples": default_samples(images, s), "negative": default_negative(s),
            "project": project["name"]}


def _snapshot_folder(p: dict[str, Any], dest: Path, max_side: int) -> dict[str, Any]:
    from PIL import Image

    count, excluded, samples = 0, [], []
    for f in _folder_images(p["folder"] or ""):
        cap_file = f.with_suffix(".txt")
        caption = cap_file.read_text(encoding="utf-8", errors="replace").strip() if cap_file.is_file() else ""
        reason = policy_flag(split_tags(caption), None, caption)  # 和標註專案相同的規則
        if reason:
            excluded.append({"file": f.name, "reason": reason})
            continue
        name = f"{count:05d}"
        im = None
        with Image.open(f) as src:
            if f.suffix.lower() not in TRAIN_IMAGE_EXTS or (max_side and max(src.size) > max_side):
                im = src.convert("RGB")
                if max_side:
                    im.thumbnail((max_side, max_side), Image.LANCZOS)
        if im is not None:
            im.save(dest / f"{name}.jpg", quality=95)
        else:
            shutil.copyfile(f, dest / f"{name}{f.suffix.lower()}")
        (dest / f"{name}.txt").write_text(caption + "\n", encoding="utf-8")
        if caption and len(samples) < 3:
            samples.append(caption[:400])
        count += 1
    if not count:
        raise BadRequest(t("msg.train_folder_empty"))
    return {"count": count, "excluded": excluded, "samples": samples, "negative": "", "project": p["name"]}


def _free_vram() -> list[str]:
    """釋放 LTS 自己佔的 VRAM。標註還在跑時不動（下一張圖會再載回來）；Forge 是別的服務，由使用者手動卸載。"""
    if jobs.any_active():
        return []
    try:
        return finder.free_other_vram()
    except Exception as e:  # noqa: BLE001  釋放失敗不影響訓練，只是 VRAM 比較緊
        log.warning("訓練前釋放 VRAM 失敗：%s", e)
        return []


def start_run(train_project_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """建立一次訓練，在背景準備資料後交給 AI Toolkit。"""
    p = _require_project(train_project_id)
    model = params.get("model") or p["settings"].get("model") or "sdxl"
    tp = MODELS.get(model)
    if tp is None:
        raise BadRequest(t("msg.train_bad_model", model=model, available=", ".join(MODELS)))
    st = _require_online()
    base = _resolve_base(tp, (params.get("base_model") or "").strip())
    pretrained = _pretrained(train_project_id, params.get("continue_from") or "", st)
    clean = _clean_params(tp, {**params, "model": model})
    rid = db.new_id()
    job_name = f"{slugify(p['name'], 'lora')[:60]}_{rid}"
    db.train_project_update(train_project_id, settings_=clean)
    db.train_run_add(rid, train_project_id, job_name, model, clean)
    threading.Thread(target=_prepare, args=(rid, p, tp, clean, base, pretrained, st, get_lang()), daemon=True).start()
    return run_view(db.train_run_get(rid))


def _canceled(rid: str) -> bool:
    run = db.train_run_get(rid)
    return run is None or run["status"] != "preparing"


def _prepare(rid: str, p: dict[str, Any], tp: dict[str, Any], params: dict[str, Any], base: str, pretrained: str | None,
             st: dict[str, Any], lang: str) -> None:
    with use_lang(lang):
        run = db.train_run_get(rid)
        dest = settings.aitoolkit_dir / "datasets" / run["job_name"]
        try:
            snap = _write_snapshot(p, params, dest)
            summary = {"excluded": snap["excluded"], "base": base}
            db.train_run_update(rid, image_count=snap["count"], summary=summary)
            if _canceled(rid):
                shutil.rmtree(dest, ignore_errors=True)
                return
            prompts = [x.strip() for x in params.get("sample_prompts") or [] if x and x.strip()] or snap["samples"]
            negative = snap["negative"] if params.get("sample_negative") is None else params["sample_negative"].strip()
            summary["freed"] = _free_vram()
            if _canceled(rid):  # 釋放 VRAM 可能要幾秒（VLM 休眠），這段時間也可能按了停止
                shutil.rmtree(dest, ignore_errors=True)
                return
            job = build_job(run["job_name"], f"{st['datasets']}/{run['job_name']}", st["output"], tp, base, params, prompts[:5],
                            negative, (params.get("trigger_word") or "").strip(), pretrained)
            r = _request("POST", "/api/jobs", json={"name": run["job_name"], "gpu_ids": settings.aitoolkit_gpu, "job_config": job})
            if r.status_code != 200:
                raise AIToolkitError(t("msg.aitoolkit_failed", error=f"HTTP {r.status_code} {r.text[:200]}"))
            job_id = r.json()["id"]
            # 先記下 job_id，狀態維持 preparing：AI Toolkit 剛建立的工作是 stopped，/start 之前去查會被當成已停止
            # 送出工作要一點時間，這段時間也可能按了停止：工作還沒開始，直接刪掉
            if not db.train_run_update(rid, when=("preparing",), job_id=job_id, summary=summary):
                try:
                    _request("GET", f"/api/jobs/{job_id}/delete")
                except AIToolkitError as e:
                    log.warning("刪除已停止的 AI Toolkit 工作 %s 失敗：%s", job_id, e)
                shutil.rmtree(dest, ignore_errors=True)
                return
            for path in (f"/api/jobs/{job_id}/start", f"/api/queue/{settings.aitoolkit_gpu}/start"):
                r = _request("GET", path)
                if r.status_code != 200:
                    raise AIToolkitError(t("msg.aitoolkit_failed", error=f"HTTP {r.status_code} {r.text[:200]}"))
            # 排進佇列之後才改成 queued；/start 的這段時間按了停止的話，取消剛排進佇列的工作
            if not db.train_run_update(rid, when=("preparing",), status="queued"):
                _cancel_job(job_id)
        except Exception as e:  # noqa: BLE001  顯示在訓練紀錄上
            log.exception("本機訓練 %s 準備失敗", rid)
            # 已經按了停止的不改成失敗（停止的狀態優先）
            failed = db.train_run_update(rid, when=("preparing",), status="failed", error=str(e) or e.__class__.__name__)
            cur = db.train_run_get(rid)
            if cur is not None and cur["job_id"]:
                # 工作可能已經排進佇列（例如 /start 成功、啟動佇列失敗），LTS 不再追蹤它，不能讓它之後自己開始
                _cancel_job(cur["job_id"])
            elif cur is None or not failed:  # 還沒交給 AI Toolkit 就停止：快照直接刪掉
                shutil.rmtree(dest, ignore_errors=True)


def _job(job_id: str) -> dict[str, Any] | None:
    """AI Toolkit 的工作資料（status、pid…）；工作不見了是 None。"""
    r = _request("GET", "/api/jobs", params={"id": job_id})
    if r.status_code != 200:
        raise AIToolkitError(t("msg.aitoolkit_failed", error=f"HTTP {r.status_code}"))
    return r.json()


def _cancel_job(job_id: str) -> None:
    """取消已經交給 AI Toolkit、但 LTS 這邊已經停止或失敗的工作（在背景執行緒裡跑，會等幾秒確認）。
    AI Toolkit 的 /stop 只對已經在跑（有 pid）的工作有用；還在佇列裡的工作 /stop 會被忽略、之後照樣開始，
    要用 mark_stopped 把狀態改成 stopped，worker 才不會啟動它。worker 剛好在這時啟動的話，下一輪再用 /stop 停掉。"""
    try:
        job = _job(job_id)
        for _ in range(CANCEL_CHECKS):
            if not job or job.get("status") not in _JOB_ACTIVE:
                return
            _request("GET", f"/api/jobs/{job_id}/{'stop' if job.get('pid') else 'mark_stopped'}")
            time.sleep(CANCEL_RECHECK_S)
            job = _job(job_id)
        if job and job.get("status") in _JOB_ACTIVE:
            log.warning("AI Toolkit 工作 %s 取消後仍在執行（%s）", job_id, job.get("status"))
    except (AIToolkitError, ValueError) as e:
        log.warning("取消 AI Toolkit 工作 %s 失敗：%s", job_id, e)


# ------------------------------------------------------------------ 進度與結果
def _refresh(run: dict[str, Any]) -> dict[str, Any]:
    """還在跑的訓練，向 AI Toolkit 查最新狀態。服務連不上時維持原狀，畫面會提示。"""
    # 準備中（含剛送出、還沒 /start）不查：AI Toolkit 剛建立的工作是 stopped，會被誤當成已停止
    if run["status"] not in ACTIVE or run["status"] == "preparing" or not run["job_id"]:
        return run
    try:
        r = _request("GET", "/api/jobs", params={"id": run["job_id"]})
        # 暫時的錯誤（HTTP 500、回應不是 JSON）不能當成工作不見：維持原狀，之後再查
        if r.status_code != 200:
            raise AIToolkitError(t("msg.aitoolkit_failed", error=f"HTTP {r.status_code}"))
        job = r.json()
    except AIToolkitError as e:
        return {**run, "offline": str(e)}
    except ValueError:
        return {**run, "offline": t("msg.aitoolkit_failed", error=r.text[:200])}
    if not job:  # 工作不見了（AI Toolkit 回 null）：AI Toolkit 的資料庫被清掉
        db.train_run_update(run["id"], status="failed", error=t("msg.train_job_gone"))
        return db.train_run_get(run["id"])
    new = JOB_STATUS.get(job.get("status"), run["status"])
    summary = {**run["summary"], "step": job.get("step") or 0, "total_steps": job.get("total_steps"),
               "speed": job.get("speed_string") or "", "info": job.get("info") or ""}
    if new == "running" and not summary.get("started_at") and summary["step"]:
        summary["started_at"] = time.time()
    if new not in ACTIVE:
        summary["finished_at"] = time.time()
    error = (job.get("info") or t("msg.aitoolkit_failed", error="?")) if new == "failed" else run.get("error")
    db.train_run_update(run["id"], status=new, summary=summary, error=error)
    return db.train_run_get(run["id"])


def _seconds_per_step(speed: str) -> float | None:
    """AI Toolkit 的速度字串：2.26 iter/sec 或 1.53 sec/iter。"""
    m = re.match(r"\s*([\d.]+)\s*(sec/iter|iter/sec)", speed or "")
    if not m or not float(m.group(1)):
        return None
    return float(m.group(1)) if m.group(2) == "sec/iter" else 1 / float(m.group(1))


def _progress(run: dict[str, Any]) -> dict[str, Any] | None:
    if run["status"] not in ACTIVE:
        return None
    sm, p = run["summary"], run["params"]
    total = sm.get("total_steps") or p.get("steps")
    step = sm.get("step") or 0
    sps = _seconds_per_step(sm.get("speed") or "")
    return {"step": step, "total_steps": total, "seconds_per_step": sps, "info": sm.get("info") or "",
            "percent": round(step / total * 100, 1) if total else None,
            "remaining_seconds": round((total - step) * sps) if sps and total else None,
            "epochs_total": p.get("epochs"), "epochs_done": step // _save_every(p["steps"], p["epochs"]) if p.get("steps") else 0}


def _output_dir(run: dict[str, Any]) -> Path:
    return settings.aitoolkit_dir / "output" / run["job_name"]


def _epochs(run: dict[str, Any]) -> list[dict[str, Any]]:
    """從輸出資料夾整理每個 epoch 的 LoRA 和範例圖。檔名：<工作名>_000000500.safetensors，最後一個是 <工作名>.safetensors。"""
    out, p = _output_dir(run), run["params"]
    if not out.is_dir() or not p.get("steps"):
        return []
    every = _save_every(p["steps"], p["epochs"])
    by_step: dict[int, dict[str, Any]] = {}
    name = run["job_name"]
    for f in out.glob(f"{name}*.safetensors"):
        m = re.fullmatch(re.escape(name) + r"(?:_(\d{9}))?\.safetensors", f.name)
        if m:
            step = int(m.group(1)) if m.group(1) else p["steps"]
            by_step.setdefault(step, {"step": step, "samples": []})["file"] = f.name
    for f in sorted((out / "samples").glob("*")) if (out / "samples").is_dir() else []:
        m = re.search(r"__(\d{9})_\d+\.\w+$", f.name)
        if m and not f.name.endswith(".thumb.jpg"):
            by_step.setdefault(int(m.group(1)), {"step": int(m.group(1)), "samples": []})["samples"].append(f"samples/{f.name}")
    eps = []
    for step in sorted(by_step):
        e = by_step[step]
        e["epoch"] = min(p["epochs"], max(1, math.ceil(step / every)))
        e["available"] = bool(e.get("file"))
        if e.get("file"):
            e["url"] = f"/api/train/runs/{run['id']}/files/{e['file']}"
            e["size"] = (out / e["file"]).stat().st_size
        e["samples"] = [f"/api/train/runs/{run['id']}/files/{x}" for x in e["samples"]]
        eps.append(e)
    return eps


def run_view(run: dict[str, Any]) -> dict[str, Any]:
    tp = MODELS.get(run["model"], {})
    return {**run, "model_label": tp.get("label", run["model"]), "progress": _progress(run), "epochs": _epochs(run),
            "a1111": db.a1111_imports(run["id"]), "error": t("msg.train_interrupted") if run.get("error") == "interrupted" else run.get("error")}


def _require_run(rid: str) -> dict[str, Any]:
    run = db.train_run_get(rid)
    if run is None:
        raise NotFound(t("msg.train_run_missing"))
    return run


def get_run(rid: str) -> dict[str, Any]:
    return run_view(_refresh(_require_run(rid)))


def active_runs() -> list[dict[str, Any]]:
    """進行中的訓練（訓練頁的總覽），附上訓練專案的名稱。"""
    out = []
    for r in db.train_runs_with_status(ACTIVE):
        r = _refresh(r)
        if r["status"] in ACTIVE:
            p = db.train_project_get(r["train_project_id"]) or {"name": ""}
            out.append({**run_view(r), "project_name": p["name"]})
    return out


def stop_run(rid: str) -> dict[str, Any]:
    run = _refresh(_require_run(rid))
    if run["status"] not in ACTIVE:
        raise BadRequest(t("msg.train_not_active"))
    if run["status"] == "preparing":  # 還在準備資料或正在送出：_prepare 看到就會停下，已經送出的工作由它取消
        if db.train_run_update(rid, when=("preparing",), status="canceled"):
            return get_run(rid)
        run = _refresh(_require_run(rid))  # 剛好排進佇列了：照一般的方式停止
        if run["status"] not in ACTIVE or not run["job_id"]:
            raise BadRequest(t("msg.train_not_active"))
    asked = run["summary"].get("stop_requested_at")
    if run["status"] == "queued":
        # 還在 AI Toolkit 的佇列裡：/stop 只停已經在跑的工作，排隊中的會被忽略、之後照樣開始，所以直接標記為停止
        action = "mark_stopped"
    else:
        # 再按一次：訓練行程可能已經不在了（例如 AI Toolkit 容器重啟過），這時要直接標記，佇列才不會一直等它
        action = "mark_stopped" if asked and time.time() - asked > STOP_RETRY_S else "stop"
    r = _request("GET", f"/api/jobs/{run['job_id']}/{action}")
    if r.status_code != 200:
        raise AIToolkitError(t("msg.aitoolkit_failed", error=f"HTTP {r.status_code} {r.text[:200]}"))
    db.train_run_update(rid, summary={**run["summary"], "stop_requested_at": asked or time.time()})
    return get_run(rid)


def delete_run(rid: str) -> None:
    """刪除紀錄、資料集快照和輸出（LoRA、範例圖）。"""
    run = _refresh(_require_run(rid))
    if run["status"] in ACTIVE:
        raise BadRequest(t("msg.train_run_busy"))
    if run["job_id"] and status()["online"]:
        try:
            _request("GET", f"/api/jobs/{run['job_id']}/delete")
        except AIToolkitError as e:
            log.warning("刪除 AI Toolkit 工作 %s 失敗：%s", run["job_id"], e)
    shutil.rmtree(settings.aitoolkit_dir / "datasets" / run["job_name"], ignore_errors=True)
    shutil.rmtree(_output_dir(run), ignore_errors=True)
    db.train_run_delete(rid)


def run_file(rid: str, rel: str) -> Path:
    """輸出資料夾裡的檔案（LoRA、範例圖）。"""
    root = _output_dir(_require_run(rid)).resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise NotFound(t("msg.train_file_missing"))
    return path


def to_forge(rid: str, epoch: int, overwrite: bool = False) -> dict[str, Any]:
    """把一個 epoch 的 LoRA 傳給 A1111 / Forge。"""
    from . import a1111

    run = _require_run(rid)
    ep = next((e for e in _epochs(run) if e["epoch"] == epoch and e.get("file")), None)
    if ep is None:
        raise BadRequest(t("msg.train_epoch_missing", epoch=epoch))
    p = _require_project(run["train_project_id"])
    trigger = (run["params"].get("trigger_word") or _dataset_view(p).get("trigger") or "").strip()
    family = MODELS.get(run["model"], {}).get("family", "")
    return a1111.import_local(run, _output_dir(run) / ep["file"], epoch, trigger, p["name"], FORGE_VERSIONS.get(family, "Unknown"), overwrite)


def loss(rid: str) -> dict[str, Any]:
    """loss 曲線（AI Toolkit 記錄每一步，這裡平均成最多 300 個點）。"""
    run = _require_run(rid)
    if not run["job_id"]:
        return {"steps": [], "values": []}
    r = _request("GET", f"/api/jobs/{run['job_id']}/loss")
    if r.status_code != 200:
        return {"steps": [], "values": []}
    series = (r.json().get("series") or {}).get("loss/loss") or {}
    pairs = [(s, v) for s, v in zip(series.get("steps") or [], series.get("values") or []) if v is not None]
    size = max(1, math.ceil(len(pairs) / 300))
    chunks = [pairs[i:i + size] for i in range(0, len(pairs), size)]
    return {"steps": [c[-1][0] for c in chunks], "values": [round(sum(v for _, v in c) / len(c), 5) for c in chunks]}


def log_tail(rid: str, lines: int = 200) -> str:
    """訓練紀錄檔的最後幾行（進度列的 \\r 換成換行後只留最後的狀態）。"""
    path = _output_dir(_require_run(rid)) / "log.txt"
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 200_000))
        text = f.read().decode("utf-8", errors="replace")
    return "\n".join(x.rsplit("\r", 1)[-1] for x in text.split("\n")[-lines:])
