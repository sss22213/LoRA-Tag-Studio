"""送到 Civitai 雲端訓練（Orchestration API）。

流程：
1. prepare：逐張把訓練圖片上傳成 blob（每張都經過 Civitai 的內容審核），再用 whatif=true 試算 Buzz
2. submit：使用者確認費用後送出 training workflow，開始扣 Buzz
3. 之後用 GetWorkflow 查進度，取得每個 epoch 的 LoRA 與範例圖

API key（CIVITAI_API_KEY）只留在伺服器端，不會傳到瀏覽器。
規格：https://developer.civitai.com/orchestration/ 與 openapi-snapshots/v2-consumers.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from . import db
from .config import settings
from .exporter import convert_image, dataset_images
from .i18n import get_lang, t, use_lang
from .pipeline import block_tags, caption_for
from .profiles import PROFILES
from .tagging.postprocess import canon, split_tags, training_hints

log = logging.getLogger(__name__)


class CivitaiError(Exception):
    pass


class CivitaiNotFound(CivitaiError):
    pass


# triggerWord 欄位只對這些 ecosystem 有效（規格說明）；其他（例如 Anima）靠 caption 開頭的 trigger
TRIGGER_WORD_ECOSYSTEMS = {"sd1", "sdxl", "flux1", "chroma", "zimagebase", "zimageturbo", "flux2klein"}
# 只有 SD 系列有 minSnrGamma，也只有它們真的會訓練文字編碼器（其他類型文件寫明保持 false）
SD_ECOSYSTEMS = {"sd1", "sdxl"}
# 表單參數 → AI Toolkit 輸入欄位
PARAM_FIELDS = {
    "steps": "steps", "epochs": "epochs", "batch_size": "batchSize", "lr": "lr", "lr_scheduler": "lrScheduler",
    "optimizer": "optimizerType", "network_dim": "networkDim", "network_alpha": "networkAlpha",
    "noise_offset": "noiseOffset", "flip_augmentation": "flipAugmentation", "min_snr_gamma": "minSnrGamma",
    "train_text_encoder": "trainTextEncoder", "text_encoder_lr": "textEncoderLr",
}
COMMON_FIELDS = ("steps", "epochs", "batch_size", "lr", "lr_scheduler", "optimizer", "network_dim", "network_alpha",
                 "noise_offset", "flip_augmentation", "shuffle_tokens", "keep_tokens", "continue_from",
                 "sample_negative", "sample_cfg", "sample_strength")
# 只有部分類型有的欄位；指定給不支援的類型會回錯誤
TYPE_SPECIFIC_FIELDS = ("base_model", "trigger_word", "min_snr_gamma", "train_text_encoder", "text_encoder_lr")
LR_SCHEDULERS = ("constant", "constant_with_warmup", "cosine", "linear", "step")
# 排隊優先度（WorkflowStep.priority）：Civitai 網站的「High Priority」開關送的是 normal、沒開是 low，
# 不指定時 API 也當成 low。試算的價格三者相同；high 只有 API 能選，依帳號等級可能沒有額外效果
PRIORITIES = ("low", "normal", "high")
DEFAULT_PRIORITY = "normal"


def _type(ecosystem: str, label: str, group: str = "image", extra: dict[str, Any] | None = None,
          model: str | None = None, steps: int = 2000, lr: float = 0.0001, dim: int = 32, optimizer: str = "adamw8bit",
          noise: float = 0.0, max_batch: int = 1, cfg: float | None = None, per_step: float | None = None,
          per_epoch: int = 10, documented: bool = True, preview: bool = False) -> dict[str, Any]:
    sd = ecosystem in SD_ECOSYSTEMS
    fields = list(COMMON_FIELDS)
    if model:  # 只有 sd1 / sdxl / anima 的輸入有 model 欄位，其他類型的底模由 modelVariant / version 決定
        fields.append("base_model")
    if ecosystem in TRIGGER_WORD_ECOSYSTEMS:
        fields.append("trigger_word")
    if sd:
        fields += ["min_snr_gamma", "train_text_encoder", "text_encoder_lr"]
    defaults: dict[str, Any] = {
        "steps": steps, "epochs": 10, "batch_size": 1, "lr": lr, "lr_scheduler": "cosine", "optimizer": optimizer,
        "network_dim": dim, "network_alpha": dim, "noise_offset": noise, "flip_augmentation": False,
        "sample_cfg": cfg, "sample_strength": 1.0,
    }
    if sd:
        defaults.update(min_snr_gamma=5, train_text_encoder=True, text_encoder_lr=0.00005)
    return {"ecosystem": ecosystem, "label": label, "group": group, "extra": extra or {}, "model": model,
            "fields": fields, "max_batch": max_batch, "defaults": defaults,
            "per_step": per_step, "per_epoch": per_epoch, "documented": documented, "preview": preview}


# Civitai 的 AI Toolkit 訓練類型中，可以用圖片資料集訓練的全部（音樂類的 ACE-Step / YuE2 需要音訊，不列入）。
# 預設值與價格取自官方文件 developer.civitai.com/orchestration/recipes/training-*；
# max_batch 取自規格的 batchSize 上限；cfg（範例圖 CFG）取自規格裡 Civitai 生成該底模的預設 cfgScale，
# 沒有對應數值的（schnell、MiniMax）不送，由 Civitai 決定。
# documented=False 的類型文件沒寫預設值，用 AI Toolkit 的通用預設，價格以試算為準。
TRAINING_TYPES: dict[str, dict[str, Any]] = {
    "sd1": _type("sd1", "Stable Diffusion 1.5", model="urn:air:sd1:checkpoint:civitai:127227@139180", max_batch=4,
                 cfg=7, per_step=0.2),
    "sdxl": _type("sdxl", "Stable Diffusion XL", model="urn:air:sdxl:checkpoint:civitai:101055@128078",
                  optimizer="adafactor", noise=0.1, max_batch=4, cfg=7, per_step=0.2),
    # 步數 1500 是 Civitai API 回傳的 Anima defaultSteps（文件沒寫）
    "anima": _type("anima", "Anima",
                   model="urn:air:anima:repository:huggingface:circlestone-labs/Anima-Base-v1.0-Diffusers@main.tar",
                   steps=1500, cfg=4, documented=False),
    "flux1-dev": _type("flux1", "Flux.1 [dev]", extra={"modelVariant": "dev"}, dim=16, cfg=3.5, per_step=0.9,
                       per_epoch=20),
    "flux1-schnell": _type("flux1", "Flux.1 [schnell]", extra={"modelVariant": "schnell"}, dim=16, per_step=0.9,
                           per_epoch=20),
    "flux2klein-4b": _type("flux2klein", "Flux.2 Klein 4B", extra={"modelVariant": "4b"}, max_batch=2, cfg=5,
                           per_step=0.2),
    "flux2klein-9b": _type("flux2klein", "Flux.2 Klein 9B", extra={"modelVariant": "9b"}, cfg=5, per_step=0.45),
    "chroma": _type("chroma", "Chroma1-HD", dim=16, cfg=3.5, per_step=0.95),
    "ernie": _type("ernie", "ERNIE-Image", max_batch=2, cfg=4, per_step=0.45),
    "qwen": _type("qwen", "Qwen-Image (latest / 2512)", extra={"version": "latest"}, dim=16, cfg=2.5, per_step=0.95),
    "qwen-2509": _type("qwen", "Qwen-Image 2509", extra={"version": "2509"}, dim=16, cfg=2.5, per_step=0.95),
    "qwen21": _type("qwen21", "Qwen Image 2.1", cfg=1, documented=False),
    "zimageturbo": _type("zimageturbo", "Z-Image Turbo", max_batch=2, cfg=1, per_step=0.45),
    "zimagebase": _type("zimagebase", "Z-Image Base", lr=0.000001, optimizer="automagic", max_batch=2, cfg=4,
                        per_step=0.45),
    "boogu": _type("boogu", "Boogu Image", cfg=4, documented=False),
    "hidream-o1": _type("hidream-o1", "HiDream O1", cfg=5, documented=False),
    "ideogram4": _type("ideogram4", "Ideogram 4", cfg=7, documented=False),
    "krea2": _type("krea2", "Krea 2", cfg=4, documented=False),
    "mageflow": _type("mageflow", "Mage-Flow Base", cfg=5, documented=False),
    "ming": _type("ming", "Ming Image 0.1 Design", cfg=1, documented=False),
    # 影片類型：資料集可以是圖片、影片或混合；範例會是影片，費用較高
    "ltx2": _type("ltx2", "LTX-2 (19B)", group="video", steps=3000, cfg=4, per_step=0.75, per_epoch=50),
    "ltx23": _type("ltx23", "LTX-2.3 (22B)", group="video", steps=3000, cfg=4, per_step=0.75, per_epoch=50),
    "ltx25": _type("ltx25", "LTX-2.5", group="video", steps=3000, cfg=4, documented=False),
    "wan-2.1": _type("wan", "Wan 2.1 (14B)", group="video", extra={"modelVariant": "2.1"}, cfg=4, per_step=0.5,
                     per_epoch=200, preview=True),
    "wan-2.2": _type("wan", "Wan 2.2 (14B-A14B)", group="video", extra={"modelVariant": "2.2"}, cfg=4, per_step=0.5,
                     per_epoch=200, preview=True),
    "minimaxh3": _type("minimaxh3", "MiniMax H3", group="video", documented=False),
}

# 專案底模 → (預設訓練類型, 訓練底模 AIR；None = 該類型的預設)
# Pony / Illustrious / NoobAI / Animagine 的 AIR 由 Site API（/api/v1/model-versions/{id}）查得
PROFILE_TYPES: dict[str, tuple[str, str | None]] = {
    "sd15_anime": ("sd1", None),
    "sd15_realistic": ("sd1", None),
    "sdxl_base": ("sdxl", None),
    "pony_v6": ("sdxl", "urn:air:sdxl:checkpoint:civitai:257749@290640"),  # Pony Diffusion V6 XL
    "illustrious": ("sdxl", "urn:air:sdxl:checkpoint:civitai:795765@889818"),  # Illustrious-XL v0.1
    "noobai_xl": ("sdxl", "urn:air:sdxl:checkpoint:civitai:833294@1116447"),  # NoobAI-XL eps 1.1
    "animagine_xl": ("sdxl", "urn:air:sdxl:checkpoint:civitai:1188071@1337429"),  # Animagine XL 4.0
    "anima": ("anima", None),
    "flux1_dev": ("flux1-dev", None),
}
OPTIMIZERS = ("adamw", "adamw8bit", "adam8bit", "lion", "lion8bit", "adafactor", "adagrad", "prodigy", "prodigy8bit",
              "automagic")
TERMINAL = {"succeeded", "failed", "expired", "canceled"}
MAX_CAPTION = 1024  # BlobTrainingDataItem.caption 上限
UPLOAD_FORMAT = ("jpg", "image/jpeg")  # 以 JPEG 95 上傳，比 PNG 小很多，訓練品質差異可忽略
UPLOAD_WORKERS = 4


def configured() -> bool:
    return bool(settings.civitai_api_key)


def default_price(tp: dict[str, Any]) -> int | None:
    if tp["per_step"] is None:
        return None
    return round(tp["defaults"]["steps"] * tp["per_step"] + tp["defaults"]["epochs"] * tp["per_epoch"])


def training_type(type_id: str | None, profile: str) -> tuple[str, dict[str, Any]]:
    """選擇的訓練類型；沒指定時依專案底模。"""
    type_id = type_id or PROFILE_TYPES.get(profile, ("sdxl", None))[0]
    if type_id not in TRAINING_TYPES:
        raise CivitaiError(t("msg.civitai_bad_type", type=type_id, available=", ".join(TRAINING_TYPES)))
    return type_id, TRAINING_TYPES[type_id]


def default_model_for(type_id: str, profile: str) -> str | None:
    """訓練類型的預設底模；專案底模對應到同一類型時用專案的底模（例如 Pony 用 Pony V6 XL）。"""
    p_type, p_model = PROFILE_TYPES.get(profile, (None, None))
    return p_model if p_type == type_id and p_model else TRAINING_TYPES[type_id]["model"]


def check_params(type_id: str, tp: dict[str, Any], params: dict[str, Any]) -> None:
    """拒絕這個訓練類型沒有的欄位（例如 Flux 不能換底模、只有 SD 系列有 Min SNR γ）。"""
    for key in TYPE_SPECIFIC_FIELDS:
        if params.get(key) not in (None, "") and key not in tp["fields"]:
            raise CivitaiError(t("msg.civitai_field_unsupported", field=key, type=type_id))


def info() -> dict[str, Any]:
    """給 WebUI / API 的設定資訊（不含金鑰）。"""
    return {
        "configured": configured(),
        "types": [{"id": k, **{f: v for f, v in tp.items() if f != "extra"}, "default_price": default_price(tp),
                   "trigger_word": tp["ecosystem"] in TRIGGER_WORD_ECOSYSTEMS} for k, tp in TRAINING_TYPES.items()],
        # negative：範例圖的負面提示預設用專案底模的
        "profiles": {p: {"type": tp, "model": default_model_for(tp, p), "negative": PROFILES[p]["a1111"]["negative"]}
                     for p, (tp, _) in PROFILE_TYPES.items()},
        "optimizers": list(OPTIMIZERS),
        "lr_schedulers": list(LR_SCHEDULERS),
        "priorities": list(PRIORITIES),
        "default_priority": DEFAULT_PRIORITY,
    }


# ------------------------------------------------------------------ HTTP
def _client() -> httpx.Client:
    if not configured():
        raise CivitaiError(t("msg.civitai_not_configured"))
    return httpx.Client(base_url=settings.civitai_orchestration_url, timeout=120,
                        headers={"Authorization": f"Bearer {settings.civitai_api_key}"})


def _problem(r: httpx.Response) -> str:
    """把 ProblemDetails 轉成一行說明。"""
    try:
        d = r.json()
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:300]}"
    parts = [d.get("title") or "", d.get("detail") or ""]
    for k, v in (d.get("errors") or {}).items():
        parts.append(f"{k}: {'; '.join(map(str, v)) if isinstance(v, list) else v}")
    return f"HTTP {r.status_code}: " + " — ".join(p for p in parts if p)[:600]


def _request(method: str, path: str, **kw: Any) -> httpx.Response:
    try:
        with _client() as c:
            r = c.request(method, path, **kw)
    except httpx.HTTPError as e:
        raise CivitaiError(t("msg.civitai_connect", error=e)) from e
    if r.status_code == 401:
        raise CivitaiError(t("msg.civitai_auth"))
    return r


def upload_blob(data: bytes, content_type: str) -> tuple[str | None, str | None]:
    """上傳一張圖片，回傳 (blob id, 被擋下的原因)。"""
    r = _request("POST", "/v2/consumer/blobs", content=data, headers={"Content-Type": content_type})
    if r.status_code in (200, 201):
        blob = r.json()
        if blob.get("blockedReason"):
            return None, blob["blockedReason"]
        return blob["id"], None
    if r.status_code == 422:  # 內容審核不通過
        return None, _problem(r)
    raise CivitaiError(t("msg.civitai_api_error", error=_problem(r)))


def submit_workflow(body: dict[str, Any], whatif: bool) -> dict[str, Any]:
    r = _request("POST", "/v2/consumer/workflows", params={"wait": 0, "whatif": str(whatif).lower()}, json=body)
    if r.status_code not in (200, 202):
        raise CivitaiError(t("msg.civitai_api_error", error=_problem(r)))
    return r.json()


def get_workflow(workflow_id: str) -> dict[str, Any]:
    r = _request("GET", f"/v2/consumer/workflows/{workflow_id}")
    if r.status_code == 404:
        raise CivitaiNotFound(t("msg.civitai_run_not_found", id=workflow_id))
    if r.status_code not in (200, 202):
        raise CivitaiError(t("msg.civitai_api_error", error=_problem(r)))
    return r.json()


def cancel_workflow(workflow_id: str) -> dict[str, Any]:
    r = _request("PUT", f"/v2/consumer/workflows/{workflow_id}", json={"status": "canceled"})
    if r.status_code not in (200, 202):
        raise CivitaiError(t("msg.civitai_api_error", error=_problem(r)))
    summary = refresh_run(workflow_id)
    if summary["status"] not in TERMINAL:
        # Civitai 的取消是非同步的：要等訓練機器停下來，狀態才會變成 canceled（實測要幾分鐘），先記下已送出取消
        summary["cancel_requested_at"] = time.time()
        if db.civitai_run_get(workflow_id):
            db.civitai_run_update(workflow_id, summary["status"] or "unknown", summary)
    return summary


# ------------------------------------------------------------------ 底模
_AIR_RE = re.compile(r"^urn:air:([^:]+):([^:]+):")
# 網站上 LoRA 的 AIR 前綴與訓練 ecosystem 的對應（由 Site API 查得）。Wan / LTX 在網站上用別的名稱
# （wanvideo14b_t2v、ltxv2…），對應不明確，不在這裡檢查，交給 Civitai 試算時驗證。
LORA_AIR_ECOSYSTEMS = {"sd1": "sd1", "sdxl": "sdxl", "anima": "anima", "flux1": "flux1", "flux2klein": "flux2",
                       "chroma": "chroma", "ernie": "ernie", "qwen": "qwen", "zimageturbo": "zimageturbo",
                       "zimagebase": "zimagebase"}
LORA_AIR_TYPES = {"lora", "lycoris", "dora", "locon"}


def _to_air(ref: str, bad_msg: str = "msg.civitai_bad_model") -> str | None:
    """AIR、模型版本 ID 或含 modelVersionId 的 civitai.com 網址 → AIR。"""
    ref = (ref or "").strip()
    if not ref:
        return None
    if not ref.startswith("urn:air:"):
        m = re.search(r"modelVersionId=(\d+)", ref) or re.fullmatch(r"(\d+)", ref)
        if not m:
            raise CivitaiError(t(bad_msg, ref=ref))
        try:
            r = httpx.get(f"{settings.civitai_site_url}/api/v1/model-versions/{m.group(1)}", timeout=20)
        except httpx.HTTPError as e:
            raise CivitaiError(t("msg.civitai_connect", error=e)) from e
        air = r.json().get("air") if r.status_code == 200 else None
        if not air:
            raise CivitaiError(t(bad_msg, ref=ref))
        ref = air
    return ref


def resolve_model(ref: str, ecosystem: str) -> str | None:
    """使用者輸入的訓練底模 → AIR，並檢查 ecosystem。"""
    air = _to_air(ref)
    m = _AIR_RE.match(air or "")
    if m and m.group(1) != ecosystem:
        raise CivitaiError(t("msg.civitai_model_mismatch", model=air, got=m.group(1), want=ecosystem))
    return air


def resolve_lora(ref: str, ecosystem: str) -> str | None:
    """接續訓練用的 LoRA → AIR；要是 LoRA，且（已知對應時）要和訓練類型同一個 ecosystem。"""
    air = _to_air(ref, "msg.civitai_bad_lora")
    m = _AIR_RE.match(air or "")
    if not m:
        return air
    if m.group(2) not in LORA_AIR_TYPES:
        raise CivitaiError(t("msg.civitai_not_lora", model=air, kind=m.group(2)))
    want = LORA_AIR_ECOSYSTEMS.get(ecosystem)
    if want and m.group(1) != want:
        raise CivitaiError(t("msg.civitai_lora_mismatch", model=air, got=m.group(1), want=want))
    return air


# ------------------------------------------------------------------ 訓練內容
def _fit_caption(img: dict[str, Any], s: dict[str, Any]) -> tuple[str, int]:
    """Civitai 的 caption 上限 1024 字：超過時拿掉標籤直到放得下，回傳 (caption, 拿掉幾個標籤)。
    先拿 WD14 分數最低的；沒有分數的（手動加的、VLM 的、角色）最後才拿，由後往前。
    trigger、分級、自然語言描述、附加標籤與白色色塊關鍵字（含手動加在標籤裡的）都保留（舊做法從結尾截斷，會切掉色塊關鍵字）。"""
    text = caption_for(img, s)
    if len(text) <= MAX_CAPTION:
        return text, 0
    tags = list(img.get("tags") or [])
    scores = {canon(n).lower(): sc for n, sc in (img.get("raw") or {}).get("general") or []}
    score = lambda i: scores.get(canon(tags[i]).lower())  # noqa: E731
    # 色塊關鍵字就算是手動加在標籤裡（沒偵測到色塊時），也絕不拿掉
    keep = {canon(k).lower() for k in split_tags(s.get("block_tag"))}
    order = sorted((i for i in range(len(tags)) if canon(tags[i]).lower() not in keep),
                   key=lambda i: (score(i) is None, score(i) or 0, -i))
    removed: set[int] = set()
    for i in order:
        removed.add(i)
        text = caption_for({**img, "tags": [tg for j, tg in enumerate(tags) if j not in removed]}, s)
        if len(text) <= MAX_CAPTION:
            return text, len(removed)
    # 只剩描述還是太長：截斷描述，色塊關鍵字放回最後
    suffix = ", ".join(block_tags(img, s))
    if suffix and text.endswith(suffix):
        text = text[:-len(suffix)].rstrip(", ")
    cut = text[:MAX_CAPTION - (len(suffix) + 2 if suffix else 0)]
    cut = cut[:cut.rfind(",")] if "," in cut else cut
    return (f"{cut}, {suffix}" if suffix else cut), len(removed)


def _caption(img: dict[str, Any], s: dict[str, Any]) -> str:
    return _fit_caption(img, s)[0]


def default_samples(images: list[dict[str, Any]], s: dict[str, Any], n: int = 3) -> list[str]:
    """預設範例提示：平均取幾張圖的 caption，讓每個 epoch 的範例圖貼近資料集。"""
    count = min(n, len(images))
    if not count:
        return []
    step = (len(images) - 1) / max(1, count - 1)
    picks = [images[int(i * step + 0.5)] for i in range(count)]
    out = []
    no_block = {**s, "block_tag_auto": False}  # 範例圖不要畫出白色色塊
    for img in picks:
        cap = _caption(img, no_block)[:400]
        if cap and cap not in out:
            out.append(cap)
    return out


def default_negative(s: dict[str, Any]) -> str:
    """範例圖的負面提示：底模的預設，開啟色塊關鍵字時再加上它，讓範例圖不出現白色色塊。"""
    parts = [PROFILES[s["profile"]]["a1111"]["negative"]] + ([s["block_tag"]] if s.get("block_tag_auto") else [])
    return ", ".join(p for p in parts if p)


def build_input(s: dict[str, Any], tp: dict[str, Any], params: dict[str, Any], items: list[dict[str, str]],
                model: str | None, samples: list[str]) -> dict[str, Any]:
    """組出 AI Toolkit 訓練輸入。WebUI 送出畫面上的數值；沒給的參數（例如 API 呼叫省略）用 Civitai 的預設。"""
    fields = tp["fields"]
    hints = training_hints(s)
    inp: dict[str, Any] = {"engine": "ai-toolkit", "ecosystem": tp["ecosystem"], **tp["extra"]}
    if model:
        inp["model"] = model
    for key, api_key in PARAM_FIELDS.items():
        if key in fields and params.get(key) is not None:
            inp[api_key] = params[key]
    if "batchSize" in inp:  # 超過上限 Civitai 也會自動降，這裡先降讓送出的內容與實際一致
        inp["batchSize"] = max(1, min(int(inp["batchSize"]), tp["max_batch"]))
    if not inp.get("trainTextEncoder", True):
        inp.pop("textEncoderLr", None)
    shuffle = hints["shuffle_caption"] if params.get("shuffle_tokens") is None else bool(params["shuffle_tokens"])
    keep = hints["keep_tokens"] if params.get("keep_tokens") is None else int(params["keep_tokens"])
    inp["shuffleTokens"] = shuffle
    inp["keepTokens"] = min(10, keep) if shuffle else 0
    trigger = (s.get("trigger") if params.get("trigger_word") is None else params["trigger_word"]) or ""
    if "trigger_word" in fields and trigger.strip():
        inp["triggerWord"] = trigger.strip()
    if params.get("continue_from"):
        inp["continueFrom"] = params["continue_from"]
    inp["trace"] = "events"  # 每個 epoch 的即時追蹤（步數、每步秒數），用來顯示進度與剩餘時間
    inp["trainingData"] = {"type": "blobs", "items": items}
    negative = default_negative(s) if params.get("sample_negative") is None else params["sample_negative"].strip()
    sample_cfg: dict[str, Any] = {"prompts": samples}
    if negative:
        sample_cfg["negativePrompt"] = negative
    if params.get("sample_cfg") is not None:
        sample_cfg["cfgScale"] = params["sample_cfg"]
    if params.get("sample_strength") is not None:
        sample_cfg["strength"] = params["sample_strength"]
    inp["samples"] = sample_cfg
    return inp


def build_workflow(project: dict[str, Any], inp: dict[str, Any], allow_mature: bool,
                   priority: str = DEFAULT_PRIORITY, upload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tags": ["lora-tag-studio", f"project-{project['id']}"],
        # upload：上傳設定（max_side、only_done）不在訓練輸入裡，記在 metadata，重新訓練時填回表單
        "metadata": {"source": "lora-tag-studio", "project": project["name"][:100], **(upload or {})},
        "allowMatureContent": allow_mature,
        "steps": [{"$type": "training", "priority": priority, "input": inp}],
    }


# ------------------------------------------------------------------ 結果
RUN_PARAM_KEYS = ("steps", "epochs", "batchSize", "lr", "lrScheduler", "optimizerType", "networkDim", "networkAlpha")


def _run_params(inp: dict[str, Any]) -> dict[str, Any] | None:
    """訓練實際使用的參數（顯示在訓練紀錄，方便比較每次訓練）。"""
    params = {k: inp[k] for k in RUN_PARAM_KEYS if inp.get(k) is not None}
    return params or None


def _type_of_input(inp: dict[str, Any]) -> str | None:
    for type_id, tp in TRAINING_TYPES.items():
        if tp["ecosystem"] == inp.get("ecosystem") and all(inp.get(k) == v for k, v in tp["extra"].items()):
            return type_id
    return None


def _run_request(wf: dict[str, Any], step: dict[str, Any], inp: dict[str, Any]) -> dict[str, Any] | None:
    """這次訓練的設定換回 prepare 的參數（api.CivitaiTrainRequest 的欄位），給「用相同參數重新訓練」填回表單。"""
    type_id = _type_of_input(inp)
    if not type_id:
        return None
    fields = TRAINING_TYPES[type_id]["fields"]
    req: dict[str, Any] = {"training_type": type_id}
    for key, api_key in PARAM_FIELDS.items():
        if key in fields and inp.get(api_key) is not None:
            req[key] = inp[api_key]
    if "base_model" in fields and inp.get("model"):
        req["base_model"] = inp["model"]
    if "trigger_word" in fields:
        req["trigger_word"] = inp.get("triggerWord") or ""
    if inp.get("shuffleTokens") is not None:
        req["shuffle_tokens"] = inp["shuffleTokens"]
        req["keep_tokens"] = inp.get("keepTokens") or 0
    req["continue_from"] = inp.get("continueFrom") or ""
    samples = inp.get("samples") or {}
    req["sample_prompts"] = samples.get("prompts") or []
    req["sample_negative"] = samples.get("negativePrompt") or ""
    if samples.get("cfgScale") is not None:
        req["sample_cfg"] = samples["cfgScale"]
    if samples.get("strength") is not None:
        req["sample_strength"] = samples["strength"]
    if wf.get("allowMatureContent") is not None:
        req["allow_mature"] = wf["allowMatureContent"]
    meta = wf.get("metadata") or {}
    if meta.get("max_side"):
        req["max_side"] = meta["max_side"]
    if meta.get("only_done") is not None:
        req["only_done"] = meta["only_done"]
    return req


def summarize(wf: dict[str, Any]) -> dict[str, Any]:
    step = next((st for st in wf.get("steps") or [] if st.get("$type") == "training"), {})
    out = step.get("output") or {}
    inp = step.get("input") or {}
    epochs = []
    for ep in out.get("epochs") or []:
        model = ep.get("model") or {}
        epochs.append({
            "epoch": ep.get("epochNumber"),
            "available": bool(model.get("available", bool(model.get("url")))),
            "url": model.get("url"),
            "samples": [x.get("url") for x in ep.get("samples") or [] if x.get("url")],
            "trace_url": ep.get("traceUrl"),
        })
    cost = (wf.get("cost") or {}).get("total")
    # 試算可能已套用折扣，factors.base 是折扣前的原價（實際扣款可能以原價結算）
    full = ((wf.get("cost") or {}).get("factors") or {}).get("base")
    # 實際扣款看交易紀錄：送出時預扣、結束時可能追加，取消 / 失敗的退款是 credit（訓練中 cost 會是 0）
    tx = (wf.get("transactions") or {}).get("list") or []
    debit = sum(x.get("amount") or 0 for x in tx if x.get("type") == "debit")
    credit = sum(x.get("amount") or 0 for x in tx if x.get("type") == "credit")
    queue = step.get("queuePosition") or {}
    return {
        "id": wf.get("id"),
        "status": wf.get("status") or step.get("status"),
        "step_status": step.get("status"),
        "moderation": out.get("moderationStatus"),
        "cost": cost,
        "cost_full": round(full) if full and cost is not None and full > cost else None,
        "charged": debit - credit if tx else None,
        "refunded": credit or None,
        "priority": step.get("priority"),
        "insufficient_buzz": bool((wf.get("transactions") or {}).get("insufficientBuzz")),
        "created_at": wf.get("createdAt"),
        "started_at": step.get("startedAt") or wf.get("startedAt"),
        "completed_at": wf.get("completedAt"),
        "progress_rate": step.get("estimatedProgressRate"),
        "queue": {"ahead": queue.get("precedingJobs"), "start_at": queue.get("estimatedStartAt"),
                  "complete_at": queue.get("estimatedCompleteAt")},
        "input_plan": {"steps": inp.get("steps"), "epochs": inp.get("epochs")},
        "params": _run_params(inp),
        "request": _run_request(wf, step, inp) if inp else None,
        "epochs": epochs,
    }


# ------------------------------------------------------------------ 進度
TRACE_READ_S = 5.0  # 每次最多讀幾秒的追蹤串流
TRACE_IDLE_S = 1.5  # 這麼久沒有新資料就當作已讀到最新
TRACE_TAIL_BYTES = 256 * 1024  # 只需要最後的事件，保留尾端就好
# 讀到這麼新（秒）的事件就代表已追上目前進度，不必等串流的新資料。要比事件間隔短，否則 epoch 剛開始時
# 第一段資料就算「新」，會漏掉後面的步數；步數間隔較長時由 TRACE_IDLE_S 停止
TRACE_FRESH_S = 3


def _parse_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _trace_access(url: str) -> tuple[bool, bool]:
    """追蹤串流（可以讀, 要不要帶金鑰）。Civitai 會把串流放在別的子網域（實測 orchestration-new.civitai.com，
    不需要金鑰）：同網域的 https 網址不帶金鑰讀取，金鑰只送給設定的 orchestration 主機，其他網址不讀。"""
    u, orch = httpx.URL(url), httpx.URL(settings.civitai_orchestration_url)
    if u.host == orch.host:
        return True, True
    parts = orch.host.split(".")
    domain = ".".join(parts[1:]) if len(parts) >= 3 and not orch.host.replace(".", "").isdigit() else ""
    return bool(domain and u.scheme == "https" and u.host.endswith("." + domain)), False


def _caught_up(buf: bytearray) -> bool:
    """最後一個完整事件的時間（t，毫秒）已經是剛剛：歷史事件讀完了。"""
    end = buf.rfind(b"\n")
    if end <= 0:
        return False
    try:
        ts = json.loads(bytes(buf[buf.rfind(b"\n", 0, end) + 1:end])).get("t")
    except (ValueError, AttributeError):
        return False
    return isinstance(ts, (int, float)) and time.time() - ts / 1000 < TRACE_FRESH_S


def _read_trace(url: str) -> list[dict[str, Any]]:
    """讀取 epoch 的即時追蹤（trace: "events" 的 NDJSON）。串流在 epoch 訓練期間不會結束，讀到暫時沒有新資料就停。"""
    allowed, with_key = _trace_access(url)
    if not allowed:
        return []
    buf = bytearray()
    deadline = time.monotonic() + TRACE_READ_S
    try:
        client = _client() if with_key else httpx.Client(timeout=120)
        with client as c, c.stream("GET", url, timeout=httpx.Timeout(15, read=TRACE_IDLE_S)) as r:
            if r.status_code != 200:  # 404：worker 還沒寫第一行
                return []
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > 2 * TRACE_TAIL_BYTES:
                    del buf[:-TRACE_TAIL_BYTES]
                if time.monotonic() > deadline or _caught_up(buf):
                    break
    except httpx.HTTPError:  # 讀取逾時代表目前沒有新資料，已讀到的仍可用
        pass
    events = []
    for line in bytes(buf).split(b"\n")[:-1]:  # 最後一段可能是還沒寫完的行
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def compute_progress(summary: dict[str, Any], plan: dict[str, Any], events: list[dict[str, Any]],
                     now: float | None = None) -> dict[str, Any] | None:
    """訓練進度：百分比、完成的 epoch / 步數、預估剩餘時間。trace 的 step 事件最準，沒有時退回 epoch 數或 Civitai 的估計。"""
    if summary.get("status") in TERMINAL:
        return None
    now = now or time.time()
    epochs = summary.get("epochs") or []
    epochs_total = plan.get("epochs") or len(epochs) or None
    epochs_done = sum(1 for e in epochs if e.get("available"))
    total_steps = plan.get("steps")
    started = _parse_time(summary.get("started_at"))
    queue = summary.get("queue") or {}
    out: dict[str, Any] = {
        "stage": "training" if started else "queued", "phase": None, "percent": None,
        "epochs_done": epochs_done, "epochs_total": epochs_total, "step": None, "total_steps": total_steps,
        "seconds_per_step": None, "remaining_seconds": None, "elapsed_seconds": int(now - started) if started else None,
        "queue_ahead": queue.get("ahead"), "start_at": queue.get("start_at"), "error": None,
    }
    last = {}
    for ev in events:
        last[ev.get("type")] = ev
    if last.get("phase"):
        out["phase"] = last["phase"].get("phase")
    if last.get("error"):
        out["error"] = last["error"].get("message")
    st = last.get("step")
    if st and st.get("step") is not None:
        step, max_steps = int(st["step"]), st.get("maxSteps")
        if total_steps and max_steps and epochs_total and max_steps < total_steps * 0.9:
            step += (int(st.get("epoch") or 1) - 1) * int(max_steps)  # 每個 epoch 各自從 0 數步數
        elif max_steps:
            total_steps = int(max_steps)
        out.update(step=min(step, total_steps) if total_steps else step, total_steps=total_steps,
                   seconds_per_step=st.get("secondsPerStep"))
    frac = None
    if out["step"] is not None and total_steps:
        frac = out["step"] / total_steps
    elif epochs_total and epochs_done:
        frac = epochs_done / epochs_total
    elif summary.get("progress_rate") is not None:
        rate = float(summary["progress_rate"])
        frac = rate / 100 if rate > 1 else rate
    if frac is not None:
        frac = min(max(frac, 0.0), 1.0)
        out["percent"] = round(frac * 100, 1)
    sps = out["seconds_per_step"]
    complete_at = _parse_time(queue.get("complete_at"))
    if sps and out["step"] is not None and total_steps:
        out["remaining_seconds"] = int((total_steps - out["step"]) * sps)
    elif frac and frac >= 0.02 and out["elapsed_seconds"]:
        out["remaining_seconds"] = int(out["elapsed_seconds"] * (1 - frac) / frac)
    elif complete_at:
        out["remaining_seconds"] = max(0, int(complete_at - now))
    return out


def refresh_run(workflow_id: str, max_age: float = 0) -> dict[str, Any]:
    """向 Civitai 取得最新狀態與進度並存起來；max_age 秒內更新過的直接用存著的（多個頁面同時輪詢時不重複查）。"""
    run = db.civitai_run_get(workflow_id)
    old = (run or {}).get("summary") or {}
    if run and max_age and old and time.time() - run["updated_at"] < max_age:
        return old
    summary = summarize(get_workflow(workflow_id))
    plan = old.get("plan") or {k: v for k, v in summary["input_plan"].items() if v}
    current = next((e for e in summary["epochs"] if not e["available"] and e["trace_url"]), None)
    events = _read_trace(current["trace_url"]) if current and summary["status"] not in TERMINAL else []
    summary["plan"] = plan
    summary["params"] = summary["params"] or old.get("params")
    summary["priority"] = summary["priority"] or old.get("priority")
    summary["request"] = summary["request"] or old.get("request")
    if summary["status"] not in TERMINAL and old.get("cancel_requested_at"):
        summary["cancel_requested_at"] = old["cancel_requested_at"]
    summary["progress"] = compute_progress(summary, plan, events)
    if run:
        db.civitai_run_update(workflow_id, summary["status"] or "unknown", summary)
    return summary


def _refresh_into(run: dict[str, Any], max_age: float = 0) -> None:
    try:
        run["summary"] = refresh_run(run["workflow_id"], max_age)
        run["status"] = run["summary"]["status"]
    except CivitaiNotFound as e:
        # Civitai 已經沒有這個任務（例如保存期限已過）：標成過期，之後不再查詢
        summary = {**(run["summary"] or {}), "status": "expired", "progress": None}
        db.civitai_run_update(run["workflow_id"], "expired", summary)
        run["status"], run["summary"], run["error"] = "expired", summary, str(e)
    except CivitaiError as e:
        run["error"] = str(e)


def list_runs(project_id: str, refresh: bool = True) -> list[dict[str, Any]]:
    """專案的訓練任務；還沒結束的會向 Civitai 查最新狀態與進度。"""
    runs = db.civitai_runs(project_id)
    for run in runs:
        # 還沒結束的要更新狀態；成功的也要重新取得，因為下載網址是有時效的簽名網址
        if refresh and configured() and (run["status"] not in TERMINAL or run["status"] == "succeeded"
                                         or _watch_refund(run)):
            _refresh_into(run, max_age=5)
    return runs


REFUND_WATCH_S = 3600


def _watch_refund(run: dict[str, Any]) -> bool:
    """取消 / 失敗的任務在結束後一小時內繼續查詢（退款可能晚一點才入帳）。"""
    if run["status"] not in ("canceled", "failed"):
        return False
    ended = _parse_time(((run.get("summary") or {}).get("completed_at")))
    return ended is not None and time.time() - ended < REFUND_WATCH_S


RECENT_FINISHED_S = 24 * 3600  # 首頁也顯示最近 24 小時內結束的任務，讓使用者知道訓練完成了


def active_runs() -> list[dict[str, Any]]:
    """所有專案中進行中的訓練（與最近結束的），附進度，給首頁顯示。"""
    now = time.time()
    runs = db.civitai_runs_recent(now - 7 * 24 * 3600)
    pending = [r for r in runs if r["status"] not in TERMINAL]
    if configured() and pending:
        with ThreadPoolExecutor(min(4, len(pending))) as pool:
            list(pool.map(lambda r: _refresh_into(r, max_age=15), pending))
    out = []
    for run in runs:
        if run["status"] not in TERMINAL:
            out.append(run)
            continue
        done_at = _parse_time((run["summary"] or {}).get("completed_at")) or run["updated_at"]
        if run["status"] in ("succeeded", "failed") and now - done_at < RECENT_FINISHED_S:
            out.append(run)
    for run in out:  # 首頁不需要下載連結與範例圖
        run["summary"] = {k: v for k, v in (run["summary"] or {}).items() if k not in ("epochs", "input_plan")}
    return out


# ------------------------------------------------------------------ 準備（上傳 + 試算）
@dataclass
class Prep:
    id: str
    project_id: str
    lang: str
    status: str = "uploading"  # uploading | estimating | ready | submitting | submitted | error
    training_type: str = ""
    total: int = 0
    done: int = 0
    reused: int = 0
    blocked: list[dict[str, str]] = field(default_factory=list)
    duplicates: list[dict[str, str]] = field(default_factory=list)
    excluded: int = 0
    truncated: int = 0
    truncated_tags: int = 0  # 為了放進 1024 字拿掉的標籤總數
    ecosystem: str = ""
    model: str | None = None
    continue_from: str | None = None
    priority: str = DEFAULT_PRIORITY
    force_upload: bool = False
    cost: int | None = None
    cost_full: int | None = None
    insufficient_buzz: bool = False
    workflow_id: str | None = None
    error: str | None = None
    body: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "body"}
        d["image_count"] = len(self.body["steps"][0]["input"]["trainingData"]["items"]) if self.body else 0
        d["samples"] = self.body["steps"][0]["input"]["samples"]["prompts"] if self.body else []
        return d


_preps: dict[str, Prep] = {}
_preps_lock = threading.Lock()


def _store(prep: Prep) -> None:
    with _preps_lock:
        cutoff = time.time() - 6 * 3600
        for k in [k for k, v in _preps.items() if v.created_at < cutoff]:
            _preps.pop(k, None)
        _preps[prep.id] = prep


def get_prep(prep_id: str) -> Prep:
    with _preps_lock:
        prep = _preps.get(prep_id)
    if prep is None:
        raise CivitaiError(t("msg.civitai_prep_not_found"))
    return prep


def prepare(project_id: str, params: dict[str, Any]) -> Prep:
    """在背景上傳圖片並試算費用。params 的欄位見 api.CivitaiTrainRequest；欄位因訓練類型而異（TRAINING_TYPES 的 fields）。"""
    if not configured():
        raise CivitaiError(t("msg.civitai_not_configured"))
    project, s, images, excluded = dataset_images(project_id, bool(params.get("only_done", True)))
    type_id, tp = training_type(params.get("training_type"), s["profile"])
    check_params(type_id, tp, params)
    priority = params.get("priority") or DEFAULT_PRIORITY
    if priority not in PRIORITIES:
        raise CivitaiError(t("msg.civitai_bad_priority", priority=priority, available=", ".join(PRIORITIES)))
    model = resolve_model(params.get("base_model") or "", tp["ecosystem"]) or default_model_for(type_id, s["profile"])
    params = {**params, "continue_from": resolve_lora(params.get("continue_from") or "", tp["ecosystem"])}
    prep = Prep(id=uuid.uuid4().hex[:12], project_id=project_id, lang=get_lang(), training_type=type_id,
                total=len(images), excluded=len(excluded), ecosystem=tp["ecosystem"], model=model,
                continue_from=params["continue_from"], priority=priority,
                force_upload=bool(params.get("force_upload")))
    _store(prep)
    threading.Thread(target=_run_prepare, args=(prep, project, s, images, params), daemon=True).start()
    return prep


def _upload_one(img: dict[str, Any], max_side: int, use_cache: bool = True) -> tuple[str | None, str | None, bool]:
    # blob 屬於上傳它的帳號，快取的 key 要包含金鑰指紋，換帳號就會重新上傳
    account = hashlib.sha256(settings.civitai_api_key.encode()).hexdigest()[:12]
    key = f"{account}:{img['sha1']}:{UPLOAD_FORMAT[0]}:{max_side}"
    cached = db.civitai_blob_get(key) if use_cache else None
    if cached:
        return cached, None, True
    data, _ = convert_image(img, UPLOAD_FORMAT[0], max_side)
    blob_id, blocked = upload_blob(data, UPLOAD_FORMAT[1])
    if blob_id:
        db.civitai_blob_put(key, blob_id)
    return blob_id, blocked, False


def _run_prepare(prep: Prep, project: dict[str, Any], s: dict[str, Any], images: list[dict[str, Any]],
                 params: dict[str, Any]) -> None:
    with use_lang(prep.lang):
        try:
            try:
                # 強制重新上傳：不用快取，每張都重新上傳（重新經過 Civitai 的內容審核）
                _prepare_body(prep, project, s, images, params, use_cache=not prep.force_upload)
            except CivitaiError as e:
                # 快取的 blob 可能已被 Civitai 清掉：略過快取全部重傳一次再試算
                if not prep.reused or not re.search(r"blob|\bair\b", str(e), re.IGNORECASE):
                    raise
                log.info("Civitai 不認得快取的 blob，重新上傳：%s", e)
                prep.done = prep.reused = 0
                prep.blocked, prep.duplicates, prep.truncated, prep.truncated_tags = [], [], 0, 0
                _prepare_body(prep, project, s, images, params, use_cache=False)
            prep.status = "ready"
        except (CivitaiError, ValueError, KeyError, OSError) as e:
            log.warning("Civitai 準備失敗：%s", e)
            prep.status, prep.error = "error", str(e)
        except Exception as e:  # noqa: BLE001
            log.exception("Civitai 準備失敗")
            prep.status, prep.error = "error", t("msg.civitai_api_error", error=e)


def _prepare_body(prep: Prep, project: dict[str, Any], s: dict[str, Any], images: list[dict[str, Any]],
                  params: dict[str, Any], use_cache: bool) -> None:
    """上傳圖片（可用快取）、組出 workflow，並用 whatif 試算費用。"""
    prep.status = "uploading"
    max_side = int(params.get("max_side") or 2048)
    results: dict[str, tuple[str | None, str | None, bool]] = {}
    with ThreadPoolExecutor(UPLOAD_WORKERS) as pool:
        futures = {img["id"]: pool.submit(_upload_one, img, max_side, use_cache) for img in images}
        for img in images:
            results[img["id"]] = futures[img["id"]].result()
            prep.done += 1
            prep.reused += 1 if results[img["id"]][2] else 0
    items, kept, seen = [], [], {}
    for img in images:  # 依資料集順序（順序會影響 Civitai 的重複任務判斷）
        blob_id, blocked, _ = results[img["id"]]
        if blocked:
            prep.blocked.append({"file": img["original_name"], "reason": blocked})
            continue
        # 像素完全相同的圖（例如同一張圖存了兩次、只有 metadata 不同）轉成 JPEG 後是同一個 blob，
        # Civitai 不接受同一個 blob 出現兩次：只留第一張
        if blob_id in seen:
            prep.duplicates.append({"file": img["original_name"], "same_as": seen[blob_id]})
            continue
        seen[blob_id] = img["original_name"]
        caption, cut = _fit_caption(img, s)
        if len(caption_for(img, s)) > MAX_CAPTION:
            prep.truncated += 1
            prep.truncated_tags += cut
        items.append({"air": blob_id, "caption": caption})
        kept.append(img)
    if not items:
        raise CivitaiError(t("msg.civitai_all_blocked"))
    samples = [p.strip() for p in params.get("sample_prompts") or [] if p and p.strip()] or default_samples(kept, s)
    allow_mature = params.get("allow_mature")
    if allow_mature is None:
        allow_mature = any(i.get("rating") in ("questionable", "explicit") for i in kept)
    inp = build_input(s, TRAINING_TYPES[prep.training_type], params, items, prep.model, samples[:5])
    prep.body = build_workflow(project, inp, bool(allow_mature), prep.priority,
                               {"max_side": max_side, "only_done": bool(params.get("only_done", True))})
    prep.status = "estimating"
    estimate = summarize(submit_workflow(prep.body, whatif=True))
    prep.cost, prep.cost_full, prep.insufficient_buzz = estimate["cost"], estimate["cost_full"], estimate["insufficient_buzz"]


def submit(prep_id: str) -> dict[str, Any]:
    """使用者確認費用後，正式送出訓練（開始扣 Buzz）。"""
    prep = get_prep(prep_id)
    if prep.status == "submitted" and prep.workflow_id:  # 重複送出：回傳已建立的任務，不會再扣一次 Buzz
        return refresh_run(prep.workflow_id)
    with _preps_lock:
        if prep.status != "ready":
            raise CivitaiError(t("msg.civitai_prep_not_ready"))
        prep.status = "submitting"  # 防止重複點擊送出兩次
    try:
        wf = submit_workflow(prep.body, whatif=False)
    except CivitaiError as e:
        prep.status, prep.error = "ready", str(e)
        raise
    summary = summarize(wf)
    inp = prep.body["steps"][0]["input"]
    defaults = TRAINING_TYPES[prep.training_type]["defaults"]
    summary["plan"] = {"steps": inp.get("steps") or defaults["steps"], "epochs": inp.get("epochs") or defaults["epochs"]}
    summary["params"] = summary["params"] or _run_params(inp)
    summary["progress"] = compute_progress(summary, summary["plan"], [])
    image_count = len(inp["trainingData"]["items"])
    db.civitai_run_add(summary["id"], prep.project_id, prep.training_type, prep.model, image_count,
                       summary["charged"] or summary["cost"] or prep.cost, summary["status"] or "processing",
                       summary)
    prep.status, prep.workflow_id = "submitted", summary["id"]
    return summary
