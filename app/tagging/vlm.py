"""透過視覺語言模型 (VLM) 產生自然語言描述，或（JoyCaption 的 Danbooru 模式）Danbooru 標籤。

支援兩種後端：
- openai    : 任何 OpenAI 相容的 /chat/completions 端點（Ollama、vLLM、LM Studio、OpenRouter…）
- anthropic : Claude API（官方 anthropic SDK）
"""
from __future__ import annotations

import base64
import io
import logging
import re
import threading
from typing import Any

import httpx
from PIL import Image

from ..config import settings
from ..i18n import t
from .postprocess import canon

log = logging.getLogger(__name__)


class VLMError(RuntimeError):
    pass


DETAIL_INSTRUCTIONS = {
    "short": "Write ONE concise sentence (at most 30 words) describing the main subject, what they are doing and the setting.",
    "medium": "Write 1-2 sentences (at most 60 words) describing the subject, clothing, pose/action, expression and the background.",
    "detailed": (
        "Write a detailed description in 2-5 sentences (at most 150 words). Cover the subject, appearance, clothing, "
        "pose and action, facial expression, background and environment, composition and camera angle, lighting, "
        "and the overall visual style or medium."
    ),
}


def build_prompt(project: dict[str, Any], hint_tags: list[str] | None = None) -> tuple[str, str]:
    s = project
    trigger = (s.get("trigger") or "").strip()
    lora_type = s.get("lora_type", "character")
    system = (
        "You write captions for images in a LoRA training dataset for a text-to-image model. "
        "Describe only what is visible, objectively, in plain English. "
        "Output only the caption text: no preamble such as 'This image shows', no markdown, no quotes, no lists."
    )
    lines = [DETAIL_INSTRUCTIONS.get(s.get("vlm_detail", "medium"), DETAIL_INSTRUCTIONS["medium"])]

    if lora_type == "character" and trigger:
        lines.append(
            f"The main character is named '{trigger}'. Refer to them by that name "
            f"(for example '{trigger} is sitting ...') instead of generic words like 'a girl' or 'a person'."
        )
        prune = set(s.get("prune_groups") or [])
        traits = []
        if "hair_color" in prune or "hair_length" in prune or "hair_style" in prune:
            traits.append("hair")
        if "eye_color" in prune:
            traits.append("eye color")
        if "body" in prune:
            traits.append("body features such as ears, horns, tail or skin tone")
        if traits:
            lines.append(
                f"Do not describe {trigger}'s permanent physical traits ({', '.join(traits)}); "
                "focus on clothing, pose, expression, action and the scene."
            )
    elif lora_type == "style":
        lines.append(
            "Describe only the content (subjects, actions, objects, setting, composition). "
            "Do NOT mention the art style, medium, rendering technique, color grading or any artist."
        )
    elif lora_type == "concept" and trigger:
        lines.append(
            f"The image shows the concept '{trigger}'. Use the word '{trigger}' once where that concept appears "
            "instead of describing the concept itself in detail; describe everything else normally."
        )

    if s.get("vlm_nsfw"):
        lines.append(
            "This is an adult (18+) dataset and every person depicted is an adult. If there is nudity or sexual "
            "content, describe it explicitly and accurately with direct anatomical terms (body parts, positions, "
            "acts, fluids). Do not censor, euphemize, moralize or add warnings."
        )

    if hint_tags:
        lines.append(
            "Reference tags from an automatic tagger (they may contain mistakes; only use what you can actually see): "
            + ", ".join(hint_tags[:60])
        )
    extra = (s.get("vlm_extra_prompt") or "").strip()
    if extra:
        lines.append(extra)
    return system, "\n".join(lines)


_PREAMBLE_RE = re.compile(
    r"^(caption\s*:\s*|here(?:'s| is) (?:a |the )?(?:caption|description)[^:]*:\s*|"
    r"(?:the|this) (?:image|picture|illustration|photo(?:graph)?) (?:shows|depicts|features|is of|presents)\s+|"
    r"in (?:the|this) (?:image|picture|illustration|photo),?\s+)",
    re.IGNORECASE,
)


_REFUSAL_RE = re.compile(
    r"^(i'?m sorry|i am sorry|sorry,|i can(?:not|'t)|i am unable|i'?m unable|i won'?t|as an ai|unfortunately,? i)",
    re.IGNORECASE,
)


def clean_caption(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = text.replace("**", "").replace("__", "").strip().strip('"').strip("“”").strip()
    for _ in range(2):
        text = _PREAMBLE_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _encode(image: Image.Image) -> str:
    img = image.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.thumbnail((settings.vlm_max_side, settings.vlm_max_side), Image.LANCZOS)
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- OpenAI 相容
def _caption_openai(image: Image.Image, system: str, prompt: str, temperature: float = 0.2,
                    max_tokens: int = 600, top_p: float | None = None) -> str:
    payload = {
        "model": settings.vlm_model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(image)}"}},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    headers = {"Authorization": f"Bearer {settings.vlm_api_key}"} if settings.vlm_api_key else {}
    try:
        r = httpx.post(f"{settings.vlm_base_url}/chat/completions", json=payload, headers=headers,
                       timeout=settings.vlm_timeout)
    except httpx.HTTPError as e:
        raise VLMError(t("msg.vlm_connect_failed", url=settings.vlm_base_url, error=e)) from e
    if r.status_code >= 400:
        hint = ""
        if r.status_code == 404 and "ollama" in settings.vlm_base_url:
            hint = t("msg.vlm_model_missing_hint", model=settings.vlm_model)
        raise VLMError(t("msg.vlm_http_error", status=r.status_code, body=r.text[:300]) + hint)
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as e:
        raise VLMError(t("msg.vlm_bad_response", body=str(data)[:300])) from e


# ---------------------------------------------------------------- Claude
_anthropic_client = None
_anthropic_lock = threading.Lock()
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")


def _anthropic():
    global _anthropic_client
    with _anthropic_lock:
        if _anthropic_client is None:
            import anthropic

            _anthropic_client = anthropic.Anthropic(timeout=settings.vlm_timeout)
        return _anthropic_client


def _caption_anthropic(image: Image.Image, system: str, prompt: str) -> str:
    import anthropic

    model = settings.anthropic_model
    kwargs: dict[str, Any] = {}
    if "haiku" not in model:
        kwargs["output_config"] = {"effort": settings.anthropic_effort}
    if model in _FALLBACK_MODELS:
        # 伺服器端 fallback：主模型拒答時自動改由其他模型接手
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    try:
        resp = _anthropic().beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _encode(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
            **kwargs,
        )
    except anthropic.AuthenticationError as e:
        raise VLMError(t("msg.claude_auth")) from e
    except anthropic.RateLimitError as e:
        raise VLMError(t("msg.claude_rate_limit", error=e.message)) from e
    except anthropic.APIStatusError as e:
        raise VLMError(t("msg.claude_error", status=e.status_code, error=e.message)) from e
    except anthropic.APIConnectionError as e:
        raise VLMError(t("msg.claude_connect", error=e)) from e
    if resp.stop_reason == "refusal":
        category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise VLMError(t("msg.claude_refusal", category=category or "-"))
    return "".join(b.text for b in resp.content if b.type == "text")


def caption_image(image: Image.Image, project: dict[str, Any], hint_tags: list[str] | None = None) -> str:
    backend = settings.vlm_backend
    if backend in ("", "none", "off"):
        raise VLMError(t("msg.vlm_disabled"))
    system, prompt = build_prompt(project, hint_tags)
    if backend == "anthropic":
        raw = _caption_anthropic(image, system, prompt)
    else:
        raw = _caption_openai(image, system, prompt)
    text = clean_caption(raw)
    if not text:
        raise VLMError(t("msg.vlm_empty"))
    if _REFUSAL_RE.match(text):
        raise VLMError(t("msg.vlm_refused", text=text[:120]))
    return text


# ---------------------------------------------------------------- Danbooru 標籤
# JoyCaption Beta One 官方「Danbooru tag list」提示與 system prompt（模型以這段文字訓練，不要改寫）
BOORU_SYSTEM = "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions."
BOORU_PROMPT = (
    "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, "
    "`character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, "
    "expression, actions, background. Use precise Danbooru syntax. No extra text."
)
VLM_TAG_GROUPS = ("character", "copyright", "artist", "general")
_NAMESPACE_RE = re.compile(r"^(artist|copyright|character|meta|general|species|lore|rating)\s*:\s*(.+)$", re.IGNORECASE)


def parse_booru_tags(text: str) -> dict[str, list[str]]:
    """把 VLM 輸出的 Danbooru 標籤清單拆成 {character, copyright, artist, general}（正規形式、去重）。

    官方標示這個模式約 3% 會輸出異常（整句話、無限重複），過長的項目與重複會被丟掉。
    分級（rating:）交給 WD14；meta / species / lore 視為一般標籤。
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip().strip("`")
    out: dict[str, list[str]] = {k: [] for k in VLM_TAG_GROUPS}
    seen: set[str] = set()
    for item in re.split(r"[,\n]", text):
        item = item.strip().rstrip(".").strip()
        m = _NAMESPACE_RE.match(item)
        group, name = (m.group(1).lower(), m.group(2)) if m else ("general", item)
        if group == "rating":
            continue
        if group not in out:
            group = "general"
        tag = canon(name).lower()
        if not tag or len(tag) > 60 or len(tag.split()) > 6 or tag in seen:
            continue
        seen.add(tag)
        out[group].append(tag)
    return out


def tag_image(image: Image.Image) -> dict[str, list[str]]:
    """用 VLM 產生 Danbooru 標籤（建議 JoyCaption Beta One）。"""
    backend = settings.vlm_backend
    if backend in ("", "none", "off"):
        raise VLMError(t("msg.vlm_disabled"))
    if backend == "anthropic":
        raw = _caption_anthropic(image, "You tag images for a text-to-image training dataset.", BOORU_PROMPT)
    else:
        # 取樣參數照 JoyCaption 官方預設；temperature 0 容易陷入重複
        raw = _caption_openai(image, BOORU_SYSTEM, BOORU_PROMPT, temperature=0.6, top_p=0.9, max_tokens=512)
    text = (raw or "").strip()
    if not text:
        raise VLMError(t("msg.vlm_empty"))
    if _REFUSAL_RE.match(text):
        raise VLMError(t("msg.vlm_refused", text=text[:120]))
    tags = parse_booru_tags(text)
    if sum(len(v) for v in tags.values()) < 3:
        raise VLMError(t("msg.vlm_bad_tags", text=text[:160]))
    return tags


def status() -> dict[str, Any]:
    backend = settings.vlm_backend
    info: dict[str, Any] = {"backend": backend, "available": False}
    if backend in ("", "none", "off"):
        info["message"] = t("msg.vlm_status_disabled")
        return info
    if backend == "anthropic":
        import os

        info["model"] = settings.anthropic_model
        info["available"] = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        info["message"] = t("msg.vlm_status_key_ok") if info["available"] else t("msg.vlm_status_key_missing")
        return info
    info["model"] = settings.vlm_model
    info["base_url"] = settings.vlm_base_url
    try:
        headers = {"Authorization": f"Bearer {settings.vlm_api_key}"} if settings.vlm_api_key else {}
        r = httpx.get(f"{settings.vlm_base_url}/models", headers=headers, timeout=3)
        r.raise_for_status()
        models = [m.get("id") for m in r.json().get("data", [])]
        info["models"] = models
        want = settings.vlm_model
        info["available"] = not models or any(m in (want, f"{want}:latest") for m in models)
        info["message"] = t("msg.vlm_status_ok") if info["available"] else t("msg.vlm_status_model_missing", model=want)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # 連線被拒或主機名稱查不到：服務沒在跑，或 vLLM 還在載入模型（HTTP 埠載入完才會開）
        info["message"] = t("msg.vlm_status_starting", url=settings.vlm_base_url)
    except Exception as e:  # noqa: BLE001
        info["message"] = t("msg.vlm_status_unreachable", error=e.__class__.__name__)
    return info
