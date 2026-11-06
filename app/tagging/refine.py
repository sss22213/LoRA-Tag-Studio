"""用 Claude / Codex CLI 看圖修正既有的標註：拿掉錯的標籤、補上明顯漏掉的、修正自然語句裡的錯誤。

模型只回 JSON（tags / caption / changes），回來的標籤再經過專案的修剪群組、黑名單、觸發詞與數量上限，
所以不會把刻意拿掉的特徵（例如角色 LoRA 的髮色）加回去。修正前的內容存在 raw["refine_prev"]，可以還原。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from PIL import Image

from ..i18n import t
from . import cli
from .postprocess import apply_filters, canon, split_tags
from .vlm import VLMError, ask_cli, clean_caption, _REFUSAL_RE

log = logging.getLogger(__name__)

TARGETS = ("both", "tags", "caption")

SYSTEM = ("You review and correct the captions of a LoRA training dataset for a text-to-image model. You get one "
          "image with its current tags and caption, written by automatic taggers that make mistakes. Fix what is "
          "wrong and keep what is right. Reply with JSON only.")


def _prune_words(groups: list[str]) -> str:
    return ", ".join(g.replace("_", " ") for g in groups)


def build_prompt(s: dict[str, Any], project_name: str, tags: list[str], nl: str, target: str,
                 instructions: str = "") -> tuple[str, list[str]]:
    """提示與要模型回的欄位。target：both / tags / caption（沒有自然語句時只修標籤）。"""
    do_tags = target in ("both", "tags")
    do_caption = target in ("both", "caption") and bool(nl.strip())
    trigger = (s.get("trigger") or "").strip()
    lora = s.get("lora_type", "character")
    lines = [f"Dataset: {project_name} ({lora} LoRA)."]
    if trigger:
        lines.append(f"Trigger word: {trigger}. It is added to every caption automatically: never put it in the tags."
                     + (f" In the caption, keep calling the main character '{trigger}'." if lora == "character" else ""))
    lines.append("Current tags: " + (", ".join(tags) if tags else "(none)"))
    if nl.strip():
        lines.append("Current caption: " + nl.strip())
    keys: list[str] = []
    if do_tags:
        keys.append("tags")
        rules = [
            "Remove tags that are wrong: not visible in the image, contradicting another tag (keep only the correct "
            "one, e.g. not both long hair and short hair), or naming a different character or series than the one "
            "in the image.",
            "Add clearly visible, important tags that are missing (Danbooru style: lowercase, spaces instead of "
            "underscores). Do not add guesses.",
            "Order: people count (1girl, solo…), character, series, then everything else.",
        ]
        limit = int(s.get("max_tags") or 0)
        if limit:
            rules.append(f"At most {limit} tags.")
        prune = list(s.get("prune_groups") or [])
        if prune:
            rules.append(f"Do not add tags of these kinds, they are removed on purpose: {_prune_words(prune)}.")
        lines.append("Tags:\n- " + "\n- ".join(rules))
    if do_caption:
        keys.append("caption")
        lines.append("Caption:\n- Correct only the parts that are wrong or not visible; keep everything that is right.\n"
                     "- Keep its language, style, number of sentences and length; do not shorten it.\n"
                     "- Remove tag lists, notes or instructions that do not belong in a caption (such as "
                     "'Reference tags: …').")
    if s.get("vlm_nsfw"):
        lines.append("This is an adult (18+) dataset and every person depicted is an adult. Describe nudity or "
                     "sexual content accurately and directly; do not censor or add warnings.")
    if instructions.strip():
        lines.append("Instructions from the dataset owner (follow them): " + instructions.strip())
    example = {"tags": "[...]", "caption": '"..."'}
    shape = ", ".join(f'"{k}": {example[k]}' for k in keys)
    lines.append(f'Reply with JSON only, no markdown: {{{shape}, "changes": "one short sentence about what you changed"}}')
    return "\n".join(lines), keys


def parse(text: str) -> dict[str, Any]:
    """從回答裡取出第一個完整的 JSON 物件（容忍 ```json 區塊、前後多餘的文字，以及 JSON 後面又接的說明）。"""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            data, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(data, dict) and ("tags" in data or "caption" in data):
            return data
    log.warning("修正標註：回答不是可用的 JSON：%s", text[:2000])
    raise VLMError(t("msg.refine_bad_reply", text=text[:160]))


def refine_image(image: Image.Image, img: dict[str, Any], s: dict[str, Any], project_name: str, engine: str,
                 model: str = "", target: str = "both", instructions: str = "") -> dict[str, Any]:
    """回傳 {tags, nl_caption, changes}；沒要求修正的欄位維持原樣。"""
    if engine not in cli.ENGINES:
        raise VLMError(t("msg.cli_bad_engine", engines=", ".join(cli.ENGINES)))
    tags, nl = list(img.get("tags") or []), img.get("nl_caption") or ""
    prompt, keys = build_prompt(s, project_name, tags, nl, target, instructions)
    if not keys:
        return {"tags": tags, "nl_caption": nl, "changes": ""}
    reply = ask_cli(engine, model, SYSTEM, prompt, image)
    if _REFUSAL_RE.match(reply.strip()):
        raise VLMError(t("msg.vlm_refused", text=reply.strip()[:120]))
    data = parse(reply)
    out = {"tags": tags, "nl_caption": nl, "changes": str(data.get("changes") or "")[:300]}
    if "tags" in keys:
        raw = data.get("tags")
        if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
            raise VLMError(t("msg.refine_bad_reply", text=str(raw)[:160]))
        trigger = canon(s.get("trigger") or "").lower()
        new = [x for x in apply_filters(split_tags(raw), s) if x.lower() != trigger]
        limit = int(s.get("max_tags") or 0)
        new = new[:limit] if limit else new
        if not new:  # 全部拿掉通常是模型出錯，不採用
            raise VLMError(t("msg.refine_empty"))
        out["tags"] = new
    if "caption" in keys:
        cap = data.get("caption")
        if not isinstance(cap, str) or not clean_caption(cap):
            raise VLMError(t("msg.refine_bad_reply", text=str(cap)[:160]))
        out["nl_caption"] = clean_caption(cap)
    return out
