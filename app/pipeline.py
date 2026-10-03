"""單張圖片的標註流程：WD14 標籤 →（選用）VLM Danbooru 標籤 →（選用）VLM 自然語句 → 後處理。"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from .i18n import t
from .profiles import default_settings, normalize_settings
from .tagging import vlm
from .tagging.blocks import has_blocks
from .tagging.postprocess import (build_caption, count_tokens_estimate, select_tags, split_tags, top_rating,
                                  vlm_tag_groups)
from .tagging.wd14 import get_tagger

log = logging.getLogger(__name__)


def needs_vlm(s: dict[str, Any]) -> bool:
    """是否要用 VLM 產生自然語句（只有 natural / hybrid 模式會用到）。"""
    return bool(s.get("use_vlm")) and s.get("caption_mode") in ("natural", "hybrid")


def needs_vlm_tags(s: dict[str, Any]) -> bool:
    """是否要用 VLM 產生 Danbooru 標籤。"""
    return s.get("vlm_tags", "off") != "off"


def config_problem(s: dict[str, Any]) -> str | None:
    """設定組合在標註時不會產生任何內容（WD14 關閉、VLM 也不會執行）時回傳說明。

    WD14 關閉時 tag_pil 會保留既有標籤，所以這種組合標註後什麼都不會改變。
    trigger_only 模式本來就不需要標籤，不算問題。
    """
    if s.get("caption_mode") == "trigger_only" or s.get("use_wd14", True) or needs_vlm(s) or needs_vlm_tags(s):
        return None
    return t("msg.nothing_to_generate")


def tag_pil(im: Image.Image, s: dict[str, Any], prev: dict[str, Any] | None = None) -> dict[str, Any]:
    """對 PIL 圖片執行標註，回傳要寫回的欄位。"""
    prev = prev or {}
    raw = prev.get("raw")
    tags = list(prev.get("tags") or [])
    nl = prev.get("nl_caption") or ""
    errors: list[str] = []

    fresh = False  # 有新的 WD14 / VLM 結果才重組標籤，否則保留原本的（例如匯入的 .txt）
    if s.get("use_wd14", True):
        raw = get_tagger(s.get("wd14_model") or None).predict(im)
        fresh = True

    if needs_vlm_tags(s):
        # VLM 標籤存進 raw["vlm"]，之後「重新套用」可以不必再呼叫 VLM
        try:
            raw = {**(raw or {}), "vlm": vlm.tag_image(im)}
            fresh = True
        except vlm.VLMError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("VLM 標籤失敗")
            errors.append(t("msg.vlm_failed", error=e))

    if fresh:
        tags = select_tags(raw or {}, s)

    if needs_vlm(s):
        try:
            nl = vlm.caption_image(im, s, hint_tags=tags or None)
        except vlm.VLMError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("VLM caption 失敗")
            errors.append(t("msg.vlm_failed", error=e))

    return {
        "raw": raw,
        "tags": tags,
        "nl_caption": nl,
        "rating": top_rating(raw),
        "status": "error" if errors else "done",
        "error": "；".join(errors) or None,
    }


def caption_for(img: dict[str, Any], s: dict[str, Any]) -> str:
    raw = img.get("raw") or {}
    # 分級標籤要放在「人數 + 角色 / 作品 / 畫師」之後的底模，需要知道哪些標籤屬於這些類別
    vlm = vlm_tag_groups(raw, s)
    chars = [n for n, _ in raw.get("character", [])] + vlm["character"] + vlm["copyright"] + vlm["artist"]
    return build_caption(s, img.get("tags") or [], img.get("nl_caption") or "", img.get("rating"), chars,
                         block_tags(img, s))


def block_tags(img: dict[str, Any], s: dict[str, Any]) -> list[str]:
    """有白色色塊、且專案開啟「自動加入色塊關鍵字」時要加的標籤。"""
    return split_tags(s.get("block_tag")) if s.get("block_tag_auto") and has_blocks(img.get("blocks")) else []


def quick_tag(im: Image.Image, profile: str = "illustrious", lora_type: str = "character",
              **overrides: Any) -> dict[str, Any]:
    """不建立專案，直接對單張圖片產生 caption（給 API / LLM 使用）。"""
    s = normalize_settings(default_settings(profile, lora_type, **overrides))
    result = tag_pil(im, s)
    caption = caption_for(result, s)
    return {
        "caption": caption,
        "tags": result["tags"],
        "nl_caption": result["nl_caption"],
        "rating": result["rating"],
        "token_estimate": count_tokens_estimate(caption),
        "error": result["error"],
        "settings": s,
    }
