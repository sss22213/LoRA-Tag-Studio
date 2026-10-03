"""各底模 (base model) 的標註預設值與使用指南。

這裡只放「與語言無關」的資料：預設設定、分級對應、範例 caption、生成 prompt 範本。
名稱、說明、訓練建議等文字在 app/locales/<lang>.json 的 server.profiles.<key>。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .i18n import t, tr

# WD14 輸出的四種分級
WD14_RATINGS = ("general", "sensitive", "questionable", "explicit")

PONY_RATINGS = {
    "general": "rating_safe",
    "sensitive": "rating_questionable",
    "questionable": "rating_questionable",
    "explicit": "rating_explicit",
}
DANBOORU_NEW_RATINGS = {  # NoobAI / Illustrious 系
    "general": "general",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit",
}
SIMPLE_NSFW_RATINGS = {  # SD1.5 / SDXL 常見寫法：只在 NSFW 圖片加上 nsfw
    "questionable": "nsfw",
    "explicit": "nsfw",
}
ANIMAGINE_RATINGS = {  # Animagine / Anima
    "general": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "explicit",
}

# 所有專案設定的鍵與基本預設值；profile 只覆寫需要不同的部分
BASE_SETTINGS: dict[str, Any] = {
    "profile": "illustrious",
    "lora_type": "character",  # character | style | concept
    "trigger": "",
    "class_word": "",  # 例如 1girl / woman / style，kohya 資料夾命名與提示用
    "caption_mode": "tags",  # tags | natural | hybrid | trigger_only
    "use_wd14": True,
    "use_vlm": False,
    "wd14_model": "",  # 空字串 = 使用伺服器預設
    "general_threshold": 0.35,
    "character_threshold": 0.85,
    "max_tags": 40,
    "include_character_tags": True,
    "include_rating": False,
    "prefix_tags": "",  # 固定加在 trigger 後面的標籤，例如 source_anime
    "append_tags": "",  # 固定加在最後的標籤
    "block_tag": "white rectangle",  # 有白色色塊的圖片自動加上的關鍵字（生成時放進負面提示）
    "block_tag_auto": False,
    "add_quality_tags": False,
    "quality_tags": "",
    "underscore_to_space": True,
    "escape_parentheses": False,
    "prune_groups": [],
    "blacklist": "",
    "vlm_detail": "medium",  # short | medium | detailed
    "vlm_extra_prompt": "",
    "vlm_nsfw": False,  # 允許 VLM 用直白的成人詞彙描述（需使用無審查模型）
    "nl_position": "before_tags",  # hybrid 模式：自然語句在標籤前或後
    "vlm_tags": "off",  # 用 VLM（JoyCaption 的 Danbooru 模式）產生標籤：off | extra | merge | only
    "vlm_artist_tags": False,  # 是否採用 VLM 判斷的畫師標籤（常猜錯，預設不用）
}

# LoRA 類型 → 建議修剪的特徵群組（文字在 server.lora_types）
LORA_TYPES: dict[str, dict[str, Any]] = {
    "character": {"prune_groups": ["hair_color", "eye_color"]},
    "style": {"prune_groups": ["style_medium"]},
    "concept": {"prune_groups": []},
}

CAPTION_MODES = ("tags", "natural", "hybrid", "trigger_only")

# VLM 標籤模式：off 只用 WD14；extra 加入 VLM 的角色 / 作品 / 畫師；merge 再加入 VLM 的一般標籤；
# only 只用 VLM 的標籤（WD14 若開啟只用來判斷分級）
VLM_TAG_MODES = ("off", "extra", "merge", "only")

# 各架構文字編碼器一次讀取的 token 數（CLIP 75；T5 / Qwen3 512），WebUI 用來提示 caption 是否過長
FAMILY_TOKEN_LIMITS = {"sd15": 75, "sdxl": 75, "anima": 512, "flux": 512}

# profile 可選欄位：
#   rating_position  "prefix"：分級標籤緊接在 trigger / 前綴標籤後（Pony、Anima）
#                    "after_character"（預設）：放在人數 + 角色標籤之後（Animagine、NoobAI…）
#   shuffle_caption  False：建議不要 shuffle（kohya 參數提示）
#   artist_prefix    畫師標籤的前綴（Anima 要求 @畫師名）

PROFILES: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------ SD 1.5
    "sd15_anime": {
        "family": "sd15",
        "defaults": {"caption_mode": "tags", "max_tags": 30, "general_threshold": 0.35, "include_rating": True},
        "rating_map": SIMPLE_NSFW_RATINGS,
        "example": "mychar, 1girl, solo, long hair, school uniform, serafuku, smile, looking at viewer, outdoors, cherry blossoms",
        "a1111": {
            "positive": "masterpiece, best quality, <lora:{lora}:0.8>, {trigger}, 1girl, solo, ...",
            "negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        },
    },
    "sd15_realistic": {
        "family": "sd15",
        "defaults": {"caption_mode": "hybrid", "use_vlm": True, "vlm_detail": "short", "max_tags": 20,
                     "general_threshold": 0.4, "include_character_tags": False},
        "rating_map": SIMPLE_NSFW_RATINGS,
        "example": "ohwx woman, a photo of a woman with long brown hair smiling on a beach at sunset, white dress, upper body",
        "a1111": {
            "positive": "RAW photo, <lora:{lora}:0.8>, {trigger}, ..., 8k uhd, dslr, soft lighting, high quality, film grain",
            "negative": "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, blurry, bad anatomy",
        },
    },
    # ------------------------------------------------------------------ SDXL
    "sdxl_base": {
        "family": "sdxl",
        "defaults": {"caption_mode": "hybrid", "use_vlm": True, "vlm_detail": "medium", "max_tags": 20,
                     "general_threshold": 0.4, "include_character_tags": False},
        "rating_map": SIMPLE_NSFW_RATINGS,
        "example": "ohwx man, a close-up portrait of a man with short black hair and a beard wearing a denim jacket, standing in a neon-lit street at night, shallow depth of field, cinematic lighting",
        "a1111": {
            "positive": "<lora:{lora}:0.8>, {trigger}, photo of ..., highly detailed, sharp focus, natural lighting",
            "negative": "lowres, worst quality, low quality, blurry, deformed, bad anatomy, bad hands, watermark, text, signature",
        },
    },
    "pony_v6": {
        "family": "sdxl",
        "defaults": {"caption_mode": "tags", "max_tags": 40, "general_threshold": 0.35, "include_rating": True,
                     "prefix_tags": "source_anime", "quality_tags": "score_9, score_8_up, score_7_up",
                     "add_quality_tags": False},
        "rating_map": PONY_RATINGS,
        "rating_position": "prefix",
        "example": "mychar, source_anime, rating_safe, 1girl, solo, long hair, maid, maid headdress, smile, looking at viewer, indoors",
        "a1111": {
            "positive": "score_9, score_8_up, score_7_up, source_anime, rating_safe, <lora:{lora}:1>, {trigger}, 1girl, solo, ...",
            "negative": "score_6, score_5, score_4, source_pony, source_furry, 3d, worst quality, low quality, bad anatomy",
        },
    },
    "illustrious": {
        "family": "sdxl",
        "defaults": {"caption_mode": "tags", "max_tags": 40, "general_threshold": 0.35, "include_rating": False,
                     "quality_tags": "masterpiece, best quality"},
        "rating_map": DANBOORU_NEW_RATINGS,
        "example": "mychar, 1girl, solo, white hair, long hair, black dress, frills, sitting, looking at viewer, smile, indoors, window",
        "a1111": {
            "positive": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, <lora:{lora}:0.8>, {trigger}, 1girl, solo, ...",
            "negative": "lowres, worst quality, bad quality, bad anatomy, sketch, jpeg artifacts, signature, watermark, old, oldest, censored, bar censor",
        },
    },
    "noobai_xl": {
        "family": "sdxl",
        "defaults": {"caption_mode": "tags", "max_tags": 40, "general_threshold": 0.35, "include_rating": True,
                     "quality_tags": "masterpiece, best quality, newest, absurdres, highres"},
        "rating_map": DANBOORU_NEW_RATINGS,
        "example": "mychar, 1girl, solo, general, animal ears, fox girl, kimono, holding umbrella, rain, outdoors",
        "a1111": {
            "positive": "masterpiece, best quality, newest, absurdres, highres, <lora:{lora}:0.8>, {trigger}, 1girl, solo, ...",
            "negative": "worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro",
        },
    },
    "animagine_xl": {
        "family": "sdxl",
        "defaults": {"caption_mode": "tags", "max_tags": 40, "general_threshold": 0.35, "include_rating": True,
                     "quality_tags": "masterpiece, high score, great score, absurdres"},
        "rating_map": ANIMAGINE_RATINGS,
        "example": "mychar, 1girl, solo, safe, blonde hair, blue eyes, armor, holding sword, castle, sky",
        "a1111": {
            "positive": "<lora:{lora}:0.8>, {trigger}, 1girl, solo, ..., safe, masterpiece, high score, great score, absurdres",
            "negative": "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry",
        },
    },
    # ------------------------------------------------------------------ Anima（Cosmos-Predict2 2B + Qwen3 0.6B）
    "anima": {
        "family": "anima",
        # 官方順序：[品質 / meta / 年份 / 分級] [1girl…] [角色] [作品] [畫師] [一般標籤]
        "rating_position": "prefix",
        # sd-scripts 的 Anima 範例使用 --cache_text_encoder_outputs，此時不能 shuffle_caption
        "shuffle_caption": False,
        "artist_prefix": "@",
        "defaults": {"caption_mode": "tags", "max_tags": 40, "general_threshold": 0.35, "include_rating": True,
                     "quality_tags": "masterpiece, best quality, score_7"},
        "rating_map": ANIMAGINE_RATINGS,
        "example": "mychar, safe, 1girl, solo, long hair, white hair, red eyes, black dress, frills, sitting, looking at viewer, smile, indoors, window",
        "a1111": {
            "positive": "masterpiece, best quality, score_7, safe, <lora:{lora}:1>, {trigger}, 1girl, solo, ...",
            "negative": "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration",
        },
    },
    # ------------------------------------------------------------------ FLUX
    "flux1_dev": {
        "family": "flux",
        "defaults": {"caption_mode": "natural", "use_vlm": True, "vlm_detail": "detailed", "max_tags": 30,
                     "general_threshold": 0.4, "include_character_tags": False},
        "rating_map": None,
        "example": "mychar, a young woman with long silver hair and red eyes, wearing a black trench coat, stands on a rainy city street at night. She looks over her shoulder at the viewer with a calm expression. Neon signs reflect on the wet pavement, cinematic lighting, anime illustration style.",
        "a1111": {
            "positive": "{trigger}, a young woman with ..., <lora:{lora}:1>",
            "negative": "",  # Flux dev 不使用負向提示（說明文字在語系檔）
        },
    },
}


def get_profile(key: str) -> dict[str, Any]:
    if key not in PROFILES:
        raise KeyError(t("msg.unknown_profile", profile=key, available=", ".join(PROFILES)))
    return PROFILES[key]


def profile_text(key: str) -> dict[str, Any]:
    """目前語言的 profile 文字（name / summary / civitai_base / caption_style / …）。"""
    return tr(f"profiles.{key}", default={}) or {}


def profile_name(key: str) -> str:
    return profile_text(key).get("name", key)


def list_profiles() -> list[dict[str, Any]]:
    out = []
    for key, p in PROFILES.items():
        tx = profile_text(key)
        out.append({
            "key": key,
            "name": tx.get("name", key),
            "family": p["family"],
            "token_limit": FAMILY_TOKEN_LIMITS[p["family"]],
            "summary": tx.get("summary", ""),
            "civitai_base": tx.get("civitai_base", ""),
            "caption_mode": p["defaults"].get("caption_mode", BASE_SETTINGS["caption_mode"]),
        })
    return out


def profile_view(key: str) -> dict[str, Any]:
    """合併資料與目前語言文字，給 API / WebUI 使用。"""
    p = get_profile(key)
    tx = profile_text(key)
    labels = tr("training_labels", default={})
    return {
        "key": key,
        "name": tx.get("name", key),
        "family": p["family"],
        "token_limit": FAMILY_TOKEN_LIMITS[p["family"]],
        "summary": tx.get("summary", ""),
        "civitai_base": tx.get("civitai_base", ""),
        "rating_map": p["rating_map"],
        "guide": {
            "caption_style": tx.get("caption_style", ""),
            "tag_order": tx.get("tag_order", []),
            "example": p["example"],
            "training": {labels.get(k, k): v for k, v in (tx.get("training") or {}).items()},
            "a1111": {
                "positive": p["a1111"]["positive"],
                "negative": p["a1111"]["negative"],
                "negative_note": tx.get("a1111_negative_note", ""),
                "settings": tx.get("a1111_settings", ""),
            },
            "nsfw": tx.get("nsfw", ""),
            "notes": tx.get("notes", []),
        },
        "defaults": default_settings(key),
    }


def lora_types_view() -> dict[str, dict[str, Any]]:
    return {k: {**(tr(f"lora_types.{k}", default={}) or {}), "prune_groups": v["prune_groups"]}
            for k, v in LORA_TYPES.items()}


def lora_type_name(key: str) -> str:
    return tr(f"lora_types.{key}.name", default=key)


def caption_modes_view() -> dict[str, str]:
    return {k: tr(f"caption_modes.{k}", default=k) for k in CAPTION_MODES}


def default_settings(profile: str, lora_type: str = "character", **overrides: Any) -> dict[str, Any]:
    """依底模與 LoRA 類型產生專案預設設定。"""
    p = get_profile(profile)
    s = deepcopy(BASE_SETTINGS)
    s.update(deepcopy(p["defaults"]))
    s["profile"] = profile
    if lora_type not in LORA_TYPES:
        lora_type = "character"
    s["lora_type"] = lora_type
    s["prune_groups"] = list(LORA_TYPES[lora_type]["prune_groups"])
    if lora_type == "style":
        s["include_character_tags"] = False
    for k, v in overrides.items():
        if k in BASE_SETTINGS and v is not None:
            s[k] = v
    return s


def normalize_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """補齊缺漏鍵、丟棄未知鍵並修正型別。"""
    s = deepcopy(BASE_SETTINGS)
    for k, v in (raw or {}).items():
        if k in s and v is not None:
            s[k] = v
    if s["profile"] not in PROFILES:
        s["profile"] = BASE_SETTINGS["profile"]
    if s["lora_type"] not in LORA_TYPES:
        s["lora_type"] = "character"
    if s["caption_mode"] not in CAPTION_MODES:
        s["caption_mode"] = "tags"
    if s["vlm_tags"] not in VLM_TAG_MODES:
        s["vlm_tags"] = "off"
    for k in ("general_threshold", "character_threshold"):
        s[k] = min(1.0, max(0.0, float(s[k])))
    s["max_tags"] = max(0, int(s["max_tags"]))
    if not isinstance(s["prune_groups"], list):
        s["prune_groups"] = [g.strip() for g in str(s["prune_groups"]).split(",") if g.strip()]
    for k in ("use_wd14", "use_vlm", "include_character_tags", "include_rating", "add_quality_tags",
              "underscore_to_space", "escape_parentheses", "vlm_nsfw", "vlm_artist_tags", "block_tag_auto"):
        s[k] = bool(s[k])
    s["block_tag"] = ", ".join(x.strip() for x in str(s["block_tag"]).split(",") if x.strip()) or BASE_SETTINGS["block_tag"]
    s["trigger"] = str(s["trigger"]).strip()
    return s


def guide_markdown(key: str) -> str:
    """目前語言的 Markdown 版指南（給 LLM / 文件）。"""
    v = profile_view(key)
    g = v["guide"]
    h = lambda k, **kw: t(f"guide_md.{k}", **kw)  # noqa: E731
    mode = PROFILES[key]["defaults"].get("caption_mode", "tags")
    lines = [
        h("title", name=v["name"]),
        "",
        f"- {h('civitai_base')}: {v['civitai_base']}",
        f"- {h('default_mode')}: {caption_modes_view()[mode]}",
        "",
        f"## {h('caption_style')}",
        g["caption_style"],
        "",
        f"## {h('tag_order')}",
        " → ".join(g["tag_order"]),
        "",
        f"## {h('example')}",
        f"```\n{g['example']}\n```",
        "",
        f"## {h('training')}",
    ]
    lines += [f"- {k}: {val}" for k, val in g["training"].items()]
    negative = f"`{g['a1111']['negative']}`" if g["a1111"]["negative"] else g["a1111"]["negative_note"]
    lines += [
        "",
        f"## {h('a1111')}",
        f"- Positive: `{g['a1111']['positive']}`",
        f"- Negative: {negative}",
        f"- {h('settings')}: {g['a1111']['settings']}",
    ]
    if v["rating_map"]:
        lines += ["", f"## {h('rating_map')}"]
        lines += [f"- {k} → `{val}`" for k, val in v["rating_map"].items()]
    if g["nsfw"]:
        lines += ["", f"## {h('nsfw')}", g["nsfw"], "", t("nsfw_common")]
    if g["notes"]:
        lines += ["", f"## {h('notes')}"] + [f"- {n}" for n in g["notes"]]
    lines += ["", f"## {h('lora_types')}"]
    for lt in lora_types_view().values():
        lines.append(f"### {lt.get('name', '')}")
        lines += [f"- {tip}" for tip in lt.get("tips", [])]
    return "\n".join(lines)
