"""標籤後處理：門檻、排序、特徵修剪、黑名單，以及依底模組合最終 caption。

內部儲存的標籤一律是「正規形式」：空格分隔、括號不跳脫（例如 `tokai teio (umamusume)`）。
底線 / 跳脫等輸出格式只在 build_caption() 時套用。
"""
from __future__ import annotations

import fnmatch
import re
from typing import Any, Iterable

from ..i18n import t
from ..profiles import get_profile

# 含底線的顏文字標籤，不能把底線換成空格
KAOMOJIS = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}

COUNT_TAG_RE = re.compile(r"^((\d|6\+)(girl|boy|other)s?|solo|solo focus|multiple girls|multiple boys|multiple others|no humans)$")

_COLORS = (
    "aqua|black|blonde|blue|brown|green|grey|gray|orange|pink|purple|red|silver|white|yellow|"
    "light brown|light blue|light purple|light green|light pink|dark blue|dark green|dark red|dark brown|dark purple|"
    "multicolored|two-tone|gradient|streaked|split-color|colored inner|rainbow"
)

# 可選擇修剪的特徵群組（顯示名稱在語系檔 server.prune_groups）。角色 LoRA 常刪除固定特徵，讓 trigger 學會它們。
PRUNE_GROUPS: dict[str, dict[str, Any]] = {
    "hair_color": {
        "patterns": [rf"({_COLORS}) hair", r"colored tips", r"streaked hair", r"colored inner hair"],
    },
    "hair_length": {
        "patterns": [r"(very short|short|medium|long|very long|absurdly long) hair"],
    },
    "hair_style": {
        "patterns": [
            r"(low |short |high )?twintails", r"(side |high |low )?ponytail", r"(single |twin |french |crown |side )?braids?",
            r"(single |double )?hair bun", r"drill hair", r"twin drills", r"ahoge", r"antenna hair",
            r"((blunt|swept|parted|crossed|asymmetrical) )?bangs", r"sidelocks", r"hair between eyes", r"hair intakes",
            r"bob cut", r"hime cut", r"(wavy|curly|straight|messy|spiked) hair", r"hair over (one eye|eyes)",
            r"one side up", r"two side up",
        ],
    },
    "eye_color": {
        "patterns": [
            rf"({_COLORS}) eyes", r"heterochromia", r"slit pupils", r"symbol-shaped pupils",
            r"(star|heart)-shaped pupils", r"ringed eyes", r"tareme", r"tsurime",
        ],
    },
    "body": {
        "patterns": [
            r"flat chest", r"(small|medium|large|huge|gigantic) breasts",
            r"animal ears", r"(cat|dog|fox|wolf|rabbit|horse|bear|mouse|tiger|lion) ears", r"animal ear fluff", r"extra ears",
            r"pointy ears", r"(single |demon |dragon )?horns?", r"(cat |fox |dog |wolf |demon |dragon )?tail",
            r"(angel |demon |dragon |bat |feathered )?wings", r"halo", r"fangs?", r"skin fang",
            r"mole( under (eye|mouth))?", r"freckles", r"dark skin", r"dark-skinned (female|male)", r"tan", r"muscular( female| male)?",
        ],
    },
    "style_medium": {
        "patterns": [
            r"anime coloring", r"realistic", r"photorealistic", r"photo \(medium\)", r"sketch", r"lineart",
            r"monochrome", r"greyscale", r"limited palette", r"flat color", r"cel shading",
            r".+ \(medium\)", r"traditional media", r"faux traditional media", r"pixel art", r"3d", r"official art",
            r"game cg", r"official style", r"oekaki", r"retro artstyle", r"\d{4}s \(style\)", r"impressionism",
            r"ukiyo-e", r"parody", r"style parody", r"blending", r"chromatic aberration", r"film grain",
        ],
    },
    "nsfw": {
        "patterns": [
            r"nude", r"completely nude", r"nipples?", r"areolae", r"pussy", r"penis", r"testicles", r"anus", r"clitoris",
            r"sex", r"vaginal", r"anal", r"oral", r"fellatio", r"cunnilingus", r"paizuri", r"handjob", r"footjob",
            r"masturbation", r"cum( .+)?", r"ejaculation", r"pubic hair", r"erection", r"spread legs", r"spread pussy",
            r"pussy juice", r"after sex", r"group sex", r"threesome", r"nsfw", r"explicit", r"rating[ _].+",
        ],
    },
    "censorship": {
        "patterns": [r"censored", r"uncensored", r"mosaic censoring", r"bar censor", r"convenient censoring",
                     r"novelty censor", r"heart censor", r"blank censor", r"light censor", r"steam censor"],
    },
    "meta": {
        "patterns": [
            r"signature", r"watermark", r"artist name", r"dated", r"(twitter|patreon|pixiv) username", r"web address",
            r"copyright name", r"character name", r"logo", r"(english |japanese |chinese )?text", r"speech bubble",
            r"jpeg artifacts", r"commentary( request)?", r"translated", r"lowres", r"highres", r"absurdres",
        ],
    },
}

_PRUNE_RE = {
    key: re.compile("^(" + "|".join(g["patterns"]) + ")$")
    for key, g in PRUNE_GROUPS.items()
}


# Civitai 政策：未成年特徵 + 性內容的組合一律不可上傳，這類圖片會被標記並排除匯出
MINOR_TAGS = {
    "loli", "shota", "child", "children", "female child", "male child", "toddler", "baby",
    "kindergarten uniform", "elementary school student", "randoseru", "aged down",
}
SEXUAL_TAGS = {
    "nsfw", "explicit", "rating_explicit", "rating_questionable", "nude", "completely nude", "sex", "vaginal",
    "anal", "oral", "fellatio", "pussy", "penis", "nipples", "cum", "masturbation", "paizuri", "handjob",
    "cunnilingus", "erection", "spread pussy", "after sex", "group sex",
}


def policy_flag(tags: Iterable[str], rating: str | None = None, nl_caption: str = "") -> str | None:
    """偵測「未成年特徵 + 性內容」組合，回傳原因（None 表示沒問題）。"""
    keys = {canon(t).lower() for t in tags}
    minor = keys & MINOR_TAGS
    if not minor:
        return None
    sexual = rating in ("questionable", "explicit") or bool(keys & SEXUAL_TAGS)
    if not sexual and nl_caption:
        sexual = bool(re.search(r"\b(nude|naked|sex|sexual|genital|penis|vagina|pussy|nipples?)\b", nl_caption, re.I))
    if sexual:
        return t("msg.policy_flag", tags=", ".join(sorted(minor)))
    return None


# Pony 等模型的特殊標籤必須保留底線
PROTECTED_RE = re.compile(r"^(score_\d+(_up)?|source_[a-z0-9]+|rating_[a-z]+)$", re.IGNORECASE)


def canon(tag: str) -> str:
    """把任意格式的標籤轉為正規形式（空格、不跳脫）。"""
    t = tag.strip().replace("\\(", "(").replace("\\)", ")")
    if t in KAOMOJIS or PROTECTED_RE.match(t):
        return t
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip()


def split_tags(text: str | Iterable[str] | None) -> list[str]:
    if text is None:
        return []
    if isinstance(text, str):
        items = text.replace("\n", ",").split(",")
    else:
        items = list(text)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        t = canon(str(item))
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def format_tag(tag: str, underscore_to_space: bool, escape_parentheses: bool) -> str:
    t = canon(tag)
    if t not in KAOMOJIS and not PROTECTED_RE.match(t) and not underscore_to_space:
        t = t.replace(" ", "_")
    if escape_parentheses:
        t = t.replace("(", "\\(").replace(")", "\\)")
    return t


def matches_prune(tag: str, groups: Iterable[str]) -> bool:
    key = canon(tag).lower()
    return any(_PRUNE_RE[g].match(key) for g in groups if g in _PRUNE_RE)


def _blacklist_matchers(blacklist: str | Iterable[str]) -> list[str]:
    return [b.lower() for b in split_tags(blacklist)]


def matches_blacklist(tag: str, patterns: list[str]) -> bool:
    key = canon(tag).lower()
    return any(fnmatch.fnmatchcase(key, p) for p in patterns)


def vlm_tag_groups(raw: dict[str, Any] | None, settings: dict[str, Any]) -> dict[str, list[str]]:
    """依 vlm_tags 模式取出要採用的 VLM 標籤（畫師已依底模加上前綴，例如 Anima 的 @）。"""
    mode = settings.get("vlm_tags", "off")
    vlm = (raw or {}).get("vlm") or {}
    if mode == "off" or not vlm:
        return {"character": [], "copyright": [], "artist": [], "general": []}
    prefix = get_profile(settings["profile"]).get("artist_prefix", "")
    vlm_chars = [canon(t) for t in vlm.get("character", [])]
    # VLM 常編出不存在的角色名與作品名（實測 JoyCaption 會把香風智乃判成別的角色）。
    # WD14 已認出角色時，只有在 VLM 的角色與 WD14 一致時才採用 VLM 的角色 / 作品；「只用 VLM」模式不檢查。
    cthr = float(settings.get("character_threshold", 0.85))
    wd14_chars = {canon(n).lower() for n, s in (raw or {}).get("character", []) if s >= cthr}
    trusted = mode == "only" or not wd14_chars or any(c.lower() in wd14_chars for c in vlm_chars)
    return {
        "character": vlm_chars if trusted and settings.get("include_character_tags", True) else [],
        "copyright": [canon(t) for t in vlm.get("copyright", [])] if trusted else [],
        "artist": [prefix + canon(t) for t in vlm.get("artist", [])] if settings.get("vlm_artist_tags") else [],
        "general": [canon(t) for t in vlm.get("general", [])] if mode in ("merge", "only") else [],
    }


def select_tags(raw: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    """由 WD14 原始預測、VLM 標籤與專案設定，產生排序好的標籤列表（正規形式）。

    raw = {"rating": {...}, "general": [[name, score], ...], "character": [[name, score], ...],
           "vlm": {"character": [...], "copyright": [...], "artist": [...], "general": [...]}}
    順序：人數 → 角色 → 作品 → 畫師 → 一般標籤（WD14 依分數，接著是 VLM 多出來的）。
    vlm_tags 為 only 時不採用 WD14 的標籤（WD14 仍可提供分級）。
    """
    gthr = float(settings["general_threshold"])
    cthr = float(settings["character_threshold"])
    prune = settings.get("prune_groups") or []
    black = _blacklist_matchers(settings.get("blacklist") or "")
    trigger = canon(settings.get("trigger") or "").lower()

    def keep(name: str) -> bool:
        t = canon(name)
        low = t.lower()
        if not t or low == trigger:
            return False
        if matches_prune(t, prune) or matches_blacklist(t, black):
            return False
        return True

    use_wd14_tags = settings.get("vlm_tags", "off") != "only"
    general = [t for t, _ in sorted(((canon(n), s) for n, s in raw.get("general", []) if s >= gthr),
                                    key=lambda x: -x[1])] if use_wd14_tags else []
    chars: list[str] = []
    if use_wd14_tags and settings.get("include_character_tags", True):
        chars = [canon(n) for n, s in sorted(raw.get("character", []), key=lambda x: -x[1]) if s >= cthr]
    vlm = vlm_tag_groups(raw, settings)
    general += vlm["general"]
    counts = [t for t in general if COUNT_TAG_RE.match(t.lower())]
    others = [t for t in general if not COUNT_TAG_RE.match(t.lower())]

    # 先去重再截斷：WD14 與 VLM 常有相同的標籤，不能佔用 max_tags 的名額
    ordered = split_tags([t for t in counts + chars + vlm["character"] + vlm["copyright"] + vlm["artist"] + others
                          if keep(t)])
    limit = int(settings.get("max_tags") or 0)
    return ordered[:limit] if limit > 0 else ordered


def apply_filters(tags: list[str], settings: dict[str, Any]) -> list[str]:
    """對既有（可能已人工編輯）的標籤套用修剪 + 黑名單，用於批次操作。"""
    prune = settings.get("prune_groups") or []
    black = _blacklist_matchers(settings.get("blacklist") or "")
    return [t for t in tags if not matches_prune(t, prune) and not matches_blacklist(t, black)]


def top_rating(raw: dict[str, Any] | None) -> str | None:
    if not raw or not raw.get("rating"):
        return None
    return max(raw["rating"].items(), key=lambda kv: kv[1])[0]


def build_caption(
    settings: dict[str, Any],
    tags: list[str],
    nl_caption: str = "",
    rating: str | None = None,
    character_tags: Iterable[str] = (),
    extra_tags: Iterable[str] = (),
) -> str:
    """依專案設定組合最終訓練 caption（寫入 .txt 的內容）。extra_tags 加在最後（例如白色色塊關鍵字）。"""
    profile = get_profile(settings["profile"])
    mode = settings.get("caption_mode", "tags")
    u2s = bool(settings.get("underscore_to_space", True))
    esc = bool(settings.get("escape_parentheses", False))
    fmt = lambda t: format_tag(t, u2s, esc)  # noqa: E731

    trigger = (settings.get("trigger") or "").strip()
    class_word = (settings.get("class_word") or "").strip()

    extra = [fmt(t) for t in extra_tags]
    if mode == "trigger_only":
        return ", ".join(x for x in (trigger, class_word, *extra) if x) or ", ".join(fmt(t) for t in tags)

    prefix = [fmt(t) for t in split_tags(settings.get("prefix_tags"))]
    append = [fmt(t) for t in split_tags(settings.get("append_tags"))] + extra
    quality = [fmt(t) for t in split_tags(settings.get("quality_tags"))] if settings.get("add_quality_tags") else []

    rating_tag = None
    rmap = profile.get("rating_map")
    if settings.get("include_rating") and rmap and rating in rmap:
        rating_tag = rmap[rating]

    body = [fmt(t) for t in tags]
    if rating_tag:
        if profile.get("rating_position") == "prefix":
            # Pony：source_*, rating_*；Anima：safe / nsfw … 緊接在 trigger 與前綴標籤後面
            prefix = prefix + [rating_tag]
        else:
            # Animagine / NoobAI：放在「人數 + 角色」之後
            char_set = {canon(c).lower() for c in character_tags}
            i = 0
            while i < len(tags) and (COUNT_TAG_RE.match(canon(tags[i]).lower()) or canon(tags[i]).lower() in char_set):
                i += 1
            body = body[:i] + [rating_tag] + body[i:]

    nl = re.sub(r"\s+", " ", (nl_caption or "")).strip()
    if trigger and nl:
        # VLM 可能改變 trigger 大小寫（T5 會區分大小寫），統一還原
        nl = re.sub(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", lambda _: trigger, nl, flags=re.IGNORECASE)
    if mode == "hybrid" and body:
        nl = nl.rstrip(".")
    head: list[str] = []
    # trigger 一律放最前面（keep_tokens 依賴這個位置）；只有純自然語句已經以 trigger 開頭時才不重複
    if trigger and not (mode == "natural" and nl and re.match(re.escape(trigger) + r"(?!\w)", nl, re.IGNORECASE)):
        head.append(trigger)

    if mode == "natural":
        parts = head + prefix + quality + ([nl] if nl else body) + append
    elif mode == "hybrid":
        if settings.get("nl_position") == "after_tags":
            parts = head + prefix + quality + body + ([nl] if nl else []) + append
        else:
            parts = head + prefix + quality + ([nl] if nl else []) + body + append
    else:  # tags
        parts = head + prefix + quality + body + append

    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p.strip())
    return ", ".join(out)


def count_tokens_estimate(caption: str) -> int:
    """粗估 token 數（~1 token / 英文單字或標點），用於提示是否超過文字編碼器上限（CLIP 75）。"""
    return len(re.findall(r"\w+|[^\w\s]", caption))


def training_hints(settings: dict[str, Any]) -> dict[str, Any]:
    """依 caption 結構推算 kohya 的 shuffle_caption / keep_tokens 建議值。"""
    mode = settings.get("caption_mode", "tags")
    profile = get_profile(settings["profile"])
    if mode in ("natural", "trigger_only") or not profile.get("shuffle_caption", True):
        return {"shuffle_caption": False, "keep_tokens": 0}
    keep = 1 if (settings.get("trigger") or "").strip() else 0
    keep += len(split_tags(settings.get("prefix_tags")))
    if settings.get("add_quality_tags"):
        keep += len(split_tags(settings.get("quality_tags")))
    if settings.get("include_rating") and profile.get("rating_map") and profile.get("rating_position") == "prefix":
        keep += 1
    if mode == "hybrid" and settings.get("nl_position", "before_tags") == "before_tags":
        keep += 1  # 自然語句視為一個固定片段
    return {"shuffle_caption": True, "keep_tokens": keep}
