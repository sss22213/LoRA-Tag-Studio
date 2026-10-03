"""多國語言：語系檔載入、語言協商與翻譯。

語系檔位於 app/locales/<code>.json，結構：
    {"_meta": {...}, "server": {...後端文字...}, "ui": {...WebUI 文字...}}
缺少的鍵會退回英文（en.json）。新增語言只要放一個新的 JSON 檔。
"""
from __future__ import annotations

import contextlib
import json
import re
import threading
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from .config import settings

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
FALLBACK = "en"

_current: ContextVar[str | None] = ContextVar("lang", default=None)
_lock = threading.Lock()
_raw: dict[str, dict[str, Any]] = {}
_merged: dict[str, dict[str, Any]] = {}


def _load_raw() -> dict[str, dict[str, Any]]:
    with _lock:
        if not _raw:
            for f in sorted(LOCALES_DIR.glob("*.json")):
                _raw[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        return _raw


def _deep_merge(base: Any, over: Any) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _deep_merge(base.get(k), v) if k in base else deepcopy(v)
        return out
    return deepcopy(over) if over is not None else deepcopy(base)


def available() -> dict[str, dict[str, Any]]:
    """{code: _meta}，依設定的順序。"""
    raw = _load_raw()
    return {code: data.get("_meta", {"code": code, "name": code}) for code, data in
            sorted(raw.items(), key=lambda kv: kv[1].get("_meta", {}).get("order", 99))}


def catalog(lang: str | None = None) -> dict[str, Any]:
    lang = lang or get_lang()
    raw = _load_raw()
    if lang not in raw:
        lang = FALLBACK if FALLBACK in raw else next(iter(raw), None)
    if lang is None:  # 沒有任何語系檔：回傳空目錄，t() 會直接回傳鍵名
        return {}
    with _lock:
        if lang not in _merged:
            _merged[lang] = _deep_merge(raw.get(FALLBACK, {}), raw[lang]) if lang != FALLBACK else raw[lang]
        return _merged[lang]


# ------------------------------------------------------------------ 語言協商
def normalize(code: str | None) -> str | None:
    """把 zh-Hant-TW / zh_CN / ja-JP / en-US 等轉成支援的語言代碼。"""
    if not code:
        return None
    c = code.strip().replace("_", "-").lower()
    codes = list(available())
    exact = {x.lower(): x for x in codes}
    if c in exact:
        return exact[c]
    if c.startswith("zh"):
        trad = any(t in c for t in ("hant", "-tw", "-hk", "-mo"))
        want = "zh-TW" if trad else "zh-CN"
        return want if want in codes else next((x for x in codes if x.startswith("zh")), None)
    base = c.split("-")[0]
    return next((x for x in codes if x.lower().split("-")[0] == base), None)


def from_accept_language(header: str | None) -> str | None:
    if not header:
        return None
    items = []
    for i, part in enumerate(header.split(",")):
        piece = part.strip().split(";")
        q = 1.0
        for p in piece[1:]:
            if p.strip().startswith("q="):
                try:
                    q = float(p.strip()[2:])
                except ValueError:
                    q = 0.0
        items.append((-q, i, piece[0]))
    for _, _, code in sorted(items):
        found = normalize(code)
        if found:
            return found
    return None


def default_lang() -> str:
    return normalize(settings.default_lang) or FALLBACK


def get_lang() -> str:
    return _current.get() or default_lang()


def set_lang(code: str | None):
    return _current.set(normalize(code) or default_lang())


def reset_lang(token) -> None:
    _current.reset(token)


@contextlib.contextmanager
def use_lang(code: str | None) -> Iterator[None]:
    token = set_lang(code)
    try:
        yield
    finally:
        _current.reset(token)


# ------------------------------------------------------------------ 翻譯
class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _lookup(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def tr(path: str, lang: str | None = None, default: Any = None) -> Any:
    """取得 server 區段中的任意值（字串、清單或物件）。"""
    value = _lookup(catalog(lang).get("server", {}), path)
    return deepcopy(value) if value is not None else default


def t(key: str, lang: str | None = None, **params: Any) -> str:
    """翻譯 server 區段的字串，{name} 以參數取代。"""
    text = tr(key, lang)
    if not isinstance(text, str):
        return key
    return text.format_map(_SafeDict({k: v for k, v in params.items()}))


def ui_catalog(lang: str) -> dict[str, Any]:
    data = catalog(lang)
    return {"lang": normalize(lang) or FALLBACK, "meta": data.get("_meta", {}), "ui": data.get("ui", {})}


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER.findall(text))
