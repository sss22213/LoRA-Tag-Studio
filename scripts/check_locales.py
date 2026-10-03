"""檢查 app/locales/*.json 是否完整。只用標準函式庫：python3 scripts/check_locales.py

- 每個語言與 en.json 的鍵、型別、清單長度、{佔位符} 是否一致
- app.js / index.html / 後端程式碼用到的鍵是否都存在
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOCALES = ROOT / "app" / "locales"
REF = "en"
PH = re.compile(r"\{(\w+)\}")

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def lookup(data: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


# ------------------------------------------------------------------ 語系檔之間的一致性
def compare(ref: Any, other: Any, path: str, lang: str) -> None:
    if isinstance(ref, dict):
        if not isinstance(other, dict):
            return err(f"[{lang}] {path}: 應為物件")
        for k in ref:
            if k not in other:
                err(f"[{lang}] 缺少 {path}.{k}".replace("[.", "["))
            else:
                compare(ref[k], other[k], f"{path}.{k}" if path else k, lang)
        for k in other.keys() - ref.keys():
            err(f"[{lang}] 多餘的鍵 {path}.{k}")
    elif isinstance(ref, list):
        if not isinstance(other, list):
            return err(f"[{lang}] {path}: 應為清單")
        if len(ref) != len(other):
            err(f"[{lang}] {path}: 清單長度 {len(other)}，en 為 {len(ref)}")
        for i, (a, b) in enumerate(zip(ref, other)):
            compare(a, b, f"{path}[{i}]", lang)
    elif isinstance(ref, str):
        if lang == REF and "lang" in PH.findall(ref):
            err(f"[{lang}] {path}: 佔位符不能叫 {{lang}}（會被當成 t() 的語言參數）")
        if not isinstance(other, str):
            return err(f"[{lang}] {path}: 應為字串")
        if not other.strip():
            err(f"[{lang}] {path}: 空字串")
        if set(PH.findall(ref)) != set(PH.findall(other)):
            err(f"[{lang}] {path}: 佔位符 {sorted(set(PH.findall(other)))}，en 為 {sorted(set(PH.findall(ref)))}")


def check_catalogs() -> dict[str, dict[str, Any]]:
    cats: dict[str, dict[str, Any]] = {}
    for f in sorted(LOCALES.glob("*.json")):
        try:
            cats[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            err(f"{f.name}: JSON 格式錯誤：{e}")
    if REF not in cats:
        err(f"找不到 {REF}.json")
        return cats
    orders: dict[Any, str] = {}
    for code, data in cats.items():
        meta = data.get("_meta", {})
        for field in ("code", "name", "html_lang", "order"):
            if field not in meta:
                err(f"[{code}] _meta 缺少 {field}")
        if meta.get("code") != code:
            err(f"[{code}] _meta.code 為 {meta.get('code')!r}，應與檔名相同")
        if meta.get("order") in orders:
            err(f"[{code}] _meta.order 與 {orders[meta['order']]} 重複")
        orders[meta.get("order")] = code
        for section in ("server", "ui"):  # en 與自己比對時只檢查空字串與保留的佔位符
            compare(cats[REF].get(section, {}), data.get(section, {}), section, code)
    return cats


# ------------------------------------------------------------------ 程式碼用到的鍵
def literal_names(path: Path, name: str) -> list[str]:
    """取出模組層級常數（tuple / list / dict 的字串鍵）。"""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == name and node.value is not None:
            v = node.value
            items = v.keys if isinstance(v, ast.Dict) else getattr(v, "elts", [])
            return [x.value for x in items if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    err(f"{path.relative_to(ROOT)}: 找不到常數 {name}")
    return []


def js_array(src: str, name: str) -> list[str]:
    m = re.search(rf"const {name} = \[(.*?)\];", src, re.S)
    if not m:
        err(f"app.js: 找不到 {name}")
        return []
    return re.findall(r"'([^']+)'", m.group(1))


def check_ui_keys(ui: dict[str, Any]) -> None:
    js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    keys = set(re.findall(r"\bth?\(\s*'([a-z0-9_.]+)'", js))
    keys |= set(re.findall(r"\blookup\(\s*'([a-z0-9_.]+)'", js))
    keys |= set(re.findall(r'data-i18n(?:-title)?="([a-z0-9_.]+)"', html))

    # 動態鍵：依 app.js 內的清單展開
    settings_src = re.search(r"const SETTINGS_SECTIONS = .*?\n\};", js, re.S)
    body = settings_src.group(0) if settings_src else ""
    keys |= {f"settings.f.{k}" for k in re.findall(r"\{ key: '(\w+)'", body)}
    keys |= {f"settings.sec.{k}" for k in re.findall(r"\{ id: '(\w+)'", body)}
    keys |= {f"settings.detail_{k}" for k in ("short", "medium", "detailed")}
    keys |= {f"rating.{k}" for k in js_array(js, "RATINGS")}
    keys |= {f"status.{k}" for k in js_array(js, "STATUSES")}
    keys |= {f"api.tools.{k}" for k in js_array(js, "MCP_TOOLS")}
    keys |= {f"projects.step{n}_{p}" for n in range(1, 5) for p in ("title", "desc")}
    keys |= {f"family.{k}" for k in re.findall(r'"family":\s*"(\w+)"', (ROOT / "app" / "profiles.py").read_text())}

    for key in sorted(keys):
        if not isinstance(lookup(ui, key), str):
            err(f"[ui] app.js / index.html 用到不存在的鍵 {key}")
    for tpl in re.findall(r"\bth?\(\s*`([a-z0-9_.]+)\$\{", js):
        # `settings.f.${k}` → settings.f；`projects.step${n}_title` → projects
        parent = tpl[:-1] if tpl.endswith(".") else tpl.rsplit(".", 1)[0]
        if not isinstance(lookup(ui, parent), dict):
            err(f"[ui] 動態鍵前綴 {tpl} 不存在")


def check_server_keys(server: dict[str, Any]) -> None:
    app = ROOT / "app"
    for f in sorted(app.rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        for key in re.findall(r"\bt\(\s*\"([a-z0-9_.]+)\"", src):
            if not isinstance(lookup(server, key), str):
                err(f"[server] {f.relative_to(ROOT)} 用到不存在的字串 {key}")
        for key in re.findall(r"\btr\(\s*\"([a-z0-9_.]+)\"", src):
            if lookup(server, key) is None:
                err(f"[server] {f.relative_to(ROOT)} 用到不存在的鍵 {key}")
        # r = lambda k, **kw: t(f"readme.{k}", ...) 這類包裝：檢查 r("xxx") 的呼叫
        for name, prefix in re.findall(r"(\w+) = lambda k, \*\*kw: t\(f\"([a-z0-9_]+)\.\{k\}\"", src):
            for key in re.findall(rf"\b{name}\(\s*['\"](\w+)['\"]", src):
                if not isinstance(lookup(server, f"{prefix}.{key}"), str):
                    err(f"[server] {f.relative_to(ROOT)} 用到不存在的字串 {prefix}.{key}")

    enums = {
        "wd14_models": [r.split("/")[-1] for r in literal_names(app / "tagging" / "wd14.py", "WD14_MODELS")],
        "export_formats": literal_names(app / "exporter.py", "EXPORT_FORMATS"),
        "bulk_actions": literal_names(app / "services.py", "BULK_ACTIONS"),
        "prune_groups": literal_names(app / "tagging" / "postprocess.py", "PRUNE_GROUPS"),
        "caption_modes": literal_names(app / "profiles.py", "CAPTION_MODES"),
        "lora_types": literal_names(app / "profiles.py", "LORA_TYPES"),
        "profiles": literal_names(app / "profiles.py", "PROFILES"),
    }
    for section, names in enums.items():
        for n in names:
            if lookup(server, f"{section}.{n}") is None:
                err(f"[server] 缺少 {section}.{n}")
        extra = set(server.get(section, {})) - set(names)
        if extra:
            err(f"[server] {section} 有程式碼未使用的鍵：{sorted(extra)}")


def main() -> int:
    cats = check_catalogs()
    if REF in cats:
        check_ui_keys(cats[REF].get("ui", {}))
        check_server_keys(cats[REF].get("server", {}))
    for e in errors:
        print("✗", e)
    langs = ", ".join(f"{c} ({d.get('_meta', {}).get('name', '?')})" for c, d in
                      sorted(cats.items(), key=lambda kv: kv[1].get("_meta", {}).get("order", 99)))
    print(f"{'✓' if not errors else '✗'} {len(cats)} 個語言：{langs}；{len(errors)} 個問題")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
