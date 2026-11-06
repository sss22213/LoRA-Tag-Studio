"""用已登入的 Claude Code CLI（`claude`）或 OpenAI Codex CLI（`codex`）看圖寫字：產生或修正 caption。

用的是使用者自己的訂閱帳號，不需要 API 金鑰。每次呼叫開一個子行程：
- claude：`claude -p --input-format stream-json --output-format stream-json`，圖片以 base64 放進訊息。
  `--tools ""` 關掉所有工具、`--safe-mode` 不載入使用者的 hooks / MCP / 外掛 / CLAUDE.md。
- codex：`codex exec --json`，圖片寫成暫存檔用 `-i` 傳入。唯讀沙箱、關掉內建工具，
  `--ignore-user-config` 不載入使用者的 config.toml（MCP 伺服器等），登入資訊仍用 CODEX_HOME。
兩者都只讀得到圖片和提示，碰不到 app 的檔案。

容器裡把主機的 ~/.claude、~/.codex 掛到 CLI_HOME 底下。app 以 root 執行時，CLI 改用這兩個資料夾擁有者的
身分執行，更新登入權杖時檔案仍屬於主機的使用者（不會變成 root、讓主機上的 CLI 讀不到）。
"""
from __future__ import annotations

import base64
import glob
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import settings
from ..i18n import t

log = logging.getLogger(__name__)

ENGINES = ("claude", "codex")
_CODEX_TOOLS_OFF = ("image_generation", "shell_tool", "unified_exec", "browser_use", "computer_use", "apps", "plugins",
                    "view_image", "web_search_request")
_USAGE_GRACE = 5.0  # codex 回答後再等這麼久收尾（它有時會接著想跑指令），之後直接結束


class CLIError(RuntimeError):
    pass


def _home() -> Path:
    return Path(settings.cli_home) if settings.cli_home else Path.home()


def _creds_dir(engine: str) -> Path:
    return _home() / (".claude" if engine == "claude" else ".codex")


def _version_key(path: str) -> list[int]:
    m = re.search(r"releases/(\d+(?:\.\d+)*)", path)
    return [int(x) for x in m.group(1).split(".")] if m else [0]


def _bin(engine: str) -> str | None:
    if engine == "claude":
        b = settings.claude_bin
        return shutil.which(b) or (b if os.path.isfile(b) else None)
    if settings.codex_bin and (shutil.which(settings.codex_bin) or os.path.isfile(settings.codex_bin)):
        return settings.codex_bin
    found = shutil.which("codex")
    if found:
        return found
    # 掛進容器的 ~/.codex 裡有 static 版本的 codex（依版本號取最新）
    cands = glob.glob(str(_creds_dir("codex") / "packages/standalone/releases/*/bin/codex"))
    return max(cands, key=_version_key) if cands else None


def _logged_in(engine: str) -> bool:
    d = _creds_dir(engine)
    return (d / (".credentials.json" if engine == "claude" else "auth.json")).is_file()


def status() -> dict[str, Any]:
    """{claude: {available, reason, models}, codex: {...}}。reason：not_installed / not_logged_in / None。"""
    out = {}
    for e in ENGINES:
        reason = None if _bin(e) else "not_installed"
        if reason is None and not _logged_in(e):
            reason = "not_logged_in"
        out[e] = {"available": reason is None, "reason": reason, "models": models(e)}
    return out


def models(engine: str) -> list[dict[str, str]]:
    """可選的模型：claude 用 CLAUDE_MODELS（別名，CLI 會對應到最新版）；codex 依帳號讀 Codex 的模型清單。"""
    if engine == "claude":
        return [{"name": m, "label": m} for m in settings.claude_models]
    if settings.codex_models:
        return [{"name": m, "label": m} for m in settings.codex_models]
    try:
        data = json.loads((_creds_dir("codex") / "models_cache.json").read_text())
    except (OSError, ValueError):
        return []
    return [{"name": m["slug"], "label": m.get("display_name") or m["slug"]} for m in data.get("models", [])
            if m.get("slug") and m.get("visibility") != "hide" and "review" not in m["slug"]]


def default_model(engine: str) -> str:
    ms = models(engine)
    return ms[0]["name"] if ms else ""


def _run_as() -> tuple[dict[str, Any], dict[str, str]]:
    """子行程的身分與環境。以 root 執行、而登入資料夾屬於別人時，改用那個使用者的身分。"""
    env = {**os.environ, "HOME": str(_home()), "CODEX_HOME": str(_creds_dir("codex")),
           "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    kw: dict[str, Any] = {}
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        for e in ENGINES:
            d = _creds_dir(e)
            if d.is_dir() and d.stat().st_uid != 0:
                kw = {"user": d.stat().st_uid, "group": d.stat().st_gid}
                break
    return kw, env


def _jpeg(image: Image.Image) -> bytes:
    img = image.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.thumbnail((settings.vlm_max_side, settings.vlm_max_side), Image.LANCZOS)
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _check(engine: str) -> str:
    b = _bin(engine)
    if not b:
        raise CLIError(t("msg.cli_not_installed", engine=engine))
    if not _logged_in(engine):
        raise CLIError(t("msg.cli_not_logged_in", engine=engine))
    return b


def ask(engine: str, model: str, system: str, prompt: str, image: Image.Image | None = None) -> str:
    """問一次（可附一張圖），回傳模型的文字回答。"""
    if engine not in ENGINES:
        raise CLIError(t("msg.cli_bad_engine", engines=", ".join(ENGINES)))
    model = model or default_model(engine)
    return (_ask_claude if engine == "claude" else _ask_codex)(_check(engine), model, system, prompt, image)


def _ask_claude(binary: str, model: str, system: str, prompt: str, image: Image.Image | None) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(_jpeg(image)).decode()}})
    payload = json.dumps({"type": "user", "message": {"role": "user", "content": content}}, ensure_ascii=False)
    args = [binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--model", model, "--tools", "", "--safe-mode", "--no-session-persistence", "--system-prompt", system]
    if settings.cli_effort and "haiku" not in model:  # Haiku 不接受 --effort
        args += ["--effort", settings.cli_effort]
    kw, env = _run_as()
    try:
        p = subprocess.run(args, input=payload.encode(), capture_output=True, timeout=settings.cli_timeout,
                           env=env, cwd=tempfile.gettempdir(), **kw)
    except subprocess.TimeoutExpired as e:
        raise CLIError(t("msg.cli_timeout", engine="claude", s=settings.cli_timeout)) from e
    except OSError as e:
        raise CLIError(t("msg.cli_failed", engine="claude", error=e)) from e
    result, texts = None, []
    for line in p.stdout.decode(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            result = ev
        elif ev.get("type") == "assistant":
            texts += [b.get("text") or "" for b in (ev.get("message") or {}).get("content") or [] if b.get("type") == "text"]
    if result is not None and result.get("is_error"):
        raise CLIError(t("msg.cli_failed", engine="claude", error=str(result.get("result") or result.get("subtype"))[:300]))
    if result is not None and isinstance(result.get("result"), str):
        return result["result"].strip()
    if texts:
        return "".join(texts).strip()
    err = p.stderr.decode(errors="replace").strip()
    raise CLIError(t("msg.cli_failed", engine="claude", error=(err or f"exit {p.returncode}")[:300]))


def _ask_codex(binary: str, model: str, system: str, prompt: str, image: Image.Image | None) -> str:
    kw, env = _run_as()
    work = tempfile.mkdtemp(prefix="lts-codex-")
    try:
        os.chmod(work, 0o755)  # 以別的使用者身分執行時要讀得到圖片
        args = [binary, "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                "--ignore-rules", "-s", "read-only", "-m", model, "-C", work,
                "-c", "model_reasoning_summary=none"]
        if settings.cli_effort:
            args += ["-c", f"model_reasoning_effort={settings.cli_effort}"]
        for f in _CODEX_TOOLS_OFF:
            args += ["-c", f"features.{f}=false"]
        if image is not None:
            path = os.path.join(work, "image.jpg")
            with open(path, "wb") as fh:
                fh.write(_jpeg(image))
            os.chmod(path, 0o644)
            args += ["-i", path]
        args.append("-")  # 提示從 stdin 讀
        errfile = open(os.path.join(work, "stderr.txt"), "w+b")  # 不用管線：stderr 寫滿時會卡住 codex
        try:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errfile,
                                    env=env, cwd=work, **kw)
        except OSError as e:
            errfile.close()
            raise CLIError(t("msg.cli_failed", engine="codex", error=e)) from e
        try:
            proc.stdin.write(f"{system}\n\n{prompt}".encode())
            proc.stdin.close()
            return _read_codex(proc, errfile)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            errfile.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _read_codex(proc: subprocess.Popen, errfile: Any) -> str:
    import selectors

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    deadline, answer, buf = time.time() + settings.cli_timeout, None, b""
    while True:
        left = deadline - time.time()
        if left <= 0:
            if answer is not None:
                return answer
            raise CLIError(t("msg.cli_timeout", engine="codex", s=settings.cli_timeout))
        if not sel.select(timeout=min(left, 1.0)):
            continue
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            kind = ev.get("type")
            item = ev.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message" and item.get("text"):
                answer = item["text"].strip()
                deadline = min(deadline, time.time() + _USAGE_GRACE)
            elif kind == "turn.completed" and answer is not None:
                return answer
            elif kind in ("turn.failed", "error", "thread.error"):
                if answer is not None:
                    return answer
                err = ev.get("error") or ev.get("message") or "codex error"
                if isinstance(err, dict):
                    err = err.get("message") or json.dumps(err, ensure_ascii=False)
                raise CLIError(t("msg.cli_failed", engine="codex", error=str(err)[:300]))
    if answer is not None:
        return answer
    proc.wait()
    errfile.seek(0)
    err = errfile.read().decode(errors="replace").strip()
    raise CLIError(t("msg.cli_failed", engine="codex", error=(err or f"exit {proc.poll()}")[:300]))
