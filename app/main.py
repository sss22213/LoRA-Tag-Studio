"""LoRA Tag Studio — FastAPI 進入點。"""
from __future__ import annotations

import contextlib
import hmac
import json
import logging
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings

from . import civitai, db, i18n, jobs, services
from .api import router
from .config import settings
from .mcp_server import mcp
from .profiles import list_profiles

__version__ = "1.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("lora-tag-studio")

# MCP：無狀態 Streamable HTTP，方便多個 LLM 客戶端同時連線。
# 服務通常透過區網 IP 存取，因此關閉 SDK 預設只允許 localhost 的 DNS rebinding 保護，改以 API_KEY 控管。
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    db.connect()
    jobs.start_worker()
    log.info("資料目錄：%s；WD14 預設模型：%s；VLM：%s", settings.data_dir, settings.wd14_default_model,
             settings.vlm_backend)
    langs = list(i18n.available())
    if not langs:
        log.error("找不到任何語系檔（%s/*.json），WebUI 會顯示鍵名；請重新 build 映像檔", i18n.LOCALES_DIR)
    else:
        log.info("語系：%s；預設：%s", ", ".join(langs), i18n.default_lang())
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="LoRA Tag Studio API",
    version=__version__,
    description=(
        "Upload images → auto-tag (WD14 + optional VLM) → edit → export a Civitai-ready LoRA training dataset. "
        "Captions follow the conventions of the selected base model (SD1.5, SDXL, Pony, Illustrious, NoobAI, "
        "Animagine, Anima, Flux). MCP endpoint: /mcp. LLM guide: /llms.txt"
    ),
    lifespan=lifespan,
)


@app.exception_handler(services.NotFound)
async def _not_found(_: Request, exc: services.NotFound) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(civitai.CivitaiError)
async def _civitai_error(_: Request, exc: civitai.CivitaiError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(services.BadRequest)
async def _bad_request(_: Request, exc: services.BadRequest) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.include_router(router)


@app.get("/llms.txt", include_in_schema=False, response_class=PlainTextResponse)
def llms_txt() -> str:
    base = settings.public_base_url or "http://<host>:7870"
    with i18n.use_lang("en"):
        profiles = "\n".join(f"- `{p['key']}`: {p['name']} — {p['summary']}" for p in list_profiles())
    auth = "Send `Authorization: Bearer <API_KEY>` (or `X-API-Key`) on every request.\n" if settings.api_key else ""
    return f"""# LoRA Tag Studio

> Turns image folders into LoRA training datasets (images + .txt captions) for the Civitai trainer,
> with captions formatted for the chosen base model.

{auth}
## Interfaces
- OpenAPI spec: {base}/openapi.json (Swagger UI: {base}/docs) — usable as OpenAI/Open WebUI tool server
- MCP (Streamable HTTP): {base}/mcp

## Base model profiles
{profiles}

## Workflow (REST)
1. `POST /api/projects` {{"name", "profile", "lora_type": "character|style|concept", "trigger"}}
2. `POST /api/projects/{{id}}/upload` (multipart files, zip allowed) or `POST /api/projects/{{id}}/import-urls` {{"urls": [...]}}
3. `POST /api/projects/{{id}}/tag` {{"only_untagged": true}} → `GET /api/jobs/{{job_id}}` until status=done
4. Review: `GET /api/projects/{{id}}/captions`, `GET /api/projects/{{id}}/stats`
5. Fix: `POST /api/projects/{{id}}/bulk` {{"action": "remove", "tags": ["watermark"]}} or
   `PATCH /api/projects/{{id}}/images/{{image_id}}` {{"tags": [...], "nl_caption": "..."}}
6. Export: `POST /api/projects/{{id}}/export` {{"format": "civitai"}} → download_url
- Guide per base model: `GET /api/profiles/{{profile}}/guide.md?lang=en` (languages: {", ".join(i18n.available())})
- One-off: `POST /api/quick-tag/json` {{"image_url": "...", "profile": "pony_v6", "trigger": "mychar"}}
"""


# 讓 /mcp（無結尾斜線）也能直接使用，避免 307 轉址造成部分 MCP 客戶端失敗
class _MCPPathFix:
    def __init__(self, app_):
        self.app = app_

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/mcp":
            scope = dict(scope, path="/mcp/", raw_path=b"/mcp/")
        await self.app(scope, receive, send)


def _headers(scope) -> dict[str, str]:
    return {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}


def _cookies(headers: dict[str, str]) -> dict[str, str]:
    c = SimpleCookie()
    try:
        c.load(headers.get("cookie", ""))
    except Exception:  # noqa: BLE001
        return {}
    return {k: m.value for k, m in c.items()}


class _LangMiddleware:
    """決定這個請求的語言：?lang= → X-Lang → cookie lts_lang → Accept-Language → DEFAULT_LANG。"""

    def __init__(self, app_):
        self.app = app_

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = _headers(scope)
        qs = parse_qs(scope.get("query_string", b"").decode())
        lang = (i18n.normalize((qs.get("lang") or [""])[0])
                or i18n.normalize(headers.get("x-lang"))
                or i18n.normalize(_cookies(headers).get("lts_lang"))
                or i18n.from_accept_language(headers.get("accept-language")))
        # MCP 與未指定語言的 API 客戶端使用 DEFAULT_LANG（瀏覽器會帶 Accept-Language）
        token = i18n.set_lang(lang)
        try:
            await self.app(scope, receive, send)
        finally:
            i18n.reset_lang(token)


class _AuthMiddleware:
    """API_KEY 驗證（純 ASGI，不影響串流回應）。"""

    PUBLIC = ("/api/health", "/api/i18n")

    def __init__(self, app_):
        self.app = app_

    @staticmethod
    def _extract(scope) -> str:
        headers = _headers(scope)
        if headers.get("x-api-key"):
            return headers["x-api-key"]
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        cookie_key = _cookies(headers).get("lts_key")
        if cookie_key:
            return cookie_key
        qs = parse_qs(scope.get("query_string", b"").decode())
        return (qs.get("api_key") or [""])[0]

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and settings.api_key:
            path = scope["path"]
            public = path in self.PUBLIC or path.startswith("/api/i18n/")
            protected = (path.startswith("/api/") and not public) or path.startswith("/mcp")
            if protected and not hmac.compare_digest(self._extract(scope).encode(), settings.api_key.encode()):
                body = json.dumps({"detail": i18n.t("msg.api_key_required")}, ensure_ascii=False).encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


app.mount("/mcp", mcp_app)
app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
app.add_middleware(_AuthMiddleware)
app.add_middleware(_LangMiddleware)
app.add_middleware(_MCPPathFix)
