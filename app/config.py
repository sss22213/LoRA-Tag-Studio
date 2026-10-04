"""環境變數設定。所有設定都可在 .env / docker-compose.yml 中覆寫。"""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        self.data_dir = Path(_env("DATA_DIR", "/data"))
        self.import_dir = Path(_env("IMPORT_DIR", "/import"))
        self.web_dir = Path(_env("WEB_DIR", str(Path(__file__).resolve().parent.parent / "web")))

        # 未指定語言時（MCP、未帶語言的 API 客戶端）使用的語言：zh-TW | en | ja | ko | zh-CN
        self.default_lang = _env("DEFAULT_LANG", "en")
        # 若設定 API_KEY，所有 /api 與 /mcp 請求都需帶上金鑰
        self.api_key = _env("API_KEY")
        # 給 LLM / MCP 回傳下載連結時使用的對外網址，例如 http://192.168.1.10:7870
        self.public_base_url = _env("PUBLIC_BASE_URL").rstrip("/")

        # WD14 (ONNX) 標註模型
        self.wd14_default_model = _env("WD14_MODEL", "SmilingWolf/wd-eva02-large-tagger-v3")
        self.ort_device = _env("ORT_DEVICE", "auto").lower()  # auto | cpu | cuda

        # waifu2x（nunif 的 ONNX 模型）放大低解析圖片。第一次使用時只從 zip 裡下載需要的模型檔
        self.waifu2x_dir = Path(_env("WAIFU2X_DIR", "/models/waifu2x"))
        self.waifu2x_models_url = _env(
            "WAIFU2X_MODELS_URL",
            "https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_onnx_models_20250502.zip")

        # 自然語言 caption 用的視覺語言模型 (VLM)
        # openai    : 任何 OpenAI 相容端點（Ollama / vLLM / LM Studio / OpenRouter / OpenAI）
        # anthropic : Claude API（需要 ANTHROPIC_API_KEY）
        # none      : 停用
        self.vlm_backend = _env("VLM_BACKEND", "openai").lower()
        self.vlm_base_url = _env("VLM_BASE_URL", "http://ollama:11434/v1").rstrip("/")
        self.vlm_model = _env("VLM_MODEL", "qwen2.5vl:7b")
        self.vlm_api_key = _env("VLM_API_KEY", "ollama")
        self.vlm_timeout = _env_int("VLM_TIMEOUT", 180)
        self.vlm_max_side = _env_int("VLM_MAX_SIDE", 1024)
        self.anthropic_model = _env("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.anthropic_effort = _env("ANTHROPIC_EFFORT", "low")

        # 同時處理幾張圖（VLM 走網路時可調高）
        self.tag_concurrency = max(1, _env_int("TAG_CONCURRENCY", 2))
        self.max_upload_mb = _env_int("MAX_UPLOAD_MB", 2048)
        # 允許「從網址匯入」存取內網位址（預設關閉以避免 SSRF）
        self.allow_private_urls = _env("ALLOW_PRIVATE_URLS", "0").lower() in ("1", "true", "yes")

        # Civitai 雲端訓練（Orchestration API）。金鑰在 civitai.com 帳號設定建立，只留在伺服器端
        self.civitai_api_key = _env("CIVITAI_API_KEY")
        self.civitai_orchestration_url = _env("CIVITAI_ORCHESTRATION_URL", "https://orchestration.civitai.com").rstrip("/")
        self.civitai_site_url = _env("CIVITAI_SITE_URL", "https://civitai.com").rstrip("/")

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.projects_dir, self.exports_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
