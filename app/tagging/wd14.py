"""WD14 (SmilingWolf wd-*-tagger-v3) ONNX 標註器。"""
from __future__ import annotations

import csv
import logging
import threading
from typing import Any

import numpy as np
from PIL import Image

from ..config import settings

log = logging.getLogger(__name__)

# 說明文字在語系檔 server.wd14_models.<repo 名稱>
WD14_MODELS: tuple[str, ...] = (
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
)


def wd14_models_view() -> dict[str, str]:
    from ..i18n import tr

    return {repo: tr(f"wd14_models.{repo.split('/')[-1]}", default=repo) for repo in WD14_MODELS}

# 原始預測只保留分數高於此值的標籤，之後調整門檻時不必重新推論
RAW_FLOOR = 0.15

_lock = threading.Lock()
_taggers: dict[str, "WD14Tagger"] = {}
_dlls_preloaded = False


def _providers() -> list[str]:
    import onnxruntime as ort

    global _dlls_preloaded
    if settings.ort_device == "cpu":
        return ["CPUExecutionProvider"]
    if not _dlls_preloaded and hasattr(ort, "preload_dlls"):
        # onnxruntime-gpu[cuda,cudnn] 透過 pip 安裝的 CUDA/cuDNN 需要先載入
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            log.warning("preload_dlls 失敗：%s", e)
        _dlls_preloaded = True
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if settings.ort_device == "cuda":
        log.warning("ORT_DEVICE=cuda 但 onnxruntime 沒有 CUDA provider，改用 CPU")
    return ["CPUExecutionProvider"]


class WD14Tagger:
    def __init__(self, repo_id: str) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.repo_id = repo_id
        log.info("下載 / 載入 WD14 模型 %s ...", repo_id)
        model_path = hf_hub_download(repo_id, "model.onnx")
        csv_path = hf_hub_download(repo_id, "selected_tags.csv")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=_providers())
        self.providers = self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[1])  # NHWC
        self.output_name = self.session.get_outputs()[0].name

        self.names: list[str] = []
        self.rating_idx: list[int] = []
        self.general_idx: list[int] = []
        self.character_idx: list[int] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                self.names.append(row["name"])
                cat = int(row["category"])
                if cat == 9:
                    self.rating_idx.append(i)
                elif cat == 4:
                    self.character_idx.append(i)
                else:
                    self.general_idx.append(i)
        log.info("WD14 %s 就緒（providers=%s, input=%d）", repo_id, self.providers, self.size)

    def _prepare(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
        w, h = image.size
        side = max(w, h)
        padded = Image.new("RGB", (side, side), (255, 255, 255))
        padded.paste(image, ((side - w) // 2, (side - h) // 2))
        if side != self.size:
            padded = padded.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(padded, dtype=np.float32)[:, :, ::-1]  # RGB → BGR
        return np.ascontiguousarray(arr[np.newaxis, ...])

    def predict(self, image: Image.Image) -> dict[str, Any]:
        probs = self.session.run([self.output_name], {self.input_name: self._prepare(image)})[0][0]
        rating = {self.names[i]: round(float(probs[i]), 4) for i in self.rating_idx}
        general = [[self.names[i], round(float(probs[i]), 4)] for i in self.general_idx if probs[i] >= RAW_FLOOR]
        character = [[self.names[i], round(float(probs[i]), 4)] for i in self.character_idx if probs[i] >= RAW_FLOOR]
        general.sort(key=lambda x: -x[1])
        character.sort(key=lambda x: -x[1])
        return {"model": self.repo_id, "rating": rating, "general": general, "character": character}


def get_tagger(repo_id: str | None = None) -> WD14Tagger:
    repo_id = repo_id or settings.wd14_default_model
    with _lock:
        tagger = _taggers.get(repo_id)
        if tagger is None:
            tagger = WD14Tagger(repo_id)
            _taggers[repo_id] = tagger
        return tagger


def loaded_models() -> list[dict[str, Any]]:
    return [{"repo_id": k, "providers": v.providers} for k, v in _taggers.items()]
