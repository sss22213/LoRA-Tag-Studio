# JoyCaption 用的 vLLM 映像檔。
# vllm/vllm-openai:v0.30.0 內附 transformers 5.17，但 0.30.0 的 Llava / Pixtral 載入器仍匯入 5.17 已改名的
# PixtralRotaryEmbedding，模型在檢查架構時就失敗（vllm-project/vllm#58755，main 已修正）。
# 這裡改裝 0.30.0 測試時使用的 transformers 5.16.1；新版 vLLM 修正後可用 JOYCAPTION_VLLM_IMAGE 換掉。
ARG VLLM_IMAGE=vllm/vllm-openai:v0.30.0
FROM ${VLLM_IMAGE}

ARG TRANSFORMERS_VERSION=5.16.1
RUN uv pip install --system --no-cache "transformers==${TRANSFORMERS_VERSION}" \
 && python3 -c "import vllm.model_executor.models.pixtral"
