# syntax=docker/dockerfile:1.7
# Build from the repository root:
# docker buildx build --platform linux/arm64 -f runtime/vllm-0.27.1/Dockerfile .
ARG BASE=vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404@sha256:ae35bb2db70814d1239ea588e1abb5288adcbd287cac1d4d00ea0f28cd2033df
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

COPY runtime/vllm-0.27.1/payload/refusal_projection.py ${SITE}/refusal_projection.py
COPY runtime/vllm-0.27.1/payload/refusal_utils.py ${SITE}/v1/worker/gpu/refusal_utils.py
COPY runtime/vllm-0.27.1/payload/refusal_api_router.py ${SITE}/entrypoints/serve/refusal/api_router.py
COPY runtime/vllm-0.27.1/payload/refusal_init.py ${SITE}/entrypoints/serve/refusal/__init__.py
COPY runtime/vllm-0.27.1/patch_vllm_0271.py /opt/qwen38/patch_vllm_0271.py

RUN python3 /opt/qwen38/patch_vllm_0271.py --site ${SITE} \
    && python3 -m py_compile \
      ${SITE}/refusal_projection.py \
      ${SITE}/model_executor/models/qwen3_5.py \
      ${SITE}/v1/core/sched/output.py \
      ${SITE}/v1/core/kv_cache_utils.py \
      ${SITE}/v1/worker/gpu/model_runner.py \
      ${SITE}/v1/worker/gpu/refusal_utils.py \
      ${SITE}/v1/worker/gpu_worker.py \
      ${SITE}/entrypoints/serve/refusal/api_router.py \
      ${SITE}/entrypoints/serve/__init__.py \
    && python3 -c 'import importlib.metadata as m; assert m.version("vllm") == "0.27.1"'

COPY hf/refusal_dirs_qwen38.safetensors /opt/refusal/refusal_dirs_qwen38.safetensors
COPY runtime/vllm-0.27.1/payload/verify_qwen38_rank1.py /opt/qwen38/verify_qwen38_rank1.py

ENV VLLM_REFUSAL_LAMBDA_INIT=0.0

LABEL org.opencontainers.image.title="vllm-qwen38-rank1" \
      org.opencontainers.image.version="v0.27.1-rank1h" \
      org.opencontainers.image.source="https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection" \
      ai.e-dani.model="unsloth/Qwen3.8-27B-NVFP4" \
      ai.e-dani.arch="linux/arm64 (sm121)"
