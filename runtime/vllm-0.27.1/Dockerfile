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

# Blocks A, B and E of the per-request acceptance test run here, exactly as
# Dockerfile.rank1g did for the 0.25.2 lineage. Dropping this gate is how the
# 0.27.1 rebuild shipped a runtime whose per-request lambda never executed.
# Blocks C and D need a GPU and are skipped: this line does not certify the whole
# fix, only that the state machine and the role buffers are not broken at the base.
COPY tools/test_per_request.py /opt/refusal/test_per_request.py
RUN cd /tmp && VLLM_REFUSAL_DIRS=/opt/refusal/refusal_dirs_qwen38.safetensors \
    python3 /opt/refusal/test_per_request.py

# THE PATCHED RUNNER MUST BE THE ONE THAT ACTUALLY RUNS, and on 0.27.1 it is not
# by default. `patch_model_runner` wires v1/worker/gpu/model_runner.py — the V2
# runner — but 0.27.1 picks the runner from the architecture:
# `Qwen3_5ForConditionalGeneration` is not among the 7 in
# default_v2_model_runner_architectures(), and the model is flagged hybrid, so
# `is_hybrid and not is_default_v2_architecture` sends it to the V1 runner, which
# carries no refusal wiring at all. The symptom is silent and total: RefusalState
# is never constructed, the per-token buffer never exists, every forward falls back
# to the global scalar, and `cache_salt: "refusal:<x>"` does nothing. Measured
# 2026-08-22 in production: identical output byte for byte at temperature 0 with
# lambda 0, 1.0, 2.5 and 4.0, while the global dial did change the answer.
#
# Pinning it in the image, not in the Deployment, keeps the guarantee with the
# artifact that needs it. This image serves one model, so the scope is right.
ENV VLLM_USE_V2_MODEL_RUNNER=1

# Fails the build if a future base ever stops offering that switch, instead of
# silently reverting to a runner where the projection is dead code.
RUN python3 -c 'import vllm.envs as e; assert e.VLLM_USE_V2_MODEL_RUNNER is True, e.VLLM_USE_V2_MODEL_RUNNER; print("model runner V2: forzado")'

ENV VLLM_REFUSAL_LAMBDA_INIT=0.0

LABEL org.opencontainers.image.title="vllm-qwen38-rank1" \
      org.opencontainers.image.version="v0.27.1-rank1h" \
      org.opencontainers.image.source="https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection" \
      ai.e-dani.model="unsloth/Qwen3.8-27B-NVFP4" \
      ai.e-dani.arch="linux/arm64 (sm121)"
