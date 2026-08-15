# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Qwen3.8-27B + runtime rank-1 refusal projection — GB10/sm121, arm64
# ---------------------------------------------------------------------------
# Builds from the PUBLIC upstream base, so this is reproducible by anyone:
#
#   ghcr.io/anemll/dspark-vllm-gx10:0.1.1   (vLLM 0.25.2.dev0+g752a3a504)
#
# Everything the mechanism needs is in this repo — the core (lambda in the
# prefix-cache hash key, the collective_rpc setter, the /admin endpoint) AND the
# Qwen-specific hook. An earlier revision of this Dockerfile started from a
# private image that already carried the core; that made the repo un-buildable
# for anyone but us, which defeats the point of publishing it.
#
# Build (from the repo root):
#   docker buildx build --platform linux/arm64 \
#     -t <your-registry>/vllm-qwen38-rank1:20260815 \
#     --push -f deploy/Dockerfile .
#
# It cross-builds to arm64 from an x86 host WITHOUT QEMU: every layer is a COPY,
# and RUN is the only thing that would need to execute target-arch binaries.
# The stale .pyc files in the venv do not need deleting — Python invalidates on
# source mtime+size, and COPY changes both.
# ---------------------------------------------------------------------------
ARG BASE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

# --- core: model-agnostic, shared with the DeepSeek-V4 implementation --------
# refusal_projection.py is the mechanism itself. The other three are the wiring
# that makes it safe and reachable:
#
#   kv_cache_utils.py  puts lambda in the block hash key. WITHOUT THIS a prefix
#                      cached at lambda=0 and reused at lambda=1 yields corrupt
#                      state, silently and with no error. Load-bearing.
#   gpu_worker.py      set/get through collective_rpc — with the mp executor the
#                      frontend shares no process with the workers, so a local
#                      setter reaches no rank.
#   serve/refusal/     POST+GET /admin/refusal_lambda. The router only mounts if
#                      VLLM_REFUSAL_DIRS is set: no hook, no endpoint to attack.
COPY vllm/refusal_projection.py            ${SITE}/refusal_projection.py
COPY vllm/v1/core/kv_cache_utils.py        ${SITE}/v1/core/kv_cache_utils.py
COPY vllm/v1/worker/gpu_worker.py          ${SITE}/v1/worker/gpu_worker.py
COPY vllm/entrypoints/serve/__init__.py    ${SITE}/entrypoints/serve/__init__.py
COPY vllm/entrypoints/serve/refusal/       ${SITE}/entrypoints/serve/refusal/

# --- the hook, inside Qwen3.5 and only Qwen3.5 ------------------------------
# Overrides `forward` on Qwen3_5DecoderLayer rather than patching qwen3_next.py,
# which every Qwen3-Next model in the image shares. The copied body is
# AST-identical to the parent's apart from the two projection lines.
# The same class builds the backbone and the MTP drafter, so one hook covers both.
COPY vllm/models_qwen3_5.py                ${SITE}/model_executor/models/qwen3_5.py

# --- backport of upstream #48846, re-derived for 0.25.2 ---------------------
# Still unfixed in 0.25.2 (both `value.strip()` intact), and the file was rewritten
# between 0.23.1 and 0.25.2, so the older patch does not apply. Without it the
# leading indentation of tool arguments is destroyed, which breaks exact-match edit
# tools and makes agentic clients retry the same call forever.
COPY vllm/parser_qwen3.py                  ${SITE}/parser/qwen3.py

# --- the 128 directions (48 out_proj + 16 o_proj + 64 down_proj), 2.6 MB ----
COPY hf/refusal_dirs_qwen38.safetensors    /opt/refusal/refusal_dirs_qwen38.safetensors

# --- startup guard, fail-closed ---------------------------------------------
COPY tools/verify_qwen38_rank1.py          /opt/qwen38/verify_qwen38_rank1.py

# VLLM_REFUSAL_DIRS is deliberately NOT set: the image boots INERT and is
# equivalent to its base. Turning it on is the deployment's job, and removing the
# variable is the rollback — no rebuild, no going back to a previous image.
ENV VLLM_REFUSAL_LAMBDA_INIT=0.0

LABEL org.opencontainers.image.title="vllm-qwen38-rank1" \
      org.opencontainers.image.description="vLLM (sm121/GB10, arm64) for Qwen3.8-27B with runtime rank-1 refusal projection and a hot-swappable lambda." \
      org.opencontainers.image.base.name="ghcr.io/anemll/dspark-vllm-gx10:0.1.1" \
      org.opencontainers.image.source="https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.e-dani.arch="linux/arm64 (sm121)"
