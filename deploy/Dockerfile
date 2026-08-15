# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Qwen3.8-27B + proyeccion ortogonal rank-1 en runtime — GB10/sm121, arm64
# ---------------------------------------------------------------------------
# BASE: la imagen que YA sirve DeepSeek-V4-Flash en este cluster.
#   harbor.lan.e-dani.com/homelab/dspark-vllm-gx10:0.1.1-rank1e
#   (vLLM 0.25.2.dev0+g752a3a504.d20260714)
#
# No es una eleccion de comodidad: esa imagen ya trae DENTRO todo el nucleo del
# mecanismo —refusal_projection.py, el lambda en la clave de hash del prefix cache
# (v1/core/kv_cache_utils.py), el setter por collective_rpc (v1/worker/gpu_worker.py)
# y el endpoint /admin/refusal_lambda— Y ADEMAS la implementacion de Qwen3.5
# (model_executor/models/qwen3_5.py + qwen3_5_mtp.py, con Qwen3_5MTP registrado en
# speculative decoding). Comprobado en la imagen, no supuesto.
#
# El qwen36 que corria antes iba sobre vllm-sm121-v5 == vLLM 0.23.1. Portar el parche
# de 0.25.2 HACIA ATRAS a 0.23.1 habria sido trabajo puro de perdida: se sube dos
# minors y el nucleo ya viene aplicado. Lo unico que faltaba es cablear el hook dentro
# de qwen3_5.py, que es lo que anade esta imagen.
#
# NO se va a 0.27.1 (11-08): exigiria compilar vLLM sm_121/arm64 desde fuente, horas de
# build y un artefacto sin validar. Esta base corre hoy en nvidia-dgx.
#
# POR QUE SOLO HAY COPY Y NINGUN RUN
# Todo el cambio es Python puro. Con solo COPY, BuildKit produce la imagen arm64 desde
# un host x86 SIN QEMU/binfmt: RUN es lo unico que exigiria ejecutar binarios del arch
# destino. El smoke test de build se sustituye por el guard de arranque
# (/opt/qwen38/verify_qwen38_rank1.py), que ademas comprueba mas cosas y las comprueba
# donde importa: en el pod que va a servir.
#
# Los .pyc viejos del venv no hay que borrarlos: Python invalida por mtime+tamano del
# fuente, y COPY los cambia.
#
# Build (desde la raiz del repo):
#   docker buildx build --platform linux/arm64 \
#     -t harbor.lan.e-dani.com/homelab/vllm-qwen38-rank1:20260815c-arm64 \
#     --push images/qwen38-rank1
# ---------------------------------------------------------------------------
ARG BASE=harbor.lan.e-dani.com/homelab/dspark-vllm-gx10:0.1.1-rank1e
FROM ${BASE}

ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

# --- el modulo del mecanismo, en su variante Qwen ---------------------------
# Cambia respecto al de DeepSeek en tres cosas: el naming de los sublayers (Qwen3.5 es
# hibrido y tiene TRES clases que escriben al residual, no una), un COEFICIENTE POR
# MODULO —la ablacion no tiene la misma fuerza en todas las capas, asi que un lambda
# uniforme no reproduce el perfil— y fail-closed en vez de avisar y seguir.
# Conserva is_enabled/set_lambda/get_lambda/lambda_hash_key con la misma firma, que es
# lo unico que le importan kv_cache_utils, gpu_worker y el api_router de la base.
COPY payload/refusal_projection.py       ${SITE}/refusal_projection.py

# --- el hook, dentro de Qwen3.5 y solo de Qwen3.5 ---------------------------
# Sobrescribe `forward` en Qwen3_5DecoderLayer en vez de parchear qwen3_next.py, que lo
# comparten TODOS los modelos Qwen3-Next de la imagen. El cuerpo copiado es
# AST-identico al del padre salvo las dos lineas de proyeccion (verificado).
# La misma clase la construyen el backbone y el drafter MTP, asi que un solo punto
# cubre los dos y quedan alineados por construccion.
COPY payload/models_qwen3_5.py           ${SITE}/model_executor/models/qwen3_5.py

# --- backport de upstream #48846, re-derivado para 0.25.2 -------------------
# El bug SIGUE sin arreglar en 0.25.2 (los dos `value.strip()` intactos), pero el
# fichero se reescribio entre 0.23.1 y 0.25.2 (8.9K -> 18.2K, ahora lleva la config de
# thinking), asi que el parche viejo del ConfigMap no aplica. Va HORNEADO aqui, no
# montado por subPath: un subPath NO refresca al cambiar el ConfigMap y deja el payload
# nuevo como no-op silencioso hasta un reinicio ajeno.
# OJO: hay que QUITAR el mount de `ornith-tool-parser-patch` sobre qwen3.py en el
# deployment, o el parser viejo de 0.23.1 tapa a este.
COPY payload/parser_qwen3.py             ${SITE}/parser/qwen3.py

# --- las 128 direcciones (48 out_proj + 16 o_proj + 64 down_proj) -----------
# Extraidas de la diferencia entre Qwen/Qwen3.8-27B y
# Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored, los dos BF16, en float64.
# Puertas medidas: energia rank-1 >= 0.988, s0/s1 39-65, cos(v0,u0^T Wbase) >= 0.996,
# lambda_eff ~= 1.000 en los 128 modulos. Mejores que el caso DeepSeek que ya corre en
# produccion (energia 0.84-0.94, s0/s1 13-23) y sin su sobredisparo: alli el checkpoint
# horneado se pasaba a lambda_eff 2.43 e INVERTIA la direccion.
# Ruta distinta a la de DeepSeek: la base ya trae /opt/refusal/refusal_dirs.safetensors.
COPY payload/refusal_dirs_qwen38.safetensors  /opt/refusal/refusal_dirs_qwen38.safetensors

# --- guard de arranque, fail-closed -----------------------------------------
COPY payload/verify_qwen38_rank1.py      /opt/qwen38/verify_qwen38_rank1.py

# NO se define VLLM_REFUSAL_DIRS: la imagen arranca INERTE y es equivalente a la base.
# Activarla es cosa del deployment, y quitar la variable es el rollback — sin rebuild y
# sin volver a la imagen anterior.
ENV VLLM_REFUSAL_LAMBDA_INIT=0.0

LABEL org.opencontainers.image.title="vllm-qwen38-rank1" \
      org.opencontainers.image.description="vLLM (sm121/GB10, arm64) para Qwen3.8-27B con proyeccion ortogonal rank-1 en runtime y lambda conmutable en caliente. Derivada de dspark-vllm-gx10:0.1.1-rank1e." \
      org.opencontainers.image.base.name="harbor.lan.e-dani.com/homelab/dspark-vllm-gx10:0.1.1-rank1e" \
      org.opencontainers.image.created="2026-08-15" \
      org.opencontainers.image.version="20260815-arm64" \
      org.opencontainers.image.vendor="homelab / e-dani.com" \
      org.opencontainers.image.source="k8s-ai-pocharlies/images/qwen38-rank1" \
      ai.e-dani.purpose="Qwen3.8-27B NVFP4+MTP residente en DGX1 con dial de refusal en runtime" \
      ai.e-dani.model="Qwen/Qwen3.8-27B (servido desde unsloth/Qwen3.8-27B-NVFP4)" \
      ai.e-dani.refusal-dirs="Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored vs Qwen/Qwen3.8-27B" \
      ai.e-dani.arch="linux/arm64 (sm121)"
