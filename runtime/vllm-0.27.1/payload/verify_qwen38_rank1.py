#!/usr/bin/env python3
"""Guard de arranque de la imagen qwen38-rank1. Falla CERRADO antes de servir.

Sustituye al `verify_tool_arg_patch.py` del ConfigMap `ornith-tool-parser-patch`, que
pineaba `EXPECTED_VLLM = "0.23.1rc1.dev301+g04c2a8dea"` y aborta con cualquier otra
version. Aqui el parser va HORNEADO en la imagen, no montado por subPath, asi que:

  - se acabo la trampa del subPath, que NO refresca cuando cambia el ConfigMap y deja
    el payload nuevo como no-op silencioso hasta un reinicio ajeno;
  - el guard deja de pinear una version y comprueba el COMPORTAMIENTO. Pinear la
    version obliga a tocar el guard en cada bump aunque el fix siga bien; comprobar que
    el fix actua sigue siendo valido cuando la imagen sube de version, y sigue fallando
    si alguien monta encima un parser viejo.

Cuatro comprobaciones, todas baratas y sin GPU.
"""

import importlib.metadata
import json
import os
import sys


def fail(msg):
    sys.exit(f"[qwen38-rank1] FATAL: {msg}")


_vllm_version = importlib.metadata.version("vllm")
if _vllm_version != "0.27.1":
    fail(f"vLLM inesperado: {_vllm_version}; esta capa exige exactamente 0.27.1")


# 0 -- FlashInfer Python y sus dos artefactos binarios tienen que ser de la misma
#      version. El bypass heredado de la base ocultaba 0.6.15 + 0.6.13 y hacia que el
#      servidor arrancase sano para morir en la primera inferencia (plan: 20 vs 19
#      argumentos). Aqui se rechaza antes de cargar el modelo o reclamar la GPU.
try:
    _flashinfer_python = importlib.metadata.version("flashinfer-python")
    _flashinfer_cubin = importlib.metadata.version("flashinfer-cubin")
    _flashinfer_jit = importlib.metadata.version("flashinfer-jit-cache")
except importlib.metadata.PackageNotFoundError as e:
    fail(f"falta un paquete FlashInfer obligatorio: {e}")

_flashinfer_jit_base = _flashinfer_jit.split("+", 1)[0]
if not (
    _flashinfer_python == _flashinfer_cubin == _flashinfer_jit_base
):
    fail(
        "versiones FlashInfer desalineadas: "
        f"python={_flashinfer_python}, cubin={_flashinfer_cubin}, jit={_flashinfer_jit}"
    )
if os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"):
    fail("FLASHINFER_DISABLE_VERSION_CHECK no puede estar activo en el runtime Qwen")

# 1 -- la superficie que el arbol de la imagen base espera de refusal_projection.
#      NO es la que usa este modelo: v1/core/sched/output.py, v1/worker/gpu/model_runner.py
#      y v1/worker/gpu_model_runner.py llaman a la maquinaria por token, que esta
#      documentada como muerta pero SIGUE CABLEADA. Quitarla rompe el arranque con un
#      ImportError desde un fichero que no tiene nada que ver, y el mensaje no apunta a
#      la causa. Se comprueba aqui, que es donde se entiende.
_REQUIRED = (
    "is_enabled", "set_lambda", "get_lambda", "lambda_hash_key",      # el dial
    "parse_request_lambda", "set_per_token_lambda", "get_per_token_lambda",  # por token
    "ensure_buffers", "fill", "fill_neutral", "get_buffer",  # CUDA-graph safe
    "RefusalProjection", "resolve_direction", "verify_all_consumed",  # este modelo
)
try:
    import vllm.refusal_projection as _rp
except Exception as e:  # noqa: BLE001
    fail(f"no puedo importar vllm.refusal_projection: {e}")
_missing = [n for n in _REQUIRED if not hasattr(_rp, n)]
if _missing:
    fail(
        f"a refusal_projection le faltan {_missing}. El arbol de la imagen base los "
        f"importa; sin ellos el arranque revienta desde otro fichero con un error que "
        f"no apunta aqui."
    )

# 2 -- el fix del whitespace en argumentos de tool (upstream #48846) esta en efecto.
#      Sin el, la indentacion de la primera linea se pierde, los editores exact-match
#      de los clientes agenticos fallan y reintentan la MISMA tool call para siempre.
try:
    from vllm.parser.qwen3 import _qwen3_arg_converter
except Exception as e:  # noqa: BLE001
    fail(f"no puedo importar vllm.parser.qwen3: {e}")

got = json.loads(_qwen3_arg_converter("<parameter=x>\n    indented\n</parameter>\n", False))["x"]
if got != "    indented":
    fail(
        f"el fix #48846 NO esta en efecto (got {got!r}, esperado '    indented'). "
        f"Alguien ha montado un parser viejo por encima del de la imagen, o la imagen "
        f"se construyo sin el payload."
    )

# 3 -- si el dial esta encendido, el fichero de direcciones tiene que cargar Y traer
#      coeficientes por modulo. Un fichero de la generacion anterior (solo r_hat) se
#      aplicaria con lam uniforme y NO reproduciria el perfil por capa del checkpoint
#      ablado: saldria un modelo ablado de forma distinta a la medida, sin ningun error.
dirs_path = os.environ.get("VLLM_REFUSAL_DIRS")
if dirs_path:
    if not os.path.exists(dirs_path):
        fail(f"VLLM_REFUSAL_DIRS={dirs_path} no existe")
    try:
        from vllm.refusal_projection import get_dirs, _ensure_loaded
        dirs, coefs = _ensure_loaded()
    except Exception as e:  # noqa: BLE001
        fail(f"no puedo cargar las direcciones de {dirs_path}: {e}")
    if not dirs:
        fail(f"{dirs_path} no tiene direcciones")
    lo, hi = min(coefs.values()), max(coefs.values())
    print(
        f"[qwen38-rank1] direcciones: {len(dirs)} modulos, coef {lo:.4f}..{hi:.4f}",
        flush=True,
    )

    # 4 -- el modo del drafter MTP tiene que ser uno de los tres. Un valor con typo
    #      caeria al defecto en silencio y nadie sabria que el drafter va sin proyectar.
    mode = os.environ.get("VLLM_REFUSAL_MTP_MODE", "off").lower()
    if mode not in ("off", "last", "mean"):
        fail(f"VLLM_REFUSAL_MTP_MODE={mode!r} no valido: off | last | mean")
    print(f"[qwen38-rank1] MTP mode: {mode}", flush=True)
else:
    print("[qwen38-rank1] VLLM_REFUSAL_DIRS sin definir: imagen INERTE", flush=True)

import vllm  # noqa: E402

print(
    f"[qwen38-rank1] OK — vLLM {vllm.__version__}, "
    f"FlashInfer {_flashinfer_python}, fix #48846 verificado",
    flush=True,
)
