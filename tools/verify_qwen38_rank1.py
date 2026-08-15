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

Tres comprobaciones, todas baratas y sin GPU.
"""

import json
import os
import sys


def fail(msg):
    sys.exit(f"[qwen38-rank1] FATAL: {msg}")


# 0 -- la superficie que el arbol de la imagen base espera de refusal_projection.
#      NO es la que usa este modelo: v1/core/sched/output.py, v1/worker/gpu/model_runner.py
#      y v1/worker/gpu_model_runner.py llaman a la maquinaria por token, que esta
#      documentada como muerta pero SIGUE CABLEADA. Quitarla rompe el arranque con un
#      ImportError desde un fichero que no tiene nada que ver, y el mensaje no apunta a
#      la causa. Se comprueba aqui, que es donde se entiende.
_REQUIRED = (
    "is_enabled", "set_lambda", "get_lambda", "lambda_hash_key",      # el dial
    "parse_request_lambda", "set_per_token_lambda", "get_per_token_lambda",  # por token
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

# 1 -- el fix del whitespace en argumentos de tool (upstream #48846) esta en efecto.
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

# 2 -- si el dial esta encendido, el fichero de direcciones tiene que cargar Y traer
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

    # 3 -- el modo del drafter MTP tiene que ser uno de los tres. Un valor con typo
    #      caeria al defecto en silencio y nadie sabria que el drafter va sin proyectar.
    mode = os.environ.get("VLLM_REFUSAL_MTP_MODE", "off").lower()
    if mode not in ("off", "last", "mean"):
        fail(f"VLLM_REFUSAL_MTP_MODE={mode!r} no valido: off | last | mean")
    print(f"[qwen38-rank1] MTP mode: {mode}", flush=True)
else:
    print("[qwen38-rank1] VLLM_REFUSAL_DIRS sin definir: imagen INERTE", flush=True)

import vllm  # noqa: E402

print(f"[qwen38-rank1] OK — vLLM {vllm.__version__}, fix #48846 verificado", flush=True)
