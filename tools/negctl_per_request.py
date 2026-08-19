"""Control negativo del port a Qwen: misma API, comportamiento VIEJO.

Reproduce el DEFECTO real —los buffers no existen cuando se construye la capa ni
cuando se captura el grafo, que es lo que pasaba porque `capture_model` no pasa
por `execute_model`— y comprueba que el bloque C lo DETECTA. Si esto saliera
"PASS", el bloque C no valdria de nada y seria un test decorativo.
"""
import os
import sys

os.environ.setdefault("VLLM_REFUSAL_DIRS", "/opt/refusal/refusal_dirs_qwen38.safetensors")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vllm.refusal_projection as rp  # noqa: E402
from vllm.v1.worker.gpu.refusal_utils import RefusalState  # noqa: E402

if not torch.cuda.is_available():
    print("SIN GPU: este control no vale")
    sys.exit(2)

DEV = torch.device("cuda")
dirs, coefs = rp._ensure_loaded()
KEY = sorted(dirs)[0]
R0, COEF = dirs[KEY].to(DEV), float(coefs[KEY])
H, N = R0.shape[0], 6

# --- VIEJO: buffers AUSENTES al construir la capa y al capturar el grafo ---
rp._buf[:] = [None] * len(rp._buf)
rp.set_lambda(0.0)
mod = rp.RefusalProjection(R0, COEF, device=DEV, role=rp.ROLE_TARGET)
assert mod._tok is None, "en este escenario el modulo NO debe tener buffer atado"

y = torch.randn(N, H, dtype=torch.bfloat16, device=DEV)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        mod(y)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = mod(y)

# Ahora si se crean los buffers y se escribe lambda=1.0 para la peticion.
st = RefusalState(max_num_reqs=4, max_num_tokens=64, device=DEV)
st.add_request(0, 1.0)
st.fill_target(np.array([0], dtype=np.int32), np.array([N], dtype=np.int32), 0.0)
g.replay()
torch.cuda.synchronize()

cambio = not torch.equal(out, y)
print(f"replay refleja el lambda por peticion: {cambio}")
if cambio:
    print("FAIL  el bloque C NO discrimina: pasaria igual con el defecto puesto")
    sys.exit(1)
print("OK    el bloque C SI detecta el defecto (buffer ausente en la captura)")
sys.exit(0)
