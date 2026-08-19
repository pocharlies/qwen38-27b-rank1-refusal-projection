#!/usr/bin/env python3
"""Lambda POR PETICION en Qwen3.8-27B. Se corre DENTRO de la imagen.

Importa de `vllm.*` instalado a proposito: `RefusalState` hace
`from vllm import refusal_projection`, asi que probar una copia suelta no probaria
nada de lo que corre en el pod.

CADA BLOQUE ESTA ESCRITO PARA FALLAR si se revierte una pieza concreta:

  A  layout de filas del target.
  B  semantica del buffer: mutacion in-place, padding al global, desborde
     fail-safe, y fill_neutral (sin el, un dummy run lee el buffer rancio).
  C  REPLAY DE GRAFO CUDA. El defecto grave y mudo: `capture_model` no pasa por
     `execute_model`, asi que el grafo tenia horneado el escalar global y todo
     decode servido por grafo ignoraba el cache_salt. Falla si se revierte.
  D  COMPILACION. Especifico de Qwen y no existe en el port de DeepSeek: aqui el
     forward vive en region FULLGRAPH de Dynamo. Se comprueba que compila con
     fullgraph=True y que cambiar el CONTENIDO del buffer NO recompila. Si el
     buffer se leyera de un global de modulo (como hace DeepSeek con view_for),
     esto es lo que lo cazaria.
  E  aislamiento de roles: el drafter no puede leer el buffer del target.
"""
import os
import sys

os.environ.setdefault("VLLM_REFUSAL_DIRS", "/opt/refusal/refusal_dirs_qwen38.safetensors")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vllm.refusal_projection as rp  # noqa: E402
from vllm.v1.worker.gpu.refusal_utils import RefusalState  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {extra}")
    if not cond:
        fails.append(name)


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dirs, coefs = rp._ensure_loaded()
KEY = sorted(dirs)[0]
R0, COEF = dirs[KEY].to(DEV), float(coefs[KEY])
H = R0.shape[0]
print(f"direccion de prueba: {KEY}  hidden={H}  coef={COEF}  device={DEV}\n")


def reset(max_num_tokens=64, max_num_reqs=4):
    rp._buf[:] = [None] * len(rp._buf)
    rp.set_lambda(0.0)
    return RefusalState(max_num_reqs=max_num_reqs, max_num_tokens=max_num_tokens,
                        device=DEV)


# --- A. layout de filas del target -------------------------------------------
st = reset()
st.add_request(0, 1.0)      # peticion con cache_salt refusal:1.0
st.add_request(1, None)     # peticion normal -> global
st.add_request(2, 0.5)
idx = np.array([0, 1, 2], dtype=np.int32)
nst = np.array([4, 2, 1], dtype=np.int32)
st.fill_target(idx, nst, global_lambda=0.0)
tgt = rp.get_buffer(rp.ROLE_TARGET)[:7].cpu().numpy()
check("target: fila i <-> token i (repeat por num_scheduled_tokens)",
      np.array_equal(tgt, np.array([1, 1, 1, 1, 0, 0, 0.5], dtype=np.float32)),
      f"got={tgt.tolist()}")

st.fill_target(idx, nst, global_lambda=2.5)
check("peticion sin salt toma el lambda global VIGENTE",
      float(rp.get_buffer(rp.ROLE_TARGET)[4]) == 2.5)

st.remove_request(0)
st.fill_target(np.array([0], dtype=np.int32), np.array([1], dtype=np.int32), 0.0)
check("remove_request limpia el slot (no hereda el lambda anterior)",
      float(rp.get_buffer(rp.ROLE_TARGET)[0]) == 0.0)


# --- B. semantica del buffer --------------------------------------------------
st = reset(max_num_tokens=64)
ptr = rp.get_buffer(rp.ROLE_TARGET).data_ptr()
st.add_request(0, 1.0)
st.fill_target(np.array([0], dtype=np.int32), np.array([3], dtype=np.int32), 0.0)
check("fill muta in-place (mismo data_ptr)",
      rp.get_buffer(rp.ROLE_TARGET).data_ptr() == ptr)
pad = rp.get_buffer(rp.ROLE_TARGET)[:8].cpu().numpy()
check("filas de padding al lambda global",
      np.array_equal(pad, np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)),
      f"got={pad.tolist()}")
st.add_request(1, 1.0)
st.fill_target(np.array([1], dtype=np.int32), np.array([999], dtype=np.int32), 0.0)
check("desborde -> buffer entero al global (fail-safe, NO recorta)",
      bool((rp.get_buffer(rp.ROLE_TARGET) == 0.0).all()))
st.fill_target(np.array([1], dtype=np.int32), np.array([3], dtype=np.int32), 0.0)
check("antes de fill_neutral hay datos reales",
      float(rp.get_buffer(rp.ROLE_TARGET)[0]) == 1.0)
st.fill_neutral(global_lambda=0.0)
check("fill_neutral deja los DOS buffers al global",
      float(rp.get_buffer(rp.ROLE_TARGET).max()) == 0.0
      and float(rp.get_buffer(rp.ROLE_DRAFT).max()) == 0.0)


def ref(y, lam_vec, r, coef):
    lv = torch.tensor(lam_vec, dtype=torch.float64, device=y.device).unsqueeze(-1)
    r64 = r.double()
    return y.double() - lv * coef * (y.double() @ r64).unsqueeze(-1) * r64


def rel(out, want):
    return float(((out.double() - want).norm(dim=1)
                  / want.norm(dim=1).clamp_min(1e-30)).max())


# --- C. replay de grafo CUDA --------------------------------------------------
if DEV.type != "cuda":
    print("\nSKIP  bloques C y D — no hay GPU en este contenedor\n")
else:
    st = reset(max_num_tokens=64)
    st.add_request(0, 0.0)
    N = 6
    mod = rp.RefusalProjection(R0, COEF, device=DEV, role=rp.ROLE_TARGET)
    y = torch.randn(N, H, dtype=torch.bfloat16, device=DEV)

    st.fill_neutral(global_lambda=0.0)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            mod(y)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = mod(y)

    # Sin recapturar: se escribe el lambda por peticion y se reproduce.
    st.add_request(0, 1.0)
    st.fill_target(np.array([0], dtype=np.int32), np.array([N], dtype=np.int32), 0.0)
    g.replay(); torch.cuda.synchronize()
    e = rel(static_out, ref(y, [1.0] * N, R0, COEF))
    check("replay ve el lambda por peticion escrito DESPUES de capturar",
          e < 5e-3, f"err_max={e:.3e}")

    out1 = static_out.clone()
    st.add_request(0, 0.0)
    st.fill_target(np.array([0], dtype=np.int32), np.array([N], dtype=np.int32), 0.0)
    g.replay(); torch.cuda.synchronize()
    check("mismo grafo, lambda distinto -> salida distinta",
          not torch.equal(static_out, out1))
    check("replay con lambda 0 devuelve y intacto", torch.equal(static_out, y))

    st2 = RefusalState(max_num_reqs=4, max_num_tokens=64, device=DEV)
    st2.add_request(0, 1.0); st2.add_request(1, 0.0)
    st2.fill_target(np.array([0, 1], dtype=np.int32),
                    np.array([3, 3], dtype=np.int32), 0.0)
    g.replay(); torch.cuda.synchronize()
    e = rel(static_out, ref(y, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], R0, COEF))
    check("lote mixto bajo replay: cada peticion con SU lambda", e < 5e-3,
          f"err_max={e:.3e}")
    check("en lote mixto las filas con lambda=0 salen intactas",
          torch.equal(static_out[3:], y[3:]))

    # --- D. COMPILACION: fullgraph y sin recompilar ---------------------------
    # El forward de Qwen3_5DecoderLayer es fullgraph. Si el buffer se leyera de un
    # global de modulo, o compilaria con graph break (fullgraph=True lo prohibe) o
    # meteria un guard sobre estado Python que recompila. Atado como atributo
    # tensor, no debe pasar ninguna de las dos.
    import torch._dynamo as dynamo
    from torch._dynamo.testing import CompileCounter

    dynamo.reset()
    st3 = reset(max_num_tokens=64)
    st3.add_request(0, 0.0)
    mod_c = rp.RefusalProjection(R0, COEF, device=DEV, role=rp.ROLE_TARGET)
    cnt = CompileCounter()
    compiled = torch.compile(mod_c, backend=cnt, fullgraph=True, dynamic=False)

    y2 = torch.randn(N, H, dtype=torch.bfloat16, device=DEV)
    try:
        o0 = compiled(y2)
        n_after_first = cnt.frame_count
        ok_fullgraph = True
        why = ""
    except Exception as exc:  # noqa: BLE001
        ok_fullgraph = False
        n_after_first = -1
        why = str(exc)[:160]
    check("compila con fullgraph=True (sin graph break)", ok_fullgraph, why)

    if ok_fullgraph:
        check("compilo exactamente una vez", n_after_first == 1,
              f"frames={n_after_first}")
        # Cambiar SOLO el contenido del buffer, misma forma: no debe recompilar.
        for lam in (1.0, 0.25, 1.0, 0.0):
            st3.add_request(0, lam)
            st3.fill_target(np.array([0], dtype=np.int32),
                            np.array([N], dtype=np.int32), 0.0)
            compiled(y2)
        check("cambiar el CONTENIDO del buffer NO recompila",
              cnt.frame_count == n_after_first,
              f"frames {n_after_first} -> {cnt.frame_count}")
        # ...y el resultado compilado sigue siendo el correcto.
        st3.add_request(0, 1.0)
        st3.fill_target(np.array([0], dtype=np.int32),
                        np.array([N], dtype=np.int32), 0.0)
        oc = compiled(y2)
        e = rel(oc, ref(y2, [1.0] * N, R0, COEF))
        check("el forward COMPILADO aplica el lambda por peticion", e < 5e-3,
              f"err_max={e:.3e}")


# --- E. aislamiento de roles --------------------------------------------------
st = reset(max_num_tokens=64)
st.add_request(0, 1.0)
st.fill_target(np.array([0], dtype=np.int32), np.array([4], dtype=np.int32), 0.0)
st.fill_draft_neutral(global_lambda=0.0)

y_t = torch.randn(4, H, dtype=torch.bfloat16, device=DEV)
mod_t = rp.RefusalProjection(R0, COEF, device=DEV, role=rp.ROLE_TARGET)
mod_d = rp.RefusalProjection(R0, COEF, device=DEV, role=rp.ROLE_DRAFT)
check("el modulo target lee el buffer del target (proyecta)",
      not torch.equal(mod_t(y_t), y_t))
check("el modulo draft lee el SUYO, que esta al global 0 -> y intacto",
      torch.equal(mod_d(y_t), y_t))
check("los dos roles apuntan a buffers DISTINTOS",
      rp.get_buffer(rp.ROLE_TARGET).data_ptr() != rp.get_buffer(rp.ROLE_DRAFT).data_ptr())

rp.set_lambda(0.0)
print()
print("TODOS OK" if not fails else "FALLOS: " + ", ".join(fails))
sys.exit(1 if fails else 0)
