#!/usr/bin/env python3
"""Algebra del port a SGLang, en CPU y sin motor. Se corre fuera del build.

    python3 runtime/sglang-dspark/test_projection_equivalence.py

Comprueba lo unico que se puede comprobar sin GPU, que es justo lo que decide si la
ablacion es correcta o es teatro:

  1. proyectar la SALIDA == editar el PESO, por modulo y con su `coef`
  2. lambda=0 es la identidad BIT A BIT (no "aproximadamente")
  3. `set_lambda` muta el tensor IN-PLACE (el mismo objeto), que es lo unico que
     sobrevive al replay de un CUDA graph ya capturado
  4. una forma que no es `hidden` levanta RuntimeError (fail-closed, no warning)
  5. el resolver mapea el naming de SGLang al del checkpoint, y CORTA los prefijos
     del drafter, que reindexa desde 0
  6. `verify_all_consumed` aborta si sobra alguna direccion

Lo que NO cubre: que las anclas existan en el arbol de la imagen (eso es fail-closed
en el propio `patch_sglang_qwen38_27b.py`, que muere si una no aparece exactamente una
vez) y que el forward parcheado sea el que corre bajo grafo (eso solo se ve en GPU).
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "payload"))

HIDDEN = 64
KEYS = [
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.3.mlp.down_proj",
]
COEFS = [1.0, 1.13, 0.87]


def _write_dirs(path: Path) -> None:
    """Escribe un fichero de direcciones con el MISMO formato que el de produccion."""
    torch.manual_seed(0)
    tensors = {}
    for k in KEYS:
        v = torch.randn(HIDDEN, dtype=torch.float32)
        tensors[k] = v / v.norm()
    tensors["__coefs__"] = torch.tensor(COEFS, dtype=torch.float32)

    header, blob, off = {}, bytearray(), 0
    for k, v in tensors.items():
        flat = v.contiguous().flatten().tolist()
        raw = struct.pack(f"<{len(flat)}f", *flat)
        header[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    header["__metadata__"] = {"coef_order": json.dumps(KEYS)}
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + bytes(blob))


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    dirs_path = tmp / "dirs.safetensors"
    _write_dirs(dirs_path)

    os.environ["SGLANG_REFUSAL_DIRS"] = str(dirs_path)
    os.environ["SGLANG_REFUSAL_LAMBDA"] = "1.0"

    import refusal_projection as R

    fails: list[str] = []

    st = R.init_from_env(HIDDEN, device=torch.device("cpu"))
    assert st is not None, "el estado no se construyo"

    # 5 -- naming: SGLang mete niveles por encima; el drafter reindexa desde 0.
    w = R.resolve("model.language_model.model.layers.0.linear_attn.out_proj")
    if w is None or w.key != KEYS[0]:
        fails.append("5) el resolver no mapea el naming de SGLang al del checkpoint")
    if R.resolve("draft_model.layers.0.linear_attn.out_proj") is not None:
        fails.append("5) el resolver NO corta el prefijo del drafter")
    if R.resolve("model.language_model.model.layers.7.mlp.down_proj") is not None:
        fails.append("5) el resolver reclama un modulo que no esta en el fichero")

    # 6 -- fail-closed con direcciones sin reclamar
    try:
        R.verify_all_consumed()
        fails.append("6) verify_all_consumed NO aborto con 2 direcciones sin reclamar")
    except RuntimeError:
        pass
    w_attn = R.resolve("model.language_model.model.layers.3.self_attn.o_proj")
    w_mlp = R.resolve("model.language_model.model.layers.3.mlp.down_proj")
    try:
        R.verify_all_consumed()
    except RuntimeError as e:
        fails.append(f"6) verify_all_consumed aborto con todo reclamado: {e}")

    # 1 -- proyectar la salida == editar el peso
    torch.manual_seed(1)
    for name, wr in (("out_proj", w), ("o_proj", w_attn), ("down_proj", w_mlp)):
        W = torch.randn(HIDDEN, 128, dtype=torch.float64)
        x = torch.randn(4, 128, dtype=torch.float64)
        r = wr.r_hat.to(torch.float64)
        lam = 1.0
        W_abl = W - lam * wr.coef * torch.outer(r, r @ W)
        y_weight = x @ W_abl.T
        y_hook = R.apply(wr, (x @ W.T).to(torch.float32)).to(torch.float64)
        err = float((y_weight - y_hook).abs().max())
        if err > 1e-4:
            fails.append(f"1) {name}: err_max={err:.3e} entre peso editado y proyeccion")
        else:
            print(f"1) {name:10s} peso editado == proyeccion   err_max={err:.3e}")

    # 2 -- lambda=0 identidad BIT A BIT
    R.set_lambda(0.0)
    y = torch.randn(8, HIDDEN, dtype=torch.bfloat16)
    if not torch.equal(R.apply(w, y), y):
        fails.append("2) lambda=0 NO es la identidad bit a bit")
    else:
        print("2) lambda=0 identidad bit a bit  OK")

    # 3 -- set_lambda muta in-place (mismo objeto tensor)
    before = w.lam
    R.set_lambda(1.5)
    if before is not w.lam or float(w.lam) != 1.5:
        fails.append("3) set_lambda NO muta in-place el tensor que ve el writer")
    else:
        print("3) set_lambda in-place  OK  (lam=%.2f, mismo tensor)" % float(w.lam))

    # 3b -- y el cambio se ve desde TODOS los writers (comparten el tensor)
    if float(w_mlp.lam) != 1.5:
        fails.append("3b) los writers no comparten el tensor de lambda")

    # 4 -- forma incorrecta -> RuntimeError, no warning
    try:
        R.apply(w, torch.randn(8, HIDDEN + 1))
        fails.append("4) una forma incorrecta NO levanto RuntimeError")
    except RuntimeError:
        print("4) forma incorrecta -> RuntimeError  OK")

    # cota de lambda
    for bad in (-2.0, 4.5):
        try:
            R.set_lambda(bad)
            fails.append(f"7) set_lambda acepto {bad}, fuera de la cota")
        except ValueError:
            pass
    R.set_lambda(0.0)

    # --- lambda POR PETICION -------------------------------------------------
    #
    # Lo que permite dos alias sobre un pod. Se prueba el parseo del sello, los tres
    # layouts que existen (prefill, decode, verify de la especulativa), el fail-SAFE
    # ante un layout desconocido, y que el kernel de verdad aplica una lambda
    # DISTINTA por fila.
    class _Req:
        def __init__(self, salt): self.extra_key = salt

    class _WB:
        def __init__(self, salts, seq_lens=None):
            self.reqs = [_Req(x) for x in salts]
            self.extend_seq_lens = seq_lens

    class _FB:
        def __init__(self, n): self.input_ids = torch.zeros(n, dtype=torch.long)

    R.set_lambda(0.0)  # global a 0: el alias normal

    # 9 -- parseo del sello
    casos = [("refusal:1.0", 1.0), ("refusal: 2.5", 2.5), ("abcrefusal:1", 1.0),
             ("hola", None), ("refusal:9", None), (None, None)]
    for salt, esperado in casos:
        got = R.parse_request_lambda(salt)
        if got != esperado:
            fails.append(f"9) parse({salt!r}) = {got}, se esperaba {esperado}")
    else:
        print("9) parseo del sello (incluye fuera de cota -> None)  OK")

    # 10 -- prefill: cada peticion aporta su tramo
    R.fill_batch(_WB(["refusal:1.0", None], seq_lens=[3, 2]), _FB(5))
    got = w.tok[:7].tolist()
    if got != [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:
        fails.append(f"10) prefill mixto mal expandido: {got}")
    else:
        print("10) prefill mixto: 3 filas a lambda=1, 2 a 0, padding al global  OK")

    # 11 -- decode (1 fila/peticion) y verify de la especulativa (draft+1 filas)
    R.fill_batch(_WB(["refusal:1.0", None]), _FB(2))
    if w.tok[:2].tolist() != [1.0, 0.0]:
        fails.append(f"11) decode mal expandido: {w.tok[:2].tolist()}")
    R.fill_batch(_WB(["refusal:1.0", None]), _FB(8))
    if w.tok[:8].tolist() != [1.0] * 4 + [0.0] * 4:
        fails.append(f"11) verify mal expandido: {w.tok[:8].tolist()}")
    if not [f for f in fails if f.startswith("11)")]:
        print("11) decode y verify de la especulativa  OK")

    # 12 -- layout desconocido: fail-SAFE al global, nunca el lambda de otra peticion
    R.fill_batch(_WB(["refusal:1.0", None]), _FB(5))
    if w.tok[:5].tolist() != [0.0] * 5:
        fails.append(f"12) un layout raro NO cayo al global: {w.tok[:5].tolist()}")
    else:
        print("12) layout desconocido -> lambda global (fail-safe)  OK")

    # 13 -- el kernel aplica una lambda DISTINTA por fila
    R.fill_batch(_WB(["refusal:1.0", None]), _FB(2))
    y = torch.randn(2, HIDDEN)
    out = R.apply(w, y)
    fila_ablada = float((out[0] - y[0]).abs().max())
    fila_intacta = float((out[1] - y[1]).abs().max())
    if not (fila_ablada > 1e-6 and fila_intacta == 0.0):
        fails.append(f"13) el kernel no separa por fila (ablada={fila_ablada:.2e}, "
                     f"intacta={fila_intacta:.2e})")
    else:
        print(f"13) misma pasada, dos lambdas: fila sellada proyectada "
              f"({fila_ablada:.3f}) y fila normal BIT A BIT intacta  OK")

    # 8 -- doble reclamacion. Va la ULTIMA porque ensucia el contador a proposito:
    # dos modulos sobre la misma direccion la proyectarian a 2*lambda, y el conteo
    # total seguiria dando 128/128. Un `set` no lo veria.
    R.resolve("model.language_model.model.layers.3.mlp.down_proj")
    try:
        R.verify_all_consumed()
        fails.append("8) verify_all_consumed NO aborto con una direccion reclamada 2 veces")
    except RuntimeError as e:
        if "MAS DE UNA VEZ" not in str(e):
            fails.append(f"8) aborto, pero por otro motivo: {e}")
        else:
            print("8) doble reclamacion -> RuntimeError  OK")

    if fails:
        print("\nFALLOS:")
        for f in fails:
            print(" -", f)
        return 1
    print("\ntodo OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
