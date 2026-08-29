#!/usr/bin/env python3
"""Aplica el port rank-1 de refusal al arbol de SGLang de `lmsysorg/sglang:qwen38-27b`.

Por ANCLAS y FAIL-CLOSED, igual que `patch_vllm_0271.py` del mismo repo y que el port
hermano de `qwen4_exp`: si el fuente se mueve, el build MUERE. Nunca una imagen a medio
parchear — un servidor que abla el prefill y no el decode no se distingue de uno sano
mirando una respuesta.

CATORCE SITIOS

  qwen3_5.py  (el modelo: `Qwen3_5ForCausalLM`, hibrido 3:1, 64 capas)
    S0  import del payload
    S1  `init_from_env` ANTES de construir las capas — las capas resuelven su direccion
        en su propio __init__, y el tensor de lambda tiene que existir antes de que se
        capturen los CUDA graphs
    S2  `verify_all_consumed()` despues de `make_layers` (solo en el modelo completo:
        con `is_nextn` se construye UNA capa y sobrarian 126 direcciones)
    S3  `Qwen3_5GatedDeltaNet.__init__`  -> resuelve `linear_attn.out_proj` (48 capas)
    S4  `Qwen3_5GatedDeltaNet.forward`   -> proyecta su salida
    S5  `Qwen3_5AttentionDecoderLayer` (attn) __init__ -> resuelve `self_attn.o_proj` (16)
    S6  el mismo, forward -> proyecta su salida

  qwen2_moe.py  (`Qwen2MoeMLP` es la MLP densa que usa Qwen3.5; 64 `mlp.down_proj`)
    M0  import del payload
    M1  __init__ -> resuelve `mlp.down_proj`
    M2  forward  -> proyecta LAS DOS salidas. La rama fusionada
        (`_enable_silu_fp4_quant_fusion`, SiLU+mul+FP4-quant de FlashInfer) hace un
        `return` PROPIO y es justo la que se enciende con el checkpoint NVFP4: parchear
        solo la de abajo dejaria el modelo sin ablar en produccion y ablado en las
        pruebas de CPU. Ese es el fallo mudo que este parche existe para no cometer.

  scheduler.py
    D0..D2  el dial en caliente por `/set_internal_state`, siguiendo el precedente de
        `dspark_force_budget_frac`: es un comando de WORKER, no un server arg. El proceso
        del servidor HTTP no es el del scheduler, y el tensor de lambda vive en el
        segundo; por eso el dial NO puede ser un setter local como en vLLM.
    D3      el readback: `get_internal_state` publica el lambda VIVO de cada rank. Es el
        unico camino de vuelta desde el scheduler, y sin el, el GET del panel no tiene
        de donde leer.

  forward_batch_info.py
    F0/F1  el lambda POR PETICION: `cache_salt: "refusal:<x>"` -> `Req.extra_key` ->
        una fila de lambda por token, rellenada al final de `ForwardBatch.init_new`.
        Es lo que permite servir DOS alias (censurado y ablado) sobre UN pod sin
        tocar el dial global. El aislamiento del prefix cache ya lo da SGLang: ese
        mismo `extra_key` entra en la clave del radix cache.

  http_server.py
    H0  `/admin/refusal_lambda` (GET y POST), la MISMA superficie que servia vLLM.
        No es azucar: LiteLLM lleva ese admin_url cableado y el panel DGX lee y conmuta
        el dial por ahi. Son un envoltorio fino sobre D0..D3 — la que manda sigue siendo
        la ruta interna del scheduler.

QUE NO SE PARCHEA
  - El drafter. Con DSpark el borrador es OTRO checkpoint y OTRA arquitectura
    (`models/dspark.py`, `Qwen3DSparkModel`), que no importa ni `Qwen3_5*` ni
    `Qwen2MoeMLP`, asi que ninguna de estas anclas dispara ahi. No tenemos direcciones
    para el. Coste conocido y aceptado: con lambda>0 target y drafter dejan de estar
    alineados y la acceptance baja (~20% medido en el lado vLLM con MTP). La SALIDA no
    cambia: el rejection sampling la fija a la del target.
  - `qwen3_5_mtp.py`. Con `--speculative-algorithm DSPARK` no se usa; y si alguien
    vuelve a MTP, `resolve()` corta por el marcador `mtp`/`nextn` del prefijo.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PAYLOAD = "refusal_projection.py"


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[qwen38-27b-rank1] {path}: el ancla aparece {found} veces, se esperaban "
            f"{count}\n--- ancla ---\n{old}\n-------------"
        )
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def patch_model(site: Path) -> None:
    p = site / "srt/models/qwen3_5.py"

    # S0 --------------------------------------------------------------- import
    replace(
        p,
        """from sglang.srt.models.qwen2_moe import (
    Qwen2MoeMLP,
    Qwen2MoeSparseMoeBlock,
    can_fuse_shared_expert,
)
""",
        """from sglang.srt.models.qwen2_moe import (
    Qwen2MoeMLP,
    Qwen2MoeSparseMoeBlock,
    can_fuse_shared_expert,
)
from sglang.srt import refusal_projection as _refusal
""",
    )

    # S1 ------------------------------------------------------------- init dial
    replace(
        p,
        """        # Decoder layers
        def get_layer(idx: int, prefix: str):
""",
        """        # rank1-refusal: se inicializa AQUI, antes de construir las capas, por
        # dos razones: cada writer resuelve SU direccion en su propio __init__, y
        # el tensor de lambda tiene que existir antes de la captura de los CUDA
        # graphs (un float de Python se hornearia en la captura y el decode se
        # quedaria con el valor de entonces para siempre, sin avisar).
        _refusal.init_from_env(int(config.hidden_size))

        # Decoder layers
        def get_layer(idx: int, prefix: str):
""",
    )

    # S2 ------------------------------------------------------- verify fail-closed
    replace(
        p,
        """        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )
""",
        """        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )
        # rank1-refusal: FAIL-CLOSED. Si alguna de las 128 direcciones no ha sido
        # reclamada por ninguna capa, el modelo quedaria A MEDIO ABLAR y no daria
        # ningun error. Se aborta el arranque. Con `is_nextn` se construye UNA
        # capa a proposito: ahi sobrarian 126 y la comprobacion no aplica.
        if not is_nextn:
            _refusal.verify_all_consumed()
""",
    )

    # S3 ----------------------------------------- GatedDeltaNet: linear_attn.out_proj
    replace(
        p,
        """            prefix=add_prefix("out_proj", prefix),
        )
""",
        """            prefix=add_prefix("out_proj", prefix),
        )
        # rank1-refusal: `linear_attn.out_proj` es uno de los tres writers que
        # escriben al residual. Se resuelve UNA vez, aqui, para que el forward no
        # lea estado Python mutable (bajo torch.compile eso recompila).
        self._refusal = _refusal.resolve(add_prefix("out_proj", prefix))
""",
    )

    # S4 ------------------------------------------------- GatedDeltaNet: forward
    replace(
        p,
        """        output, _ = self.out_proj(core_attn_out)
        return output
""",
        """        output, _ = self.out_proj(core_attn_out)
        # rank1-refusal: proyectar la SALIDA equivale a editar el peso
        # (W - lam*coef*r r^T W)x == Wx - lam*coef*r (r^T Wx), pero deja el
        # checkpoint intacto. Con lambda=0 es la identidad bit a bit.
        output = _refusal.apply(self._refusal, output)
        return output
""",
    )

    # S5 ------------------------------------------------ atencion: self_attn.o_proj
    replace(
        p,
        """            prefix=add_prefix("o_proj", prefix),
        )
""",
        """            prefix=add_prefix("o_proj", prefix),
        )
        # rank1-refusal: segundo writer (16 capas de atencion completa, N%4==3).
        self._refusal = _refusal.resolve(add_prefix("o_proj", prefix))
""",
    )

    # S6 ------------------------------------------------------ atencion: forward
    replace(
        p,
        """        output, _ = self.o_proj(attn_output)
        return output
""",
        """        output, _ = self.o_proj(attn_output)
        output = _refusal.apply(self._refusal, output)
        return output
""",
    )


def patch_mlp(site: Path) -> None:
    p = site / "srt/models/qwen2_moe.py"

    # M0 --------------------------------------------------------------- import
    replace(
        p,
        """from sglang.srt.layers.activation import SiluAndMul
""",
        """from sglang.srt.layers.activation import SiluAndMul
from sglang.srt import refusal_projection as _refusal
""",
    )

    # M1 ------------------------------------------------------- MLP: mlp.down_proj
    #
    # `Qwen2MoeMLP` la comparten varios modelos de la imagen. No importa: `resolve()`
    # devuelve None para cualquier prefijo que no este en el fichero de direcciones,
    # y esta imagen sirve un solo modelo.
    replace(
        p,
        """        self.act_fn = SiluAndMul()
""",
        """        self.act_fn = SiluAndMul()
        # rank1-refusal: tercer writer, 64 de las 128 direcciones.
        self._refusal = _refusal.resolve(add_prefix("down_proj", prefix))
""",
    )

    # M2 ---------------------------------------------------------- MLP: forward
    #
    # LAS DOS salidas. La rama fusionada es la que se usa con NVFP4.
    replace(
        p,
        """        gate_up, _ = self.gate_up_proj(x)
        if self._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):
            x, _ = self.down_proj(self._silu_fp4_quant_fused(gate_up))
            return x
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
""",
        """        gate_up, _ = self.gate_up_proj(x)
        if self._enable_silu_fp4_quant_fusion and not isinstance(gate_up, tuple):
            x, _ = self.down_proj(self._silu_fp4_quant_fused(gate_up))
            # rank1-refusal: esta rama (SiLU+mul+FP4-quant fusionado de FlashInfer)
            # es la que se enciende con el checkpoint NVFP4, o sea la de produccion.
            # Sin esta linea el modelo quedaria sin ablar justo donde importa.
            return _refusal.apply(self._refusal, x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return _refusal.apply(self._refusal, x)
""",
    )


def patch_scheduler(site: Path) -> None:
    p = site / "srt/managers/scheduler.py"

    # D0 ------------------------------------------------------------- allow-list
    replace(
        p,
        """                "dspark_clear_info_records",
            ]
        )
""",
        """                "dspark_clear_info_records",
                "refusal_lambda",
            ]
        )
""",
    )

    # D1 ------------------------------------------------------------ validacion
    replace(
        p,
        """            elif k == "dspark_clear_info_records":
""",
        """            elif k == "refusal_lambda":
                from sglang.srt import refusal_projection as _refusal

                if _refusal.get_state() is None:
                    logging.warning(
                        "refusal_lambda: la proyeccion rank-1 no esta activa en este "
                        "servidor (falta SGLANG_REFUSAL_DIRS)."
                    )
                    if_success = False
                    break
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    logging.warning(f"refusal_lambda no es numerico: {v!r}.")
                    if_success = False
                    break
                if not (_refusal.LAMBDA_MIN <= fv <= _refusal.LAMBDA_MAX):
                    logging.warning(
                        f"refusal_lambda debe estar en "
                        f"[{_refusal.LAMBDA_MIN}, {_refusal.LAMBDA_MAX}], es {fv}."
                    )
                    if_success = False
                    break
            elif k == "dspark_clear_info_records":
""",
    )

    # D3 ------------------------------------------------------------- readback
    #
    # Sin esto el GET del panel no tiene de donde leer: `get_internal_state` es el
    # unico camino de vuelta desde el scheduler, que es donde vive el tensor.
    replace(
        p,
        """        ret["startup_time"] = self.startup_time
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests
""",
        """        ret["startup_time"] = self.startup_time
        ret["effective_max_running_requests_per_dp"] = self.max_running_requests
        # rank1-refusal: el valor VIVO de este rank. `None` si la proyeccion no
        # esta activa aqui — que es informacion, no un fallo: el panel distingue
        # "dial a 0" de "esta imagen no tiene dial".
        try:
            from sglang.srt import refusal_projection as _refusal

            ret["refusal_lambda"] = _refusal.get_lambda()
        except Exception:  # noqa: BLE001
            ret["refusal_lambda"] = None
""",
    )

    # D2 --------------------------------------------------------------- aplicar
    replace(
        p,
        """            remaining = dict(server_args_dict)
            frac = remaining.pop("dspark_force_budget_frac", None)
""",
        """            remaining = dict(server_args_dict)
            # rank1-refusal: comando de WORKER, no server arg — cada rank muta SU
            # tensor persistente in-place, que es lo unico que sobrevive al replay
            # de un CUDA graph ya capturado. Se loguea con el tp_rank para poder
            # comprobar que el fan-out llega a todos.
            if "refusal_lambda" in remaining:
                from sglang.srt import refusal_projection as _refusal

                _lam = _refusal.set_lambda(float(remaining.pop("refusal_lambda")))
                logger.info(
                    "rank1-refusal: lambda -> %.4f (tp_rank=%s)", _lam, self.ps.tp_rank
                )
            frac = remaining.pop("dspark_force_budget_frac", None)
""",
    )


def patch_http_server(site: Path) -> None:
    """`/admin/refusal_lambda`, la MISMA superficie HTTP que servia vLLM.

    No es azucar: LiteLLM lleva cableado
      admin_url = http://vllm-qwen38-27b-uncensored.llm.svc.cluster.local:8000/admin/refusal_lambda
    y desde ahi el panel DGX LEE el dial (GET, y quiere `lambda`, `consistent` y
    `per_rank`) y lo CONMUTA (POST `{"lambda": x}`). Servir el dial solo por
    `/set_internal_state` obligaria a cambiar LiteLLM y el panel en el mismo golpe que
    el motor. Estas dos rutas son un envoltorio fino sobre el canal interno: la que
    manda sigue siendo la del scheduler.
    """
    p = site / "srt/entrypoints/http_server.py"

    replace(
        p,
        """@app.api_route("/set_internal_state", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def set_internal_state(
    obj: Annotated[SetInternalStateReq, Body()], request: Request
):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res
""",
        """@app.api_route("/set_internal_state", methods=["POST", "PUT"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def set_internal_state(
    obj: Annotated[SetInternalStateReq, Body()], request: Request
):
    res = await _global_state.tokenizer_manager.set_internal_state(obj)
    return res


# ---------------------------------------------------------------- rank1-refusal
# Compatibilidad EXACTA con la superficie que servia vLLM: LiteLLM y el panel DGX
# hablan con `/admin/refusal_lambda`, no con `/set_internal_state`. NO EXPONER POR
# EL INGRESS PUBLICO: cambia el comportamiento del modelo en caliente.
@app.api_route("/admin/refusal_lambda", methods=["GET"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def get_refusal_lambda(request: Request):
    states = await _global_state.tokenizer_manager.get_internal_state()
    per_rank = [s.get("refusal_lambda") for s in states]
    vals = [v for v in per_rank if v is not None]
    consistent = bool(vals) and all(abs(v - vals[0]) <= 1e-9 for v in vals)
    return ORJSONResponse(
        content={
            "lambda": vals[0] if consistent else None,
            "consistent": consistent,
            "per_rank": per_rank,
        }
    )


@app.api_route("/admin/refusal_lambda", methods=["POST"])
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def post_refusal_lambda(request: Request):
    try:
        body = await request.json()
        value = float(body["lambda"])
    except Exception:  # noqa: BLE001
        return ORJSONResponse(
            status_code=400,
            content={"error": "se esperaba un cuerpo JSON {\\"lambda\\": <float>}"},
        )
    updated = await _global_state.tokenizer_manager.set_internal_state(
        SetInternalStateReq(server_args={"refusal_lambda": value})
    )
    # Un rank que no aplica es un cluster con lambdas distintos por rank, o sea
    # basura silenciosa. Se contesta 500, igual que hacia el router de vLLM.
    if not updated or not all(updated):
        return ORJSONResponse(
            status_code=500,
            content={
                "error": "los ranks no aplicaron el mismo lambda",
                "requested": value,
                "per_rank": updated,
            },
        )
    return ORJSONResponse(content={"lambda": value, "ranks": len(updated)})
""",
    )


def patch_forward_batch(site: Path) -> None:
    """F0/F1: el lambda POR PETICION, que es lo que permite DOS alias sobre UN pod.

    El sello viaja en `cache_salt` y SGLang ya lo lleva hasta `Req.extra_key`
    (`entrypoints/openai/serving_base.py:_compute_extra_key`) y hasta la clave del
    radix cache. O sea que el aislamiento de prefijos —que es obligatorio, porque el
    KV depende de lambda— sale de serie y no hay que programarlo.

    Lo unico que falta es que el forward vea una fila de lambda por token. Se rellena
    al final de `ForwardBatch.init_new`, que corre una vez por paso y SIEMPRE fuera
    del grafo; en replay no corre Python, pero el grafo lee la MEMORIA del buffer, que
    es justo lo que acabamos de escribir.
    """
    p = site / "srt/model_executor/forward_batch_info.py"

    replace(
        p,
        """from sglang.srt.utils.common import ceil_align, is_pin_memory_available
""",
        """from sglang.srt.utils.common import ceil_align, is_pin_memory_available

from sglang.srt import refusal_projection as _refusal
""",
    )

    replace(
        p,
        """        if (
            getattr(model_runner, "dcp_size", 1) > 1
            and ret.out_cache_loc is not None
            and is_hip()
        ):
            ret.dcp_kv_mask = (
                ret.positions % model_runner.dcp_size == model_runner.dcp_rank
            )

        return ret
""",
        """        if (
            getattr(model_runner, "dcp_size", 1) > 1
            and ret.out_cache_loc is not None
            and is_hip()
        ):
            ret.dcp_kv_mask = (
                ret.positions % model_runner.dcp_size == model_runner.dcp_rank
            )

        # rank1-refusal: una fila de lambda por token para ESTE paso. Va al final, con
        # el layout ya resuelto, y fuera del grafo. Es fail-SAFE: si el layout no
        # cuadra, el lote entero usa el lambda global (nunca el de otra peticion).
        _refusal.fill_batch(batch, ret)

        return ret
""",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=Path("/sgl-workspace/sglang/python/sglang"))
    ap.add_argument("--payload", type=Path, default=Path("/opt/refusal/payload"))
    args = ap.parse_args()

    shutil.copyfile(args.payload / PAYLOAD, args.site / "srt" / PAYLOAD)
    patch_model(args.site)
    patch_mlp(args.site)
    patch_scheduler(args.site)
    patch_http_server(args.site)
    patch_forward_batch(args.site)
    print("[qwen38-27b-rank1] parche aplicado: 14 anclas + payload (target; drafter no)")


if __name__ == "__main__":
    main()
