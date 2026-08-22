# SPDX-License-Identifier: Apache-2.0
"""Control en caliente de lambda de la proyeccion rank-1.

  POST /admin/refusal_lambda   {"lambda": 0.0}
  GET  /admin/refusal_lambda

El router SOLO se monta si VLLM_REFUSAL_DIRS esta definida: sin hook no hay
endpoint. Sigue el patron de `serve/lora/api_router.py`, que tambien se
auto-desactiva cuando su funcionalidad no esta habilitada.

NO EXPONER POR EL INGRESS PUBLICO. Esto no lo puede garantizar vLLM: es una ruta
HTTP mas, y quien decide que sale a internet es el ingress / AgentGateway. La
recomendacion es que /admin/* no salga de la red del cluster.

Por que collective_rpc y no un setter local: con el executor `mp` y TP=2 el
proceso del frontend NO es el de los workers. Cambiar lambda aqui no llegaria a
ningun rank. La llamada va a todos y se comprueba que TODOS devuelven el mismo
valor antes de contestar 200; si discrepan es un 500, porque un cluster con
lambdas distintos por rank produce basura silenciosa.
"""

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger
from vllm.refusal_projection import is_enabled

logger = init_logger(__name__)
router = APIRouter()

# Cota amplia a proposito. 0 = base intacto; 1 = proyeccion ortogonal exacta;
# >1 sobredispara e invierte la componente (el checkpoint horneado de cebeuq
# esta en ~2,43); <0 AMPLIFICA la direccion en vez de eliminarla, que es raro
# pero es la unica forma de pedir un modelo mas reticente, no menos.
LAMBDA_MIN = -1.0
LAMBDA_MAX = 4.0


class RefusalLambdaRequest(BaseModel):
    lambda_: float = Field(..., alias="lambda", ge=LAMBDA_MIN, le=LAMBDA_MAX)

    model_config = {"populate_by_name": True}


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


async def _rpc(raw_request: Request, method: str, args: tuple = ()):
    return await engine_client(raw_request).collective_rpc(method, args=args)


@router.post("/admin/refusal_lambda", dependencies=[Depends(validate_json_request)])
async def set_refusal_lambda(request: RefusalLambdaRequest, raw_request: Request):
    value = float(request.lambda_)
    results = await _rpc(raw_request, "set_refusal_lambda", (value,))

    applied = [r for r in results if r is not None]
    if not applied or any(abs(r - value) > 1e-9 for r in applied):
        logger.error(
            "refusal lambda: los ranks NO coinciden tras fijar %s: %s", value, results
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "los ranks no aplicaron el mismo lambda",
                "requested": value,
                "per_rank": results,
            },
        )

    logger.info("refusal lambda fijado a %s en %d ranks", value, len(applied))
    return JSONResponse(content={"lambda": value, "ranks": len(applied)})


@router.get("/admin/refusal_lambda")
async def get_refusal_lambda(raw_request: Request):
    results = await _rpc(raw_request, "get_refusal_lambda")
    vals = [r for r in results if r is not None]
    consistent = bool(vals) and all(abs(r - vals[0]) <= 1e-9 for r in vals)
    return JSONResponse(
        content={
            "lambda": vals[0] if consistent else None,
            "consistent": consistent,
            "per_rank": results,
        }
    )


def attach_router(app: FastAPI):
    if not is_enabled():
        # Sin VLLM_REFUSAL_DIRS no hay hook que controlar: no se monta la ruta.
        return
    logger.warning(
        "Proyeccion rank-1 ACTIVA: /admin/refusal_lambda montado. Esta ruta cambia "
        "el comportamiento del modelo en caliente y NO debe salir por el ingress "
        "publico."
    )
    app.include_router(router)
