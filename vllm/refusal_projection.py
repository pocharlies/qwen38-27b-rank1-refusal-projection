# SPDX-License-Identifier: Apache-2.0
"""Proyeccion ortogonal rank-1 en runtime sobre la salida de los sublayers — Qwen3.5.

Misma identidad que la version de DeepSeek-V4, verificada alli a 9,3e-16 en float64:

    (W - lam * r r^T W) x   ==   Wx - lam * r (r^T Wx)

editar el peso y proyectar la salida dan lo mismo, pero proyectar la salida deja el
checkpoint intacto y convierte lam en un dial de runtime.

QUE CAMBIA RESPECTO A LA VERSION DEEPSEEK
-----------------------------------------
1. NAMING. DeepSeek tenia un unico modulo que escribe al residual (`attn.wo_b`).
   Qwen3.5-27B es hibrido 3:1 y tiene tres:

       layers.N.linear_attn.out_proj    (Gated DeltaNet, 48 capas)
       layers.N.self_attn.o_proj        (atencion completa, 16 capas, N%4==3)
       layers.N.mlp.down_proj           (64 capas)

2. COEFICIENTE POR MODULO. Heretic NO abla con la misma fuerza en todas las capas:
   optimiza el peso capa a capa. Un lam unico no reproduce ese perfil. El fichero de
   direcciones trae, junto al r_hat unitario, un `coef` por modulo con su lam_eff
   medido, y el kernel es

       y  <-  y - lam * coef_m * r_hat (r_hat . y)

   de forma que **lam=1 reproduce EXACTAMENTE el perfil del checkpoint ablado** y lam=0
   es el base intacto y bit-exacto. NO se hereda el lam=1.5 de DeepSeek: aquel numero
   es de aquel modelo y aquel par.

3. FAIL-CLOSED. DeepSeek avisaba por log y seguia cuando un prefix no resolvia. Con una
   ablacion DISPERSA eso es peligroso: un modelo medio-ablado no da error, solo se porta
   raro. Aqui se comprueba al terminar de construir el modelo que TODAS las direcciones
   del fichero han sido reclamadas por alguna capa, y si sobra alguna el arranque aborta.

MEDIDO sobre Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored frente a Qwen/Qwen3.8-27B:
  - la ablacion toca los 128 modulos que escriben al residual, capas 0..63 completas.
  - energia rank-1 0,9867..0,9927, s0/s1 32..77, cos(v0,u0^T Wbase) mediana 0,99987, y
    lambda_eff 0,999..1,291: quita la direccion EXACTAMENTE una vez. El checkpoint de
    DeepSeek se pasaba a 2,43, o sea la INVERTIA — de ahi su caida de acceptance.
  - **mtp.* NO esta ablado**: Ektome ni siquiera trae esos tensores. El drafter va sin
    ablar, que es el desalineamiento target/drafter que el repo de DeepSeek senala como
    sospechoso de comerse el acceptance. Ver VLLM_REFUSAL_MTP_MODE.

  (El primer candidato, trohrbaugh/Qwen3.8-27B-heretic-ara, se DESCARTO tras medirlo:
   cos 0,17..0,46 y s0/s1 1,1..4,6 con un par BF16 contra BF16, o sea sin ruido de
   cuantizacion al que culpar. Ese dW no tiene forma de proyeccion. Que un repo diga
   "abliterated" no lo hace rank-1.)

CUDA GRAPHS. lam es un TENSOR EN DEVICE que se muta in-place (`lam_.fill_(v)`), nunca un
float de Python. Un escalar de Python queda horneado en el grafo capturado y cambiarlo en
caliente no hace absolutamente nada — sin error, simplemente no pasa nada.

ACTIVACION. Sin VLLM_REFUSAL_DIRS el hook no existe: no se construye el modulo y el
forward no lleva ni una rama extra. Ese es el rollback sin rebuild.

LAMBDA POR PETICION. Implementado desde 2026-08-19 y verificado en GPU. Viaja en
`cache_salt: "refusal:<x>"`, asi que un mismo pod puede servir un alias normal y otro
ablado con los mismos pesos y en el mismo lote.

Antes NO funcionaba, y el motivo merece quedar escrito porque el fallo era MUDO:
`capture_model` no pasa por `execute_model`, asi que durante la captura el tensor por
token era None y lo que quedaba trazado DENTRO del grafo era el escalar global. En replay
no corre ni una linea de Python, de modo que todo decode servido por grafo usaba el lambda
global para siempre, sin un solo aviso.

El arreglo es un buffer PERSISTENTE por rol, creado en el __init__ del runner (o sea antes
de `capture_model`) y mutado siempre in-place. Aqui hay una diferencia con DeepSeek que no
es cosmetica: el forward de este modelo vive DENTRO de la region compilada por Dynamo
(fullgraph), donde leer un global mutable provoca recompilaciones. Por eso el buffer se ata
al MODULO en la construccion (`self._tok`) y el forward solo hace `self._tok[:n]`, un slice
de un atributo tensor: cero globals, cero lock. Medido en GPU: compila una sola vez
(frames=1) y cambiar el CONTENIDO del buffer no recompila.

El dial global de /admin/refusal_lambda sigue existiendo y manda sobre quien no trae sello.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_DIRS = "VLLM_REFUSAL_DIRS"
ENV_LAMBDA = "VLLM_REFUSAL_LAMBDA_INIT"

# Que hacer con la capa del drafter MTP, que el checkpoint ablado NO toca.
#
#   off   (defecto) - el drafter queda sin proyectar, igual que el checkpoint horneado.
#                     lam=1 reproduce la fuente EXACTAMENTE. Es el defecto por eso: lo
#                     que no se ha medido no se enciende solo.
#   last            - se le aplica la direccion de la ultima capa ablada del backbone.
#   mean            - la media normalizada de las direcciones del backbone.
#
# `last`/`mean` alinean target y drafter por construccion — capacidad que el horneado no
# tiene. Es una MEJORA PLAUSIBLE, no medida: encenderla exige comparar acceptance contra
# `off` en el mismo pod y la misma hora.
ENV_MTP_MODE = "VLLM_REFUSAL_MTP_MODE"

# --- superficie POR PETICION -------------------------------------------------
#
# El lambda por peticion viaja en `cache_salt` con el formato "refusal:<float>". Lo parsea
# v1/core/sched/output.py y lo llevan hasta el runner gpu_input_batch.py y
# v1/worker/gpu/model_runner.py, que rellenan el buffer del rol correspondiente.
#
# El slot `_per_token` de abajo es el camino LEGACY del Model Runner V1, que fija un tensor
# por paso. El V2 —el que arranca este despliegue— no lo usa: lee el buffer atado al modulo.
# Se conserva porque el arbol de la imagen base lo importa y quitarlo rompe el arranque con
# un ImportError desde un fichero que no tiene nada que ver.
#
# Comportamiento ante un layout inesperado: se avisa UNA vez y se cae al lambda GLOBAL.
# Nunca al lambda de otra peticion — un fallo fail-safe, no una mezcla silenciosa.
SALT_PREFIX = "refusal:"

_per_token: torch.Tensor | None = None

# --- lambda POR PETICION, vivo bajo grafo CUDA (port de rank1g) ---------------
#
# Lo de arriba describe el mecanismo ROTO. Esto es el arreglo, portado del de
# DeepSeek pero ADAPTADO a la restriccion que aqui es dura: el forward de este
# modelo vive dentro de una region FULLGRAPH de Dynamo.
#
# QUE FALLABA. Tres cosas, y solo una gritaba en el log:
#   1. `capture_model` no pasa por `execute_model`: en la captura el slot por
#      token era None y lo que se TRAZABA dentro del grafo era el escalar global.
#      En replay no corre Python, asi que todo decode servido por grafo aplicaba
#      el global para siempre, sin un aviso.
#   2. Target y drafter tienen layouts de fila distintos.
#   3. El slot no se limpiaba: los dummy runs leian el tensor rancio.
#
# COMO SE ARREGLA AQUI, Y EN QUE SE DIFERENCIA DE DEEPSEEK. Un buffer persistente
# por rol, creado en el __init__ del runner (antes de capture_model) y mutado
# SIEMPRE in-place: la captura hornea el puntero y el tamano, cada paso reescribe
# el contenido.
#
# La diferencia esta en COMO lo lee el forward. En DeepSeek el forward llama a
# `view_for(role, n)`, que indexa una lista global de modulo. Aqui eso NO vale:
# este forward es fullgraph y leer estado Python mutable desde dentro provoca
# recompilaciones — es la misma razon por la que este fichero ya movia r_hat al
# device en el constructor en vez de perezosamente. Asi que el buffer se ATA AL
# MODULO en la construccion (`self._tok`) y el forward solo hace `self._tok[:n]`:
# un slice de un atributo tensor. Cero globals, cero lock, cero dict.
ROLE_TARGET = 0
ROLE_DRAFT = 1
ROLE_NAMES = ("target", "draft")
_N_ROLES = 2

_buf: list[torch.Tensor | None] = [None] * _N_ROLES


def ensure_buffers(max_num_tokens: int, device: torch.device) -> None:
    """Crea los buffers por rol. DEBE llamarse ANTES de construir el modelo.

    El orden importa dos veces: antes de `capture_model` (si no, el grafo hornea
    el escalar) y antes de que se construyan las capas (si no, `RefusalProjection`
    no tiene buffer que atarse y cae al global para siempre). El runner lo cumple
    solo: crea `RefusalState` en su __init__ y carga el modelo despues.

    `inference_mode(False)` por lo mismo que el tensor de lambda global: un tensor
    nacido en inference mode no admite mutacion in-place despues.
    """
    if not is_enabled():
        return
    with _lock:
        for role in (ROLE_TARGET, ROLE_DRAFT):
            if _buf[role] is None:
                with torch.inference_mode(False):
                    _buf[role] = torch.full(
                        (max_num_tokens,),
                        float(_lam_value),
                        dtype=torch.float32,
                        device=device,
                    )
        logger.info(
            "refusal projection: buffers por rol listos (%d tokens, %s)",
            max_num_tokens,
            device,
        )


def get_buffer(role: int) -> torch.Tensor | None:
    """Buffer del rol. Se llama en la CONSTRUCCION del modulo, nunca en el forward."""
    return _buf[role]


def fill(role: int, tok: torch.Tensor, global_lambda: float) -> None:
    """Escribe `tok` en las filas reales y el lambda global en el resto.

    Las filas de padding llevan el global, y eso es lo que hace que cualquier
    n >= n_real sea semanticamente correcta — o sea lo que hace que el arreglo
    sobreviva al padding de los grafos.
    """
    buf = _buf[role]
    if buf is None:
        return
    n = int(tok.shape[0])
    cap = int(buf.shape[0])
    if n > cap:
        # Fail-safe: no se recorta. Recortar significaria aplicar a una peticion
        # el lambda de OTRA. Se deja el buffer al global.
        _warn_capacity(role, n, cap)
        fill_neutral(role, global_lambda)
        return
    with torch.inference_mode(False):
        buf[:n].copy_(tok, non_blocking=True)
        if cap > n:
            buf[n:].fill_(float(global_lambda))


def fill_neutral(role: int, global_lambda: float) -> None:
    """Todo el buffer al lambda global. Para dummy/profile runs y para el drafter.

    Sin esto un dummy run consume el contenido RANCIO del ultimo paso real.
    """
    buf = _buf[role]
    if buf is None:
        return
    with torch.inference_mode(False):
        buf.fill_(float(global_lambda))


_warned_capacity: set[tuple[int, int, int]] = set()


def _warn_capacity(role: int, want: int, cap: int) -> None:
    """Unico desajuste que sigue siendo posible con buffers por rol.

    Si aparece, el lote pide mas filas de las que el buffer tiene y se cae al
    lambda GLOBAL. Significa que `max_num_tokens` no acota el forward de ese rol
    — supuesto roto, no un caso normal.
    """
    key = (role, want, cap)
    if key not in _warned_capacity:
        _warned_capacity.add(key)
        logger.warning(
            "refusal projection: rol %s pide %d filas pero el buffer tiene %d; "
            "se usa el lambda GLOBAL. La seleccion por peticion NO esta actuando.",
            ROLE_NAMES[role],
            want,
            cap,
        )


def parse_request_lambda(cache_salt: str | None) -> float | None:
    """`refusal:<float>` -> lambda. None si el salt no es nuestro o es invalido."""
    if not cache_salt or not cache_salt.startswith(SALT_PREFIX):
        return None
    try:
        return float(cache_salt[len(SALT_PREFIX) :])
    except ValueError:
        logger.warning("cache_salt %r no parsea como lambda; se ignora", cache_salt)
        return None


def set_per_token_lambda(t: torch.Tensor | None) -> None:
    """Fija el lambda por token del lote en curso. El runner lo llama por paso."""
    global _per_token
    _per_token = t


def get_per_token_lambda() -> torch.Tensor | None:
    return _per_token


_warned_shapes: set[tuple[int, int]] = set()


def _warn_shape_mismatch(got: int, want: int) -> None:
    """Avisa UNA vez por par de formas: un log por token ahogaria el proceso, pero
    callarlo del todo es peor — ya paso, y oculto durante dos ciclos de build que la
    seleccion por peticion no hacia absolutamente nada."""
    key = (got, want)
    if key not in _warned_shapes:
        _warned_shapes.add(key)
        logger.warning(
            "refusal projection: lambda por token con %d elementos pero y tiene %d "
            "filas; se usa el lambda GLOBAL para este lote. La seleccion por peticion "
            "NO esta actuando.",
            got, want,
        )


# lam se cuantiza a entero para la clave de hash de bloque del prefix cache.
LAMBDA_HASH_SCALE = 1000

_lock = threading.Lock()
_dirs: dict[str, torch.Tensor] | None = None
_coefs: dict[str, float] | None = None
_consumed: set[str] = set()
_seen_prefixes: set[str] = set()
_lam_by_device: dict[torch.device, torch.Tensor] = {}
_lam_value: float = 0.0


def _load_dirs(path: str) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Lector safetensors minimo: no arrastra la dependencia por unos MB.

    El fichero lleva un tensor por modulo mas `__coefs__`, y en `__metadata__` el
    `coef_order` que los empareja. Emparejar por orden alfabetico implicito seria una
    bomba de relojeria: basta que cambie un nombre para que cada capa reciba el
    coeficiente de otra, sin error.
    """
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()

    meta = header.get("__metadata__", {})

    def _tensor(key):
        m = header[key]
        if m["dtype"] != "F32":
            raise ValueError(f"{path}: {key} es {m['dtype']}, se esperaba F32")
        a, b = m["data_offsets"]
        return torch.frombuffer(bytearray(blob[a:b]), dtype=torch.float32).clone()

    dirs = {
        k: _tensor(k)
        for k in header
        if k not in ("__metadata__", "__coefs__")
    }

    if "__coefs__" not in header:
        raise ValueError(
            f"{path}: falta `__coefs__`. Este hook aplica un coeficiente POR MODULO "
            f"porque Heretic abla con fuerza distinta en cada capa; un fichero sin "
            f"coeficientes es de la generacion anterior y aplicarlo con lam uniforme "
            f"NO reproduce el perfil del checkpoint ablado."
        )
    order = json.loads(meta["coef_order"])
    cv = _tensor("__coefs__")
    if len(order) != len(cv) or set(order) != set(dirs):
        raise ValueError(
            f"{path}: coef_order ({len(order)}) no casa con las direcciones ({len(dirs)})"
        )
    coefs = {m: float(cv[i]) for i, m in enumerate(order)}
    return dirs, coefs


def _ensure_loaded():
    global _dirs, _coefs
    path = os.environ.get(ENV_DIRS)
    if not path:
        return None, None
    with _lock:
        if _dirs is None:
            _dirs, _coefs = _load_dirs(path)
            logger.info(
                "refusal projection: %d direcciones cargadas de %s (coef %.4f..%.4f)",
                len(_dirs), path, min(_coefs.values()), max(_coefs.values()),
            )
    return _dirs, _coefs


def get_dirs() -> dict[str, torch.Tensor] | None:
    return _ensure_loaded()[0]


def is_enabled() -> bool:
    return bool(os.environ.get(ENV_DIRS))


def get_lambda_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Tensor de lam por device, COMPARTIDO por todas las capas y mutado in-place.

    `inference_mode(False)` NO es opcional. El tensor se crea durante el constructor
    de `RefusalProjection`, antes de que Dynamo capture `forward`. Si naciera dentro
    del primer forward seria un *inference tensor* y PyTorch prohibiria mutarlo
    in-place despues:

        RuntimeError: Inplace update to inference tensor outside InferenceMode is
        not allowed.

    Ademas, entrar en `_lock` desde el forward hace que el compilador fullgraph aborte
    con `Unsupported context manager`. Por eso esta funcion no se llama desde forward.
    """
    global _lam_value
    with _lock:
        t = _lam_by_device.get(device)
        if t is None:
            init = float(os.environ.get(ENV_LAMBDA, "0.0"))
            _lam_value = init
            with torch.inference_mode(False):
                t = torch.tensor(init, device=device, dtype=dtype)
            _lam_by_device[device] = t
        return t


def set_lambda(value: float) -> float:
    """Muta lam IN-PLACE en todos los devices. Seguro con grafos ya capturados."""
    global _lam_value
    with _lock:
        with torch.inference_mode(False):
            for t in _lam_by_device.values():
                t.fill_(value)
        _lam_value = float(value)
        return _lam_value


def get_lambda() -> float:
    return _lam_value


def lambda_hash_key() -> int | None:
    """Clave estable de lam para el hash de bloque del prefix cache.

    El KV DEPENDE de lam: un prefijo cacheado con lam=0 y reusado con lam=1 da estados
    corruptos, en silencio y sin error.
    """
    if not is_enabled():
        return None
    return int(round(_lam_value * LAMBDA_HASH_SCALE))


# ------------------------------------------------------------------ resolucion

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

# Prefijo de las claves del checkpoint. Se deriva del fichero de direcciones en vez de
# hardcodearlo, para que un checkpoint con otro esquema no exija tocar codigo.
_CKPT_STEM = "model.language_model.layers."


def _ckpt_key(prefix: str, sublayer: str) -> str | None:
    """prefix de runtime + sublayer -> clave del fichero de direcciones.

    NO se concatena el prefix tal cual. El prefijo de runtime NO es la clave del
    checkpoint, aunque se parezcan: para el modelo multimodal la cadena es

        Qwen3_5ForConditionalGeneration(prefix="model")
          -> language_model = Qwen3_5ForCausalLM(prefix="model.language_model")
            -> self.model   = Qwen3_5Model(prefix="model.language_model.model")   <-- .model. de mas
              -> make_layers(prefix="model.language_model.model.layers")

    o sea `model.language_model.model.layers.N` en runtime contra
    `model.language_model.layers.N` en el checkpoint. Concatenar el prefix dejaba las 128
    direcciones sin reclamar y el arranque abortaba — lo caza verify_all_consumed(), pero
    solo despues de cargar los pesos.

    Asi que se extrae el INDICE y se construye la clave. Es inmune al esquema de
    prefijos: da igual cuantos niveles meta vLLM por encima.
    """
    if not prefix:
        return None
    # El drafter MTP se construye con esta misma clase. Su indice tambien empieza en 0,
    # asi que sin esta comprobacion reclamaria la direccion de la capa 0 del backbone,
    # que es de otra capa y de otra profundidad. Silencioso y erroneo.
    if "mtp" in prefix.lower():
        return None
    m = _LAYER_RE.search(prefix)
    if m is None:
        return None
    return f"{_CKPT_STEM}{int(m.group(1))}.{sublayer}"


def resolve_direction(prefix: str, sublayer: str):
    """Devuelve (r_hat, coef) para ese sublayer, o None si no esta ablado.

    Devolver None NO es un error: la ablacion medida es DISPERSA (capas 26-55 de 64), asi
    que la inmensa mayoria de las capas no lleva hook y eso es lo correcto. Lo que si es
    un error es que sobre una direccion sin reclamar — lo caza verify_all_consumed().
    """
    dirs, coefs = _ensure_loaded()
    if dirs is None:
        return None
    _seen_prefixes.add(prefix)
    key = _ckpt_key(prefix, sublayer)
    if key is None or key not in dirs:
        return None
    _consumed.add(key)
    return dirs[key], coefs[key]


def resolve_mtp_direction():
    """Direccion para la capa del drafter, segun VLLM_REFUSAL_MTP_MODE.

    El checkpoint ablado no toca `mtp.*`, asi que aqui no hay nada que "resolver": hay
    que DECIDIR. Por defecto no se decide nada (`off`).
    """
    dirs, coefs = _ensure_loaded()
    if dirs is None:
        return None
    mode = os.environ.get(ENV_MTP_MODE, "off").lower()
    if mode == "off":
        return None

    backbone = sorted(k for k in dirs if k.startswith("model."))
    if not backbone:
        return None

    if mode == "last":
        key = backbone[-1]
        logger.info("refusal projection: MTP toma la direccion de %s", key)
        return dirs[key].clone(), coefs[key]
    if mode == "mean":
        acc = torch.zeros_like(next(iter(dirs.values())))
        for k in backbone:
            r = dirs[k]
            # alinear el signo antes de promediar: r y -r son la MISMA direccion (el
            # hook usa r r^T), pero sumarlas crudas se cancelaria.
            acc += r if torch.dot(r, dirs[backbone[0]]) >= 0 else -r
        acc /= torch.linalg.vector_norm(acc)
        coef = float(sum(coefs[k] for k in backbone) / len(backbone))
        logger.info("refusal projection: MTP toma la media de %d direcciones", len(backbone))
        return acc, coef
    raise ValueError(f"{ENV_MTP_MODE}={mode!r} no valido: off | last | mean")


def verify_all_consumed() -> None:
    """FAIL-CLOSED. Aborta si alguna direccion del fichero no la reclamo ninguna capa.

    Sin esto, un cambio de naming en vLLM deja capas sin proyectar y el resultado es un
    modelo A MEDIO ABLAR: no hay error, no hay log que lo delate en produccion, y el
    comportamiento solo se nota si alguien mide el refusal rate. Preferimos no arrancar.
    """
    dirs, _ = _ensure_loaded()
    if dirs is None:
        return
    orphan = sorted(set(dirs) - _consumed)
    if orphan:
        raise RuntimeError(
            f"refusal projection: {len(orphan)} direcciones sin reclamar por ninguna "
            f"capa, p.ej. {orphan[:3]}. El modelo quedaria A MEDIO ABLAR y no daria "
            f"ningun error. Casi siempre es que el naming de vLLM ya no casa con el del "
            f"checkpoint. PREFIJOS QUE SE VIERON EN RUNTIME (muestra): "
            f"{sorted(_seen_prefixes)[:3]}. Compara con las claves esperadas: "
            f"{sorted(dirs)[:2]}. NO se sirve asi."
        )
    logger.info("refusal projection: %d/%d direcciones reclamadas", len(_consumed), len(dirs))


# ------------------------------------------------------------------ el kernel


class RefusalProjection(nn.Module):
    """y <- y - lam * coef * r * (r . y), con lam como tensor en device.

    El producto escalar va en fp32 aunque `y` llegue en bf16: medido en la fase 2 de
    DeepSeek, 1,66e-3 contra 2,29e-3 haciendolo en bf16 — mejora un 28% y es gratis.
    """

    def __init__(
        self,
        r_hat: torch.Tensor,
        coef: float,
        *,
        device: torch.device,
        role: int = ROLE_TARGET,
    ) -> None:
        super().__init__()
        # Los dos tensores quedan en el device definitivo ANTES de que Dynamo vea
        # forward. Moverlos perezosamente alli obligaria a ejecutar Python (incluido
        # el lock del registro global) dentro de una region fullgraph.
        self.register_buffer(
            "r_hat",
            r_hat.to(device=device, dtype=torch.float32),
            persistent=False,
        )
        self.coef = float(coef)
        self._lam = get_lambda_tensor(device, torch.float32)
        # BUFFER POR ROL ATADO AQUI, no consultado en el forward. Esta es LA
        # diferencia con el port de DeepSeek y no es cosmetica: alli el forward
        # llama a `view_for(role, n)`, que indexa una lista global de modulo. Este
        # forward es fullgraph, y leer estado Python mutable desde dentro provoca
        # recompilaciones. Atado como atributo, el forward solo ve un tensor.
        #
        # None = los buffers no existian al construir la capa (tests, o el hook
        # activado sin runner). El forward cae al dial global, que es el
        # comportamiento de antes: degradar, nunca petar.
        self._tok: torch.Tensor | None = get_buffer(role)
        self._role = role

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        r = self.r_hat
        proj = y.to(torch.float32) @ r
        # `self._tok` se fija en __init__ y no cambia nunca, asi que este `is None`
        # lo resuelve Dynamo UNA vez al especializar: no es una rama por paso ni un
        # guard que se reevalue. Lo que cambia entre pasos es el CONTENIDO del
        # buffer, que el kernel lee de memoria — exactamente lo que sobrevive al
        # replay del grafo.
        lam = self._lam if self._tok is None else self._tok[: proj.shape[0]]
        return y - (lam * self.coef * proj).unsqueeze(-1).to(y.dtype) * r.to(y.dtype)
