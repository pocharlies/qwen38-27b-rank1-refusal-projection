"""Proyeccion ortogonal rank-1 de refusal en runtime, para Qwen3.8-27B servido con SGLang.

Port del runtime de vLLM 0.27.1 (`../vllm-0.27.1/`) al motor SGLang de la receta
`MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`. Misma identidad algebraica, mismo fichero de
direcciones, mismo significado de lambda:

    (W - lam*coef*r r^T W) x   ==   Wx - lam*coef*r (r^T Wx)

editar el peso y proyectar la salida dan lo mismo, pero proyectar la salida deja el
checkpoint intacto y convierte lam en un dial de runtime. lam=0 es el base BIT-EXACTO;
lam=1 reproduce el perfil por capa de Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored.

QUE SE CONSERVA DEL PORT DE vLLM
  - El fichero `refusal_dirs_qwen38.safetensors`: 128 direcciones (48
    `linear_attn.out_proj` + 16 `self_attn.o_proj` + 64 `mlp.down_proj`, capas 0..63,
    hidden 5120) mas un `coef` POR MODULO. Heretic no abla con la misma fuerza en todas
    las capas, asi que un lam uniforme sin `coef` NO reproduce el perfil.
  - FAIL-CLOSED: al terminar de construir el modelo se comprueba que las 128 direcciones
    han sido reclamadas por alguna capa. Un modelo a medio ablar no da error y solo se
    nota midiendo el refusal rate.
  - lam es un TENSOR EN DEVICE mutado in-place. SGLang captura CUDA graphs; un float de
    Python se hornea en la captura y el decode se queda con el valor de entonces PARA
    SIEMPRE, sin un aviso. Es la misma mina que `capture_model` en vLLM.

QUE CAMBIA RESPECTO AL PORT DE vLLM
  1. NO hay lambda POR PETICION. En vLLM viajaba en `cache_salt` y exigio cablear el
     Model Runner V2. SGLang no tiene plumbing equivalente: aqui el dial es GLOBAL, un
     escalar para toda la batch. Decision del owner (28-08-2026), no un olvido.
  2. El dial se mueve por el canal que SGLang ya tiene para comandos de worker,
     `/set_internal_state`, y no por una ruta HTTP propia: el proceso del servidor HTTP
     NO es el del scheduler, que es donde vive el tensor.
  3. El drafter (DSpark) NO se abla: es otro checkpoint, con otra arquitectura
     (`models/dspark.py`), y no tenemos direcciones para el. Consecuencia medida en el
     lado vLLM con MTP: con lambda=1 la acceptance baja ~20% en los temas en los que se
     enciende el dial. Se acepta; el rejection sampling garantiza que la SALIDA sigue
     siendo la del target, o sea que es coste de velocidad, no de correccion.

POR QUE NO SE USAN LOS `--forward-hooks` NATIVOS DE SGLANG
  `model_executor/model_runner_components/cuda_graph_setup.py` los registra DESPUES de
  capturar los grafos ("capture stays hook-free and hooks fire only on the eager forward
  path"). Con cuda graph en decode, un hook nativo abla el prefill y NO abla el decode:
  ablacion parcial y muda. Por eso esto va DENTRO del forward del modulo.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

ENV_DIRS = "SGLANG_REFUSAL_DIRS"       # ruta al .safetensors de 128 direcciones + coefs
ENV_LAMBDA = "SGLANG_REFUSAL_LAMBDA"   # lambda inicial; 0 = base intacto

# Cota amplia a proposito, la misma que el router de vLLM: 0 = base; 1 = el perfil del
# checkpoint ablado; >1 sobredispara e INVIERTE la componente (medido: a 2.5 el refusal
# vuelve casi a la base y la generacion se desboca); <0 amplifica la direccion, o sea
# pide un modelo MAS reticente.
LAMBDA_MIN = -1.0
LAMBDA_MAX = 4.0

# Prefijo de las claves del fichero de direcciones. Se conserva del port de vLLM: el
# fichero se emitio contra el naming del checkpoint, no contra el de ningun motor.
_CKPT_STEM = "model.language_model.layers."

# El drafter especulativo (DSpark, DFlash, NEXTN/MTP) tiene sus propias capas y su
# propio indice, que tambien empieza en 0. Sin este filtro reclamaria la direccion de la
# capa 0 del backbone —otra capa, otra profundidad— en silencio y mal.
_DRAFT_MARKERS = ("mtp", "draft", "dspark", "dflash", "nextn", "eagle")

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")

# --- lambda POR PETICION -------------------------------------------------------
#
# Viaja en `cache_salt`, igual que en el port de vLLM. En SGLang eso NO hay que
# inventarlo: el layer OpenAI ya concatena `cache_salt` (+ `extra_key`) en
# `Req.extra_key` (entrypoints/openai/serving_base.py:_compute_extra_key) y ese
# mismo valor entra en la clave del radix cache
# (mem_cache/radix_cache.py: "extra key (e.g. lora_id, cache_salt)").
#
# Lo segundo importa tanto como lo primero: el KV DEPENDE de lambda. Un prefijo
# cacheado con lambda=0 y reusado con lambda=1 daria estados corruptos, en silencio.
# Como el sello viaja en la clave, los dos alias no comparten prefijo: el aislamiento
# sale gratis, no hay que programarlo.
_SALT_RE = re.compile(r"refusal:\s*([-+]?\d*\.?\d+)")

# 131072 filas = 512 KB. Mas anchas que cualquier forward plausible (prefill troceado
# 2048, decode 4 slots x 8 tokens de verify). Si algun dia se queda corto, `fill_batch`
# cae al global y avisa UNA vez: nunca mezcla el lambda de una peticion con otra.
TOKEN_BUFFER_ROWS = 131072

_lock = threading.Lock()
_state: Optional["_State"] = None
_dirs: Optional[dict] = None
_coefs: Optional[dict] = None
# Contador, no un `set`: un `set` solo ve la falta. Si DOS modulos reclamasen la
# misma direccion —otro subarbol cuyo nombre tambien casa con
# `layers.N.<a>.<b>`— el conteo seguiria dando 128/128 y esa capa quedaria
# proyectada DOS veces, o sea a 2*lambda. Tan invisible como quedarse a medias,
# que es justo lo que este fichero existe para no permitir.
_consumed: dict = {}
_seen_prefixes: set = set()


# --------------------------------------------------------------------------- carga


def _load_dirs(path: str):
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

    dirs = {k: _tensor(k) for k in header if k not in ("__metadata__", "__coefs__")}

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


class _State:
    """Estado del hook: el dial GLOBAL y el buffer POR TOKEN.

    El global es el que mueven `/admin/refusal_lambda` y `/set_internal_state`, y es
    el que se aplica a quien no trae sello. El buffer por token es el que permite
    servir dos lambdas distintos EN EL MISMO LOTE, y por tanto dos alias sobre un
    solo pod. Los dos son tensores en device mutados in-place: bajo CUDA graph, el
    grafo hornea el PUNTERO, no el contenido.
    """

    __slots__ = ("lam", "tok", "hidden", "device", "value")

    def __init__(self, lam: float, hidden: int, device: torch.device):
        # inference_mode(False) NO es opcional: un tensor nacido en inference mode no
        # admite mutacion in-place despues, y el dial es exactamente eso.
        with torch.inference_mode(False):
            self.lam = torch.tensor(float(lam), device=device, dtype=torch.float32)
            # Filas de sobra a proposito (512 KB): el forward mas ancho es un prefill
            # troceado (chunked-prefill-size, 2048 hoy) y los grafos de decode
            # rellenan hasta su bs capturado. Se crea AQUI, en la construccion del
            # modelo, porque asignar durante la captura de grafos no vale.
            self.tok = torch.full(
                (TOKEN_BUFFER_ROWS,), float(lam), device=device, dtype=torch.float32
            )
        self.hidden = int(hidden)
        self.device = device
        self.value = float(lam)

    def set_lambda(self, value: float) -> None:
        with torch.inference_mode(False):
            self.lam.fill_(float(value))
            # El buffer por token tambien: quien no trae sello usa el global, y las
            # filas de padding de los grafos leen de aqui.
            self.tok.fill_(float(value))
        self.value = float(value)


class _Writer:
    """Una direccion ya colocada en device, con su coeficiente y el lambda compartido.

    Se construye UNA vez, en el __init__ del modulo que la usa, para que el forward no
    tenga que tocar estado Python mutable (globals, locks, dicts): bajo `torch.compile`
    —que la receta DSpark enciende— eso provoca recompilaciones.
    """

    __slots__ = ("r_hat", "coef", "lam", "tok", "hidden", "key")

    def __init__(self, key: str, r_hat: torch.Tensor, coef: float, st: "_State"):
        self.key = key
        self.r_hat = r_hat.to(device=st.device, dtype=torch.float32).contiguous()
        self.coef = float(coef)
        self.lam = st.lam
        # El buffer por token se ATA aqui, en la construccion, y el forward solo hace
        # un slice de un atributo tensor: cero globals, cero dicts, cero lock. Bajo
        # `torch.compile` —que la receta DSpark enciende— leer estado Python mutable
        # desde el forward provocaria recompilaciones.
        self.tok = st.tok
        self.hidden = st.hidden


# ------------------------------------------------------------------------ ciclo de vida


def init_from_env(hidden_size: int, device=None) -> Optional[_State]:
    """Se llama en el __init__ del modelo, ANTES de construir las capas.

    Dos razones para que sea ahi y no perezoso: (1) las capas resuelven su direccion en
    su propio __init__ y necesitan el estado ya montado; (2) el tensor de lambda tiene
    que existir antes de la captura de grafos.

    Sin `SGLANG_REFUSAL_DIRS` no hay estado y `apply()` es la identidad — la rama se
    hornea en la captura y no cuesta nada. Con la variable puesta y el fichero ilegible
    ABORTA: un servidor que arranca "casi" ablado es peor que uno que no arranca.
    """
    global _state, _dirs, _coefs
    with _lock:
        if _state is not None:
            # IDEMPOTENTE A PROPOSITO: si una segunda construccion (drafter, o un
            # segundo runner en el mismo proceso) creara un `_State` nuevo, los grafos
            # ya capturados seguirian apuntando al tensor `lam` viejo y el dial dejaria
            # de mover nada, en silencio.
            if _state.hidden != int(hidden_size):
                raise RuntimeError(
                    f"rank1-refusal: ya inicializado con hidden={_state.hidden} y ahora "
                    f"se pide {hidden_size}. Dos modelos distintos en el mismo proceso "
                    f"no pueden compartir un dial."
                )
            return _state

        path = os.environ.get(ENV_DIRS)
        if not path:
            logger.info("rank1-refusal: %s sin definir, proyeccion DESACTIVADA", ENV_DIRS)
            return None

        dirs, coefs = _load_dirs(path)
        for k, v in dirs.items():
            if v.ndim != 1 or v.shape[0] != int(hidden_size):
                raise RuntimeError(
                    f"rank1-refusal: {k} tiene shape {tuple(v.shape)}, se esperaba "
                    f"({hidden_size},)"
                )
            norm = float(v.norm())
            if not (0.99 <= norm <= 1.01):
                raise RuntimeError(f"rank1-refusal: {k} no es unitaria (||r||={norm:.6f})")

        lam = float(os.environ.get(ENV_LAMBDA, "0"))
        if not (LAMBDA_MIN <= lam <= LAMBDA_MAX):
            raise RuntimeError(
                f"rank1-refusal: {ENV_LAMBDA}={lam} fuera de [{LAMBDA_MIN}, {LAMBDA_MAX}]"
            )

        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        _dirs, _coefs = dirs, coefs
        _state = _State(lam, int(hidden_size), device)
        logger.info(
            "rank1-refusal: ACTIVA  dirs=%s (%d modulos, coef %.4f..%.4f) hidden=%d "
            "lambda_inicial=%.4f device=%s",
            path, len(dirs), min(coefs.values()), max(coefs.values()),
            hidden_size, lam, device,
        )
        return _state


def get_state() -> Optional[_State]:
    return _state


def is_enabled() -> bool:
    return _state is not None


def resolve(prefix: str) -> Optional[_Writer]:
    """prefix de runtime de un writer -> `_Writer`, o None si ese modulo no esta ablado.

    NO se concatena el prefix tal cual contra el fichero: el naming de SGLang no es el
    del checkpoint (aqui la torre de texto cuelga de `model.language_model.model...`).
    Se extrae el INDICE de capa y los DOS ultimos componentes del camino, que es
    exactamente la forma de las claves (`linear_attn.out_proj`, `self_attn.o_proj`,
    `mlp.down_proj`). Asi da igual cuantos niveles meta el motor por encima.

    Devolver None no es un error por si mismo: lo que si lo es —y lo caza
    `verify_all_consumed()`— es que al final sobre una direccion sin reclamar.
    """
    if _state is None or not prefix:
        return None
    _seen_prefixes.add(prefix)
    lowered = prefix.lower()
    if any(m in lowered for m in _DRAFT_MARKERS):
        # El drafter reindexa desde 0: sin este corte reclamaria direcciones del
        # backbone que no le corresponden.
        return None
    m = _LAYER_RE.search(prefix)
    if m is None:
        return None
    tail = m.group(2).split(".")
    if len(tail) < 2:
        return None
    key = f"{_CKPT_STEM}{int(m.group(1))}.{tail[-2]}.{tail[-1]}"
    if key not in _dirs:
        return None
    _consumed[key] = _consumed.get(key, 0) + 1
    return _Writer(key, _dirs[key], _coefs[key], _state)


def verify_all_consumed() -> None:
    """FAIL-CLOSED. Aborta si alguna direccion no la reclamo ninguna capa.

    Sin esto, un cambio de naming en SGLang deja capas sin proyectar y el resultado es un
    modelo A MEDIO ABLAR: no hay error, no hay log que lo delate en produccion, y el
    comportamiento solo se nota si alguien mide el refusal rate. Preferimos no arrancar.
    """
    if _state is None:
        return
    twice = sorted(k for k, n in _consumed.items() if n > 1)
    if twice:
        raise RuntimeError(
            f"rank1-refusal: {len(twice)} direcciones reclamadas MAS DE UNA VEZ, p.ej. "
            f"{[(k, _consumed[k]) for k in twice[:3]]}. Esas capas se proyectarian dos "
            f"veces (2*lambda) y el conteo total seguiria cuadrando. NO se sirve asi."
        )
    orphan = sorted(set(_dirs) - set(_consumed))
    if orphan:
        raise RuntimeError(
            f"rank1-refusal: {len(orphan)} direcciones sin reclamar por ninguna capa, "
            f"p.ej. {orphan[:3]}. El modelo quedaria A MEDIO ABLAR y no daria ningun "
            f"error. Casi siempre es que el naming de SGLang ya no casa con el del "
            f"checkpoint. PREFIJOS VISTOS EN RUNTIME (muestra): "
            f"{sorted(_seen_prefixes)[:3]}. Claves esperadas: {sorted(_dirs)[:2]}. "
            f"NO se sirve asi."
        )
    logger.info(
        "rank1-refusal: %d/%d direcciones reclamadas, cada una exactamente una vez",
        len(_consumed), len(_dirs),
    )


def parse_request_lambda(extra_key) -> Optional[float]:
    """`cache_salt: "refusal:<x>"` -> lambda. None si no lo trae o no es valido.

    Se busca por REGEX y no por prefijo porque el layer OpenAI CONCATENA
    `cache_salt` + `extra_key` sin separador: un sello valido puede quedar pegado a
    otra cosa. Un valor fuera de la cota se ignora (se avisa una vez) y esa peticion
    usa el global: nunca el lambda de otra.
    """
    if not extra_key or not isinstance(extra_key, str):
        return None
    m = _SALT_RE.search(extra_key)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if not (LAMBDA_MIN <= v <= LAMBDA_MAX):
        _warn_once("salt_range", f"cache_salt pide lambda {v}, fuera de "
                                 f"[{LAMBDA_MIN}, {LAMBDA_MAX}]; se usa el global")
        return None
    return v


_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("rank1-refusal: %s", msg)


def fill_batch(worker_batch, forward_batch) -> None:
    """Escribe una fila de lambda POR TOKEN para el lote en curso.

    Se llama al final de `ForwardBatch.init_new`, o sea UNA vez por paso y SIEMPRE
    fuera del grafo: en replay no corre Python, pero esto corre antes, y lo que el
    grafo lee es la memoria del buffer.

    FAIL-SAFE, no fail-closed: ante cualquier layout que no cuadre se deja el buffer
    entero al lambda GLOBAL y se avisa una vez. Recortar o adivinar significaria
    aplicarle a una peticion el lambda de OTRA, que es peor que ignorar el sello.
    """
    st = _state
    if st is None:
        return
    reqs = getattr(worker_batch, "reqs", None) or []
    lams = [parse_request_lambda(getattr(r, "extra_key", None)) for r in reqs]
    if not lams or all(l is None for l in lams):
        # Nadie trae sello: el buffer ya esta al global salvo que el lote anterior lo
        # ensuciara. Un fill es un kernel sobre 512 KB, ni se nota.
        with torch.inference_mode(False):
            st.tok.fill_(st.value)
        return

    vals = [st.value if l is None else l for l in lams]
    if len(set(vals)) == 1:
        # Todas iguales: un solo kernel, sin copia host->device.
        with torch.inference_mode(False):
            st.tok.fill_(float(vals[0]))
        return

    ids = getattr(forward_batch, "input_ids", None)
    n = int(ids.shape[0]) if ids is not None else 0
    bs = len(vals)
    per_token = None
    seq_lens = getattr(worker_batch, "extend_seq_lens", None)
    if seq_lens is not None and len(seq_lens) == bs and int(sum(seq_lens)) == n:
        # Prefill: cada peticion aporta su tramo.
        per_token = [v for v, l in zip(vals, seq_lens) for _ in range(int(l))]
    elif n and bs and n % bs == 0:
        # Decode, y tambien el paso de verify de la especulativa: mismas filas por
        # peticion (1 en decode, draft+1 al verificar).
        rep = n // bs
        per_token = [v for v in vals for _ in range(rep)]
    if per_token is None or len(per_token) != n or n > st.tok.shape[0]:
        _warn_once(
            "layout",
            f"lote de {n} filas y {bs} peticiones con lambdas distintos que no encaja "
            f"en ningun layout conocido (buffer {st.tok.shape[0]}); se usa el lambda "
            f"GLOBAL para este lote. La seleccion por peticion NO esta actuando.",
        )
        with torch.inference_mode(False):
            st.tok.fill_(st.value)
        return

    with torch.inference_mode(False):
        # El resto del buffer al global: son las filas de padding de los grafos.
        st.tok.fill_(st.value)
        st.tok[:n].copy_(
            torch.tensor(per_token, dtype=torch.float32), non_blocking=True
        )


def set_lambda(value: float) -> float:
    """Dial en caliente. Lo llama `Scheduler.set_internal_state` en CADA rank."""
    if _state is None:
        raise RuntimeError(
            f"rank1-refusal: proyeccion desactivada en este servidor (falta {ENV_DIRS})"
        )
    v = float(value)
    if not (LAMBDA_MIN <= v <= LAMBDA_MAX):
        raise ValueError(f"lambda {v} fuera de [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    _state.set_lambda(v)
    return v


def get_lambda() -> Optional[float]:
    return None if _state is None else float(_state.lam.item())


# ------------------------------------------------------------------------------ kernel


def apply(w: Optional[_Writer], y: torch.Tensor) -> torch.Tensor:
    """y <- y - lam * coef * r (r . y).  Con `w=None` es la identidad.

    `w` se fija en el __init__ del modulo y no cambia nunca, asi que este `is None` lo
    resuelve el compilador UNA vez al especializar: no es una rama por paso. Lo que si
    cambia entre pasos es el CONTENIDO de `w.lam`, que el kernel lee de memoria —
    exactamente lo que sobrevive al replay del grafo.

    El producto escalar va en fp32 aunque `y` llegue en bf16: medido en el port de
    DeepSeek, 1,66e-3 contra 2,29e-3 haciendolo en bf16 — 28% mejor y gratis.
    """
    if w is None:
        return y
    if y.shape[-1] != w.hidden:
        # FAIL-CLOSED, y es carga, no adorno: si un ancla acabase disparando sobre un
        # tensor shardeado o a medio gather, el servidor tiene que MORIR. Un servidor
        # que abla la mitad no se distingue de uno sano mirando una respuesta.
        raise RuntimeError(
            f"rank1-refusal[{w.key}]: se esperaba ultimo eje {w.hidden}, visto "
            f"{tuple(y.shape)}"
        )
    proj = y.to(torch.float32) @ w.r_hat
    # UNA fila de lambda por token. Las filas de padding que meten los grafos de CUDA
    # leen del mismo buffer, donde `fill_batch` deja el lambda GLOBAL — por eso
    # cualquier n >= n_real es semanticamente correcta y el arreglo sobrevive al
    # padding. Si el lote pide mas filas que el buffer, `fill_batch` ya cayo al global.
    lam = w.tok[: proj.shape[0]] if proj.shape[0] <= w.tok.shape[0] else w.lam
    return y - ((proj * lam * w.coef).unsqueeze(-1) * w.r_hat).to(y.dtype)
