# SPDX-License-Identifier: Apache-2.0
"""Estado por peticion del lambda de la proyeccion rank-1, para el Model Runner V2.

QUE HABIA AQUI ANTES. `make_token_lambdas`, que construia una vista del buffer
del tamano de los tokens REALES y se la pasaba a `set_per_token_lambda`. No
funcionaba, y de tres maneras distintas:

  1. `capture_model` no pasa por `execute_model`, asi que durante la captura el
     slot estaba en None y lo que quedaba TRAZADO dentro del grafo era el escalar
     global. En replay no corre Python: todo decode servido por grafo aplicaba el
     lambda global para siempre, sin un solo aviso.
  2. La vista se dimensionaba a los tokens reales mientras `y` llega PADEADO para
     los grafos, asi que ni las formas casaban.
  3. Nadie limpiaba el buffer: los dummy runs leian el contenido del paso anterior.

QUE HACE AHORA. Escribe el CONTENIDO de un buffer persistente por rol, creado en
este __init__ — o sea antes de `capture_model` Y antes de que se construyan las
capas, que es lo que permite a `RefusalProjection` atarselo como atributo. La
vista la calcula el forward sobre ese storage fijo.

Sigue calcado de `LoraState` (lora_utils.py) en lo que importa: el mismo
`idx_mapping` / `num_scheduled_tokens` del lote.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm import refusal_projection

# NaN = la peticion no trajo lambda propio -> usa el global del servidor.
NO_LAMBDA = np.nan


class RefusalState:
    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):
        self.lambdas = np.full(max_num_reqs, NO_LAMBDA, dtype=np.float32)
        self.max_num_tokens = max_num_tokens
        refusal_projection.ensure_buffers(max_num_tokens, device)

    def add_request(self, req_index: int, refusal_lambda: float | None) -> None:
        self.lambdas[req_index] = NO_LAMBDA if refusal_lambda is None else refusal_lambda

    def remove_request(self, req_index: int) -> None:
        # El slot se reutiliza; dejarlo limpio evita que una peticion nueva herede
        # el modo de la anterior si algun camino no llamara a add_request.
        self.lambdas[req_index] = NO_LAMBDA

    def _per_req(self, idx_mapping: np.ndarray, global_lambda: float) -> np.ndarray:
        """Lambda de cada peticion del lote, con NaN resuelto al global VIGENTE."""
        lam = self.lambdas[idx_mapping]
        return np.where(np.isnan(lam), np.float32(global_lambda), lam)

    def fill_target(
        self,
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        global_lambda: float,
    ) -> None:
        """Buffer del TARGET: fila i <-> token i del lote."""
        tok = np.repeat(self._per_req(idx_mapping, global_lambda), num_scheduled_tokens)
        refusal_projection.fill(
            refusal_projection.ROLE_TARGET,
            torch.from_numpy(np.ascontiguousarray(tok)),
            global_lambda,
        )

    def fill_draft_neutral(self, global_lambda: float) -> None:
        """Drafter MTP: SIEMPRE al lambda global. No es pereza, es lo correcto hoy.

        DOS RAZONES, y la segunda es la que lo decide.

        1. Con `VLLM_REFUSAL_MTP_MODE=off` (el valor por defecto) el drafter no
           lleva proyeccion en absoluto: `resolve_mtp_direction` no devuelve nada
           y las capas `mtp.*` se construyen SIN hook. Este buffer no lo lee nadie.

        2. Y si se encendiera, el drafter MTP autoregresivo tiene DOS layouts de
           fila en la misma peticion: el primer pase avanza con `num_tokens` (el
           mismo layout que el target) y los pasos de decode con `num_reqs` filas,
           una por peticion ("Each request produces exactly 1 token per draft
           generation step", autoregressive/speculator.py). Un unico buffer no
           puede servir a los dos sin adivinar en cual estamos, y adivinar mal
           significa aplicarle a una peticion el lambda de otra. Eso es
           EXACTAMENTE el defecto que este arreglo elimina, asi que no se
           reintroduce por el otro lado.

        El coste de dejarlo al global es acotado y conocido: con muestreo por
        rechazo probabilistico los tokens aceptados siguen la distribucion del
        TARGET, asi que un drafter con lambda distinto cuesta ACCEPTANCE RATE, no
        correccion. Encender MTP_MODE exige antes resolver el doble layout.
        """
        refusal_projection.fill_neutral(refusal_projection.ROLE_DRAFT, global_lambda)

    def fill_neutral(self, global_lambda: float) -> None:
        """Los dos buffers al global. Para dummy/profile runs."""
        refusal_projection.fill_neutral(refusal_projection.ROLE_TARGET, global_lambda)
        refusal_projection.fill_neutral(refusal_projection.ROLE_DRAFT, global_lambda)
