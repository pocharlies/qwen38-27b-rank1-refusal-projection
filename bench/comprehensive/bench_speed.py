#!/usr/bin/env python3
"""Bateria de velocidad y acceptance MTP para Qwen3.8-27B.

Mide NON-STREAMING y a temperatura 0, que es lo que pide el criterio de
aceptacion. Contar chunks SSE con decodificacion especulativa mide steps/s, no
tok/s, y las cifras salen ~4x bajas: aqui se usa usage.completion_tokens del
propio servidor, que es el unico numero que no miente.

El acceptance rate NO se estima desde el tok/s: se lee de /metrics, que expone
los contadores de spec-decode de vLLM. Se toma un delta alrededor de cada fase
para no promediar con el warmup.

Uso:  python3 bench_speed.py --base http://<head>:8888 [--out resultados.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

MODEL = "qwen38-27b"

# ── prompts ───────────────────────────────────────────────────────────────────
# Dos familias, porque el criterio las separa: "codigo/estructurado" y "media
# sobre prompts variados". Los textos son deliberadamente largos de RESPUESTA
# (no de prompt) para que el decode domine y el tok/s sea el del steady state.
CODE_PROMPTS = [
    ("py-refactor",
     "Escribe una clase Python `RateLimiter` con token bucket thread-safe, "
     "type hints completos, docstrings y tres tests de pytest. Codigo completo."),
    ("sql-window",
     "Escribe una consulta PostgreSQL que, por cliente, saque el ticket medio "
     "movil de 3 pedidos usando window functions, y explica cada CTE."),
    ("ts-types",
     "Implementa en TypeScript un tipo `DeepReadonly<T>` recursivo que funcione "
     "con arrays, tuplas, Map y Set, con casos de prueba que compilen."),
    ("k8s-yaml",
     "Escribe un Deployment de Kubernetes con initContainer, probes, "
     "securityContext sin privilegios y limites, y comenta cada bloque."),
    ("algo",
     "Implementa Dijkstra con heap binario en Rust, con manejo de errores "
     "idiomatico y tests. Codigo completo, sin elidir."),
]

VARIED_PROMPTS = [
    ("explica-tecnico",
     "Explica como funciona la atencion dispersa comprimida frente a la "
     "atencion densa, y por que reduce el KV cache. Detallado."),
    ("resumen-largo",
     "Resume las implicaciones operativas de servir un MoE de 284B con 13B "
     "activos en dos nodos unidos por RoCE, y los modos de fallo."),
    ("prosa",
     "Escribe un informe de incidencia de 600 palabras sobre una caida de un "
     "servicio de inferencia por agotamiento de memoria unificada."),
    ("razonamiento",
     "Tengo 119 GiB por nodo, pesos de 83.5 GB por nodo y un KV cache "
     "comprimido. Calcula el contexto maximo servible y razona los pasos."),
    ("traduccion",
     "Traduce al ingles tecnico y luego al aleman: 'el arbitro de GPU escala a "
     "cero los residentes del nodo elegido antes de arrancar la carga'."),
]

# Profundidades para la curva de prefill. El relleno es texto real repetido, no
# un token repetido: un token repetido se comprime distinto en la atencion
# dispersa y da una curva optimista.
FILLER = ("El planificador de vLLM agrupa peticiones en lotes continuos y el "
          "cache de KV se pagina en bloques de tamano fijo para evitar la "
          "fragmentacion externa de la memoria del acelerador. ")
PREFILL_DEPTHS = [1_000, 4_000, 16_000, 64_000, 131_072, 262_144]


def _post(base: str, path: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _metrics(base: str) -> dict[str, float]:
    """Contadores de spec-decode de /metrics. Devuelve {} si no estan."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as r:
            body = r.read().decode()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] /metrics no accesible: {e}", file=sys.stderr)
        return {}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        m = re.match(r"^(\S+?)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def _request_load(base: str) -> tuple[float, float]:
    """Gauge global de peticiones. Sirve para detectar contaminacion externa.

    Los contadores de spec-decode son globales al head: si otro alias entra durante
    una fase, su acceptance se suma al delta y la corrida deja de ser atribuible al
    benchmark. El propio benchmark es secuencial, asi que durante su POST esperamos
    como maximo running=1 y fuera del POST running=0; waiting siempre debe ser 0.
    """
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as r:
        body = r.read().decode()
    vals = {"running": 0.0, "waiting": 0.0}
    for line in body.splitlines():
        for key in vals:
            if line.startswith(f"vllm:num_requests_{key}"):
                try:
                    vals[key] += float(line.rsplit(None, 1)[-1])
                except (ValueError, IndexError):
                    pass
    return vals["running"], vals["waiting"]


def _wait_idle(base: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running, waiting = _request_load(base)
        if running == 0 and waiting == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"el head no quedo ocioso en {timeout}s: running={running} waiting={waiting}")
        time.sleep(0.5)


class _LoadMonitor:
    """Muestrea carga mientras una fase conoce si SU request esta activa."""

    def __init__(self, base: str):
        self.base = base
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.own_request_active = False
        self.max_running = 0.0
        self.max_waiting = 0.0
        self.contaminated = False
        self.samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set_own_request(self, active: bool) -> None:
        with self.lock:
            self.own_request_active = active

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                running, waiting = _request_load(self.base)
                with self.lock:
                    expected = 1.0 if self.own_request_active else 0.0
                    self.samples += 1
                    self.max_running = max(self.max_running, running)
                    self.max_waiting = max(self.max_waiting, waiting)
                    if running > expected or waiting > 0:
                        self.contaminated = True
            except Exception:  # noqa: BLE001
                # No poder verificar aislamiento invalida la fase: fail closed.
                with self.lock:
                    self.contaminated = True
            self.stop_event.wait(0.5)

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=5)
        with self.lock:
            return {
                "contaminated": self.contaminated,
                "max_running": self.max_running,
                "max_waiting": self.max_waiting,
                "samples": self.samples,
            }


def _spec_delta(before: dict, after: dict) -> dict:
    """acceptance rate y mean acceptance length a partir del delta.

    vLLM v1 expone:
      vllm:spec_decode_num_draft_tokens_total     tokens propuestos
      vllm:spec_decode_num_accepted_tokens_total  tokens aceptados
      vllm:spec_decode_num_drafts_total           numero de drafts
    mean acceptance length = aceptados/drafts + 1  (el +1 es el token del target,
    que siempre sale aunque se rechace todo el draft).
    """
    def d(k: str) -> float:
        for name in (f"vllm:spec_decode_num_{k}_total", f"vllm:spec_decode_num_{k}"):
            if name in after:
                return after.get(name, 0.0) - before.get(name, 0.0)
        return 0.0

    draft, acc, drafts = d("draft_tokens"), d("accepted_tokens"), d("drafts")
    res: dict[str, float | None] = {
        "draft_tokens": draft, "accepted_tokens": acc, "drafts": drafts,
    }
    res["acceptance_rate"] = round(acc / draft, 4) if draft > 0 else None
    res["mean_acceptance_length"] = round(acc / drafts + 1, 3) if drafts > 0 else None
    return res


def _sum_spec_deltas(deltas: list[dict]) -> dict:
    """Aggregate only request-local, load-clean speculative counter deltas."""
    draft = sum(float(row.get("draft_tokens") or 0) for row in deltas)
    accepted = sum(float(row.get("accepted_tokens") or 0) for row in deltas)
    drafts = sum(float(row.get("drafts") or 0) for row in deltas)
    return {
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "drafts": drafts,
        "acceptance_rate": round(accepted / draft, 4) if draft > 0 else None,
        "mean_acceptance_length": (
            round(accepted / drafts + 1, 3) if drafts > 0 else None
        ),
    }


def run_group(base: str, name: str, prompts, max_tokens: int,
              temperature: float = 0.0, top_p: float | None = None,
              max_contaminated_retries: int = 5) -> dict:
    """`temperature` por defecto 0: es lo que pedia el criterio de aceptacion y
    con lo que se midio TODA la evidencia previa de este directorio.

    Temp 0 no representa todos los regimenes de produccion. Para comparar metodos
    de muestreo del drafter la diferencia es cualitativa, no de grado:

      temp == 0  -> rejection_sampler_utils.py:234 "Greedy sampling. Only the
                    target max/argmax are needed". Los logits del draft NO entran
                    en la aceptacion, que es justamente lo unico que aporta
                    `draft_sample_method=probabilistic`.
      temp  > 0  -> rama HAS_DRAFT_LOGITS: la aceptacion es el test de ratio p/q
                    con la softmax del draft.

    O sea que un A/B de draft_sample_method medido solo a temp 0 mide `greedy` en
    el unico regimen donde no puede perder. De ahi este flag.
    """
    print(f"\n=== {name} (temp={temperature}"
          + (f", top_p={top_p}" if top_p is not None else "") + ") ===")
    rows = []
    clean_spec_deltas = []
    discarded = []
    unresolved_contamination = False
    for pid, text in prompts:
        for attempt in range(max_contaminated_retries + 1):
            _wait_idle(base)
            monitor = _LoadMonitor(base)
            monitor.start()
            before = _metrics(base)
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": text}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,          # NON-streaming, a proposito
            }
            if top_p is not None:
                payload["top_p"] = top_p
            t0 = time.perf_counter()
            monitor.set_own_request(True)
            response = None
            request_error = None
            try:
                response = _post(base, "/v1/chat/completions", payload)
            except Exception as exc:  # noqa: BLE001
                request_error = exc
            dt = time.perf_counter() - t0
            # Keep expected=1 through the final metrics read. The response can
            # reach the client a few milliseconds before vLLM drops its own
            # running gauge; lowering expected earlier creates false
            # contamination. A real overlapping request still yields running=2.
            after = _metrics(base)
            monitor.set_own_request(False)
            request_load = monitor.finish()
            if request_load["contaminated"]:
                discarded.append({
                    "id": pid,
                    "attempt": attempt,
                    "wall_s": round(dt, 3),
                    "load_guard": request_load,
                })
                print(
                    f"  {pid:18} intento {attempt} DESCARTADO por trafico "
                    f"concurrente: {request_load}")
                if attempt == max_contaminated_retries:
                    unresolved_contamination = True
                    rows.append({
                        "id": pid,
                        "error": "contamination_retries_exhausted",
                    })
                continue
            if request_error is not None:
                print(f"  {pid:18} FALLO: {request_error}")
                rows.append({"id": pid, "error": str(request_error)})
                break

            assert response is not None
            spec_delta = _spec_delta(before, after)
            clean_spec_deltas.append(spec_delta)
            u = response.get("usage") or {}
            ct = u.get("completion_tokens") or 0
            # Respuesta vacia = fallo, no exito. Pero OJO: con reasoning_parser
            # deepseek_v4 el texto puede venir en reasoning_content, y si max_tokens
            # corta dentro del bloque de pensamiento los dos campos salen vacios con
            # tokens > 0. Eso es truncamiento, no un backend muerto: solo es fallo de
            # verdad cuando el servidor no genero NADA.
            msg = response["choices"][0].get("message") or {}
            body = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            if ct == 0:
                print(f"  {pid:18} FALLO: 0 tokens generados")
                rows.append({"id": pid, "error": "empty_response", "wall_s": round(dt, 2)})
                break
            truncated = not body.strip()
            tps = ct / dt
            rows.append({
                "id": pid,
                "attempt": attempt,
                "completion_tokens": ct,
                "prompt_tokens": u.get("prompt_tokens"),
                "wall_s": round(dt, 3),
                "tok_s": round(tps, 2),
                "truncated_in_think": truncated,
                "spec": spec_delta,
            })
            print(f"  {pid:18} {ct:6d} tok  {dt:7.2f}s  {tps:6.2f} tok/s"
                  f"{'  (cortado en el bloque de think)' if truncated else ''}")
            break

    load_guard = {
        "contaminated": unresolved_contamination,
        "discarded_attempts": len(discarded),
        "discarded": discarded,
    }
    ok = [r["tok_s"] for r in rows if "tok_s" in r]
    return {
        "rows": rows,
        "tok_s_mean": round(statistics.mean(ok), 2) if ok else None,
        "tok_s_median": round(statistics.median(ok), 2) if ok else None,
        "spec": _sum_spec_deltas(clean_spec_deltas),
        "load_guard": load_guard,
    }


def run_ttft(base: str, n: int = 5) -> dict:
    """TTFT de prompt corto. Es el unico caso donde SI se hace streaming: es la
    unica forma de cronometrar el primer token. No se usa para medir tok/s."""
    print("\n=== TTFT (prompt corto) ===")
    lat = []
    for i in range(n):
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "Di 'hola'."}],
                   "temperature": 0, "max_tokens": 16, "stream": True}
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                for raw in r:
                    s = raw.decode(errors="ignore").strip()
                    if not s.startswith("data:") or s.endswith("[DONE]"):
                        continue
                    try:
                        ch = json.loads(s[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    d = (ch.get("choices") or [{}])[0].get("delta") or {}
                    # OJO: esta build emite el pensamiento en `reasoning`, NO en
                    # `reasoning_content`, y el primer chunk trae content:"" (falsy)
                    # solo para abrir el rol. Mirar solo content/reasoning_content
                    # hace que TTFT no capture NADA y salga vacio sin error.
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        lat.append(time.perf_counter() - t0)
                        break
        except Exception as e:  # noqa: BLE001
            print(f"  intento {i+1}: FALLO {e}")
    for i, v in enumerate(lat):
        print(f"  intento {i+1}: {v:.3f}s")
    return {"samples_s": [round(v, 3) for v in lat],
            "median_s": round(statistics.median(lat), 3) if lat else None}


def run_prefill(base: str, depths) -> dict:
    """Curva de prefill por profundidad de contexto. max_tokens=1 para aislar
    el prefill del decode."""
    print("\n=== curva de prefill ===")
    rows = []
    unit = max(1, len(FILLER) // 4)  # ~4 chars/token aprox
    for d in depths:
        filler = FILLER * max(1, d // unit)
        prompt = (filler + "\n\nEn una sola palabra, di 'listo'.")
        payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 1, "stream": False}
        t0 = time.perf_counter()
        try:
            r = _post(base, "/v1/chat/completions", payload)
        except Exception as e:  # noqa: BLE001
            print(f"  {d:>8} objetivo  FALLO: {e}")
            rows.append({"target_depth": d, "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        pt = (r.get("usage") or {}).get("prompt_tokens") or 0
        rows.append({"target_depth": d, "prompt_tokens": pt, "wall_s": round(dt, 3),
                     "prefill_tok_s": round(pt / dt, 1) if dt > 0 else None})
        print(f"  {d:>8} objetivo -> {pt:>8} tok reales  {dt:7.2f}s  {pt/dt:8.1f} tok/s")
    return {"rows": rows}


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="p.ej. http://100.73.153.70:8888")
    ap.add_argument("--model", default=MODEL, help="ID exacto expuesto por /v1/models")
    ap.add_argument("--out", default="bench_speed_results.json")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--skip-prefill", action="store_true")
    # Solo afectan a los grupos de decode (code/varied), que son los que miden
    # acceptance. ttft y prefill se quedan a temp 0 a proposito: miden latencia de
    # prefill, no muestreo, y moverlos romperia la comparabilidad con el historico.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-contaminated-retries", type=int, default=5)
    a = ap.parse_args()
    MODEL = a.model

    try:
        with urllib.request.urlopen(a.base.rstrip("/") + "/v1/models", timeout=30) as r:
            served = [m["id"] for m in json.loads(r.read()).get("data", [])]
        print(f"modelos servidos: {served}")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: el servidor no responde en {a.base}: {e}", file=sys.stderr)
        return 1

    res = {
        "base": a.base,
        "served_models": served,
        # En el JSON, para que la evidencia se explique sola: sin esto no se
        # distingue una corrida a temp 0 de una a temp 1 mirando el fichero.
        "sampling": {"temperature": a.temperature, "top_p": a.top_p},
        "code": run_group(a.base, "codigo / estructurado", CODE_PROMPTS, a.max_tokens,
                          a.temperature, a.top_p, a.max_contaminated_retries),
        "varied": run_group(a.base, "prompts variados", VARIED_PROMPTS, a.max_tokens,
                            a.temperature, a.top_p, a.max_contaminated_retries),
        "ttft": run_ttft(a.base),
    }
    if not a.skip_prefill:
        res["prefill"] = run_prefill(a.base, PREFILL_DEPTHS)

    # Veredicto contra los criterios de aceptacion.
    spec = res["code"]["spec"]
    ar = spec.get("acceptance_rate")
    mal = spec.get("mean_acceptance_length")
    verdict = {
        "tok_s_code":    (res["code"]["tok_s_mean"],   45, 55),
        "tok_s_varied":  (res["varied"]["tok_s_mean"], 40, 45),
        "acceptance_rate": (ar, 0.55, 0.70),
        "mean_accept_len": (mal, 3.5, 4.5),
    }
    print("\n=== VEREDICTO ===")
    gates = {}
    for k, (val, mn, obj) in verdict.items():
        if val is None:
            print(f"  {k:18} SIN DATO"); gates[k] = None; continue
        ok = val >= mn
        gates[k] = ok
        print(f"  {k:18} {val:>8}  min {mn}  obj {obj}   {'OK' if ok else 'POR DEBAJO'}")
    if res["ttft"]["median_s"] is not None:
        ok = res["ttft"]["median_s"] < 3.0
        gates["ttft"] = ok
        print(f"  {'ttft_median_s':18} {res['ttft']['median_s']:>8}  max 3.0        {'OK' if ok else 'POR DEBAJO'}")
    res["gates"] = gates

    if gates.get("acceptance_rate") is False:
        print("\n*** PARAR: acceptance < 55%. Es fallo de carga del drafter, no de "
              "sampling. NO tocar LiteLLM hasta resolverlo. ***")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nescrito {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
