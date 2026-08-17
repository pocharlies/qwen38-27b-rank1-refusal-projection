#!/usr/bin/env python3
"""Comparacion extensa entre dos lambdas: acceptance, velocidad y calidad.

Por que existe: todas las medidas anteriores tienen n=1 o n=3 por brazo, y este
banco tiene un ruido que se come cualquier veredicto de una sola corrida (medido:
el MISMO lambda=1 dio acceptance 0,5383 y 0,5944 en dos pasadas). Esto corre lo
mismo muchas veces y ALTERNANDO los brazos, para que ninguna deriva del sistema
—termica, cache, carga— se cuele como si fuera efecto de lambda.

Ejes:
  1. acceptance de MTP     (bench_speed, contador de /metrics)
  2. velocidad             (tok/s codigo y variado, TTFT)
  3. calidad               NIAH 32k/128k y tool-calling 5 fases

Salida: JSON con todas las corridas + resumen con media, desviacion y t de Welch.
Un delta sin su desviacion al lado no significa nada; por eso van juntos.

Uso: python3 compare_full.py --base http://<head>:8888 --reps 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics as st
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def post(base, path, payload, timeout=120):
    r = urllib.request.Request(
        base.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def get(base, path, timeout=60):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as f:
        return json.loads(f.read())


def set_lambda(base, lam):
    post(base, "/admin/refusal_lambda", {"lambda": lam})
    chk = get(base, "/admin/refusal_lambda")
    got = chk.get("lambda")
    # `or` NO vale aqui: 0.0 es falsy y 0 es justo el brazo de control.
    if not chk.get("consistent") or got is None or abs(got - lam) > 1e-9:
        raise RuntimeError(f"lambda no quedo fijado: {chk}")


def wait_idle(base, timeout=600):
    deadline = time.monotonic() + timeout
    while True:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as f:
            body = f.read().decode()
        running = waiting = 0.0
        for line in body.splitlines():
            if line.startswith("vllm:num_requests_running{"):
                running += float(line.rsplit(None, 1)[-1])
            elif line.startswith("vllm:num_requests_waiting{"):
                waiting += float(line.rsplit(None, 1)[-1])
        if running == 0 and waiting == 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"el runtime no quedo ocioso en {timeout}s: "
                f"running={running} waiting={waiting}")
        time.sleep(0.5)


def _run(cmd, *, timeout, accepted_returncodes=(0,)):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode not in accepted_returncodes:
        raise RuntimeError(
            f"fallo ({p.returncode}) ejecutando {' '.join(map(str, cmd))}\n"
            f"stdout:\n{p.stdout[-4000:]}\nstderr:\n{p.stderr[-4000:]}")
    return p


def run_speed(base, model, tag, results_dir, max_tokens):
    out = results_dir / f"cf_speed_{tag}.json"
    if not out.exists():
        _run([sys.executable, str(HERE / "bench_speed.py"), "--base", base,
              "--model", model, "--max-tokens", str(max_tokens),
              "--skip-prefill", "--out", str(out)], timeout=1800)
    else:
        print(f"  reutilizando speed artefacto existente: {out}", flush=True)
    d = json.loads(out.read_text())
    # Estructura real de bench_speed: code/varied llevan tok_s_mean y un sub-dict
    # `spec` con el acceptance leido de los contadores de /metrics. El acceptance
    # de la puerta es el de `code`, que es el perfil con el que se calibro el
    # suelo de 0,55.
    code = d.get("code") or {}
    varied = d.get("varied") or {}
    spec = code.get("spec") or {}
    code_load = code.get("load_guard") or {}
    varied_load = varied.get("load_guard") or {}
    clean_code = sum("tok_s" in row for row in (code.get("rows") or []))
    clean_varied = sum("tok_s" in row for row in (varied.get("rows") or []))
    # Under live production traffic, one prompt can exhaust all clean-window
    # retries even though the other four in its family are valid.  Four clean
    # 600-token prompts still expose thousands of draft decisions; retain that
    # aggregate and label it partial instead of discarding eight good samples.
    partial = bool(code_load.get("contaminated") or varied_load.get("contaminated"))
    contaminated = bool(
        (code_load.get("contaminated") and clean_code < 4)
        or (varied_load.get("contaminated") and clean_varied < 4)
    )
    return {
        "acceptance": spec.get("acceptance_rate"),
        "accept_len": spec.get("mean_acceptance_length"),
        "acceptance_varied": (varied.get("spec") or {}).get("acceptance_rate"),
        "tok_s_code": code.get("tok_s_mean"),
        "tok_s_varied": varied.get("tok_s_mean"),
        "ttft": (d.get("ttft") or {}).get("median_s"),
        "contaminated": contaminated,
        "partial_due_to_contamination": partial,
        "clean_prompt_counts": {"code": clean_code, "varied": clean_varied},
        "load_guard": {"code": code_load, "varied": varied_load},
    }


def run_niah(base, model, tag, results_dir):
    out = results_dir / f"cf_niah_{tag}.json"
    _run([sys.executable, str(HERE / "bench_niah.py"), "--base", base,
          "--model", model,
          "--lengths", "32000,128000", "--depths", "0,25,50,75,100",
          "--lambdas", "0", "--no-lambda-control", "--out", str(out)], timeout=3000)
    rs = json.loads(out.read_text())["results"]
    return {"hits": sum(r["hits"] for r in rs), "n": sum(r["n"] for r in rs),
            "invalid": sum(r.get("errors", 0) + r.get("empty", 0) for r in rs)}


def run_tooling(base, model, tag, results_dir):
    out = results_dir / f"cf_tool_{tag}.json"
    # returncode=1 significa que una prueba de calidad fallo; es un resultado del
    # benchmark, no un fallo del harness. El JSON estructurado sigue siendo valido.
    _run([sys.executable, str(HERE / "bench_tooling.py"), "--base", base,
          "--model", model,
          "--out", str(out)], timeout=1800, accepted_returncodes=(0, 1))
    d = json.loads(out.read_text())
    return {k: d.get(k) for k in ("score", "passed", "total", "by_family")}


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    se = math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))
    return (st.mean(a) - st.mean(b)) / se if se else 0.0


def balanced_orders(values):
    """Williams design: balances position and first-order carryover.

    ``itertools.permutations(values)[:reps]`` is not balanced for four arms:
    its first six rows all start with the first lambda.  For an even number of
    arms this construction yields exactly N orders in which every lambda occurs
    once in every position and every directed adjacent pair occurs once.
    """
    n = len(values)
    if n == 2:
        return [list(values), list(reversed(values))]
    base = [0]
    for offset in range(1, n):
        base.append((offset + 1) // 2 if offset % 2 else n - offset // 2)
    orders = [
        [values[(index + shift) % n] for index in base]
        for shift in range(n)
    ]
    if n % 2:
        orders += [list(reversed(order)) for order in orders]
    return orders


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--lambdas", default="0,1,1.5")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--max-contaminated-retries", type=int, default=3)
    ap.add_argument("--idle-timeout", type=float, default=600)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--skip-lambdas", default="")
    ap.add_argument("--skip-niah", action="store_true")
    args = ap.parse_args()
    lams = [float(x) for x in args.lambdas.split(",")]
    if len(lams) < 2:
        raise SystemExit("--lambdas debe contener al menos dos valores")
    if len(set(lams)) != len(lams):
        raise SystemExit("--lambdas contiene valores duplicados")
    if any(lam < 0 or lam > 2.5 for lam in lams):
        raise SystemExit("--lambdas solo admite valores entre 0 y 2.5")
    if args.idle_timeout <= 0:
        raise SystemExit("--idle-timeout debe ser positivo")
    if args.reps <= 0:
        raise SystemExit("--reps debe ser positivo")
    skipped_lams = {
        float(value) for value in args.skip_lambdas.split(",") if value.strip()
    }
    if not skipped_lams.issubset(set(lams)):
        raise SystemExit("--skip-lambdas debe ser subconjunto de --lambdas")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(args.results_dir or (HERE / f"ab_lambda_{stamp}"))
    out_path = Path(args.out) if args.out else results_dir / "compare_full.json"
    initial = get(args.base, "/admin/refusal_lambda")
    initial_lambda = initial.get("lambda")
    if initial_lambda is None or not initial.get("consistent"):
        raise RuntimeError(f"estado inicial de lambda invalido: {initial}")
    if args.resume_existing:
        if not out_path.exists():
            raise RuntimeError(f"no existe resultado para reanudar: {out_path}")
        data = json.loads(out_path.read_text())
        meta = data.get("_meta") or {}
        for key, expected in (
            ("base", args.base), ("model", args.model), ("lambdas", lams),
            ("max_tokens", args.max_tokens),
        ):
            if meta.get(key) != expected:
                raise RuntimeError(
                    f"no se puede reanudar: {key}={meta.get(key)!r}, "
                    f"esperado={expected!r}")
        meta["resumed_at"] = datetime.now(timezone.utc).isoformat()
        meta["reps_target"] = args.reps
        meta["skipped_lambdas"] = sorted(skipped_lams)
        meta["niah_skipped"] = bool(args.skip_niah)
        data["_meta"] = meta
    else:
        results_dir.mkdir(parents=True, exist_ok=False)
        data = {str(l): {"speed": []} for l in lams}
        data["_meta"] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "base": args.base,
            "model": args.model,
            "lambdas": lams,
            "reps": args.reps,
            "reps_target": args.reps,
            "max_tokens": args.max_tokens,
            "initial_lambda": initial_lambda,
            "initial_per_rank": initial.get("per_rank"),
            "results_dir": str(results_dir),
            "skipped_lambdas": sorted(skipped_lams),
            "niah_skipped": bool(args.skip_niah),
        }
    t0 = time.time()

    try:
        # ── 1+2) acceptance y velocidad. Recorremos las permutaciones para que
        # cada lambda ocupe de forma equilibrada posiciones tempranas y tardias.
        # Con dos brazos esto degenera exactamente en AB/BA; con tres y reps=6
        # recorre las seis ordenaciones una vez.
        orders = balanced_orders(lams)
        data["_meta"]["orders"] = orders
        for i in range(args.reps):
            order = orders[i % len(orders)]
            for lam in order:
                if lam in skipped_lams:
                    continue
                if len(data[str(lam)]["speed"]) > i:
                    print(f"  rep{i+1} lambda={lam}: ya completa; se reutiliza", flush=True)
                    continue
                attempt = 0
                while True:
                    wait_idle(args.base, timeout=args.idle_timeout)
                    set_lambda(args.base, lam)
                    try:
                        r = run_speed(
                            args.base, args.model, f"{lam}_{i}_attempt{attempt}",
                            results_dir, args.max_tokens)
                    finally:
                        # Never hold a benchmark lambda while waiting for the
                        # next production-idle window.
                        set_lambda(args.base, float(initial_lambda))
                    if not r["contaminated"]:
                        break
                    data["_meta"].setdefault("discarded_contaminated", []).append({
                        "rep": i + 1, "lambda": lam, "attempt": attempt,
                        "load_guard": r["load_guard"],
                    })
                    out_path.write_text(json.dumps(data, indent=2))
                    print(f"  DESCARTADA rep{i+1} lambda={lam}: trafico concurrente "
                          f"{r['load_guard']}", flush=True)
                    attempt += 1
                    if attempt > args.max_contaminated_retries:
                        raise RuntimeError(
                            f"demasiadas corridas contaminadas para lambda={lam}, rep={i+1}")
                data[str(lam)]["speed"].append(r)
                out_path.write_text(json.dumps(data, indent=2))
                print(f"  rep{i+1} lambda={lam:<4} acceptance={r.get('acceptance')} "
                      f"tok_s={r.get('tok_s_code')}", flush=True)

        # ── 3) calidad, una pasada por brazo (son caras)
        for lam in lams:
            if lam in skipped_lams:
                continue
            need_niah = not args.skip_niah and "niah" not in data[str(lam)]
            need_tooling = "tooling" not in data[str(lam)]
            if not need_niah and not need_tooling:
                print(f"  lambda={lam}: niah/tooling ya completos; se reutilizan", flush=True)
                continue
            wait_idle(args.base, timeout=args.idle_timeout)
            set_lambda(args.base, lam)
            try:
                if need_niah:
                    data[str(lam)]["niah"] = run_niah(
                        args.base, args.model, str(lam), results_dir)
                if need_tooling:
                    data[str(lam)]["tooling"] = run_tooling(
                        args.base, args.model, str(lam), results_dir)
            finally:
                set_lambda(args.base, float(initial_lambda))
            out_path.write_text(json.dumps(data, indent=2))
            print(f"  lambda={lam}: niah={data[str(lam)].get('niah')} "
                  f"tooling={data[str(lam)].get('tooling')}", flush=True)
    finally:
        # El dial es global de produccion: restaurar SIEMPRE, tambien ante timeout,
        # JSON corrupto, Ctrl-C o un fallo de una prueba.
        set_lambda(args.base, float(initial_lambda))
        restored = get(args.base, "/admin/refusal_lambda")
        data["_meta"]["restored"] = restored
        out_path.write_text(json.dumps(data, indent=2))
        print(f"  lambda restaurado: {restored}", flush=True)

    # ── resumen
    print("\n" + "=" * 74)
    print(f"{'metrica':22s} " + " ".join(f"{'l='+str(l):>16s}" for l in lams))
    print("=" * 74)
    for key, label in (("acceptance", "acceptance (code)"),
                       ("acceptance_varied", "acceptance (variado)"),
                       ("accept_len", "accept_len"),
                       ("tok_s_code", "tok/s codigo"), ("tok_s_varied", "tok/s variado"),
                       ("ttft", "TTFT mediana")):
        cols, series = [], []
        for l in lams:
            vals = [s[key] for s in data[str(l)]["speed"] if s.get(key) is not None]
            series.append(vals)
            cols.append(f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}" if len(vals) > 1
                        else (f"{vals[0]:.4f}" if vals else "—"))
        print(f"{label:22s} " + " ".join(f"{c:>16s}" for c in cols))
        for left_index, right_index in itertools.combinations(range(len(lams)), 2):
            t = welch(series[left_index], series[right_index])
            if t is not None:
                print(f"{'  Welch l='+str(lams[left_index])+' vs '+str(lams[right_index]):22s} "
                      f"t={t:+.2f}")
    print("-" * 74)
    for l in lams:
        n = data[str(l)].get("niah")
        print(f"{'NIAH l='+str(l):22s} " +
              (f"{n['hits']}/{n['n']} (invalidas {n['invalid']})" if n else "—"))
        tool = data[str(l)].get("tooling")
        print(f"{'tool-calling l='+str(l):22s} {tool.get('score') if tool else '—'}")

    print(f"\n|t| < 2 => indistinguible. Suelo de acceptance: 0.55")
    print(f"tiempo total: {(time.time()-t0)/60:.1f} min")
    data["_meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    data["_meta"]["elapsed_minutes"] = round((time.time() - t0) / 60, 2)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"escrito {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
