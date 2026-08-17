#!/usr/bin/env python3
"""Measure dangerous-prompt refusal at several runtime lambdas.

Uses the pinned 60-prompt StrongREJECT Small set (10 prompts in each of six
categories) with no jailbreak wrapper.  Raw prompts and responses are written
only to the results directory; stdout contains aggregate counts, never harmful
response text.  The first-pass refusal label exactly reproduces the official
StrongREJECT string-matching evaluator and is complemented by a learned
classifier in score_refusal_classifier.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_COMMIT = "3432b2d696b428f242bd507df96d80f686571d5e"
DATASET_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
    f"{DATASET_COMMIT}/strongreject_dataset/strongreject_small_dataset.csv"
)
DATASET_SHA256 = "3051340e3e89a3598d764dde497d5fcda80e258ac05cc35e6bd87228ac3d467c"
MARKERS_COMMIT = "f9b7371a5cc4d3de54ffad160028976e0cdd01c8"
MARKERS_URL = (
    "https://raw.githubusercontent.com/dsbowen/strong_reject/"
    f"{MARKERS_COMMIT}/strong_reject/eval_files/string_matching.txt"
)
MARKERS_SHA256 = "9e7129b46498f807b3be954058ef61537cea5fc6e31bf626c65555b2964f3e1d"
MODEL = "qwen38-27b"


def get_json(base: str, path: str, timeout: float = 60) -> dict[str, Any]:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as response:
        return json.loads(response.read())


def post_json(base: str, path: str, payload: dict[str, Any],
              timeout: float = 600) -> dict[str, Any]:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def set_lambda(base: str, value: float) -> dict[str, Any]:
    post_json(base, "/admin/refusal_lambda", {"lambda": value}, timeout=120)
    state = get_json(base, "/admin/refusal_lambda")
    got = state.get("lambda")
    if not state.get("consistent") or got is None or abs(float(got) - value) > 1e-9:
        raise RuntimeError(f"lambda no quedo consistente en {value}: {state}")
    return state


def wait_idle(base: str, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while True:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as response:
            body = response.read().decode()
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


def wait_for_queue(base: str, max_waiting: int, max_running: int | None,
                   poll_seconds: float) -> None:
    """Wait indefinitely until the production backlog is below the threshold."""
    last: tuple[float, float] | None = None
    while True:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as response:
            body = response.read().decode()
        running = waiting = 0.0
        for line in body.splitlines():
            if line.startswith("vllm:num_requests_running{"):
                running += float(line.rsplit(None, 1)[-1])
            elif line.startswith("vllm:num_requests_waiting{"):
                waiting += float(line.rsplit(None, 1)[-1])
        state = (running, waiting)
        if state != last:
            print(
                f"[strongreject] esperando ventana: running={running:g} "
                f"waiting={waiting:g} umbrales="
                f"running<={max_running} waiting<={max_waiting}",
                flush=True,
            )
            last = state
        if waiting <= max_waiting and (max_running is None or running <= max_running):
            return
        time.sleep(poll_seconds)


def download_verified(url: str, expected_sha256: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"sha256 inesperado para {url}: {actual}")
    return data


def wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [centre - radius, centre + radius]


def exact_mcnemar(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_refusal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    def usable(rows: list[dict[str, Any]]) -> dict[str, bool]:
        return {
            str(row["prompt_id"]): bool(row["strongreject_string_refusal"])
            for row in rows
            if "error" not in row and not row.get("empty_content")
        }

    left_map, right_map = usable(left), usable(right)
    common = sorted(left_map.keys() & right_map.keys())
    left_only = sum(left_map[key] and not right_map[key] for key in common)
    right_only = sum(right_map[key] and not left_map[key] for key in common)
    left_refused = sum(left_map[key] for key in common)
    right_refused = sum(right_map[key] for key in common)
    n = len(common)
    return {
        "n_common_nonempty": n,
        "left_refused": left_refused,
        "right_refused": right_refused,
        "delta_right_minus_left": (right_refused - left_refused) / n if n else None,
        "left_only": left_only,
        "right_only": right_only,
        "mcnemar_exact_p_two_sided": exact_mcnemar(left_only, right_only),
    }


def query_one(base: str, row: dict[str, str], markers: list[str], max_tokens: int,
              retries: int, request_timeout: float) -> dict[str, Any]:
    prompt = row["forbidden_prompt"]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    last_error: str | None = None
    response: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        try:
            response = post_json(
                base, "/v1/chat/completions", payload, timeout=request_timeout)
            break
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2**attempt)
    elapsed = time.perf_counter() - started
    prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if response is None:
        return {
            "prompt_id": prompt_id,
            "category": row["category"],
            "source": row["source"],
            "forbidden_prompt": prompt,
            "error": last_error,
            "wall_s": round(elapsed, 3),
        }

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    usage = response.get("usage") or {}
    matched = [marker for marker in markers if marker in content]
    return {
        "prompt_id": prompt_id,
        "category": row["category"],
        "source": row["source"],
        "forbidden_prompt": prompt,
        "content": content,
        "reasoning_content": reasoning,
        "empty_content": not content.strip(),
        "strongreject_string_refusal": bool(matched),
        "strongreject_matched_markers": matched,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "wall_s": round(elapsed, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group)
        errors = sum("error" in row for row in group)
        valid = [row for row in group if "error" not in row]
        empty = sum(row["empty_content"] for row in valid)
        nonempty = [row for row in valid if not row["empty_content"]]
        refused = sum(row["strongreject_string_refusal"] for row in nonempty)
        return {
            "n": total,
            "errors": errors,
            "valid": len(valid),
            "empty": empty,
            "empty_rate": empty / len(valid) if valid else None,
            "nonempty": len(nonempty),
            "string_refusals": refused,
            "string_refusal_rate_nonempty": refused / len(nonempty) if nonempty else None,
            "string_refusal_ci95_wilson": wilson(refused, len(nonempty)),
        }

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "overall": one(rows),
        "by_category": {
            category: one(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
    }


def main() -> int:
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default=MODEL, help="ID exacto expuesto por /v1/models")
    parser.add_argument("--lambdas", default="0,1,1.5,2.5")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--restore-lambda", type=float)
    parser.add_argument(
        "--resume", action="store_true",
        help="Reutiliza brazos completos y ejecuta solo los que faltan",
    )
    parser.add_argument(
        "--skip-idle-wait", action="store_true",
        help="No espera ociosidad antes de un brazo (solo para reanudar el lambda ya activo)",
    )
    parser.add_argument(
        "--max-waiting-before-start", type=int,
        help="Espera sin timeout hasta que la cola sea <= N antes de cambiar lambda",
    )
    parser.add_argument(
        "--max-running-before-start", type=int,
        help="Junto a --max-waiting-before-start, exige tambien running <= N",
    )
    parser.add_argument("--queue-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    MODEL = args.model

    lambdas = [float(value) for value in args.lambdas.split(",")]
    if len(lambdas) < 2 or len(set(lambdas)) != len(lambdas):
        raise SystemExit("--lambdas requiere al menos dos valores unicos")
    if any(lam < 0 or lam > 2.5 for lam in lambdas):
        raise SystemExit("--lambdas solo admite valores entre 0 y 2.5")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout debe ser positivo")
    if args.max_waiting_before_start is not None and args.max_waiting_before_start < 0:
        raise SystemExit("--max-waiting-before-start no puede ser negativo")
    if args.max_running_before_start is not None and args.max_running_before_start < 0:
        raise SystemExit("--max-running-before-start no puede ser negativo")
    if args.max_running_before_start is not None and args.max_waiting_before_start is None:
        raise SystemExit("--max-running-before-start requiere --max-waiting-before-start")
    if args.queue_poll_seconds <= 0:
        raise SystemExit("--queue-poll-seconds debe ser positivo")
    if args.resume:
        if not args.results_dir.is_dir():
            raise SystemExit(f"--resume requiere un directorio existente: {args.results_dir}")
    else:
        args.results_dir.mkdir(parents=True, exist_ok=False)

    dataset_bytes = download_verified(DATASET_URL, DATASET_SHA256)
    markers_bytes = download_verified(MARKERS_URL, MARKERS_SHA256)
    dataset_text = dataset_bytes.decode()
    markers = [line.strip() for line in markers_bytes.decode().splitlines() if line.strip()]
    dataset = list(csv.DictReader(io.StringIO(dataset_text)))
    category_counts = Counter(row["category"] for row in dataset)
    if len(dataset) != 60 or sorted(category_counts.values()) != [10] * 6:
        raise RuntimeError(f"StrongREJECT Small inesperado: n={len(dataset)} {category_counts}")
    dataset_path = args.results_dir / "strongreject_small_dataset.csv"
    markers_path = args.results_dir / "strongreject_string_matching.txt"
    if args.resume:
        if dataset_path.read_bytes() != dataset_bytes:
            raise RuntimeError(f"dataset previo no coincide: {dataset_path}")
        if markers_path.read_bytes() != markers_bytes:
            raise RuntimeError(f"marcadores previos no coinciden: {markers_path}")
    else:
        dataset_path.write_bytes(dataset_bytes)
        markers_path.write_bytes(markers_bytes)

    initial = get_json(args.base, "/admin/refusal_lambda")
    initial_lambda = initial.get("lambda")
    if initial_lambda is None or not initial.get("consistent"):
        raise RuntimeError(f"estado inicial de lambda invalido: {initial}")
    restore_lambda = args.restore_lambda if args.restore_lambda is not None else float(initial_lambda)
    metadata_path = args.results_dir / "metadata.json"
    if args.resume:
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "base": args.base,
            "model": MODEL,
            "lambdas": lambdas,
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"configuracion de resume incompatible: esperado={expected} actual={actual}")
        metadata.setdefault("resume_history", []).append({
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "initial": initial,
            "concurrency": args.concurrency,
            "retries": args.retries,
            "request_timeout": args.request_timeout,
        })
        metadata["restore_lambda"] = restore_lambda
        report_path = args.results_dir / "summary.json"
        report = json.loads(report_path.read_text())
        report["pairwise"] = {}
    else:
        metadata = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "base": args.base,
            "model": MODEL,
            "lambdas": lambdas,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "retries": args.retries,
            "request_timeout": args.request_timeout,
            "dataset": {
                "name": "StrongREJECT Small",
                "commit": DATASET_COMMIT,
                "url": DATASET_URL,
                "sha256": DATASET_SHA256,
                "rows": len(dataset),
                "category_counts": dict(sorted(category_counts.items())),
                "jailbreak": "none",
            },
            "string_evaluator": {
                "name": "StrongREJECT string_matching",
                "commit": MARKERS_COMMIT,
                "url": MARKERS_URL,
                "sha256": MARKERS_SHA256,
                "markers": len(markers),
            },
            "initial": initial,
            "restore_lambda": restore_lambda,
        }
        report = {"arms": {}, "pairwise": {}}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows_by_lambda: dict[str, list[dict[str, Any]]] = {}
    partial_rows_by_lambda: dict[str, list[dict[str, Any]]] = {}

    expected_prompt_ids = {
        hashlib.sha256(row["forbidden_prompt"].encode()).hexdigest()[:16]
        for row in dataset
    }
    for lam in lambdas:
        output_path = args.results_dir / f"responses_lambda_{lam}.jsonl"
        if not args.resume or not output_path.exists():
            continue
        rows = [json.loads(line) for line in output_path.read_text().splitlines() if line]
        prompt_ids = [str(row.get("prompt_id")) for row in rows]
        unique_prompt_ids = set(prompt_ids)
        if (len(rows) > len(dataset)
                or len(unique_prompt_ids) != len(prompt_ids)
                or not unique_prompt_ids.issubset(expected_prompt_ids)):
            raise RuntimeError(
                f"brazo previo corrupto lambda={lam}: "
                f"rows={len(rows)} ids={len(unique_prompt_ids)}")
        if len(rows) == len(dataset) and unique_prompt_ids == expected_prompt_ids:
            rows_by_lambda[str(lam)] = rows
            report["arms"][str(lam)] = summarize(rows)
        else:
            partial_rows_by_lambda[str(lam)] = rows

    try:
        for lam in lambdas:
            if str(lam) in rows_by_lambda:
                print(f"[strongreject] lambda={lam} ya completo; se reutiliza", flush=True)
                continue
            existing_rows = partial_rows_by_lambda.get(str(lam), [])
            existing_ids = {str(row["prompt_id"]) for row in existing_rows}
            pending_dataset = [
                row for row in dataset
                if hashlib.sha256(row["forbidden_prompt"].encode()).hexdigest()[:16]
                not in existing_ids
            ]
            if existing_rows:
                print(
                    f"[strongreject] lambda={lam} parcial {len(existing_rows)}/"
                    f"{len(dataset)}; se reanuda",
                    flush=True,
                )
            if args.max_waiting_before_start is not None:
                wait_for_queue(
                    args.base, args.max_waiting_before_start,
                    args.max_running_before_start, args.queue_poll_seconds)
            elif not args.skip_idle_wait:
                wait_idle(args.base)
            state = set_lambda(args.base, lam)
            print(f"[strongreject] lambda={lam} state={state}", flush=True)
            output_path = args.results_dir / f"responses_lambda_{lam}.jsonl"
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {
                    executor.submit(
                        query_one, args.base, row, markers, args.max_tokens,
                        args.retries, args.request_timeout): row
                    for row in pending_dataset
                }
                rows = list(existing_rows)
                with output_path.open("a") as incremental_output:
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        rows.append(row)
                        incremental_output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        incremental_output.flush()
                        os.fsync(incremental_output.fileno())
                        completed = len(rows)
                        if completed % 10 == 0 or completed == len(dataset):
                            print(
                                f"[strongreject] lambda={lam} {completed}/{len(dataset)}",
                                flush=True,
                            )
            order = {
                hashlib.sha256(row["forbidden_prompt"].encode()).hexdigest()[:16]: index
                for index, row in enumerate(dataset)
            }
            rows.sort(key=lambda row: order[str(row["prompt_id"])])
            with output_path.open("w") as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_by_lambda[str(lam)] = rows
            report["arms"][str(lam)] = summarize(rows)
            (args.results_dir / "summary.json").write_text(
                json.dumps(report, indent=2) + "\n")
            overall = report["arms"][str(lam)]["overall"]
            print(
                f"[strongreject] lambda={lam} refusals={overall['string_refusals']}/"
                f"{overall['nonempty']} empty={overall['empty']} errors={overall['errors']}",
                flush=True,
            )
        for left_index, left_lambda in enumerate(lambdas):
            for right_lambda in lambdas[left_index + 1:]:
                key = f"{left_lambda}_vs_{right_lambda}"
                report["pairwise"][key] = {
                    "left_lambda": left_lambda,
                    "right_lambda": right_lambda,
                    **paired_refusal(
                        rows_by_lambda[str(left_lambda)],
                        rows_by_lambda[str(right_lambda)],
                    ),
                }
        (args.results_dir / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n")
    finally:
        # Restoration is control-plane state, not a latency measurement.  Do it
        # immediately: waiting for an idle production endpoint can otherwise
        # leave a benchmark lambda active for ten extra minutes.
        restored = set_lambda(args.base, restore_lambda)
        metadata["restored"] = restored
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[strongreject] lambda restaurado: {restored}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
