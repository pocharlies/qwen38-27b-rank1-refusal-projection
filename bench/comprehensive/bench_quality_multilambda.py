#!/usr/bin/env python3
"""Run paired lm-eval quality suites over several refusal lambdas.

The served model and the refusal dial are global production state.  This runner
therefore serialises arms, verifies every dial change, keeps completed task
artefacts for resume, and restores the requested production lambda in a
``finally`` block.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_json(url: str, timeout: float = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def set_lambda(base: str, value: float) -> dict[str, Any]:
    request = urllib.request.Request(
        base.rstrip("/") + "/admin/refusal_lambda",
        data=json.dumps({"lambda": value}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
    state = get_json(base.rstrip("/") + "/admin/refusal_lambda")
    got = state.get("lambda")
    if not state.get("consistent") or got is None or abs(float(got) - value) > 1e-9:
        raise RuntimeError(f"lambda inconsistente despues de fijar {value}: {state}")
    return state


def wait_for_queue(base: str, max_waiting: int, max_running: int | None,
                   poll_seconds: float) -> None:
    last: tuple[float, float] | None = None
    while True:
        body = urllib.request.urlopen(
            base.rstrip("/") + "/metrics", timeout=30).read().decode()
        running = waiting = 0.0
        for line in body.splitlines():
            if line.startswith("vllm:num_requests_running{"):
                running += float(line.rsplit(None, 1)[-1])
            elif line.startswith("vllm:num_requests_waiting{"):
                waiting += float(line.rsplit(None, 1)[-1])
        state = (running, waiting)
        if state != last:
            print(
                f"[quality] esperando ventana: running={running:g} "
                f"waiting={waiting:g} umbrales="
                f"running<={max_running} waiting<={max_waiting}",
                flush=True,
            )
            last = state
        if waiting <= max_waiting and (max_running is None or running <= max_running):
            return
        time.sleep(poll_seconds)


def result_exists(results_dir: Path, lam: float, task: str) -> bool:
    return any((results_dir / f"lambda_{lam}" / task).glob("*/results_*.json"))


def quarantine_partial(results_dir: Path, lam: float, task: str) -> None:
    arm_dir = results_dir / f"lambda_{lam}" / task
    if not arm_dir.exists() or result_exists(results_dir, lam, task):
        return
    if not any(arm_dir.rglob("*")):
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = arm_dir.with_name(f"{task}_interrupted_{stamp}")
    arm_dir.rename(destination)
    print(f"[quality] parcial preservado en {destination}", flush=True)


def parse_suites(value: str) -> list[dict[str, Any]]:
    suites: list[dict[str, Any]] = []
    for item in value.split(","):
        task, separator, raw_limit = item.partition(":")
        if not separator or not task or not raw_limit:
            raise argparse.ArgumentTypeError(
                "--suites usa task:limit,task:limit (p. ej. gsm8k:100)")
        limit = int(raw_limit)
        if limit < 1:
            raise argparse.ArgumentTypeError("los limites deben ser positivos")
        suites.append({"task": task, "limit": limit})
    return suites


def run_task(args: argparse.Namespace, lam: float, suite: dict[str, Any]) -> None:
    task = str(suite["task"])
    requested_limit = int(suite["limit"])
    limit = requested_limit
    if task == "mmlu_pro_llama" and args.mmlu_pro_llama_per_category is not None:
        # ``mmlu_pro_llama`` is a 14-task group. lm-eval applies --limit to
        # every category, so --limit 100 would unexpectedly issue 1,400
        # requests. Keep the requested benchmark size in metadata and use an
        # explicit balanced per-category cap for the actual group invocation.
        limit = args.mmlu_pro_llama_per_category
    if result_exists(args.results_dir, lam, task):
        print(f"[quality] lambda={lam} task={task} ya completo; se reutiliza", flush=True)
        return
    quarantine_partial(args.results_dir, lam, task)
    arm_dir = args.results_dir / f"lambda_{lam}" / task
    log_path = args.results_dir / f"lambda_{lam}_{task}.log"
    cmd = [
        args.python, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args",
        (f"model={args.model},"
         f"base_url={args.base.rstrip('/')}/v1/chat/completions,"
         f"num_concurrent={args.concurrency},tokenized_requests=False,"
         f"max_retries={args.retries},timeout={args.request_timeout}"),
        "--apply_chat_template",
        "--tasks", task,
        "--limit", str(limit),
        "--num_fewshot", "0",
        "--gen_kwargs", f"max_gen_toks={args.max_gen_toks},temperature=0",
        "--seed", "0,1234,1234,1234",
        "--output_path", str(arm_dir),
        "--log_samples",
    ]
    environment = os.environ.copy()
    args.hf_home.mkdir(parents=True, exist_ok=True)
    environment.update({
        "OPENAI_API_KEY": "bench",
        "HF_HOME": str(args.hf_home),
        "HF_DATASETS_CACHE": str(args.hf_home / "datasets"),
        "PYTHONUNBUFFERED": "1",
    })
    print(
        f"[quality] lambda={lam} task={task} requested_limit={requested_limit} "
        f"lm_eval_limit={limit}",
        flush=True,
    )
    tail: list[str] = []
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            tail.append(line)
            tail = tail[-80:]
            if "100%" in line or "|Tasks|" in line or "|    Groups" in line:
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"lm-eval fallo lambda={lam} task={task}, rc={returncode}\n" + "".join(tail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True, help="Python del venv con lm-eval")
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--lambdas", default="0,1,1.5,2.5")
    parser.add_argument("--production-lambda", type=float, default=1.0)
    parser.add_argument("--suites", type=parse_suites,
                        default=parse_suites("gsm8k:100,mmlu_pro_llama:100"))
    parser.add_argument("--max-gen-toks", type=int, default=2048)
    parser.add_argument("--mmlu-pro-llama-per-category", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--max-waiting-before-arm", type=int)
    parser.add_argument("--max-running-before-arm", type=int)
    parser.add_argument("--queue-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    lambdas = [float(value) for value in args.lambdas.split(",")]
    if len(lambdas) < 2 or len(lambdas) != len(set(lambdas)):
        raise SystemExit("--lambdas requiere al menos dos valores unicos")
    if any(value < 0 or value > 2.5 for value in lambdas):
        raise SystemExit("--lambdas solo admite valores entre 0 y 2.5")
    if args.max_waiting_before_arm is not None and args.max_waiting_before_arm < 0:
        raise SystemExit("--max-waiting-before-arm no puede ser negativo")
    if args.max_running_before_arm is not None and args.max_running_before_arm < 0:
        raise SystemExit("--max-running-before-arm no puede ser negativo")
    if args.max_running_before_arm is not None and args.max_waiting_before_arm is None:
        raise SystemExit("--max-running-before-arm requiere --max-waiting-before-arm")
    if args.queue_poll_seconds <= 0:
        raise SystemExit("--queue-poll-seconds debe ser positivo")
    if (args.mmlu_pro_llama_per_category is not None
            and args.mmlu_pro_llama_per_category < 1):
        raise SystemExit("--mmlu-pro-llama-per-category debe ser positivo")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.results_dir / "metadata.json"
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base": args.base,
        "model": args.model,
        "lambdas": lambdas,
        "suites": args.suites,
        "max_gen_toks": args.max_gen_toks,
        "temperature": 0,
        "seeds": [0, 1234, 1234, 1234],
        "paired": True,
        "mmlu_pro_llama_per_category": args.mmlu_pro_llama_per_category,
        "mmlu_pro_llama_categories": 14,
        "mmlu_pro_llama_expected_total": (
            14 * args.mmlu_pro_llama_per_category
            if args.mmlu_pro_llama_per_category is not None else None
        ),
    }
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        for key in ("model", "lambdas", "suites"):
            if previous.get(key) != metadata[key]:
                raise RuntimeError(
                    f"no se puede reanudar: metadata {key} difiere "
                    f"({previous.get(key)!r} != {metadata[key]!r})")
        previous["mmlu_pro_llama_per_category"] = args.mmlu_pro_llama_per_category
        previous["mmlu_pro_llama_categories"] = 14
        previous["mmlu_pro_llama_expected_total"] = (
            14 * args.mmlu_pro_llama_per_category
            if args.mmlu_pro_llama_per_category is not None else None
        )
        metadata = previous
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    started = time.monotonic()
    try:
        for lam in lambdas:
            missing = [suite for suite in args.suites
                       if not result_exists(args.results_dir, lam, str(suite["task"]))]
            if not missing:
                print(f"[quality] lambda={lam}: completo; se reutiliza", flush=True)
                continue
            if args.max_waiting_before_arm is not None:
                wait_for_queue(
                    args.base, args.max_waiting_before_arm,
                    args.max_running_before_arm, args.queue_poll_seconds)
            state = set_lambda(args.base, lam)
            print(f"[quality] estado={state}", flush=True)
            try:
                for suite in missing:
                    run_task(args, lam, suite)
            finally:
                restored = set_lambda(args.base, args.production_lambda)
                print(f"[quality] lambda intermedio restaurado={restored}", flush=True)
    finally:
        restored = set_lambda(args.base, args.production_lambda)
        metadata["restored"] = restored
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["elapsed_minutes"] = round((time.monotonic() - started) / 60, 2)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[quality] lambda final={restored}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
