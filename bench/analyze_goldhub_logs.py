#!/usr/bin/env python3
"""Extract reproducible speed and MTP statistics from Goldhub's published logs."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(
    r"^═══ \[(?P<number>\d+)-(?P<name>[^]]+)] ═══\n"
    r"Max tokens: (?P<max_tokens>\d+) \| Temp: (?P<temperature>[\d.]+)\n"
    r"Duration: (?P<duration>[\d.]+)s \| Prompt: (?P<prompt>\d+) tok "
    r"\| Output: (?P<output>\d+) tok\n"
    r"Speed: (?P<speed>[\d.]+) tok/s \| Finish: (?P<finish>\w+)",
    re.MULTILINE,
)
METRIC_RE = re.compile(
    r"^(?P<time>\d\d:\d\d:\d\d) 🔔\s*(?P<prompt>[\d.]+).*?"
    r"⚡\s*(?P<acceptance>\d+)%.*?"
    r"✎﹏\s*(?P<draft>[\d.]+)\s*➠\s*(?P<accepted>[\d.]+)",
)
PROMPT_RE = re.compile(r"^(?P<time>\d\d:\d\d:\d\d) 🔔\s*(?P<prompt>[\d.]+)")


def parse_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in RESULT_RE.finditer(path.read_text(errors="replace")):
        values = match.groupdict()
        rows.append({
            "number": int(values["number"]),
            "name": values["name"].lower().replace("-", "_"),
            "max_tokens": int(values["max_tokens"]),
            "temperature": float(values["temperature"]),
            "duration_s": float(values["duration"]),
            "prompt_tokens": int(values["prompt"]),
            "output_tokens": int(values["output"]),
            "reported_tok_s": float(values["speed"]),
            "finish": values["finish"],
        })
    if len(rows) != 8:
        raise RuntimeError(f"se esperaban 8 pruebas publicadas, hay {len(rows)}")
    return rows


def parse_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        values = match.groupdict()
        rows.append({
            "time": values["time"],
            "prompt": float(values["prompt"]),
            "instant_acceptance_pct": float(values["acceptance"]),
            "draft_tokens": float(values["draft"]),
            "accepted_tokens": float(values["accepted"]),
        })
    if not rows:
        raise RuntimeError("no se encontraron filas MTP")
    return rows


def parse_prompt_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = PROMPT_RE.search(line)
        if not match or float(match.group("prompt")) <= 0:
            continue
        events.append({"time": match.group("time"), "prompt": float(match.group("prompt"))})
    return events


def prompt_start(events: list[dict[str, Any]], prompt_tokens: int) -> int:
    candidates = [
        index for index, row in enumerate(events)
        if row["prompt"] > 0 and abs(row["prompt"] - prompt_tokens) <= 0.5
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"inicio ambiguo para prompt_tokens={prompt_tokens}: {candidates}")
    return candidates[0]


def mtp_window(metrics: list[dict[str, Any]], events: list[dict[str, Any]],
               start: int) -> dict[str, Any]:
    start_time = events[start]["time"]
    end_time = events[start + 1]["time"] if start + 1 < len(events) else None
    window = [
        row for row in metrics
        if row["time"] >= start_time and (end_time is None or row["time"] < end_time)
    ]
    if not window:
        raise RuntimeError(f"ventana MTP vacia desde {start_time} hasta {end_time}")
    draft = sum(row["draft_tokens"] for row in window)
    accepted = sum(row["accepted_tokens"] for row in window)
    return {
        "start": window[0]["time"],
        "end": window[-1]["time"],
        "samples": len(window),
        "draft_tokens_sampled": draft,
        "accepted_tokens_sampled": accepted,
        "weighted_acceptance": accepted / draft if draft else None,
        "mean_instant_acceptance": statistics.mean(
            row["instant_acceptance_pct"] for row in window) / 100,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logs = args.repo_dir / "quantization_logs"
    results = parse_results(
        logs / "Qwen3.8-27B-INT4-W4A16-AutoRound_bench_results_20260814_160538.log")
    metrics_path = (
        logs / "Qwen3.8-27B-INT4-W4A16-AutoRound_bench_vLLM_Metrics_20260814_160538.log")
    metrics = parse_metrics(metrics_path)
    events = parse_prompt_events(metrics_path)
    for result in results:
        result["mtp"] = mtp_window(
            metrics, events, prompt_start(events, result["prompt_tokens"]))

    draft = sum(row["draft_tokens"] for row in metrics)
    accepted = sum(row["accepted_tokens"] for row in metrics)
    index = json.loads((args.repo_dir / "model.safetensors.index.json").read_text())
    quant_log = (
        logs / "Qwen3.8-27B-INT4-W4A16-AutoRound.QUANTIZATION_LOG.txt"
    ).read_text(errors="replace")
    quantized = re.search(r"Summary: quantized (\d+)/(\d+) in the model", quant_log)
    report = {
        "source": {
            "model": "goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound",
            "published_log_date": "2026-08-14",
        },
        "reported_speed": {
            "tests": results,
            "mean_tok_s": statistics.mean(row["reported_tok_s"] for row in results),
            "median_tok_s": statistics.median(row["reported_tok_s"] for row in results),
            "min_tok_s": min(row["reported_tok_s"] for row in results),
            "max_tok_s": max(row["reported_tok_s"] for row in results),
        },
        "mtp_global_sampled": {
            "samples": len(metrics),
            "draft_tokens_sampled": draft,
            "accepted_tokens_sampled": accepted,
            "weighted_acceptance": accepted / draft,
        },
        "checkpoint": {
            "tensor_bytes": index["metadata"]["total_size"],
            "chat_template_bytes": (args.repo_dir / "chat_template.jinja").stat().st_size,
            "quantized_layers": int(quantized.group(1)) if quantized else None,
            "total_quantizable_layers": int(quantized.group(2)) if quantized else None,
            "mtp_missing_then_rtn_quantized": (
                "Found 15 tensor(s) in the source checkpoint that are absent" in quant_log
                and "Applying WOQ[RTN] to 7 missing Linear weight(s)" in quant_log
            ),
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
