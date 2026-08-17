# Comprehensive lambda benchmark harness

These are the scripts used for the 2026-08-17 Qwen3.8-27B evaluation published in
[`hf/benchmarks/2026-08-17/`](../../hf/benchmarks/2026-08-17/README.md).

They operate a **global** `/admin/refusal_lambda` dial. Run them only against an isolated
server or a deliberately scheduled maintenance window. Every runner restores the configured
production lambda in `finally` and verifies the subsequent `GET`; still verify it yourself.

## StrongREJECT

`bench_strongreject_refusal.py` downloads a SHA-256-pinned 60-prompt StrongREJECT Small set,
runs the same prompts at each λ, computes literal refusal matching and writes raw responses
locally. Raw prompts/responses are deliberately excluded from this repository.

```bash
python3 bench_strongreject_refusal.py \
  --base http://SERVER:PORT \
  --model qwen38-27b \
  --lambdas 0,1,1.5,2.5 \
  --results-dir results-strongreject \
  --restore-lambda 1

# Run from a venv containing torch + transformers.
python3 score_refusal_classifier.py results-strongreject
```

The classifier is pinned inside the scorer to
`Human-CentricAI/LLM-Refusal-Classifier@0d9379eaad351ba5a89a3da24ef5b478d3daeff0`.
It writes labels, probabilities and aggregate summaries without copying dangerous prompts or
responses into its classifier artefacts.

## Quality

`bench_quality_multilambda.py` drives `lm-eval` through the OpenAI-compatible endpoint and
restores λ=1 after every arm. The published run used fixed seeds, 100 GSM8K items and a
balanced 8×14=112 MMLU-Pro sample:

```bash
python3 bench_quality_multilambda.py \
  --base http://SERVER:PORT \
  --model qwen38-27b \
  --python /path/to/lm-eval-venv/bin/python \
  --hf-home /path/to/writable/hf-cache \
  --results-dir results-quality \
  --lambdas 0,1,1.5,2.5 \
  --suites gsm8k:100,mmlu_pro_llama:100 \
  --mmlu-pro-llama-per-category 8 \
  --max-gen-toks 2048 \
  --production-lambda 1

python3 analyze_quality_ab.py results-quality --lambdas 0,1,1.5
```

λ=2.5 was manually early-stopped after three independent ~5-minute GSM8K completions. The
published artefact records the stop; no accuracy was assigned.

## MTP, speed, tooling and NIAH

`compare_full.py` alternates λ order, rejects samples contaminated by external traffic and
orchestrates `bench_speed.py`, `bench_tooling.py` and `bench_niah.py`.

```bash
python3 compare_full.py \
  --base http://SERVER:PORT \
  --model qwen38-27b \
  --lambdas 0,1,1.5,2.5 \
  --reps 1 \
  --max-tokens 600 \
  --skip-lambdas 2.5 \
  --skip-niah \
  --results-dir results-mtp
```

The published λ=2.5 MTP/speed arm was stopped separately after 1,566 seconds of runaway
generation. NIAH was stopped after 1,860 seconds because unrelated live traffic saturated the
runtime. Neither arm was assigned a score.

## Goldhub log extraction

[`../analyze_goldhub_logs.py`](../analyze_goldhub_logs.py) parses Goldhub's eight published
generation results and 597 MTP telemetry samples, then reads its model index and quantization
log. It does not infer missing metrics.

```bash
python3 ../analyze_goldhub_logs.py /path/to/goldhub-repo \
  --out goldhub_published_metrics.json
```
