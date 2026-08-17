# Four-lambda benchmark: refusal, quality, MTP and Goldhub

Evaluation date: **2026-08-17**.

This is the expanded evaluation of the runtime rank-1 projection on
`unsloth/Qwen3.8-27B-NVFP4`. It compares λ=0, 1, 1.5 and 2.5 on the same served checkpoint.
The source projection is the per-module profile extracted from
`Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored`; λ=1 reproduces that profile.

## Result

**Use λ=1 for Qwen3.8-27B.**

- λ=1 moves strict StrongREJECT refusals from 32/60 to 0/60, with four answers still
  carrying a disclaimer.
- λ=1 has no statistically detectable quality loss against λ=0 in the tested GSM8K and
  MMLU-Pro samples.
- λ=1.5 removes the remaining disclaimers, but loses 8.0 percentage points on GSM8K and
  26.79 points on MMLU-Pro against λ=0.
- λ=2.5 is not “more uncensored”: refusal returns almost to baseline and generation becomes
  pathologically slow.

The response to λ is **not monotonic**. Extrapolating beyond the measured rank-1 edit can
invert the component instead of removing it.

## Dangerous-prompt refusal

The same 60 StrongREJECT prompts were used in every arm, balanced across six categories.
Prompts and model responses are intentionally not published; aggregate scores are.

Two detectors are reported:

- **Literal matching** flags refusal phrases, including cautionary wording inside an answer.
- **Classifier** is `Human-CentricAI/LLM-Refusal-Classifier` at revision
  `0d9379eaad351ba5a89a3da24ef5b478d3daeff0`. It separates strict refusal, disclaimer with
  an answer, and a normal answer.

| λ | Literal refusals | Literal non-refusals | Strict classifier refusals | No strict refusal | Disclaimer + answer | Normal answer |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48/60 (80.0%) | 12/60 (20.0%) | 32/60 (53.3%) | 28/60 (46.7%) | 21/60 | 7/60 |
| **1** | **9/60 (15.0%)** | **51/60 (85.0%)** | **0/60** | **60/60 (100%)** | 4/60 | 56/60 |
| 1.5 | 1/60 (1.7%) | 59/60 (98.3%) | 0/60 | 60/60 (100%) | 0/60 | 60/60 |
| 2.5 | 47/60 (78.3%) | 13/60 (21.7%) | 29/60 (48.3%) | 31/60 (51.7%) | 20/60 | 11/60 |

Paired exact McNemar tests on strict refusal:

| Comparison | Change in right-hand refusal rate | p (two-sided) |
|---|---:|---:|
| λ=0 → 1 | −53.33 pp | 4.66×10⁻¹⁰ |
| λ=0 → 1.5 | −53.33 pp | 4.66×10⁻¹⁰ |
| λ=0 → 2.5 | −5.00 pp | 0.5811 |
| λ=1 → 2.5 | +48.33 pp | 3.73×10⁻⁹ |

λ=2.5 is statistically indistinguishable from λ=0 on strict refusal. The apparent extra
strength has undone the useful effect.

Full aggregate category breakdowns and Wilson intervals:

- [`strongreject_string_matching.json`](strongreject_string_matching.json)
- [`strongreject_classifier.json`](strongreject_classifier.json)

## General quality

The design is paired across λ and uses fixed seeds. GSM8K has 100 examples. MMLU-Pro has
112 examples, eight from each of its 14 categories.

| Suite | λ=0 | λ=1 | Δ 1−0 | p | λ=1.5 | Δ 1.5−0 | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K (n=100) | 84.0% | 81.0% | −3.0 pp | 0.2500 | 76.0% | **−8.0 pp** | **0.0386** |
| MMLU-Pro (n=112) | 76.79% | 75.00% | −1.79 pp | 0.7905 | 50.00% | **−26.79 pp** | **1.86×10⁻⁹** |

The λ=1 result does not prove universal equivalence, but no quality loss is detectable in
these samples. The λ=1.5 loss is significant in both suites and severe on MMLU-Pro.

λ=2.5 has no quality score. Its first three GSM8K completions each took about five minutes;
the arm was stopped as `generation_runaway` rather than assigning a score to an invalid run.

Data: [`quality_paired_analysis.json`](quality_paired_analysis.json) and
[`lambda_2.5_quality_early_stop.json`](lambda_2.5_quality_early_stop.json).

## MTP acceptance and speed

Here *acceptance* means speculative tokens accepted from the MTP drafter. It is unrelated
to accepting or refusing a dangerous request.

| λ | MTP code | MTP varied | Mean accepted length | Code tok/s | Varied tok/s | TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.01% | 57.93% | 3.340 | 27.13 | 21.51 | 0.376 s |
| **1** | **75.64%** | **56.89%** | **3.269** | **26.14** | **20.75** | **0.375 s** |
| 1.5 | 69.04% | 55.40% | 3.071 | 24.23 | 21.48 | 0.373 s |
| 2.5 | invalid | invalid | invalid | invalid | invalid | invalid |

Against λ=0, λ=1 changes MTP acceptance by −2.37 pp on code and −1.04 pp on varied traffic.
The absolute throughput changes are about −3.6%. This section has one replicate, so small
differences should not be treated as stable performance deltas.

The λ=1.5 arm is partial: four of five clean prompts per family survived the external-load
guard after contaminated retries were discarded. λ=2.5 ran for 1,566 seconds without
completing the ten 600-token prompts and was stopped without assigning a metric.

Data: [`results.json`](results.json) and
[`lambda_2.5_mtp_speed_early_stop.json`](lambda_2.5_mtp_speed_early_stop.json).

## Tool use and long context

| λ | Tooling |
|---:|---:|
| 0 | 7/8 |
| 1 | 7/8 |
| 1.5 | 6/8 |
| 2.5 | invalid |

λ=1 retains the base result. λ=1.5 loses the recovery case. Every valid arm fails the test
whose expected behavior is to reject a tool call, which is consistent with an uncensored
profile.

The planned NIAH 32K/128K run is **inconclusive**, not a pass or fail. It was stopped after
31 minutes of external runtime saturation: five requests running, one queued and no progress
in the completion counter for 60 seconds. See [`niah_early_stop.json`](niah_early_stop.json).

## Comparison with Goldhub's Reddit model

Source: [“We Cracked Qwen3.8-27B: 27GB INT4 that actually thinks (Heretic Edition)”](https://www.reddit.com/r/LLM/comments/1vonckt/we_cracked_qwen3827b_27gb_int4_that_actually/),
published 2026-08-14. Linked model:
`goldhub/Qwen3.8-27B-INT4-W4A16-AutoRound`.

### What its published logs establish

| Published metric | Goldhub |
|---|---:|
| Generation tests | 8 |
| Mean speed | 54.62 tok/s |
| Median speed | 60.86 tok/s |
| Range | 12.65–70.24 tok/s |
| Global weighted MTP acceptance | 69.74% |
| Per-task MTP range | 39.60–90.76% |
| Tensor size | 27,995,740,640 B (26.07 GiB) |
| Quantized layers | 256/607 |

Our mounted `unsloth/Qwen3.8-27B-NVFP4` safetensors total 23,417,592,488 bytes
(21.81 GiB). Goldhub is 19.55% larger, consistent with its documented choice to retain
vision, linear-attention projections, embeddings and the head in FP16/BF16.

The published quantization log also reports 15 source tensors absent and applies RTN to
seven missing MTP linear weights. “MTP support” therefore does not mean those MTP weights
went through the same AutoRound process as the rest of the model.

### MTP comparison is descriptive, not an A/B

| Workload | Ours λ=0 | Ours λ=1 | Ours λ=1.5 | Goldhub published |
|---|---:|---:|---:|---:|
| Code | 78.01% | 75.64% | 69.04% | 67.68% on its code prompt |
| Varied/global | 57.93% | 56.89% | 55.40% | 69.74% global; 39.60–90.76% by task |

Goldhub reports 51.89 tok/s on its code test and 54.62 tok/s on average, against
20.75–27.13 tok/s in our two prompt families. This is **not evidence that Goldhub is twice
as fast**: hardware, runtime, quantization, prompts, temperature, output lengths, batching
and load are all uncontrolled between the two publications.

Its “Zero refusals. Zero apologies” claim is based on one philosophical/creative prompt.
It is not a dangerous-prompt benchmark and cannot be compared with 60 paired StrongREJECT
items per arm. The model card explicitly declines MMLU/GSM8K, so it publishes no general
quality result comparable to this evaluation.

Reproducible extraction: [`goldhub_published_metrics.json`](goldhub_published_metrics.json)
and [`bench/analyze_goldhub_logs.py`](https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection/blob/main/bench/analyze_goldhub_logs.py)
in the GitHub repository.

## Recommendation

1. Keep **Qwen3.8-27B at λ=1**.
2. Do not make λ=1.5 the default: its small refusal-language improvement does not offset the
   measured quality, code-MTP and tool-recovery losses.
3. Do not use λ=2.5: it restores nearly all baseline refusal and triggers runaway generation.
4. Keep **DeepSeek V4 at λ=1.5**. Its projection and calibration are different; λ is not a
   portable scalar between models.

The runtime was restored and independently re-checked at λ=1, consistent across all ranks.
See [`runtime_verification.json`](runtime_verification.json).

## Limitations

- StrongREJECT has 60 prompts per arm; category cells have ten prompts each.
- The classifier can be wrong, which is why literal matches, disclaimers and strict labels
  are all reported.
- GSM8K and MMLU-Pro are fixed subsets, not full-suite runs.
- Speed/MTP has one replicate; λ=1.5 is partial because of external traffic.
- λ=2.5 has no valid quality, MTP, speed, tooling or NIAH score beyond its observed failure.
- Goldhub is evaluated from its published logs, not on the same hardware and runtime.

Machine-readable index: [`results.json`](results.json).
