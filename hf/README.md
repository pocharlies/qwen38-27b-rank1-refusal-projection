---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - uncensored
  - abliterated
  - refusal-direction
  - abliteration
  - activation-steering
  - interpretability
  - vllm
  - qwen3.8
library_name: safetensors
---

# Qwen3.8-27B — rank-1 refusal directions

**2.6 MB instead of a second 23 GB checkpoint.** 128 unit vectors in ℝ⁵¹²⁰ that let you serve
the *clean* `Qwen/Qwen3.8-27B` and turn abliteration on and off at runtime with a single HTTP
call — no restart, no second copy on disk, and `λ=0` bit-exact to the original.

```bash
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 1}'   # ablation on
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 0}'   # off, bit-exact base
```

Since 2026-08-19 λ can also be set **per request**, so one pod serves a normal alias and an
ablated one from the same weights, in the same batch:

```bash
curl -XPOST localhost:8888/v1/chat/completions \
  -d '{"model":"qwen38-27b","messages":[...],"cache_salt":"refusal:1.0"}'
```

Requests without a salt keep using the global dial. This was documented as *dead* until now:
the obvious implementation is silently wrong, because CUDA graph capture bakes in the global
scalar and replay runs no Python — every graph-served decode used the global λ with no
warning. The fix binds a per-role buffer to the module at construction, which also keeps it
compatible with Dynamo's `fullgraph` region (measured: compiles once, `frames=1`, and
mutating the buffer does not recompile). Details and the failing-capable tests are in the
GitHub README.

Code, patch and benchmarks: **https://github.com/pocharlies/qwen38-27b-rank1-refusal-projection**

## What is in the file

`refusal_dirs_qwen38.safetensors` — 128 tensors of `F32[5120]`, plus a `__coefs__` tensor and a
`coef_order` entry in `__metadata__` that pairs them.

| module kind | count |
|---|---:|
| `linear_attn.out_proj` | 48 |
| `self_attn.o_proj` | 16 |
| `mlp.down_proj` | 64 |

Layers 0–63 complete — every sublayer that writes to the residual stream.

The hook is

```
y ← y − λ · coef_m · r̂ (r̂ · y)
```

applied to each sublayer output. `coef_m` is that module's measured `λ_eff`, so **λ=1 reproduces
the source ablation profile exactly** and λ=0 is the untouched base. The per-module coefficient
is not cosmetic: `λ_eff` ranges 0.999–1.291, and collapsing it to a single mean would introduce
up to **21.8 %** error on some modules.

## How they were derived

SVD of `ΔW = W_abl − W_base` in float64, per module, between two publicly released checkpoints:

- base: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- abliterated: [`Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored`](https://huggingface.co/Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored)

Both BF16, so there is no requantization noise in the difference.

### Gates (128 modules)

| gate | value |
|---|---|
| rank-1 energy `s₀²/Σs²` | 0.9867 – 0.9927 |
| `s₀/s₁` | 32.4 – 77.1 |
| `cos(v₀, u₀ᵀW_base)` | 0.936 – 0.9999 (median 0.99987) |
| edit subtracts | 128/128 |
| `λ_eff` | 0.999 – 1.291 |
| projection vs edited weight (float64) | **7.9e-16** |

`λ_eff ≈ 1` matters: this source removes the direction *exactly once*. Baked abliterations often
overshoot — the DeepSeek-V4 reference checkpoint sits at λ_eff ≈ 2.43, i.e. it **inverts** the
direction, which measurably costs speculative-decoding acceptance.

### Two candidates that did not survive measurement

| candidate | rank-1 energy | s₀/s₁ | cos |
|---|---:|---:|---:|
| `trohrbaugh/Qwen3.8-27B-heretic-ara` | 0.14 – 0.87 | 1.14 – 4.59 | 0.17 – 0.46 |
| `orwelian84/Qwen3.8-27B-OBLITERATUS-Advanced` | 0.32 – 0.55 | 1.19 – 1.90 | 0.9997 |

An `s₀/s₁` near 1 means no dominant direction exists. **"Abliterated" in a repo name does not
make the edit rank-1** — measure before you build on it.

## Measured behaviour

Same pod, same prompts, `temperature 0`, seconds between arms:

| | λ=0 | λ=1 |
|---|---:|---:|
| Refusal rate (5 low-harm triggers) | **5/5** | **0/5** |
| Benign control falsely refused | 0/1 | 0/1 |
| Tool-calling | OK | OK |
| Throughput, alternated (tok/s) | 20.2 / 20.4 | 20.6 / 20.3 |
| MTP acceptance length | 2.80 median (62 samples), max 4.00 at k=3 | |

The dial costs nothing measurable in throughput on benign work. MTP acceptance is a more
nuanced story — see below.

**Do not read ~20 tok/s as this model's speed on a DGX Spark.** That figure comes from a node
whose GB10 is *shared* by five workloads — this model (33 GB), a vision model (30 GB), two BGE
pools and a TTS — hence `gpu_memory_utilization: 0.35`. The λ=0 vs λ=1 comparison holds,
because both arms ran under identical contention seconds apart; only the absolute number is
depressed. On a **dedicated** Spark, an independent user running this exact recipe reported
**~40 tok/s** (third-party report, not measured here).

### Expanded four-lambda benchmark (2026-08-17)

The deployment smoke above has now been followed by a paired evaluation at λ=0, 1, 1.5 and
2.5: 60 StrongREJECT prompts per arm, 100 GSM8K items, 112 balanced MMLU-Pro items, MTP/speed
prompt families and an eight-case tool battery.

| metric | λ=0 | λ=1 | λ=1.5 | λ=2.5 |
|---|---:|---:|---:|---:|
| Strict classifier refusals | 32/60 | **0/60** | 0/60 | 29/60 |
| Normal answers / disclaimers | 7 / 21 | **56 / 4** | 60 / 0 | 11 / 20 |
| GSM8K (n=100) | 84% | **81%** | 76% | invalid: runaway |
| MMLU-Pro (n=112) | 76.79% | **75.00%** | 50.00% | invalid: runaway |
| MTP acceptance, code / varied | 78.01 / 57.93% | **75.64 / 56.89%** | 69.04 / 55.40% | invalid |
| Tool battery | 7/8 | **7/8** | 6/8 | invalid |

Against λ=0, the λ=1 quality differences are not statistically detectable in these samples
(GSM8K `p=0.25`, MMLU-Pro `p=0.7905`). λ=1.5 significantly degrades both suites. At λ=2.5,
refusal returns almost to baseline and generation becomes pathologically slow. Abliteration
strength is **not monotonic** after the calibrated edit has been applied once.

**Operating point: λ=1.** Full methodology, Wilson intervals, paired tests, early-stop records,
machine-readable aggregates and the controlled interpretation of Goldhub's Reddit results are
in [`benchmarks/2026-08-17/`](benchmarks/2026-08-17/README.md).

### The drafter is not projected, and on refusal topics that costs ~20%

Measured on the served pod, 300 tokens, `temperature 0`:

| prompt | λ | MTP acceptance |
|---|---:|---:|
| benign (write quicksort) | 0 | 2.84 |
| benign (write quicksort) | 1 | 3.00 |
| **refusal trigger (phishing email)** | **1** | **2.41** |

Acceptance holds on benign work but drops ~20% on exactly the topics someone turns the dial
on for. This was predicted before it was measured: *"the patch only touches the main model,
not the drafter, so on rejected topics the mini-model keeps refusing and acceptance
collapses — the patch has to change the drafter's lambda too."*

**The effect is real. The proposed cause is not.** Projecting the drafter with a backbone
direction (`VLLM_REFUSAL_MTP_MODE=mean`) does not recover it — measured, 3 reps per cell:

| | refusal λ=1 | benign λ=1 | gap |
|---|---:|---:|---:|
| drafter unprojected | 2.41 | 3.00 | **0.59** |
| drafter projected | 2.43 / 2.54 / 2.44 → 2.47 | 2.77 / 3.16 / 3.14 → 3.02 | **0.55** |

0.59 → 0.55 is noise: the benign column alone spans 0.39 across reps.

The control that settles it: at **λ=0** the projection is multiplied by zero, so `mean` and
`off` must be *identical*. They measured 2.94 and 2.84. That 0.10 is the bench's noise floor,
and every "improvement" visible at n=1 sat below it. Do not trust a single sample here.

Best remaining explanation: the backbone already hands the drafter an ablated hidden state,
so little refusal signal survives into the drafter's own layer, and the gap is about how
hard that *content* is to predict rather than about who is ablated.

`off` is the default: it reproduces the source ablation exactly and `mean` buys nothing
measurable.

### …and it is not a defect, it is the price of the content

Sweeping λ on the live dial (no restart — that is what the dial is for), 4 triggers each,
acceptance measured on a long generation for the same refusal topic:

| λ | refusals | acceptance |
|---:|---:|---:|
| 0.3 | 4/4 | 2.72 |
| 0.5 | 3/4 | 2.87 |
| 0.7 | 1/4 | 2.41 |
| **1.0** | **0/4** | 2.61 |

Acceptance is *highest* at the λ where the model still refuses, and drops once it starts
complying. A refusal is formulaic text the drafter predicts easily; the content produced by
complying is novel and it does not. **Acceptance tracks what is being generated, not λ** —
note λ=1.0 scores above λ=0.7, which rules out a monotonic penalty in λ.

So there is no λ that both removes refusal and keeps benign-level acceptance, because the
drop *is* the cost of generating the non-refusal content. Nothing to fix here.

Useful by-product: the dose-response on refusal is clean — 4/4 → 3/4 → 1/4 → 0/4 — and
**λ=1.0 is the operating point**. No reason to go lower, and unlike the DeepSeek-V4 reference
(which needed 1.5 and sat at λ_eff 2.43), no reason to go higher.

### One direction, not 128

All 128 extracted directions are effectively the same vector — cos against layer 63 is
**0.9996 minimum, 1.0000 median**, and `cos(mean, layer63) = 1.0000`. Refusal is mediated by
a single global direction in the residual stream, measured here on a 64-layer hybrid with
linear and full attention interleaved. It also makes `last` and `mean` the same choice.

## Limitations

- **General capability is sampled, not exhausted.** The expanded run covers 100 GSM8K and
  112 balanced MMLU-Pro examples; HumanEval and the full benchmark suites were not run.
- **Refusal is measured on 60 StrongREJECT prompts per arm**, balanced across six categories.
  Category cells still contain only ten prompts. The 5-trigger/1-control table above is the
  earlier deployment smoke, not the final rate estimate.
- **Long-context retrieval is inconclusive.** The planned 32K/128K NIAH run was stopped after
  31 minutes of external runtime saturation and received no score.
- **Speed/MTP has one replicate**, and λ=1.5 retained four of five clean prompts per family
  after external-load retries were exhausted.
- **The MTP drafter is not ablated.** Ektome *does* ship all 15 `mtp.*` tensors, but they are
  **byte-identical to the base** (`mtp.layers.0.self_attn.o_proj` sha256 `9165a16183…`), so
  there is no direction to extract from them. Acceptance stays high anyway. *(Corrected: this
  line previously said the tensors were absent — presence is not ablation.)*
- **The vision tower is untouched, exactly.** 333/333 `model.visual.*` tensors byte-identical
  between base and ablation, including `merger.linear_fc2.weight`, the only one whose output
  reaches the residual stream. The gap is zero, not merely small.
- `λ=0` is bit-exact in output but **not free in compute**: the dot product and subtraction run
  in all 128 modules on every token. For zero cost, unset `VLLM_REFUSAL_DIRS`.

## Serving on a DGX Spark: one command

A [`sparkrun`](https://sparkrun.dev) recipe ships next to these vectors —
`qwen38-27b-nvfp4-refusal-dial.yaml`, the exact settings every number above was measured with:

```bash
sparkrun run qwen38-27b-nvfp4-refusal-dial.yaml

curl -XPOST localhost:8101/admin/refusal_lambda -d '{"lambda": 1}'   # ablation on
curl -XPOST localhost:8101/admin/refusal_lambda -d '{"lambda": 0}'   # off, bit-exact
```

It pulls the **stock** `unsloth/Qwen3.8-27B-NVFP4` at `model_revision: main` — no second
checkpoint anywhere. Set `container:` to your own build of the Dockerfile in the GitHub repo
(COPY-only over an existing vLLM image, so it cross-builds to arm64 from x86 without QEMU).

> Note on the revision: Unsloth super-squashes their repos, so old pinned SHAs go 404.
> Don't pin a sha — `main` is the only revision guaranteed to still exist tomorrow.

Two lines in it are load-bearing: `pre_exec` runs the fail-closed guard **before** serving, and
`--attention-backend triton_attn` is not optional on vLLM 0.25.2 — see below.

## Serving by hand

Requires the vLLM patch in the GitHub repo. Serve the **clean** checkpoint (or a clean
quantization of it — this was validated against `unsloth/Qwen3.8-27B-NVFP4`) with

```
VLLM_REFUSAL_DIRS=/path/refusal_dirs_qwen38.safetensors
VLLM_REFUSAL_LAMBDA_INIT=0.0
--attention-backend triton_attn
```

`flashinfer` does not serve Qwen3.5-architecture models on vLLM 0.25.2: it starts fine and fails
on the first real inference with `plan(): Mismatched number of arguments`.

## Security

Reducing a model's resistance to instructions reduces its resistance to **injected**
instructions arriving inside untrusted content. Prompt injection and refusal route through
overlapping machinery.

- λ>0 should not share credentials with write-capable tools.
- Keep `/admin/refusal_lambda` off any public ingress.

## Citation

```bibtex
@misc{arditi2024refusal,
  title  = {Refusal in Language Models Is Mediated by a Single Direction},
  author = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and
            Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  year   = {2024},
  note   = {NeurIPS 2024},
  eprint = {2406.11717},
  archivePrefix = {arXiv}
}
```

License: Apache-2.0 for the extraction code. The vectors are derived from the difference between
two publicly released checkpoints and inherit the Qwen license.
