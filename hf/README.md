---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
tags:
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

The dial costs nothing measurable in throughput, and MTP acceptance holds even though the
drafter itself is not projected.

## Limitations

- **General capability is unmeasured** — no MMLU-Pro, GSM8K or HumanEval. No long-context
  retrieval either.
- **Small refusal sample**: 5 triggers, 1 control, 1 rep. Clean separation, not a precise rate.
- **The MTP drafter is not ablated.** Ektome ships no `mtp.*` tensors, so there is no direction
  to extract for it. Acceptance stays high anyway.
- `λ=0` is bit-exact in output but **not free in compute**: the dot product and subtraction run
  in all 128 modules on every token. For zero cost, unset `VLLM_REFUSAL_DIRS`.

## Serving on a DGX Spark: one command

A [`sparkrun`](https://sparkrun.dev) recipe ships next to these vectors —
`qwen38-27b-nvfp4-refusal-dial.yaml`, the exact settings every number above was measured with:

```bash
sparkrun launch qwen38-27b-nvfp4-refusal-dial

curl -XPOST localhost:8000/admin/refusal_lambda -d '{"lambda": 1}'   # ablation on
curl -XPOST localhost:8000/admin/refusal_lambda -d '{"lambda": 0}'   # off, bit-exact
```

It pulls the **stock** `unsloth/Qwen3.8-27B-NVFP4` at a pinned revision — no second checkpoint
anywhere. Set `container:` to your own build of the Dockerfile in the GitHub repo (COPY-only
over an existing vLLM image, so it cross-builds to arm64 from x86 without QEMU).

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
