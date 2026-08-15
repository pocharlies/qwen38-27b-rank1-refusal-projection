# Runtime rank-1 refusal projection for Qwen3.8-27B

Serving an abliterated model **without shipping abliterated weights**: 2.6 MB of direction
vectors, base checkpoint untouched, and the ablation strength λ as a hot-swappable dial with
no restart.

Measured on a DGX Spark GB10 (sm_121, 128 GB unified), vLLM 0.25.2, TP=1, native MTP
speculative decoding k=3, `--max-model-len 65536`.

**Headline result — λ=1 removes refusal completely at no measurable cost:**

| metric | λ=0 | λ=1 | verdict |
|---|---:|---:|---|
| Refusal rate (5 triggers) | **5/5 (100 %)** | **0/5 (0 %)** | eliminated |
| Benign controls falsely refused | 0/1 | 0/1 | classifier sane |
| Tool-calling | OK | OK | no regression |
| Throughput, alternated (tok/s) | 20.2 / 20.4 | 20.6 / 20.3 | indistinguishable (shared GPU — see note) |
| MTP acceptance length | 2.80 median (62 samples), max 4.00 at k=3 | | drafter intact |

Same pod, same prompts, `temperature 0`, seconds apart. λ=0 opens every trigger with
"No puedo generar…" / "No puedo proporcionar…"; λ=1 answers all five. The benign control is
answered in **both** arms, so the classifier is not inflating the number.

> **Do not read ~20 tok/s as this model's speed on a DGX Spark.** Those numbers come from a
> node whose GB10 is *shared* by five workloads — this model (33 GB), a vision model (30 GB),
> two BGE pools and a TTS — which is why the pod runs at `gpu_memory_utilization: 0.35`.
> The λ=0 vs λ=1 comparison is still valid, because both arms ran under identical contention
> seconds apart; only the absolute figure is depressed. On a **dedicated** Spark, an
> independent user running this exact recipe reported **~40 tok/s** (third-party report, not
> measured here). Re-measured on the shared node while writing this note: 18.73 / 17.18 /
> 19.72 tok/s, median 18.73.

---

## 0. In plain terms

**The usual way to run an abliterated model:** download a second, complete copy — another
23 GB. Two copies on disk, mutually exclusive on the same GPU. Want the normal model back?
Stop the server, load different weights, wait. Every in-flight request dies.

**This way:** one checkpoint, plus a **2.6 MB** file next to it. One server process.
Switching is one HTTP call:

```bash
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 1}'   # ablation on
curl -XPOST localhost:8888/admin/refusal_lambda -d '{"lambda": 0}'   # ablation off
```

It takes effect on the **next request**. λ=0 is bit-exact to the unmodified model — when the
dial is at zero it *is* the original.

---

## 1. The identity everything rests on

Editing a weight and projecting the sublayer output are the same function:

```
(W − λ·r̂r̂ᵀW)·x  ≡  W·x − λ·r̂·(r̂ᵀ·W·x)
```

Verified here against the **real abliterated checkpoint**, not against theory:

| module | hook(λ=1) vs ideal projection | λ_eff |
|---|---:|---:|
| `layers.20.linear_attn.out_proj` | **7.893e-16** | 1.002466 |
| `layers.31.self_attn.o_proj` | **7.894e-16** | 0.999810 |

Distance to the baked checkpoint is 1.5e-3 — the BF16 storage floor. Ektome's ablation is
essentially the ideal projection.

Method origin: Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*,
NeurIPS 2024.

---

## 2. Picking the pair: "abliterated" in the name does not make it rank-1

Three published Qwen3.8 ablations were measured before choosing one. Only one is a clean
rank-1 directional edit:

| candidate | rank-1 energy | s₀/s₁ | cos(v₀, u₀ᵀW_base) | verdict |
|---|---:|---:|---:|---|
| **Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored** | **0.9867 – 0.9927** | **32 – 77** | **0.936 – 0.9999** | chosen |
| trohrbaugh/Qwen3.8-27B-heretic-ara | 0.14 – 0.87 | 1.14 – 4.59 | 0.17 – 0.46 | rejected |
| orwelian84/Qwen3.8-27B-OBLITERATUS-Advanced | 0.32 – 0.55 | 1.19 – 1.90 | 0.9997 | rejected |

The pairs are **BF16 against BF16**, so there is no requantization noise to blame. An `s₀/s₁`
of 1.4 means the top singular value barely exceeds the second: no dominant direction exists.
`trohrbaugh` also has modules at `δ ≈ 0.0002` — the fingerprint of a fine-tune, not a
directional ablation. `orwelian84` is projection-*shaped* (cos ≈ 1) but not rank-1.

`tools/probe_rank1_candidates.py` does this in ~500 MB per candidate via HTTP range requests,
without downloading any checkpoint.

### What Ektome edits

All **128** modules that write to the residual stream, layers 0–63 complete:
48 `linear_attn.out_proj` + 16 `self_attn.o_proj` + 64 `mlp.down_proj`.

Its `λ_eff` is **0.999 – 1.291**: it removes the direction *exactly once*. For contrast, the
DeepSeek-V4 baked checkpoint sits at λ_eff ≈ 2.43 — it **inverts** the direction and leaves it
at ~1.4× pointing the other way, which was the measured cause of its acceptance drop. There is
no overshoot to compensate here.

---

## 3. Why the coefficient is per-module

`λ_eff` varies from 0.999 to 1.291 depending on the layer. With a single λ — the mean, 1.0093 —
the error would reach **21.8 %** on some modules and the served profile would not be the one
the author tuned. So the file carries a `__coefs__` tensor and the kernel is

```
y ← y − λ · coef_m · r̂ (r̂ · y)
```

**λ=1 reproduces the source profile exactly**, λ=0 is the bit-exact base, and values in between
genuinely interpolate.

---

## 4. Three things that will silently break this

**1 · The runtime prefix is NOT the checkpoint key.** For the multimodal path vLLM inserts one
extra level:

```
Qwen3_5ForConditionalGeneration(prefix="model")
  → language_model = Qwen3_5ForCausalLM("model.language_model")
    → self.model   = Qwen3_5Model("model.language_model.model")   ← the extra .model.
```

Concatenating the prefix leaves all 128 directions unclaimed. Extract the layer **index**
instead — that is immune to the prefix scheme.

**2 · The hook lives INSIDE the `torch.compile` region.** `Qwen3_5Model` carries
`@support_torch_compile`, whereas DeepSeek's module sits in `splitting_ops` and falls outside.
Inheriting its lazy λ initialisation dies with
`Unsupported context manager: Dynamo does not know how to enter a 'lock'`. The `forward` must be
tensor operations only: no locks, no lazy init, no reading mutable globals. Build both buffers
in `__init__` with `torch.zeros(...)+copy_` — inside vLLM's device context a *new* tensor is born
on the GPU, while an existing one (`torch.frombuffer` over the safetensors, i.e. CPU) is never
moved, because vLLM does not call `.to()` on the model.

**3 · λ must enter the prefix-cache hash key**, and must be a device tensor mutated in place.
A Python scalar gets baked into the captured CUDA graph and changing it does nothing — silently.

### Fail-closed, and what it bought

If any direction in the file is not claimed by a layer, startup **aborts**. Both bugs above were
caught by this before a single request was served. With the "warn and continue" behaviour of the
DeepSeek repo, the first one would have served with **zero layers projected**: λ=1 and λ=0
identical, everything green, and the failure only detectable by measuring the refusal rate by
hand. Cost of the guard: one restart each.

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

---

## 5. One environment trap, not related to the dial

`--attention-backend flashinfer` **does not serve Qwen3.5 on vLLM 0.25.2**. It starts, captures
CUDA graphs, and blows up on the *first real inference*:

```
_build_attention_metadata → flashinfer/prefill.py plan(...)
TypeError: Mismatched number of arguments
```

Serve with `--attention-backend triton_attn`.

How it slipped through: the first smoke test checked `/health`, `/v1/models` and
`/admin/refusal_lambda` — all three green — but **never sent a prompt**. A smoke test without a
generation is not a smoke test.

---

## 6. One-command launch on a DGX Spark (`sparkrun`)

A ready recipe is in [`recipes/`](recipes/qwen38-27b-nvfp4-refusal-dial.yaml) — the same
settings every number on this page was measured with:

```bash
# The image is public — no login, no build step:
#   pocharlies/vllm-qwen38-rank1:20260815   (arm64 / sm_121 only)

sparkrun run qwen38-27b-nvfp4-refusal-dial.yaml

curl -XPOST localhost:8101/admin/refusal_lambda -d '{"lambda": 1}'   # ablation on
curl -XPOST localhost:8101/admin/refusal_lambda -d '{"lambda": 0}'   # off, bit-exact
```

It pulls the **stock** `unsloth/Qwen3.8-27B-NVFP4` at a pinned revision, so there is no second
checkpoint anywhere in the flow. To build the image yourself instead, see
[`deploy/Dockerfile`](deploy/Dockerfile): COPY-only over the public upstream base, so it
cross-builds to arm64 from an x86 host without QEMU (~3 min).

Two things in that file are load-bearing, not stylistic:

- `pre_exec` runs the fail-closed guard **before** serving. If any of the 128 directions is
  unclaimed, it aborts. A half-ablated model raises no error.
- `--attention-backend triton_attn` is not optional on vLLM 0.25.2 (see §5).

`VLLM_REFUSAL_LAMBDA_INIT` is `0.0`: it comes up **censored**, and uncensored is something you
turn on.

---

## 7. Reproducing it

```bash
# 1 — which published ablation is actually rank-1? (~500 MB per candidate, no full download)
python3 tools/probe_rank1_candidates.py

# 2 — directions (streams only the edited tensors over HTTP range requests, ~31 GB)
python3 tools/extract_refusal_dirs_qwen.py \
  --base Qwen/Qwen3.8-27B \
  --abl  Zynerji/Ektome-Qwen3.8-27B-PristinelyUncensored \
  --out  refusal_dirs_qwen38.safetensors \
  --report docs/extraction-report.json

# 3 — offline equivalence (must pass before touching a deployment)
python3 tools/verify_projection.py     # gate: ~1e-16 in float64, λ=0 bit-exact

# 4 — build and serve
docker buildx build --platform linux/arm64 -t <your-registry>/vllm-qwen38-rank1 --push deploy/
VLLM_REFUSAL_DIRS=/opt/refusal/refusal_dirs_qwen38.safetensors \
  vllm serve <clean-checkpoint> --attention-backend triton_attn \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' ...

# 5 — equality gate BEFORE trusting anything: λ=0 vs the image with the dial unset,
#     same prompt, temperature 0. Outputs must match.

# 6 — A/B
python3 bench/bench_refusal.py --base http://<pod>:8888 --model <served-name> --lambdas 0,1
```

---

## 8. What this does **not** establish

- **General capability is unmeasured.** MMLU-Pro, GSM8K, HumanEval were not run. Tool-calling,
  throughput and MTP acceptance are covered; general reasoning is not.
- **Long-context retrieval is unmeasured** on this model. NIAH was not run.
- **The refusal sample is small** — 5 triggers, 1 control, 1 rep. It is a clean separation
  (5/5 → 0/5), not a precise rate.
- **The MTP drafter is not ablated.** Ektome does not ship `mtp.*` tensors at all, so the
  drafter runs unprojected — same as any baked checkpoint. Acceptance stays at 3.29–3.57
  regardless, so aligning it is an unmeasured improvement, not a fix. It is available behind
  `VLLM_REFUSAL_MTP_MODE=last|mean`, defaulting to `off`.

---

## 9. Security: this cuts in an uncomfortable direction

Reducing a model's resistance to instructions reduces its resistance to **injected**
instructions arriving inside untrusted content. Prompt injection and refusal are not
independent failure modes.

- **λ>0 must not share credentials with write-capable tools.** Separate deployment, or enforced
  λ=0 on any request whose context contains scraped content or inbound mail.
- **Keep `/admin/refusal_lambda` off the public ingress.** vLLM cannot enforce this.

The better the dial works, the more the isolation matters.

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

## License

Code: Apache-2.0. The direction vectors are derived from the difference between two publicly
released checkpoints and inherit the Qwen license.
