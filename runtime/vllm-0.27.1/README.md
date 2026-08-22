# vLLM 0.27.1 port

This is the exact fail-closed port used to move the Qwen3.8-27B NVFP4 runtime from
vLLM 0.25.2 to the official ARM64 vLLM 0.27.1 image. It preserves the runtime
rank-1 dial, per-request lambda, prefix-cache isolation, tool parser and native MTP
configuration. The patcher checks every upstream anchor and aborts if the base tree
does not match 0.27.1.

Build from the repository root:

```sh
docker buildx build --platform linux/arm64 \
  -f runtime/vllm-0.27.1/Dockerfile \
  -t your-registry/vllm-qwen38-rank1:v0.27.1 --push .
```

## The per-request lambda requires the V2 model runner

`patch_model_runner` wires `v1/worker/gpu/model_runner.py`, the **V2** runner. On
0.27.1 that runner is chosen by architecture, and `Qwen3_5ForConditionalGeneration`
is not among the seven in `default_v2_model_runner_architectures()`; because the
model is flagged hybrid, `is_hybrid and not is_default_v2_architecture` sends it to
the V1 runner, which carries no refusal wiring. The failure is silent and total:
`RefusalState` is never constructed, the per-token buffer never exists, and every
forward falls back to the global scalar, so `cache_salt: "refusal:<x>"` does nothing
while `/admin/refusal_lambda` keeps working.

Measured in production on 2026-08-22, temperature 0, same prompt:

| request | output |
|---|---|
| global dial 0.0 | baseline |
| global dial 1.0 | **different** — the projection itself is fine |
| dial 0.0 + `cache_salt: refusal:1.0` | byte-for-byte the baseline |
| dial 0.0 + `cache_salt: refusal:4.0` | byte-for-byte the baseline |

The Dockerfile now sets `VLLM_USE_V2_MODEL_RUNNER=1`, which short-circuits the
architecture gate. `_validate_v2_model_runner` still runs, so an unsupported
configuration aborts at startup rather than degrading silently; the only feature it
gives up for this deployment is the `thinking_token_budget` request parameter.

Two independent ways to confirm a running pod applies per-request lambda:
`Using V2 Model Runner` and `refusal projection: buffers por rol listos` must both
appear in the engine log, and the same refusal-triggering prompt at temperature 0
must produce different output with `cache_salt: refusal:1.0` than with `refusal:0`.

The deployment configuration used for the A/B stayed unchanged: Unsloth NVFP4,
native MTP `k=3`, 262144 context, 6 sequences and 16384 batched tokens. On the same
single-stream tests, 0.27.1 was functional but not a speed upgrade:

| workload | vLLM 0.25.2 | vLLM 0.27.1 | MTP acceptance, old → new |
|---|---:|---:|---:|
| code | 28.19 tok/s | 26.01 tok/s | 85.25% → 80.34% |
| varied | 21.72 tok/s | 21.33 tok/s | 56.53% → 54.82% |

The full machine-readable result is in
[`hf/benchmarks/2026-08-22/vllm-0.27.1.json`](../../hf/benchmarks/2026-08-22/vllm-0.27.1.json).
The release is therefore published as a compatibility update, not as a performance
recommendation over the previous runtime.
