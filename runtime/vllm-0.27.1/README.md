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
