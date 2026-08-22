#!/usr/bin/env python3
"""Apply the Qwen38 rank-1 runtime port to the exact vLLM 0.27.1 tree.

The patch is deliberately anchor-based and fail-closed.  A source drift must make the
image build fail instead of silently producing a partially patched vLLM runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace(path: Path, old: str, new: str, *, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(
            f"[vllm-0.27.1-port] {path}: anchor count {found}, expected {count}"
        )
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def patch_qwen(site: Path) -> None:
    path = site / "model_executor/models/qwen3_5.py"
    replace(
        path,
        """from vllm.distributed import (
    get_pp_group,
)
""",
        """from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
""",
    )
    replace(
        path,
        """from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
""",
        """from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.refusal_projection import (
    ROLE_DRAFT as refusal_ROLE_DRAFT,
    ROLE_TARGET as refusal_ROLE_TARGET,
    RefusalProjection,
    is_enabled as refusal_is_enabled,
    resolve_direction as refusal_resolve_direction,
    resolve_mtp_direction as refusal_resolve_mtp_direction,
    verify_all_consumed as refusal_verify_all_consumed,
)
from vllm.sequence import IntermediateTensors
""",
    )
    replace(
        path,
        """    maybe_fuse_shared_experts,
    maybe_prefix,
)
""",
        """    maybe_fuse_shared_experts,
    maybe_prefix,
    sequence_parallel_chunk,
)
""",
    )

    layer_anchor = """            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile"""
    layer_patch = """            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

        self.refusal_attn: nn.Module | None = None
        self.refusal_mlp: nn.Module | None = None
        if refusal_is_enabled():
            attn_sub = (
                "linear_attn.out_proj"
                if self.layer_type == "linear_attention"
                else "self_attn.o_proj"
            )
            lowered_prefix = prefix.lower()
            is_draft = "mtp" in lowered_prefix or "draft" in lowered_prefix
            if is_draft:
                got_attn = refusal_resolve_mtp_direction()
                got_mlp = refusal_resolve_mtp_direction()
            else:
                got_attn = refusal_resolve_direction(prefix, attn_sub)
                got_mlp = refusal_resolve_direction(prefix, "mlp.down_proj")
            projection_device = self.input_layernorm.weight.device
            role = refusal_ROLE_DRAFT if is_draft else refusal_ROLE_TARGET
            if got_attn is not None:
                self.refusal_attn = RefusalProjection(
                    *got_attn, device=projection_device, role=role
                )
            if got_mlp is not None:
                self.refusal_mlp = RefusalProjection(
                    *got_mlp, device=projection_device, role=role
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        # Keep the v0.27.1 Qwen3NextDecoderLayer forward byte-for-byte in shape,
        # inserting the projections only on the raw attention and MLP outputs.
        full_num_tokens = positions.shape[-1]
        input_is_sequence_parallel = (
            self.use_attn_reduce_scatter_for_moe
            and residual is not None
            and hidden_states.shape[0] != full_num_tokens
        )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.refusal_attn is not None:
            hidden_states = self.refusal_attn(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                residual = sequence_parallel_chunk(residual)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.refusal_mlp is not None:
            hidden_states = self.refusal_mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile"""
    replace(path, layer_anchor, layer_patch)

    replace(
        path,
        """        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
""",
        """        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        if refusal_is_enabled() and (
            self.start_layer == 0 and self.end_layer == config.num_hidden_layers
        ):
            refusal_verify_all_consumed()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
""",
    )


def patch_scheduler(site: Path) -> None:
    path = site / "v1/core/sched/output.py"
    replace(
        path,
        "from vllm.config.ec_manager_config import EncoderCacheManagerMetadata\n",
        """from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.refusal_projection import parse_request_lambda as parse_refusal_lambda
""",
    )
    replace(
        path,
        """    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
""",
        """    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
    # Per-request rank-1 strength encoded as cache_salt="refusal:<float>".
    refusal_lambda: float | None = None
""",
    )
    replace(
        path,
        """            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
        )
""",
        """            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
            refusal_lambda=parse_refusal_lambda(request.cache_salt),
        )
""",
    )


def patch_model_runner(site: Path) -> None:
    path = site / "v1/worker/gpu/model_runner.py"
    replace(
        path,
        """from vllm.v1.worker.gpu.lora_utils import (
""",
        """from vllm import refusal_projection
from vllm.v1.worker.gpu.refusal_utils import RefusalState
from vllm.v1.worker.gpu.lora_utils import (
""",
    )
    replace(
        path,
        """        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        self.lora_capture_cases = [0]
""",
        """        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        self.refusal_state = RefusalState(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        self.lora_capture_cases = [0]
""",
    )
    replace(
        path,
        """        self.lora_state.remove_request(req_id)
        return True
""",
        """        self.lora_state.remove_request(req_id)
        self.refusal_state.remove_request(req_idx)
        return True
""",
    )
    replace(
        path,
        """            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
""",
        """            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)
            self.refusal_state.add_request(req_index, new_req_data.refusal_lambda)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
""",
    )
    replace(
        path,
        """                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
""",
        """                self._set_active_loras(*lora_inputs)
            if refusal_projection.is_enabled():
                global_lambda = refusal_projection.get_lambda()
                self.refusal_state.fill_target(
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                    global_lambda,
                )
                self.refusal_state.fill_draft_neutral(global_lambda)
        else:
            if refusal_projection.is_enabled():
                self.refusal_state.fill_neutral(refusal_projection.get_lambda())
            # No actual tokens to run. A dummy run for DP or memory profiling.
""",
    )


def patch_kv_cache(site: Path) -> None:
    path = site / "v1/core/kv_cache_utils.py"
    replace(
        path,
        "from vllm.logger import init_logger\n",
        """from vllm.logger import init_logger
from vllm.refusal_projection import lambda_hash_key as refusal_lambda_hash_key
""",
    )
    replace(
        path,
        """        or (request.cache_salt is not None)
    )
""",
        """        or (request.cache_salt is not None)
        or (refusal_lambda_hash_key() is not None)
    )
""",
    )
    replace(
        path,
        """    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )
""",
        """    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )
    lam_key = refusal_lambda_hash_key()
    refusal_keys: list[str] = (
        [f"refusal_lambda={lam_key}"]
        if start_token_idx == 0 and lam_key is not None
        else []
    )

    extra_keys: list[Any] = (
        lora_extra_keys
        + mm_extra_keys
        + cache_salt_keys
        + prompt_embeds_keys
        + refusal_keys
    )
""",
    )


def patch_worker(site: Path) -> None:
    path = site / "v1/worker/gpu_worker.py"
    replace(
        path,
        """    def sleep(self, level: int = 1) -> None:
""",
        """    def set_refusal_lambda(self, value: float) -> float:
        from vllm.refusal_projection import set_lambda

        applied = set_lambda(float(value))
        logger.info("refusal lambda = %s (rank %s)", applied, self.rank)
        return applied

    def get_refusal_lambda(self) -> float:
        from vllm.refusal_projection import get_lambda

        return get_lambda()

    def sleep(self, level: int = 1) -> None:
""",
    )


def patch_serve(site: Path) -> None:
    path = site / "entrypoints/serve/__init__.py"
    replace(
        path,
        """    attach_tokenize_router(app)


def register_vllm_dev_api_routers(app: FastAPI):
""",
        """    attach_tokenize_router(app)

    from vllm.entrypoints.serve.refusal.api_router import (
        attach_router as attach_refusal_router,
    )

    attach_refusal_router(app)


def register_vllm_dev_api_routers(app: FastAPI):
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    site = args.site.resolve()
    version_file = site / "version.py"
    # The release wheel obtains its version from generated package metadata rather
    # than embedding 0.27.1 in version.py.  The image digest and the exact anchor set
    # below are therefore the source identity gate; Dockerfile separately asserts
    # importlib.metadata.version("vllm") == "0.27.1".
    if not version_file.exists():
        raise SystemExit(f"[vllm-0.27.1-port] unexpected base at {site}")

    patch_qwen(site)
    patch_scheduler(site)
    patch_model_runner(site)
    patch_kv_cache(site)
    patch_worker(site)
    patch_serve(site)
    print("[vllm-0.27.1-port] all anchors applied")


if __name__ == "__main__":
    main()
