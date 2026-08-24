from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    task_id: str
    tier: str
    seed: int
    parameters: dict[str, Any]
    tags: tuple[str, ...] = field(default_factory=tuple)


CASES = {
    "rmsnorm": [
        CaseSpec("rms-public-1024", "rmsnorm", "public", 11, {"tokens": 4, "hidden": 1024, "dtype": "float16"}, ("aligned",)),
        CaseSpec("rms-hidden-111", "rmsnorm", "hidden", 12, {"tokens": 19, "hidden": 111, "dtype": "float16"}, ("tail",)),
        CaseSpec("rms-hidden-bf16", "rmsnorm", "hidden", 13, {"tokens": 38, "hidden": 4096, "dtype": "bfloat16"}, ("production",)),
    ],
    "merge_state": [
        CaseSpec("merge-public", "merge_state", "public", 21, {"tokens": 16, "heads": 8, "head_size": 64}, ("finite",)),
        CaseSpec("merge-hidden-tail", "merge_state", "hidden", 22, {"tokens": 13, "heads": 3, "head_size": 48}, ("tail",)),
        CaseSpec("merge-hidden-inf", "merge_state", "hidden", 23, {"tokens": 4, "heads": 2, "head_size": 32, "inject_inf": True}, ("inf",)),
    ],
    "sampling": [
        CaseSpec("sampling-topk", "sampling", "public", 31, {"batch": 3, "vocab": 128, "top_k": 10}, ("top_k",)),
        CaseSpec("sampling-topp-tail", "sampling", "hidden", 32, {"batch": 2, "vocab": 111, "top_p": 0.1}, ("top_p", "tail")),
        CaseSpec("sampling-joint", "sampling", "hidden", 33, {"batch": 4, "vocab": 257, "top_k": 32, "top_p": 0.8, "min_p": 0.05}, ("joint",)),
    ],
    "online_softmax": [
        CaseSpec("softmax-public", "online_softmax", "public", 41, {"rows": 4, "cols": 128, "dtype": "float16"}, ("aligned",)),
        CaseSpec("softmax-hidden-tail", "online_softmax", "hidden", 42, {"rows": 3, "cols": 1537, "dtype": "float16", "logit_offset": 10000.0}, ("tail", "large_logits")),
        CaseSpec("softmax-hidden-mask", "online_softmax", "hidden", 43, {"rows": 3, "cols": 127, "dtype": "float32", "mask": True}, ("masked", "all_masked_row")),
    ],
    "hpc_rmsnorm_scale": [
        CaseSpec("scaled-rms-public", "hpc_rmsnorm_scale", "public", 51, {"tokens": 4, "hidden": 320, "scale": 2.5}, ("aligned",)),
        CaseSpec("scaled-rms-hidden", "hpc_rmsnorm_scale", "hidden", 52, {"tokens": 17, "hidden": 111, "scale": 0.125}, ("tail", "small_scale")),
    ],
    "hpc_route_gemm": [
        CaseSpec("route-public", "hpc_route_gemm", "public", 61, {"m": 16, "k": 128, "n": 8}, ("aligned",)),
        CaseSpec("route-hidden-tail", "hpc_route_gemm", "hidden", 62, {"m": 5, "k": 127, "n": 7}, ("tail",)),
        CaseSpec("route-hidden-residual", "hpc_route_gemm", "hidden", 63, {"m": 8, "k": 257, "n": 16, "small_residual": True}, ("precision",)),
    ],
    "silu_mul": [
        CaseSpec("silu-public", "silu_mul", "public", 71, {"tokens": 4, "d": 64, "dtype": "float16"}, ("aligned",)),
        CaseSpec("silu-hidden-odd", "silu_mul", "hidden", 72, {"tokens": 5, "d": 57, "dtype": "float16"}, ("odd_d",)),
        CaseSpec("silu-hidden-bf16", "silu_mul", "hidden", 73, {"tokens": 6, "d": 200, "dtype": "bfloat16"}, ("production",)),
    ],
    "fused_add_rmsnorm": [
        CaseSpec("fadd-rms-public", "fused_add_rmsnorm", "public", 81, {"tokens": 4, "hidden": 256, "dtype": "float16"}, ("aligned",)),
        CaseSpec("fadd-rms-hidden-tail", "fused_add_rmsnorm", "hidden", 82, {"tokens": 17, "hidden": 111, "dtype": "float16"}, ("tail",)),
        CaseSpec("fadd-rms-hidden-fp32", "fused_add_rmsnorm", "hidden", 83, {"tokens": 6, "hidden": 320, "dtype": "float32"}, ("fp32",)),
    ],
    "moe_topk_softmax": [
        CaseSpec("moe-topk-public", "moe_topk_softmax", "public", 91, {"tokens": 8, "experts": 64, "topk": 2}, ("aligned",)),
        CaseSpec("moe-topk-hidden-narrow", "moe_topk_softmax", "hidden", 92, {"tokens": 5, "experts": 12, "topk": 10}, ("narrow",)),
        CaseSpec("moe-topk-hidden-renorm", "moe_topk_softmax", "hidden", 93, {"tokens": 6, "experts": 32, "topk": 4, "renormalize": True, "dtype": "bfloat16"}, ("renormalize", "bf16")),
    ],
    "tiled_gemm_bias": [
        CaseSpec("gemm-bias-public", "tiled_gemm_bias", "public", 101, {"m": 128, "k": 128, "n": 128}, ("aligned",)),
        CaseSpec("gemm-bias-hidden-tail", "tiled_gemm_bias", "hidden", 102, {"m": 17, "k": 33, "n": 9}, ("tail",)),
        CaseSpec("gemm-bias-hidden-gelu", "tiled_gemm_bias", "hidden", 103, {"m": 12, "k": 64, "n": 20, "activation": "gelu"}, ("activation",)),
    ],
    "per_token_group_quant": [
        CaseSpec("ptq-public", "per_token_group_quant", "public", 111, {"tokens": 4, "hidden": 512, "group_size": 128}, ("aligned",)),
        CaseSpec("ptq-hidden-tail", "per_token_group_quant", "hidden", 112, {"tokens": 6, "hidden": 300, "group_size": 128}, ("tail",)),
        CaseSpec("ptq-hidden-zero-group", "per_token_group_quant", "hidden", 113, {"tokens": 5, "hidden": 256, "group_size": 64, "zero_group": True}, ("zero_group",)),
    ],
    "flashattention_small": [
        CaseSpec("flash-public", "flashattention_small", "public", 121, {"batch": 2, "seq": 64, "heads": 4, "dim": 64, "dtype": "float16"}, ("aligned",)),
        CaseSpec("flash-hidden-tail", "flashattention_small", "hidden", 122, {"batch": 1, "seq": 127, "heads": 3, "dim": 48, "dtype": "float16"}, ("tail", "causal_boundary")),
        CaseSpec("flash-hidden-bf16", "flashattention_small", "hidden", 123, {"batch": 2, "seq": 32, "heads": 2, "dim": 32, "dtype": "bfloat16"}, ("production", "bf16")),
    ],
    "paged_kv_gather": [
        CaseSpec("pk-public-contiguous", "paged_kv_gather", "public", 131, {"num_physical_pages": 4, "page_size": 16, "heads": 2, "dim": 16, "dtype": "float16", "page_table": [0, 1, 2], "positions": list(range(48))}, ("aligned", "contiguous")),
        CaseSpec("pk-hidden-fragmented-partial", "paged_kv_gather", "hidden", 132, {"num_physical_pages": 4, "page_size": 16, "heads": 2, "dim": 8, "dtype": "float16", "page_table": [3, 0, 1], "positions": list(range(40))}, ("fragmented", "partial_page")),
        CaseSpec("pk-hidden-shuffled-empty", "paged_kv_gather", "hidden", 133, {"num_physical_pages": 2, "page_size": 8, "heads": 1, "dim": 4, "dtype": "float16", "page_table": [1, 0], "positions": [5, 0, 10, 3, 15, 8, 12, 1]}, ("shuffled_positions",)),
        CaseSpec("pk-hidden-empty-sequence", "paged_kv_gather", "hidden", 134, {"num_physical_pages": 2, "page_size": 8, "heads": 1, "dim": 4, "dtype": "float16", "page_table": [0, 1], "positions": []}, ("empty_sequence",)),
    ],
    "moe_align": [
        CaseSpec("moe-align-public-balanced", "moe_align", "public", 141, {"topk_ids": [[0], [1], [2], [3], [0], [1], [2], [3]], "num_experts": 4, "block_size": 4}, ("aligned", "balanced")),
        CaseSpec("moe-align-hidden-empty-expert", "moe_align", "hidden", 142, {"topk_ids": [[0], [1], [0], [2], [1], [0]], "num_experts": 4, "block_size": 4}, ("empty_expert",)),
        CaseSpec("moe-align-hidden-topk2-same-expert", "moe_align", "hidden", 143, {"topk_ids": [[0, 0], [1, 0], [0, 1], [2, 2]], "num_experts": 3, "block_size": 4}, ("topk_gt_1", "same_expert")),
    ],
    "hpc_rope_norm_store_kv": [
        CaseSpec("rope-public", "hpc_rope_norm_store_kv", "public", 151, {"tokens": 16, "heads": 4, "dim": 32, "page_size": 16, "num_physical_pages": 1, "page_table": [0]}, ("aligned", "no_norm")),
        CaseSpec("rope-hidden-norm-fragmented", "hpc_rope_norm_store_kv", "hidden", 152, {"tokens": 12, "heads": 3, "dim": 16, "page_size": 8, "num_physical_pages": 2, "page_table": [1, 0], "use_norm": True}, ("norm", "fragmented")),
        CaseSpec("rope-hidden-decode-skip", "hpc_rope_norm_store_kv", "hidden", 153, {"tokens": 10, "heads": 2, "dim": 16, "page_size": 8, "num_physical_pages": 2, "page_table": [0, 1], "skip_indices": [1, 3, 5, 7, 9]}, ("decode", "skip_slots")),
        CaseSpec("rope-hidden-tail", "hpc_rope_norm_store_kv", "hidden", 154, {"tokens": 9, "heads": 5, "dim": 8, "page_size": 4, "num_physical_pages": 3, "page_table": [2, 0, 1]}, ("tail", "partial_page")),
    ],
    "hpc_group_gemm_fp8": [
        CaseSpec("group-gemm-public-balanced", "hpc_group_gemm_fp8", "public", 161, {"group_sizes": [8, 8, 8, 8], "k": 32, "n": 16}, ("aligned", "balanced")),
        CaseSpec("group-gemm-hidden-skewed", "hpc_group_gemm_fp8", "hidden", 162, {"group_sizes": [3, 17, 6], "k": 24, "n": 12}, ("skewed",)),
        CaseSpec("group-gemm-hidden-empty-group", "hpc_group_gemm_fp8", "hidden", 163, {"group_sizes": [5, 0, 9], "k": 16, "n": 8}, ("empty_group",)),
    ],
    "hpc_attention_decode": [
        CaseSpec("decode-public-uniform", "hpc_attention_decode", "public", 171, {"batch": 4, "heads": 4, "dim": 32, "dtype": "float16", "page_size": 8, "num_physical_pages": 8, "lengths": [16, 16, 16, 16], "page_table": [[0, 1], [2, 3], [4, 5], [6, 7]]}, ("aligned", "uniform_lengths")),
        CaseSpec("decode-hidden-mixed-lengths", "hpc_attention_decode", "hidden", 172, {"batch": 3, "heads": 2, "dim": 16, "dtype": "float16", "page_size": 4, "num_physical_pages": 6, "lengths": [3, 7, 10], "page_table": [[0, 0, 0], [1, 2, 0], [3, 4, 5]]}, ("mixed_lengths",)),
        CaseSpec("decode-hidden-short-request", "hpc_attention_decode", "hidden", 173, {"batch": 2, "heads": 3, "dim": 8, "dtype": "float16", "page_size": 4, "num_physical_pages": 3, "lengths": [1, 4], "page_table": [[0], [1]]}, ("short_request",)),
        CaseSpec("decode-hidden-fragmented-bf16", "hpc_attention_decode", "hidden", 174, {"batch": 2, "heads": 2, "dim": 16, "dtype": "bfloat16", "page_size": 4, "num_physical_pages": 4, "lengths": [5, 8], "page_table": [[3, 1], [0, 2]]}, ("fragmented", "bf16")),
    ],
}


def cases_for(task_id: str, tier: str | None = None) -> list[CaseSpec]:
    cases = CASES.get(task_id, [])
    return [case for case in cases if tier is None or case.tier == tier]
