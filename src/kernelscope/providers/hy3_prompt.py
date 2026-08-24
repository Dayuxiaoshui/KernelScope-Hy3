from __future__ import annotations

import json
import re

from ..models import TaskSpec

CANDIDATE_SIGNATURES: dict[str, str] = {
    "rmsnorm": (
        "def candidate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor\n"
        "x is [tokens, hidden] fp16/bf16, weight is [hidden]. Return the RMS-normalized "
        "output with the same shape and dtype as x."
    ),
    "merge_state": (
        "def candidate(prefix_output, prefix_lse, suffix_output, suffix_lse) "
        "-> tuple[torch.Tensor, torch.Tensor]\n"
        "All four inputs share the leading [tokens, heads] shape; *_output has an extra "
        "head_size dim. Return (merged_output, merged_lse) using numerically stable "
        "log-sum-exp merging; positive-infinity LSE must be treated as an empty state."
    ),
    "sampling": (
        "def candidate(logits, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0) "
        "-> torch.Tensor\n"
        "logits is [batch, vocab] fp32. Return renormalized probabilities after applying "
        "top_k, then top_p, then min_p filtering (skip a filter when it is inactive)."
    ),
    "online_softmax": (
        "def candidate(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor\n"
        "x is [rows, cols]. Return the row-wise softmax (same shape/dtype as x); masked-out "
        "positions (mask == False) must not contribute to the row sum."
    ),
    "hpc_rmsnorm_scale": (
        "def candidate(x, weight, scale, is_moe: bool = False) -> torch.Tensor | tuple\n"
        "x is [tokens, hidden] bf16, weight is [hidden] bf16, scale is a float or fp32 "
        "scale tensor. When is_moe is False, return a single fp8_e4m3fn tensor = "
        "(rmsnorm(x, weight) / scale). When is_moe is True, return "
        "(normalized_fp32, quantized_fp8_scale0, quantized_fp8_scale1)."
    ),
    "hpc_route_gemm": (
        "def candidate(activations, weights) -> torch.Tensor\n"
        "activations is [M,K] bf16, weights is [K,N] fp32. Decompose weights into a bf16 "
        "high part and a bf16-scaled low residual (scale=1/256), and return the fp32 "
        "GEMM output activations @ weights reconstructed from high+low."
    ),
    "silu_mul": (
        "def candidate(x: torch.Tensor) -> torch.Tensor\n"
        "x is [tokens, 2D] fp16/bf16. Let D = x.shape[-1] // 2. Return "
        "silu(x[..., :D]) * x[..., D:2*D], shape [tokens, D], same dtype as x. D may be odd."
    ),
    "fused_add_rmsnorm": (
        "def candidate(x, residual, weight) -> tuple[torch.Tensor, torch.Tensor]\n"
        "x/residual are [tokens, hidden] fp16/bf16/fp32, weight is [hidden]. Return "
        "(rmsnorm(x + residual, weight), x + residual) -- the RMS-normalized sum and the "
        "updated residual, both same shape/dtype as x."
    ),
    "moe_topk_softmax": (
        "def candidate(gating, topk: int, renormalize: bool = False) "
        "-> tuple[torch.Tensor, torch.Tensor]\n"
        "gating is [tokens, experts] fp16/bf16/fp32. Softmax over experts, then take the "
        "top-k weights (fp32) and indices (int32). Break ties the way torch.topk does "
        "(lower index wins). If renormalize is True, divide the top-k weights by their sum."
    ),
    "tiled_gemm_bias": (
        "def candidate(A, B, bias, activation: str = \"relu\") -> torch.Tensor\n"
        "A is [M,K] fp16, B is [K,N] fp16, bias is [N] fp32. Return "
        "activation(fp32(A) @ fp32(B) + bias) cast to fp16, where activation is one of "
        "\"identity\", \"relu\", \"gelu\". Accumulate the GEMM in fp32."
    ),
    "per_token_group_quant": (
        "def candidate(x, group_size: int) -> tuple[torch.Tensor, torch.Tensor]\n"
        "x is [tokens, hidden] bf16/fp16. Split hidden into contiguous groups of group_size "
        "(the last group may be shorter). For each group, scale = max(abs(group)) / 448.0 "
        "(fp32), quantized = (group / scale).to(torch.float8_e4m3fn). Return "
        "(quantized [tokens, hidden] fp8_e4m3fn, scales [tokens, num_groups] fp32)."
    ),
    "flashattention_small": (
        "def candidate(qkv: torch.Tensor) -> torch.Tensor\n"
        "qkv is [3, batch, seq, heads, dim] fp16/bf16 (qkv[0]/qkv[1]/qkv[2] = q/k/v). Return "
        "causal scaled dot-product attention output softmax(q @ k^T / sqrt(dim) + "
        "causal_mask) @ v, shape [batch, seq, heads, dim], same dtype as qkv. Accumulate the "
        "score/softmax/output matmuls in fp32."
    ),
    "paged_kv_gather": (
        "def candidate(cache, page_table, positions) -> torch.Tensor\n"
        "cache is [num_physical_pages, page_size, ...] fp16/bf16. page_table is "
        "[num_logical_pages] int, mapping a logical page index to a physical page index "
        "(need not be contiguous). positions is [num_tokens] int, the logical token position "
        "for each output row (need not be sorted or contiguous). Return output[i] = "
        "cache[page_table[positions[i] // page_size], positions[i] % page_size], shape "
        "[num_tokens, ...], same dtype as cache."
    ),
    "moe_align": (
        "def candidate(topk_ids, num_experts: int, block_size: int = 16) "
        "-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]\n"
        "topk_ids is [tokens, topk] int in [0, num_experts). Flatten to positions "
        "token*topk+slot; group positions by expert in ascending expert-id order, "
        "preserving ascending position order within an expert (this preserves token "
        "multiplicity). Pad each expert's group with sentinel=topk_ids.numel() up to a "
        "multiple of block_size; an expert with zero assignments contributes zero blocks. "
        "Return (sorted_ids [padded_total] int32, expert_ids [num_blocks] int32 giving the "
        "owning expert of each block_size-sized block, num_tokens_post_pad [scalar] int32 "
        "equal to padded_total)."
    ),
    "hpc_rope_norm_store_kv": (
        "def candidate(q, k, v, cos, sin, k_cache, v_cache, page_table, slot_mapping, "
        "q_norm_weight=None, k_norm_weight=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]\n"
        "q/k/v are [tokens, heads, dim] bf16 (a packed batch of prefill+decode tokens across "
        "requests). cos/sin are [tokens, dim] precomputed RoPE angles (NeoX rotate-half "
        "convention: split dim in half into x1/x2, rotated = [-x2, x1], out = x*cos + "
        "rotated*sin). If q_norm_weight/k_norm_weight ([dim] bf16) are given, RMSNorm q/k "
        "before RoPE; otherwise skip normalization. k_cache/v_cache are "
        "[num_physical_pages, page_size, heads, dim]; page_table maps a logical page index "
        "to a physical page index the same way as paged_kv_gather. slot_mapping is [tokens] "
        "int giving each token's logical position, or -1 to skip writing that token "
        "entirely (its cache slot must be left untouched, not zeroed). Return (q_out, "
        "k_cache, v_cache): q_out is the normalized+rotated Q; k_cache/v_cache reflect "
        "rotated K (also normalized if k_norm_weight given) and raw V written into their "
        "mapped slots."
    ),
    "hpc_group_gemm_fp8": (
        "def candidate(A, B, scale, group_sizes) -> torch.Tensor\n"
        "A is [total_M, K] bf16; B is [num_groups, K, N] fp8_e4m3fn; scale is [num_groups] "
        "fp32 per-group dequant scales; group_sizes is [num_groups] int (row counts, "
        "summing to total_M). Group g owns rows A[offset:offset+group_sizes[g]] (offset is "
        "the exclusive prefix sum of group_sizes); its output rows are fp32(A_g) @ "
        "(fp32(B_g) * scale_g). Concatenate all groups' outputs in order; an empty group "
        "(size 0) contributes zero rows. Return [total_M, N] fp32."
    ),
    "hpc_attention_decode": (
        "def candidate(q, k_cache, v_cache, page_table, lengths, page_size: int) -> torch.Tensor\n"
        "q is [batch, heads, dim], one query per request (the newest/decode token -- "
        "attention is trivially causal, it may attend to every key in its own request's "
        "cache). k_cache/v_cache are [num_physical_pages, page_size, heads, dim], addressed "
        "the same way as paged_kv_gather. page_table is [batch, max_logical_pages] int "
        "mapping request b's logical page index to a physical page index; only the first "
        "ceil(lengths[b] / page_size) columns of page_table[b] are read. lengths is [batch] "
        "int, the valid KV length per request (> 0). Return [batch, heads, dim], same dtype "
        "as q, accumulating the score/softmax/output matmuls in fp32."
    ),
}

_SYSTEM_PROMPT = """You are Hy3, a GPU kernel generation model used inside the KernelScope-Hy3 \
evaluation harness. You must respond with ONLY a single JSON object (no markdown fences, no \
prose before or after) matching this schema:

{
  "task_understanding": {<free-form object describing your reading of the task>},
  "steps": [
    {
      "id": "S1",
      "type": "<free-form step type, e.g. spec_analysis|algorithm|numerics|implementation>",
      "claim": "<what this step asserts about the kernel>",
      "depends_on": ["<earlier step ids>"],
      "code_symbols": ["<identifiers in final_kernel that back this claim>"],
      "evidence_expected": ["<zero or more of: masked_load, masked_store, fp32_accumulator, shared_memory, warp_reduction, tensor_core>"]
    }
  ],
  "complexity": {<free-form object, e.g. {"time": "O(n)", "memory": "O(1)"}>},
  "final_kernel": "<a full, self-contained Python source string>",
  "launch_config": {<free-form object, e.g. {"block_size": 128}>}
}

Rules:
- Every value in "evidence_expected" is checked by a static regex over final_kernel, so only \
claim an evidence tag your code actually demonstrates (masked_load/masked_store look for \
mask|boundary|offs..<|tl.load|tl.store; fp32_accumulator looks for float|fp32|torch.float32; \
shared_memory looks for __shared__|shared_memory|smem; warp_reduction looks for \
__shfl|warp|reduce; tensor_core looks for wmma|mma|dot|hmma|wgmma). Do not claim a tag your \
code does not contain, or the step will be marked contradicted.
- "final_kernel" must be plain, self-contained PyTorch source code (it will be exec()'d on \
CPU tensors, not compiled as CUDA/Triton) that defines exactly one top-level function named \
"candidate" with the exact signature given to you for this task. Assume "torch" is already \
imported in the execution namespace.
- Do not read/write files, access the network, or import anything beyond torch.
- Output raw JSON only.
"""

_TRITON_RULES = """\
Rules:
- Every value in "evidence_expected" is checked by a static regex over final_kernel, so only \
claim an evidence tag your code actually demonstrates (masked_load/masked_store look for \
mask|boundary|offs..<|tl.load|tl.store; fp32_accumulator looks for float|fp32|torch.float32; \
shared_memory looks for __shared__|shared_memory|smem; warp_reduction looks for \
__shfl|warp|reduce; tensor_core looks for wmma|mma|dot|hmma|wgmma). Do not claim a tag your \
code does not contain, or the step will be marked contradicted.
- "final_kernel" may be EITHER (a) plain self-contained PyTorch source, OR (b) a real Triton \
GPU kernel: a module-level function decorated with @triton.jit implementing the compute, plus \
a thin Python wrapper function named "candidate" with the exact signature given to you that \
launches the Triton kernel (grid, block/BLOCK_SIZE, etc.) and returns the result. "torch", \
"triton", and "triton.language" (as "tl") are already imported in the execution namespace — do \
not import them yourself. All input tensors are already real CUDA tensors on a single GPU; do \
not call .cuda()/.to("cuda") or move tensors across devices. Correctness against an independent \
oracle is a hard gate checked before any performance measurement; a correct, efficient Triton \
kernel is preferred over a naive PyTorch fallback, but only if it is correct.
- Do not read/write files, access the network, or import anything beyond torch/triton.
- Output raw JSON only.
"""

_SYSTEM_PROMPT_TRITON = _SYSTEM_PROMPT.rsplit("Rules:", 1)[0] + _TRITON_RULES


def build_messages(task: TaskSpec, *, allow_triton: bool = False) -> list[dict]:
    signature = CANDIDATE_SIGNATURES.get(task.task_id, "no fixed signature for this task")
    user_prompt = json.dumps(
        {
            "task_id": task.task_id,
            "title": task.title,
            "difficulty": task.difficulty,
            "description": task.description,
            "input_spec": task.input_spec,
            "output_spec": task.output_spec,
            "constraints": task.constraints,
            "test_plan": task.test_plan,
            "required_candidate_signature": signature,
        },
        ensure_ascii=False,
        indent=2,
    )
    system_prompt = _SYSTEM_PROMPT_TRITON if allow_triton else _SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict:
    stripped = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        snippet = stripped[:2000]
        raise ValueError(f"hy3 response is not valid JSON: {exc}. Response was:\n{snippet}") from exc
