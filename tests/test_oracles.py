import torch

from kernelscope.oracles import (
    causal_attention_reference,
    decode_attention_reference,
    filter_probabilities,
    fused_add_rmsnorm,
    group_gemm_fp8,
    merge_attention_states,
    moe_align_block_size,
    moe_topk_softmax,
    online_softmax_reference,
    paged_gather,
    per_token_group_quant,
    rmsnorm,
    rmsnorm_with_scale,
    rope_norm_store_kv,
    route_gemm_reference,
    silu_mul,
    tiled_gemm_bias,
)


def test_rmsnorm_non_aligned_matches_formula():
    torch.manual_seed(1)
    x = torch.randn(3, 111, dtype=torch.float16)
    weight = torch.randn(111, dtype=torch.float16)
    actual = rmsnorm(x, weight)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()).half()
    torch.testing.assert_close(actual, expected)


def test_rmsnorm_scale_keeps_independent_fp32_branch():
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    fp32, quantized, quantized_2 = rmsnorm_with_scale(x, torch.ones(4, dtype=torch.bfloat16), 0.1, is_moe=True)
    assert fp32.dtype == torch.float32
    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized_2.dtype == torch.float8_e4m3fn
    assert torch.isfinite(quantized.float()).all()
    assert quantized_2.float().abs().mean() < quantized.float().abs().mean()


def test_merge_state_is_symmetric_and_handles_empty_states():
    p = torch.tensor([[[1.0, 3.0]]])
    s = torch.tensor([[[5.0, 7.0]]])
    p_lse = torch.tensor([[0.0]])
    s_lse = torch.tensor([[0.0]])
    out, lse = merge_attention_states(p, p_lse, s, s_lse)
    reverse, reverse_lse = merge_attention_states(s, s_lse, p, p_lse)
    torch.testing.assert_close(out, torch.tensor([[[3.0, 5.0]]]))
    torch.testing.assert_close(out, reverse)
    torch.testing.assert_close(lse, reverse_lse)
    empty, empty_lse = merge_attention_states(p, torch.tensor([[float("inf")]]), s, torch.tensor([[float("inf")]]))
    assert torch.equal(empty, torch.zeros_like(empty))
    assert torch.isneginf(empty_lse).all()


def test_online_softmax_mask_and_shift_invariance():
    x = torch.tensor([[10000.0, 10001.0, -3.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    actual = online_softmax_reference(x, mask)
    shifted = online_softmax_reference(x + 1234, mask)
    torch.testing.assert_close(actual, shifted)
    torch.testing.assert_close(actual[0].sum(), torch.tensor(1.0))
    assert torch.equal(actual[1], torch.zeros(3))


def test_sampling_filter_is_normalized_and_respects_thresholds():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    probs = filter_probabilities(logits, top_k=3, top_p=0.8, min_p=0.1)
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))
    assert torch.count_nonzero(probs) <= 3
    assert probs[0, 0] > 0


def test_route_gemm_decomposition_improves_weight_reconstruction():
    torch.manual_seed(2)
    a = torch.randn(5, 17, dtype=torch.bfloat16)
    w = torch.randn(17, 7, dtype=torch.float32)
    actual, high, low = route_gemm_reference(a, w)
    target = a.float() @ w
    high_only = a.float() @ high.float()
    assert (actual - target).abs().mean() <= (high_only - target).abs().mean()
    reconstructed = high.float() + low.float() / 256
    assert (reconstructed - w).abs().max() < 1e-4


def test_silu_mul_handles_odd_d_and_matches_formula():
    torch.manual_seed(3)
    x = torch.randn(5, 2 * 57, dtype=torch.float16)
    actual = silu_mul(x)
    d = 57
    expected = (torch.nn.functional.silu(x[:, :d].float()) * x[:, d : 2 * d].float()).half()
    assert actual.shape == (5, d)
    torch.testing.assert_close(actual, expected)


def test_fused_add_rmsnorm_matches_add_then_rmsnorm():
    torch.manual_seed(4)
    x = torch.randn(4, 111, dtype=torch.float16)
    residual = torch.randn(4, 111, dtype=torch.float16)
    weight = torch.randn(111, dtype=torch.float16)
    y, new_residual = fused_add_rmsnorm(x, residual, weight)
    torch.testing.assert_close(new_residual, x + residual)
    torch.testing.assert_close(y, rmsnorm(x + residual, weight))


def test_moe_topk_softmax_sums_to_one_without_renormalize_and_renormalizes_when_asked():
    torch.manual_seed(5)
    gating = torch.randn(6, 32)
    values, indices = moe_topk_softmax(gating, topk=4)
    probs = torch.softmax(gating, dim=-1)
    assert values.sum(-1).max() <= 1.0 + 1e-5
    assert indices.dtype == torch.int32
    renorm_values, renorm_indices = moe_topk_softmax(gating, topk=4, renormalize=True)
    torch.testing.assert_close(renorm_values.sum(-1), torch.ones(6))
    assert torch.equal(indices, renorm_indices)
    top_value = probs.max(-1).values
    assert torch.allclose(values.max(-1).values, top_value, atol=1e-6)


def test_tiled_gemm_bias_applies_activation_after_fp32_accumulation():
    torch.manual_seed(6)
    a = torch.randn(17, 33, dtype=torch.float16)
    b = torch.randn(33, 9, dtype=torch.float16)
    bias = torch.randn(9, dtype=torch.float32)
    relu_out = tiled_gemm_bias(a, b, bias, activation="relu")
    identity_out = tiled_gemm_bias(a, b, bias, activation="identity")
    assert relu_out.dtype == torch.float16
    torch.testing.assert_close(relu_out, torch.relu(identity_out.float()).half())
    torch.testing.assert_close(identity_out.float(), (a.float() @ b.float() + bias), atol=2e-3, rtol=2e-3)


def test_per_token_group_quant_handles_tail_group_and_bounds_dequant_error():
    torch.manual_seed(7)
    x = torch.randn(6, 300, dtype=torch.bfloat16)
    group_size = 128
    quantized, scales = per_token_group_quant(x, group_size=group_size)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scales.shape == (6, 3)
    for group in range(3):
        start, end = group * group_size, min((group + 1) * group_size, 300)
        chunk = x[:, start:end].float()
        scale = scales[:, group : group + 1]
        dequantized = quantized[:, start:end].float() * scale
        error = (dequantized - chunk).abs()
        # fp8_e4m3 has 3 mantissa bits: worst-case relative rounding error near a
        # group's own max magnitude is ~2^-3, so bound error against that, not `scale`
        # itself (scale is max_abs/448, far smaller than one quantization step).
        max_abs = chunk.abs().amax(-1, keepdim=True)
        assert (error <= 0.15 * max_abs + 1e-3).all()


def test_causal_attention_masks_future_tokens_and_matches_brute_force_softmax():
    torch.manual_seed(8)
    qkv = torch.randn(3, 2, 5, 2, 4, dtype=torch.float32)
    out = causal_attention_reference(qkv)
    assert out.shape == (2, 5, 2, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    expected = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), v)
    torch.testing.assert_close(out, expected)

    # perturbing the last key/value must not change any earlier query's output
    perturbed = qkv.clone()
    perturbed[1:, :, -1] += 100.0
    out_perturbed = causal_attention_reference(perturbed)
    torch.testing.assert_close(out[:, :-1], out_perturbed[:, :-1])


def test_paged_gather_matches_contiguous_view_and_supports_shuffled_positions():
    torch.manual_seed(9)
    cache = torch.randn(3, 4, 2, dtype=torch.float32)
    contiguous_table = torch.tensor([0, 1, 2])
    flat = cache.reshape(12, 2)

    gathered = paged_gather(cache, contiguous_table, torch.arange(12))
    torch.testing.assert_close(gathered, flat)

    shuffled_positions = torch.tensor([5, 0, 10, 3])
    shuffled = paged_gather(cache, contiguous_table, shuffled_positions)
    torch.testing.assert_close(shuffled, flat[shuffled_positions])

    # non-contiguous page_table: logical page 1 lives at physical page 0
    fragmented_table = torch.tensor([2, 0, 1])
    fragmented = paged_gather(cache, fragmented_table, torch.tensor([4, 5]))
    torch.testing.assert_close(fragmented, cache[0, :2])

    empty = paged_gather(cache, contiguous_table, torch.tensor([], dtype=torch.int64))
    assert empty.shape == (0, 2)


def test_moe_align_block_size_pads_per_expert_preserves_multiplicity_and_skips_empty_experts():
    topk_ids = torch.tensor([[0], [1], [0], [2], [1], [0]], dtype=torch.int32)
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(topk_ids, num_experts=4, block_size=4)
    assert expert_ids.tolist() == [0, 1, 2]  # expert 3 has zero assignments, contributes no block
    assert num_tokens_post_pad.item() == sorted_ids.numel() == 12
    sentinel = topk_ids.numel()
    real_positions = sorted_ids[sorted_ids != sentinel]
    assert sorted(real_positions.tolist()) == [0, 1, 2, 3, 4, 5]

    # topk > 1 with a token selecting the same expert twice: both slots must appear
    topk2_ids = torch.tensor([[0, 0], [1, 0], [0, 1], [2, 2]], dtype=torch.int32)
    sorted_ids2, expert_ids2, num_tokens_post_pad2 = moe_align_block_size(topk2_ids, num_experts=3, block_size=4)
    assert expert_ids2.tolist() == [0, 1, 2]
    assert num_tokens_post_pad2.item() == sorted_ids2.numel() == 12
    sentinel2 = topk2_ids.numel()
    real_positions2 = sorted_ids2[sorted_ids2 != sentinel2]
    # token 0 selected expert 0 twice (flat positions 0 and 1) -- both must be preserved
    assert sorted(real_positions2.tolist()) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_rope_norm_store_kv_scatters_into_fresh_cache_and_skips_marked_slots():
    torch.manual_seed(10)
    tokens, heads, dim = 6, 2, 8
    page_size, num_physical_pages = 4, 2
    q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
    half = dim // 2
    angle = torch.arange(tokens).float().unsqueeze(-1) * torch.ones(half)
    cos = torch.cat([angle.cos(), angle.cos()], dim=-1)
    sin = torch.cat([angle.sin(), angle.sin()], dim=-1)
    k_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=torch.bfloat16)
    v_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=torch.bfloat16)
    k_cache_orig, v_cache_orig = k_cache.clone(), v_cache.clone()
    page_table = torch.tensor([1, 0])
    slot_mapping = torch.tensor([0, 1, -1, 3, 4, -1])

    q_out, k_cache_new, v_cache_new = rope_norm_store_kv(
        q, k, v, cos, sin, k_cache, v_cache, page_table, slot_mapping
    )

    # inputs must not be mutated in place
    torch.testing.assert_close(k_cache, k_cache_orig)
    torch.testing.assert_close(v_cache, v_cache_orig)

    def rope(x):
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos.unsqueeze(1).to(x.dtype) + rotated * sin.unsqueeze(1).to(x.dtype)

    torch.testing.assert_close(q_out, rope(q))
    k_rot = rope(k)

    for i, position in enumerate(slot_mapping.tolist()):
        if position == -1:
            continue
        page, offset = position // page_size, position % page_size
        physical = page_table[page].item()
        torch.testing.assert_close(k_cache_new[physical, offset], k_rot[i])
        torch.testing.assert_close(v_cache_new[physical, offset], v[i])

    # token 2's logical position (2) is never targeted by any valid slot -- untouched
    page, offset = 2 // page_size, 2 % page_size
    physical = page_table[page].item()
    torch.testing.assert_close(k_cache_new[physical, offset], k_cache_orig[physical, offset])
    torch.testing.assert_close(v_cache_new[physical, offset], v_cache_orig[physical, offset])


def test_group_gemm_fp8_matches_per_group_dequantized_matmul_and_skips_empty_groups():
    torch.manual_seed(11)
    group_sizes = torch.tensor([3, 0, 5], dtype=torch.int32)
    k, n = 6, 4
    total_m = int(group_sizes.sum().item())
    A = torch.randn(total_m, k, dtype=torch.bfloat16)
    B = torch.randn(3, k, n).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.3, 0.7, 0.2])

    out = group_gemm_fp8(A, B, scale, group_sizes)
    assert out.shape == (total_m, n)
    assert out.dtype == torch.float32

    expected_group0 = A[0:3].float() @ (B[0].float() * scale[0])
    expected_group2 = A[3:8].float() @ (B[2].float() * scale[2])
    torch.testing.assert_close(out[0:3], expected_group0)
    torch.testing.assert_close(out[3:8], expected_group2)

    all_empty = group_gemm_fp8(A.new_zeros((0, k)), B, scale, torch.zeros(3, dtype=torch.int32))
    assert all_empty.shape == (0, n)


def test_decode_attention_reference_matches_manual_softmax_and_respects_variable_lengths():
    torch.manual_seed(12)
    heads, dim, page_size, num_physical_pages = 2, 4, 4, 3
    k_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=torch.float32)
    v_cache = torch.randn(num_physical_pages, page_size, heads, dim, dtype=torch.float32)
    q = torch.randn(2, heads, dim, dtype=torch.float32)
    page_table = torch.tensor([[0, 1], [2, 0]])
    lengths = torch.tensor([5, 1])

    out = decode_attention_reference(q, k_cache, v_cache, page_table, lengths, page_size=page_size)
    assert out.shape == (2, heads, dim)

    # batch 0: 5 keys spanning physical page 0 (slots 0-3) then physical page 1 (slot 0)
    keys0 = torch.cat([k_cache[0], k_cache[1, :1]], dim=0)
    values0 = torch.cat([v_cache[0], v_cache[1, :1]], dim=0)
    scale = dim ** -0.5
    scores0 = torch.einsum("hd,lhd->hl", q[0], keys0) * scale
    expected0 = torch.einsum("hl,lhd->hd", torch.softmax(scores0, dim=-1), values0)
    torch.testing.assert_close(out[0], expected0)

    # batch 1 attends to a single key (physical page 2, slot 0) -- softmax over one key is trivially 1
    torch.testing.assert_close(out[1], v_cache[2, 0])
