import torch

from kernelscope.oracles import (
    filter_probabilities,
    merge_attention_states,
    online_softmax_reference,
    rmsnorm,
    rmsnorm_with_scale,
    route_gemm_reference,
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
