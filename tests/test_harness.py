import torch

from kernelscope.cases import CASES, cases_for
from kernelscope.harness import run_case
from kernelscope.oracles import (
    filter_probabilities,
    merge_attention_states,
    online_softmax_reference,
    rmsnorm,
    rmsnorm_with_scale,
    route_gemm_reference,
)


CANDIDATES = {
    "rmsnorm": rmsnorm,
    "merge_state": merge_attention_states,
    "sampling": filter_probabilities,
    "online_softmax": online_softmax_reference,
    "hpc_rmsnorm_scale": rmsnorm_with_scale,
    "hpc_route_gemm": lambda **inputs: route_gemm_reference(**inputs)[0],
}


def test_all_reference_candidates_pass_all_cases():
    assert sum(len(cases) for cases in CASES.values()) >= 15
    for task_id, candidate in CANDIDATES.items():
        results = [run_case(case, candidate) for case in cases_for(task_id)]
        assert results
        assert all(result.passed for result in results), results


def test_wrong_shape_is_rejected():
    case = cases_for("rmsnorm", "public")[0]
    result = run_case(case, lambda x, weight: x[:, :1])
    assert result.passed is False
    assert result.max_abs_error == float("inf")


def test_unstable_softmax_is_caught():
    def unstable(x, mask=None):
        values = torch.exp(x.float())
        if mask is not None:
            values = values * mask
        return (values / values.sum(-1, keepdim=True)).to(x.dtype)

    case = cases_for("online_softmax", "hidden")[0]
    result = run_case(case, unstable)
    assert result.passed is False
