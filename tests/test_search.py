from kernelscope.search import CandidateFeedback, SearchTrace

def test_invalid_fast_candidate_cannot_win():
    invalid_fast = CandidateFeedback("fast-invalid", True, False, False, performance_ratio=1.2)
    valid = CandidateFeedback("valid", True, True, True, performance_ratio=0.7, boundary_coverage=1.0, claim_consistency=1.0)
    trace = SearchTrace("rmsnorm")
    trace.add_round([invalid_fast, valid])
    assert trace.best().candidate_id == "valid"
    assert invalid_fast.reward() == 0.0

def test_search_trace_serializes_reward_and_rounds():
    trace = SearchTrace("online_softmax")
    trace.add_round([CandidateFeedback("c1", True, True, False, performance_ratio=1.0)])
    payload = trace.to_dict()
    assert payload["best_candidate_id"] == "c1"
    assert payload["rounds"][0][0]["reward"] > 0
