from kernelscope.execution_diagnosis import classify_case_failure, classify_execution_errors


def test_passed_case_has_no_classification_needed():
    assert classify_execution_errors([{"case_id": "a", "passed": True, "max_abs_error": 0.0, "detail": "tolerance=0.01"}]) == []


def test_shape_exception_is_impl():
    report = {"case_id": "a", "passed": False, "max_abs_error": float("inf"),
               "detail": "RuntimeError: The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 1"}
    assert classify_case_failure(report) == "IMPL.shape_mismatch"


def test_inf_error_without_exception_is_spec_violation():
    report = {"case_id": "a", "passed": False, "max_abs_error": float("inf"), "detail": "tolerance=1e-06"}
    assert classify_case_failure(report) == "SPEC.output_contract_violation"


def test_near_miss_is_numerical_precision():
    report = {"case_id": "a", "passed": False, "max_abs_error": 1.9e-5, "detail": "tolerance=1e-05"}
    assert classify_case_failure(report) == "NUM.precision_residual"


def test_far_off_finite_error_is_algo():
    report = {"case_id": "a", "passed": False, "max_abs_error": 5.0, "detail": "tolerance=1e-05"}
    assert classify_case_failure(report) == "ALGO.result_mismatch"


def test_index_exception_is_mem():
    report = {"case_id": "a", "passed": False, "max_abs_error": float("inf"),
               "detail": "IndexError: index 10 is out of bounds for dimension 0 with size 5"}
    assert classify_case_failure(report) == "MEM.out_of_bounds"


def test_classify_execution_errors_dedupes_and_skips_passed():
    reports = [
        {"case_id": "a", "passed": True, "max_abs_error": 0.0, "detail": "tolerance=0.01"},
        {"case_id": "b", "passed": False, "max_abs_error": float("inf"), "detail": "tolerance=0.01"},
        {"case_id": "c", "passed": False, "max_abs_error": float("inf"), "detail": "tolerance=0.01"},
    ]
    assert classify_execution_errors(reports) == ["SPEC.output_contract_violation"]
