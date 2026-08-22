from kernelscope.metrics import LocalizationSample, evaluate_localization

def test_localization_metrics_cover_required_outputs():
    rows = [
        LocalizationSample("w1", "S2", "S2", False, False, "NUM", "NUM"),
        LocalizationSample("w2", "S1", "S3", False, False, "MEM", "ALGO"),
        LocalizationSample("r1", None, None, True, True),
        LocalizationSample("r2", None, "S1", True, False),
    ]
    metrics = evaluate_localization(rows)
    assert metrics["error_detection_recall"] == 1.0
    assert metrics["localization_top1"] == 0.5
    assert metrics["correct_result_invalid_process_recall"] == 0.5
    assert 0.0 <= metrics["error_type_macro_f1"] <= 1.0

def test_empty_metrics_are_defined():
    metrics = evaluate_localization([])
    assert metrics["samples"] == 0
    assert metrics["localization_top1"] == 0.0
