from ppd.optimizer.ppd_decision_engine import PPDDecisionEngine


def test_decision_metadata_explains_short_append_bypass(tmp_path):
    engine = PPDDecisionEngine(str(tmp_path), base_config="2P_2D")

    decision, metadata = engine.decide_with_metadata(
        turn=2,
        input_tokens=64,
        output_tokens=1024,
        current_qps=3.7,
        context_tokens=10000,
    )

    assert decision is True
    assert metadata["decision_source"] == "short_append_bypass"
    assert metadata["context_class"] == "huge"
    assert metadata["workload_type"] == "very_long_gen"
    assert metadata["selected_qps"] == 4.0


def test_decision_metadata_marks_missing_calibration_default(tmp_path):
    engine = PPDDecisionEngine(str(tmp_path), base_config="2P_2D")

    decision, metadata = engine.decide_with_metadata(
        turn=2,
        input_tokens=1024,
        output_tokens=32,
        current_qps=4.0,
        context_tokens=2048,
    )

    assert decision is True
    assert metadata["decision_source"] == "default_missing_calibration_pair"
    assert metadata["selected_calibration_cell"] == {
        "context_class": "large",
        "workload_type": "huge_paste",
        "qps": 4.0,
    }
    assert metadata["calibration_pair_available"] is False
