from app.investigation.scenarios import DemoOutcome, get_scenario, get_scenarios


def test_demo_suite_contains_hero_flag_clear_and_context_cases():
    scenarios = get_scenarios()
    outcomes = {scenario.outcome for scenario in scenarios}

    assert DemoOutcome.NOT_ESTABLISHED in outcomes
    assert DemoOutcome.FLAG in outcomes
    assert DemoOutcome.CLEAR in outcomes


def test_non_flag_case_is_explicitly_supported_by_corpus_sources():
    scenario = get_scenario("rtd_7714_2_process_control")

    assert scenario.outcome is DemoOutcome.CLEAR
    assert "28_Temperature_Sensor_Replacement_2018.txt" in scenario.evidence_files
    assert "26_Temperature_Sensor_Calibration_History.csv" in scenario.evidence_files
    assert "no process-control issue" in scenario.outcome_description


def test_hero_case_preserves_not_established_safety_behavior():
    scenario = get_scenario("press07_42_3_basis")

    assert scenario.outcome is DemoOutcome.NOT_ESTABLISHED
    assert "13_Process_Requalification_2014.txt" in scenario.evidence_files
    assert "25_SOP_Cure_Process_v8_Current.txt" in scenario.evidence_files


def test_unknown_scenario_raises_key_error():
    try:
        get_scenario("does_not_exist")
    except KeyError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("Expected unknown scenario to raise KeyError")
