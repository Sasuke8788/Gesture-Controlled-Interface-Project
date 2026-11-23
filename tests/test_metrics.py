import pandas as pd

from src import metrics as mt


def log():
    base = dict(distance="1m", lighting="bright", orientation="up", speed="normal", background="plain",
                mode="SIMULATED", scenario="normal", user_id="u1", mean_fps=20.0)
    rows = [
        dict(trial_type="command", recognized_correctly=True, outcome="success", latency_ms=200, n_executed=1, duration_s=4),
        dict(trial_type="command", recognized_correctly=True, outcome="success", latency_ms=300, n_executed=1, duration_s=4),
        dict(trial_type="command", recognized_correctly=False, outcome="missed", latency_ms=None, n_executed=0, duration_s=4),
        dict(trial_type="command", recognized_correctly=True, outcome="repeated", latency_ms=250, n_executed=2, duration_s=4),
        dict(trial_type="idle", recognized_correctly=False, outcome="false_activation", latency_ms=None, n_executed=2, duration_s=60),
        dict(trial_type="idle", recognized_correctly=False, outcome="clean", latency_ms=None, n_executed=0, duration_s=60),
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_summary_values():
    s = mt.reliability_summary(log())
    assert s["n_command_trials"] == 4 and s["n_idle_trials"] == 2
    assert s["command_success_rate"] == 0.5 and s["missed_command_rate"] == 0.25
    assert s["repeated_command_rate"] == 0.25 and s["recognition_rate"] == 0.75
    assert s["false_activations"] == 2 and abs(s["false_activations_per_min"] - 1.0) < 1e-9
    assert s["latency_ms_median"] == 250 and s["fps_mean"] == 20.0


def test_empty_log_has_no_invented_numbers():
    assert mt.reliability_summary(pd.DataFrame()) == {"n_trials": 0}


def test_no_idle_gives_none_not_zero():
    s = mt.reliability_summary(log().query("trial_type=='command'"))
    assert s["false_activations_per_min"] is None


def test_outcome_classification_and_breakdown():
    assert mt.classify_outcome(1, 0) == "success" and mt.classify_outcome(0, 0) == "missed"
    assert mt.classify_outcome(2, 0) == "repeated" and mt.classify_outcome(1, 1) == "incorrect"
    bd = mt.breakdown(log(), "lighting")
    assert bd.iloc[0]["trials"] == 4
