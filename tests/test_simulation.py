"""Unit tests for the synthetic scenario generator in simulation.py.

These tests confirm that each demonstration scenario reliably produces the
classification the dashboard promises, using the detection engine as an
independent checker.
"""

import math

import pytest

from detection_engine import (
    MEASUREMENT_KEYS,
    PLAUSIBLE_RANGES,
    VerificationStatus,
    verify,
)
from simulation import (
    DEFAULT_EXPECTED,
    DEFAULT_SEED,
    DEFAULT_TOLERANCES,
    SCENARIOS,
    ProcessSimulator,
    SimulatorConfig,
    generate_sample,
    make_scenario_simulator,
)


def _classify(sample, expected=None, tolerances=None):
    # A sample may carry its own expected response (e.g. the claim-mismatch
    # scenario claims a different process state). Honour it the same way the
    # dashboard does, so the comparison is physically meaningful.
    if expected is None:
        expected = sample.expected
    return verify(
        sample.observed,
        DEFAULT_EXPECTED if expected is None else expected,
        DEFAULT_TOLERANCES if tolerances is None else tolerances,
        sample.claimed_state,
    )


def test_all_five_required_scenarios_exist():
    required = {
        "normal",
        "sensor_mismatch",
        "claim_mismatch",
        "noisy",
        "insufficient",
    }
    assert required.issubset(set(SCENARIOS))


def test_every_scenario_has_human_readable_metadata():
    for sid, meta in SCENARIOS.items():
        assert meta["name"], sid
        assert meta["hint"], sid
        assert meta["cause"], sid


# --------------------------------------------------------------------------- #
# 18 / 1. Normal operation stays NORMAL
# --------------------------------------------------------------------------- #

def test_normal_scenario_is_normal_under_default_config():
    sample = generate_sample("normal")
    assert _classify(sample).status is VerificationStatus.NORMAL


def test_normal_scenario_claim_is_non_empty():
    assert generate_sample("normal").claimed_state.strip()


# --------------------------------------------------------------------------- #
# 12. Noisy measurements inside tolerances stay NORMAL
# --------------------------------------------------------------------------- #

def test_noisy_scenario_is_normal_for_many_seeds():
    for seed in range(25):
        sample = generate_sample("noisy", seed=seed)
        result = _classify(sample)
        assert result.status is VerificationStatus.NORMAL, f"seed={seed}"


def test_noisy_scenario_is_deterministic_for_same_seed():
    first = generate_sample("noisy", seed=7)
    second = generate_sample("noisy", seed=7)
    assert first.observed == second.observed


def test_noisy_scenario_varies_with_seed():
    first = generate_sample("noisy", seed=1)
    second = generate_sample("noisy", seed=2)
    assert first.observed != second.observed


# --------------------------------------------------------------------------- #
# 13-14. The two mismatch scenarios reliably trigger an anomaly
# --------------------------------------------------------------------------- #

def test_sensor_mismatch_triggers_anomaly():
    sample = generate_sample("sensor_mismatch")
    result = _classify(sample)
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY
    assert _ph_failed(result)


def test_sensor_mismatch_is_stable_across_seeds():
    for seed in range(5):
        sample = generate_sample("sensor_mismatch", seed=seed)
        assert _classify(sample).status is VerificationStatus.INTEGRITY_ANOMALY


def test_claim_mismatch_triggers_anomaly():
    sample = generate_sample("claim_mismatch")
    result = _classify(sample)
    assert result.status is VerificationStatus.INTEGRITY_ANOMALY


def test_claim_mismatch_claims_target_reached():
    sample = generate_sample("claim_mismatch")
    assert "target" in sample.claimed_state.lower()
    assert _classify(sample).status is VerificationStatus.INTEGRITY_ANOMALY


def test_mismatch_scenarios_work_for_custom_config():
    expected = {"ph": 7.2, "conductivity": 400.0, "temperature": 20.0}
    tolerances = {"ph": 0.3, "conductivity": 50.0, "temperature": 1.0}
    for sid in ("sensor_mismatch", "claim_mismatch"):
        sample = generate_sample(sid, expected=expected, tolerances=tolerances)
        assert (
            _classify(sample, expected, tolerances).status
            is VerificationStatus.INTEGRITY_ANOMALY
        ), sid


def _ph_failed(result):
    check = next(c for c in result.checks if c.key == "ph")
    return check.within_tolerance is False


# --------------------------------------------------------------------------- #
# 15. Insufficient-data scenario
# --------------------------------------------------------------------------- #

def test_insufficient_scenario_is_insufficient_data():
    sample = generate_sample("insufficient")
    result = _classify(sample)
    assert result.status is VerificationStatus.INSUFFICIENT_DATA
    assert result.status is not VerificationStatus.NORMAL


def test_insufficient_scenario_has_missing_or_nan_values():
    sample = generate_sample("insufficient")
    assert sample.observed["ph"] is None
    assert math.isnan(sample.observed["conductivity"])


# --------------------------------------------------------------------------- #
# Generated readings stay inside plausible physical ranges
# --------------------------------------------------------------------------- #

def test_generated_readings_are_physically_plausible():
    for sid in SCENARIOS:
        sample = generate_sample(sid)
        for key in MEASUREMENT_KEYS:
            value = sample.observed[key]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue  # the insufficient scenario intentionally breaks data
            low, high = PLAUSIBLE_RANGES[key]
            assert low <= value <= high, f"{sid}/{key}"


def test_scenario_metadata_labels_values_as_simulated():
    sample = generate_sample("sensor_mismatch")
    assert "simulat" in sample.injected_cause.lower()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #

def test_unknown_scenario_raises_value_error():
    with pytest.raises(ValueError):
        generate_sample("does_not_exist")


def test_incomplete_configuration_raises_value_error():
    # The stateful simulator validates its tolerances at construction time.
    with pytest.raises(ValueError):
        ProcessSimulator(SimulatorConfig(), tolerances={"ph": 8.0})


def test_scenario_ids_are_ordered():
    assert list(SCENARIOS) == [
        "normal",
        "sensor_mismatch",
        "claim_mismatch",
        "noisy",
        "stale",
        "insufficient",
    ]


# --------------------------------------------------------------------------- #
# Stateful time-series simulation (ProcessSimulator)
# --------------------------------------------------------------------------- #

def _drift_sim(scenario="normal", seed=DEFAULT_SEED, **overrides):
    """Bare simulator config (no scenario faults) - for drift/delay tests."""
    config = SimulatorConfig(scenario_id=scenario, seed=seed, **overrides)
    return ProcessSimulator(config)


def _preset_sim(scenario, seed=DEFAULT_SEED):
    """Simulator with the catalogue preset applied (faults included)."""
    return make_scenario_simulator(scenario, seed=seed)


def test_step_returns_chronological_indices_and_fixed_clock():
    sim = _drift_sim()
    stamps = [sim.step().index for _ in range(5)]
    assert stamps == [1, 2, 3, 4, 5]
    # The simulated clock is fixed and deterministic (not wall-clock).
    times = [s.timestamp for s in sim.samples]
    assert times == sorted(times)
    assert times[0].year == 2026


def test_successive_samples_are_coherent_not_random_jump():
    sim = _drift_sim()
    ph = [sim.step().observed["ph"] for _ in range(40)]
    # Successive true readings wander only slightly; no wild jumps.
    gaps = [abs(ph[i + 1] - ph[i]) for i in range(len(ph) - 1)]
    assert max(gaps) < 1.0


def test_normal_simulator_stays_normal_across_many_samples():
    for seed in range(6):
        sim = _drift_sim(seed=seed)
        for _ in range(60):
            s = sim.step()
            r = verify(
                s.observed, DEFAULT_EXPECTED, DEFAULT_TOLERANCES,
                s.claimed_state, data_quality=s.quality,
            )
            assert r.status is VerificationStatus.NORMAL, (seed, s.index)


def test_sensor_mismatch_anomaly_is_persistent_across_samples():
    sim = _preset_sim("sensor_mismatch")
    for _ in range(40):
        s = sim.step()
        r = verify(
            s.observed, DEFAULT_EXPECTED, DEFAULT_TOLERANCES, s.claimed_state
        )
        assert r.status is VerificationStatus.INTEGRITY_ANOMALY
        assert _ph_failed(r)


def test_stale_scenario_reports_insufficient_data_and_stale_quality():
    sim = _preset_sim("stale")
    for _ in range(11):
        s = sim.step()
        assert s.quality["ph"] == "stale"    # frozen from the first sample
        r = verify(
            s.observed, DEFAULT_EXPECTED, DEFAULT_TOLERANCES,
            s.claimed_state, data_quality=s.quality,
        )
        assert r.status is VerificationStatus.INSUFFICIENT_DATA


def test_insufficient_scenario_alternates_invalid_conductivity():
    sim = _preset_sim("insufficient")
    for i in range(6):
        s = sim.step()
        assert s.quality["ph"] == "missing" and s.observed["ph"] is None
        assert s.quality["conductivity"] == ("invalid" if i % 2 == 0 else "ok")
        r = verify(
            s.observed, DEFAULT_EXPECTED, DEFAULT_TOLERANCES,
            s.claimed_state, data_quality=s.quality,
        )
        assert r.status is VerificationStatus.INSUFFICIENT_DATA


def test_slow_drift_eventually_triggers_anomaly():
    sim = _drift_sim(drift_enabled=True, drift_fraction=0.10)
    first = None
    for _ in range(60):
        s = sim.step()
        r = verify(
            s.observed, DEFAULT_EXPECTED, DEFAULT_TOLERANCES, s.claimed_state
        )
        if r.status is VerificationStatus.INTEGRITY_ANOMALY and first is None:
            first = s.index
    assert first is not None and first > 3


def test_measurement_delay_tracks_older_true_values():
    sim = _drift_sim(delay_samples=3, noise_fraction=0.0, noise_cap_fraction=0.0)
    samples = [sim.step() for _ in range(6)]
    from detection_engine import MEASUREMENT_KEYS as _KEYS
    for t in range(3, 6):
        for k in _KEYS:
            assert abs(samples[t].observed[k] - samples[t - 3].true_values[k]) < 1e-9


def test_simulator_is_reproducible_for_same_seed():
    a, b = _drift_sim("noisy"), _drift_sim("noisy")
    seq_a = [a.step().observed for _ in range(12)]
    seq_b = [b.step().observed for _ in range(12)]
    assert seq_a == seq_b


def test_different_seeds_produce_different_sequences():
    a, b = _drift_sim("noisy", seed=1), _drift_sim("noisy", seed=2)
    assert [a.step().observed for _ in range(6)] != [
        b.step().observed for _ in range(6)
    ]


def test_restart_reproduces_identical_sequence():
    sim = _drift_sim("noisy")
    first = [sim.step().observed for _ in range(8)]
    sim.restart()
    second = [sim.step().observed for _ in range(8)]
    assert first == second


def test_make_scenario_simulator_makes_all_catalogue_scenarios():
    for sid in SCENARIOS:
        sim = make_scenario_simulator(sid)
        s = sim.step()
        assert s.quality and s.observed is not None


def test_noisy_scenario_remains_within_tolerance_across_samples():
    sim = _drift_sim("noisy")
    for _ in range(80):
        s = sim.step()
        for k in MEASUREMENT_KEYS:
            assert abs(s.observed[k] - DEFAULT_EXPECTED[k]) <= DEFAULT_TOLERANCES[k]


def test_claim_mismatch_carries_target_reached_expected_profile():
    sample = generate_sample("claim_mismatch")
    assert "target" in sample.claimed_state.lower()
    assert sample.expected is not None
    assert sample.expected["conductivity"] < DEFAULT_EXPECTED["conductivity"]
