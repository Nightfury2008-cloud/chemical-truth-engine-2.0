"""Stateful time-series simulator for Chemical Truth Engine 2.0.

Everything in this module is a SOFTWARE SIMULATION. No real chemical process,
sensor, pump or other hardware is involved. The values produced here are
synthetic examples and are NOT experimental measurements.

How the simulation works
------------------------
:class:`ProcessSimulator` keeps a small process state and advances it one
sample at a time via :meth:`ProcessSimulator.step`. Each new sample depends
on the previous state, so the output forms coherent trends instead of
unrelated random numbers:

* a bounded, mean-reverting "wander" keeps the underlying process values
  near an illustrative profile (see process_profiles.py);
* small bounded Gaussian sensor noise is added on top;
* optional slow sensor drift, optional measurement delay (a value delayed by
  N samples), and optional missing / invalid / stale readings can be enabled.

Determinism
-----------
All randomness comes from a seeded ``random.Random``. The same
(configuration, seed) pair always produces the same sequence of samples.
Timestamps use a fixed simulated clock (not the wall clock), so sequences
are reproducible.

HONESTY NOTES
-------------
* Process profiles are illustrative demo values (see process_profiles.py).
  No acid-base chemistry is modelled and no universal dose-to-pH formula is
  used or implied.
* The "injected cause" of a scenario is known only because this simulator
  introduces it on purpose. The detection engine can only compare numbers;
  it cannot infer WHY a mismatch occurred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite
from random import Random
from typing import Any, Mapping

from detection_engine import MEASUREMENT_KEYS, PLAUSIBLE_RANGES
from process_profiles import expected_for_state, profile_label

__all__ = [
    "DEFAULT_EXPECTED",
    "DEFAULT_SEED",
    "DEFAULT_TOLERANCES",
    "PRESETS",
    "SCENARIOS",
    "ProcessSimulator",
    "Sample",
    "ScenarioSample",
    "SimulatorConfig",
    "generate_sample",
    "make_scenario_simulator",
    "scenario_ids",
]

# --------------------------------------------------------------------------- #
# Synthetic software-demo defaults (NOT calibrated, NOT laboratory measurements)
# --------------------------------------------------------------------------- #

#: Expected values for the default claimed state. Kept identical to the
#: "steady_operation" process profile so the dashboard defaults stay in sync.
DEFAULT_EXPECTED: dict[str, float] = expected_for_state("steady_operation")

DEFAULT_TOLERANCES: dict[str, float] = {
    "ph": 0.5,
    "conductivity": 80.0,
    "temperature": 2.0,
}

#: Base seed so demo runs are reproducible by default.
DEFAULT_SEED = 20261018

#: Data-quality flags used by samples (mirrors detection_engine vocabulary).
QUALITY_OK = "ok"
QUALITY_MISSING = "missing"
QUALITY_INVALID = "invalid"
QUALITY_STALE = "stale"

_CLAIM_STEADY = "Steady operation"
_CLAIM_TARGET_REACHED = "Target state reached"

#: Simulated clock start (fixed, so sequences are reproducible).
_SIM_START = datetime(2026, 1, 1, 12, 0, 0)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Sample:
    """One timestamped sample produced by the stateful simulator."""

    index: int
    timestamp: datetime                 # simulated clock, not wall time
    process_state: str                  # profile the process actually follows
    claimed_state: str                  # what the simulated controller claims
    true_values: dict[str, float]       # underlying illustrative process values
    observed: dict[str, float | None]   # what the "sensors" report
    quality: dict[str, str]             # ok | missing | invalid | stale
    expected: dict[str, float] | None   # expected response for the CLAIM
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioSample:
    """One synthetic verification sample (single-step convenience view)."""

    scenario_id: str
    scenario_name: str
    injected_cause: str
    claimed_state: str
    observed: dict[str, float | None]
    notes: str
    quality: dict[str, str] | None = None
    expected: dict[str, float] | None = None


@dataclass
class SimulatorConfig:
    """Configuration for :class:`ProcessSimulator`.

    All "fraction" parameters are fractions of the configured measurement
    tolerance, which keeps each scenario's intent stable even when the user
    changes tolerances.
    """

    scenario_id: str = "normal"
    seed: int = DEFAULT_SEED
    sample_interval_seconds: float = 5.0
    start_time: datetime = field(default_factory=lambda: _SIM_START)
    # Sensor model ---------------------------------------------------------
    noise_fraction: float = 0.10        # gaussian sigma, as fraction of tolerance
    noise_cap_fraction: float = 0.50    # hard cap on |noise|, fraction of tolerance
    wander_fraction: float = 0.15       # per-step process wander, fraction of tolerance
    wander_max_fraction: float = 0.30   # cap on |wander|, fraction of tolerance
    drift_enabled: bool = False
    drift_fraction: float = 0.05        # slow drift added per step, fraction of tolerance
    delay_samples: int = 0              # report the value from N steps ago
    # Fault injection ------------------------------------------------------
    sensor_bias_fractions: dict[str, float] = field(default_factory=dict)
    missing_probability: float = 0.0
    invalid_probability: float = 0.0
    missing_keys: frozenset[str] = frozenset()      # missing on every step
    invalid_odd_keys: frozenset[str] = frozenset()  # invalid on odd steps
    stale_from: dict[str, int] = field(default_factory=dict)  # key -> step#
    stale_probability: float = 0.0
    stale_hold_samples: int = 3
    # Claim / state --------------------------------------------------------
    process_state_id: str = "steady_operation"
    claimed_state: str = _CLAIM_STEADY
    expected_state_id: str | None = None  # profile for the expected response
    note: str = ""


# --------------------------------------------------------------------------- #
# Scenario catalogue and presets
# --------------------------------------------------------------------------- #

SCENARIOS: dict[str, dict[str, str]] = {
    "normal": {
        "name": "1. Normal operation",
        "hint": (
            "No fault is injected. The process wanders slightly around its "
            "profile and the sensor adds only small noise, so every reading "
            "stays inside its tolerance."
        ),
        "cause": (
            "None. The simulator deliberately kept readings inside all "
            "configured tolerances, so a correct engine reports NORMAL."
        ),
    },
    "sensor_mismatch": {
        "name": "2. Sensor-data inconsistency",
        "hint": (
            "A constant +3x-tolerance bias is applied to the pH sensor "
            "while the controller claim stays unchanged - a classic "
            "miscalibrated or manipulated sensor reading."
        ),
        "cause": (
            "Simulator-injected sensor bias: the pH sensor reports about "
            "three tolerances high on every sample. This cause is known "
            "only because the simulator introduced it."
        ),
    },
    "claim_mismatch": {
        "name": "3. Controller-claim inconsistency",
        "hint": (
            "The controller claims 'Target state reached' while the process "
            "is actually still in the steady state, so the readings "
            "disagree with the expected response for the claimed state."
        ),
        "cause": (
            "Simulator-injected claim mismatch: the claim points at the "
            "'target reached' profile but the process follows the steady "
            "profile. This cause is known only because the simulator "
            "introduced it."
        ),
    },
    "noisy": {
        "name": "4. Noisy measurements (within tolerances)",
        "hint": (
            "The sensor noise is turned up (bounded to at most 0.8x each "
            "tolerance overall), which must NOT trigger an alert."
        ),
        "cause": (
            "Simulator-injected bounded noise (seeded, clipped inside the "
            "tolerances). A correct engine reports NORMAL."
        ),
    },
    "stale": {
        "name": "5. Stale sensor data",
        "hint": (
            "The pH sensor reports a frozen value with no fresh update, so "
            "its data-quality flag reads STALE from the very first sample "
            "onward and no fresh conclusion can be drawn."
        ),
        "cause": (
            "Simulator-injected staleness: the pH reading never updates. "
            "A correct engine reports INSUFFICIENT DATA for stale data - "
            "not NORMAL, and not automatically an attack."
        ),
    },
    "insufficient": {
        "name": "6. Insufficient or invalid data",
        "hint": (
            "pH is missing on every sample and conductivity turns invalid "
            "(NaN) on alternating samples, so no conclusion can be drawn."
        ),
        "cause": (
            "Simulator-injected data-quality problem. The engine must "
            "answer INSUFFICIENT DATA - not NORMAL, and not automatically "
            "an attack."
        ),
    },
}

#: Per-scenario simulator settings. Values not listed keep the neutral
#: defaults of SimulatorConfig.
PRESETS: dict[str, dict[str, Any]] = {
    "normal": {
        "noise_fraction": 0.10,
        "note": "Nominal sensor noise only.",
    },
    "sensor_mismatch": {
        "noise_fraction": 0.10,
        "sensor_bias_fractions": {"ph": 3.0},
        "note": "pH sensor biased +3x tolerance on every sample.",
    },
    "claim_mismatch": {
        "noise_fraction": 0.05,
        "noise_cap_fraction": 0.15,
        "wander_fraction": 0.05,
        "wander_max_fraction": 0.10,
        "claimed_state": _CLAIM_TARGET_REACHED,
        "expected_state_id": "target_reached",
        "note": (
            "Controller claims the target state while the process stays "
            "steady; expected response comes from the target_reached "
            "profile."
        ),
    },
    "noisy": {
        "noise_fraction": 0.40,
        "noise_cap_fraction": 0.50,
        "note": "Heavier but bounded noise; still inside every tolerance.",
    },
    "stale": {
        "noise_fraction": 0.10,
        "stale_from": {"ph": 1},
        "note": "pH sensor is frozen/stale on every sample.",
    },
    "insufficient": {
        "noise_fraction": 0.10,
        "missing_keys": frozenset({"ph"}),
        "invalid_odd_keys": frozenset({"conductivity"}),
        "note": "pH missing always; conductivity invalid on odd samples.",
    },
}


def scenario_ids() -> list[str]:
    """Scenario identifiers in display order."""
    return list(SCENARIOS)


def _clamp(key: str, value: float) -> float:
    """Keep a generated value inside the plausible physical range."""
    low, high = PLAUSIBLE_RANGES[key]
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# The stateful simulator
# --------------------------------------------------------------------------- #

class ProcessSimulator:
    """Stateful, deterministic time-series simulator.

    Parameters
    ----------
    config:
        Simulator settings; defaults to a neutral "normal" configuration.
    base_values:
        Underlying process values the process wanders around. Defaults to
        the illustrative steady-operation profile.
    tolerances:
        Active tolerances; faults and noise are scaled as fractions of
        these, so scenario intent survives tolerance changes.
    """

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        *,
        base_values: Mapping[str, float] | None = None,
        tolerances: Mapping[str, float] | None = None,
    ) -> None:
        self.config = config if config is not None else SimulatorConfig()
        self.base = dict(DEFAULT_EXPECTED if base_values is None else base_values)
        self.tolerances = dict(
            DEFAULT_TOLERANCES if tolerances is None else tolerances
        )
        for key in MEASUREMENT_KEYS:
            if key not in self.base:
                raise ValueError(f"base_values must include {key!r}.")
            if key not in self.tolerances:
                raise ValueError(f"tolerances must include {key!r}.")
            if self.tolerances[key] < 0:
                raise ValueError(f"tolerance for {key!r} must not be negative.")
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        """Reset the simulation to sample zero (same seed -> same sequence)."""
        self._rng = Random(self.config.seed)
        self.index = 0
        self._wander: dict[str, float] = {k: 0.0 for k in MEASUREMENT_KEYS}
        self._drift: dict[str, float] = {k: 0.0 for k in MEASUREMENT_KEYS}
        self._true_history: list[dict[str, float]] = []
        self._reported_last: dict[str, float | None] = {
            k: self.base[k] for k in MEASUREMENT_KEYS
        }
        self._stale_frozen: dict[str, float] = {}
        self._stale_count: dict[str, int] = {k: 0 for k in MEASUREMENT_KEYS}
        self.samples: list[Sample] = []

    def restart(self) -> None:
        """Restart the simulation (identical sequence for the same seed)."""
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _next_true_values(self) -> dict[str, float]:
        """Advance the underlying process with a bounded, mean-reverting walk."""
        values: dict[str, float] = {}
        for key in MEASUREMENT_KEYS:
            tol = self.tolerances[key]
            step = self._rng.uniform(
                -self.config.wander_fraction * tol,
                self.config.wander_fraction * tol,
            )
            # Damping (0.6) pulls the wander back toward zero: mean-reverting.
            wander = self._wander[key] * 0.6 + step
            limit = self.config.wander_max_fraction * tol
            wander = max(-limit, min(limit, wander))
            self._wander[key] = wander
            values[key] = _clamp(key, self.base[key] + wander)
        return values

    def step(self) -> Sample:
        """Produce the next timestamped sample and remember the state."""
        self.index += 1
        cfg = self.config
        timestamp = cfg.start_time + timedelta(
            seconds=cfg.sample_interval_seconds * (self.index - 1)
        )

        # 1. Underlying (illustrative) process values.
        true_values = self._next_true_values()
        self._true_history.append(dict(true_values))

        # 2. What the sensor physically sees (optional delay).
        if cfg.delay_samples > 0 and len(self._true_history) > cfg.delay_samples:
            sensed = dict(
                self._true_history[-1 - cfg.delay_samples]
            )
        else:
            sensed = dict(true_values)

        quality: dict[str, str] = {}
        observed: dict[str, float | None] = {}
        notes: list[str] = []

        for key in MEASUREMENT_KEYS:
            tol = self.tolerances[key]

            # 3a. Deterministic per-key faults.
            if key in cfg.missing_keys:
                observed[key] = None
                quality[key] = QUALITY_MISSING
                continue
            if key in cfg.invalid_odd_keys and self.index % 2 == 1:
                observed[key] = float("nan")
                quality[key] = QUALITY_INVALID
                continue

            # 3b. Random per-sample faults.
            if self._rng.random() < cfg.missing_probability:
                observed[key] = None
                quality[key] = QUALITY_MISSING
                continue
            if self._rng.random() < cfg.invalid_probability:
                observed[key] = float("nan")
                quality[key] = QUALITY_INVALID
                continue

            # 3c. Stale: freeze at the configured step (or randomly), then
            # repeat the frozen value and flag it STALE.
            stale_from_step = cfg.stale_from.get(key)
            becomes_stale = (
                (stale_from_step is not None and self.index >= stale_from_step)
                or self._rng.random() < cfg.stale_probability
            )
            if becomes_stale:
                if key not in self._stale_frozen:
                    self._stale_frozen[key] = sensed[key]
                observed[key] = self._stale_frozen[key]
                quality[key] = QUALITY_STALE
                self._stale_count[key] += 1
                continue

            # 3d. Normal chain: bias + slow drift + bounded sensor noise.
            noise = self._rng.gauss(0.0, cfg.noise_fraction * tol)
            noise = max(
                -cfg.noise_cap_fraction * tol,
                min(cfg.noise_cap_fraction * tol, noise),
            )
            bias = cfg.sensor_bias_fractions.get(key, 0.0) * tol
            if cfg.drift_enabled:
                self._drift[key] += cfg.drift_fraction * tol
            value = sensed[key] + bias + self._drift[key] + noise
            observed[key] = _clamp(key, value)
            quality[key] = QUALITY_OK
            self._reported_last[key] = observed[key]

        if cfg.delay_samples > 0:
            notes.append(
                f"Measurements are delayed by {cfg.delay_samples} sample(s)."
            )
        if cfg.drift_enabled:
            notes.append("Slow sensor drift is enabled.")
        if cfg.note:
            notes.append(cfg.note)

        expected = (
            expected_for_state(cfg.expected_state_id)
            if cfg.expected_state_id is not None
            else None
        )

        sample = Sample(
            index=self.index,
            timestamp=timestamp,
            process_state=cfg.process_state_id,
            claimed_state=cfg.claimed_state,
            true_values=true_values,
            observed=observed,
            quality=quality,
            expected=expected,
            notes=tuple(notes),
        )
        self.samples.append(sample)
        return sample


# --------------------------------------------------------------------------- #
# Single-sample convenience wrapper (kept for compatibility and quick tests)
# --------------------------------------------------------------------------- #

def _config_for_scenario(
    scenario_id: str, seed: int
) -> SimulatorConfig:
    """Build a SimulatorConfig for one of the catalogue scenarios."""
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario_id {scenario_id!r}. "
            f"Valid options: {', '.join(scenario_ids())}."
        )
    preset = dict(PRESETS[scenario_id])
    note = preset.pop("note", "")
    stale_from = dict(preset.get("stale_from", {}))
    return SimulatorConfig(
        scenario_id=scenario_id,
        seed=seed,
        note=note,
        stale_from=stale_from,
        **{k: v for k, v in preset.items() if k != "stale_from"},
    )


def generate_sample(
    scenario_id: str,
    expected: Mapping[str, float] | None = None,
    tolerances: Mapping[str, float] | None = None,
    seed: int = DEFAULT_SEED,
) -> ScenarioSample:
    """Generate one deterministic synthetic sample for a scenario.

    This is a convenience wrapper that runs a single step of a fresh
    :class:`ProcessSimulator`. Use the simulator directly for multi-sample
    time-series behaviour.

    Parameters
    ----------
    scenario_id:
        Key into :data:`SCENARIOS`.
    expected:
        Accepted for signature compatibility but not used for generation:
        the simulated process follows the illustrative profiles in
        process_profiles.py, independently of any dashboard configuration.
    tolerances:
        Active tolerances (faults and noise scale with them).
    seed:
        The same (tolerances, seed) pair always produces the same sample.

    Raises
    ------
    ValueError
        If ``scenario_id`` is unknown or the tolerances are incomplete.
    """
    del expected  # generation is profile-driven, see module docstring
    if tolerances is not None:
        for key in MEASUREMENT_KEYS:
            if key not in tolerances:
                raise ValueError(f"tolerances must include {key!r}.")
    tol = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)

    config = _config_for_scenario(scenario_id, seed)
    simulator = ProcessSimulator(config, tolerances=tol)
    sample = simulator.step()
    meta = SCENARIOS[scenario_id]
    return ScenarioSample(
        scenario_id=scenario_id,
        scenario_name=meta["name"],
        injected_cause=meta["cause"],
        claimed_state=sample.claimed_state,
        observed=dict(sample.observed),
        notes=meta["cause"],
        quality=dict(sample.quality),
        expected=sample.expected,
    )


def make_scenario_simulator(
    scenario_id: str,
    *,
    seed: int = DEFAULT_SEED,
    tolerances: Mapping[str, float] | None = None,
) -> ProcessSimulator:
    """Create a fresh stateful simulator preset for a catalogue scenario.

    This is the dashboard's entry point for multi-sample simulation: each
    "advance one sample" click calls ``step()`` on the returned simulator.
    Restarting the simulation simply means creating a new simulator with the
    same arguments (the same seed reproduces the same sequence).
    """
    config = _config_for_scenario(scenario_id, seed)
    return ProcessSimulator(config, tolerances=tolerances)
