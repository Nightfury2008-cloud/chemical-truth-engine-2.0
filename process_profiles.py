"""Illustrative process-response profiles for Chemical Truth Engine 2.0.

SCIENTIFIC-HONESTY NOTE
-----------------------
The values in this module are ILLUSTRATIVE SOFTWARE-DEMO VALUES. They are
not measured, not calibrated and not scientifically validated.

* pH depends on the actual chemistry and conditions of a process. It cannot
  be predicted from a universal linear "dose -> pH" formula, and no such
  model is used or implied here. The pH values below are plain example
  numbers chosen to make the demonstration readable.
* Conductivity and temperature values are likewise invented examples.

WHY THIS MODULE EXISTS
----------------------
It is the single, clearly marked seam between the synthetic demonstration and
future real calibration data. The detection engine (detection_engine.py) and
the dashboard (app.py) never hard-code process values; they ask this module
(or the user's configuration) for the expected response of a claimed state.
Replacing these profiles with experimentally calibrated data later requires
no changes to the engine, the simulator architecture or the dashboard.

Each profile maps a controller-claimed process state to the expected sensor
readings for that state, in the project's standard units:
pH dimensionless, conductivity in uS/cm, temperature in degrees C.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PROFILE_IDS",
    "PROFILES",
    "ProcessProfile",
    "expected_for_state",
    "profile_label",
]


@dataclass(frozen=True)
class ProcessProfile:
    """Expected sensor response for one claimed process state (illustrative)."""

    state_id: str
    label: str
    ph: float
    conductivity: float
    temperature: float
    description: str

    def expected_values(self) -> dict[str, float]:
        """Expected readings as a {measurement: value} mapping."""
        return {
            "ph": self.ph,
            "conductivity": self.conductivity,
            "temperature": self.temperature,
        }


#: Illustrative profiles. "steady_operation" deliberately matches the
#: dashboard's synthetic demo defaults (pH 8.0, 650 uS/cm, 25.5 C).
PROFILES: dict[str, ProcessProfile] = {
    "steady_operation": ProcessProfile(
        state_id="steady_operation",
        label="Steady operation",
        ph=8.0,
        conductivity=650.0,
        temperature=25.5,
        description=(
            "Illustrative baseline state: the process is sitting at its "
            "configured set point. These numbers are invented demo values, "
            "not measurements."
        ),
    ),
    "target_reached": ProcessProfile(
        state_id="target_reached",
        label="Target state reached",
        ph=7.2,
        conductivity=520.0,
        temperature=26.2,
        description=(
            "Illustrative 'target state' the controller may claim to have "
            "reached. The differences from the steady profile (a somewhat "
            "lower pH and conductivity, slightly warmer) are invented for "
            "the demonstration. No chemical reaction is modelled and no "
            "measured transition time is implied."
        ),
    ),
}

#: Profile identifiers in display order.
PROFILE_IDS: tuple[str, ...] = tuple(PROFILES)


def expected_for_state(state_id: str) -> dict[str, float]:
    """Return the expected readings for a claimed process state.

    Raises
    ------
    ValueError
        If the state is unknown. Callers must never silently fall back to
        other values for an unknown claim.
    """
    profile = PROFILES.get(state_id)
    if profile is None:
        raise ValueError(
            f"Unknown process state {state_id!r}. "
            f"Known states: {', '.join(PROFILE_IDS)}."
        )
    return profile.expected_values()


def profile_label(state_id: str) -> str:
    """Human-readable label for a process state."""
    profile = PROFILES.get(state_id)
    return profile.label if profile is not None else state_id
