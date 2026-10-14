# rl/state_encoder.py

from typing import Any

import numpy as np


# These IDs must match the action IDs in the dataset.
NUM_ACTIONS = 7

VITAL_KEYS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "temperature_c",
    "spo2",
    "gcs",
]

LAB_STATUS_OPTIONS = [
    "not_ordered",
    "ordered",
    "available",
    "reviewed",
    "unknown",
]

DIAGNOSTICS_STATUS_OPTIONS = [
    "not_ordered",
    "ordered",
    "available",
    "completed",
    "reviewed",
    "unknown",
]


def _number(value: Any, default: float = 0.0) -> float:
    """Safely convert a value into a finite float."""

    if isinstance(value, bool):
        return float(value)

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(result):
        return default

    return result


def _one_hot(value: str, categories: list[str]) -> list[float]:
    """One-hot encode a categorical value."""

    if value not in categories:
        value = "unknown"

    return [
        1.0 if value == category else 0.0
        for category in categories
    ]


def encode_state(state: dict[str, Any]) -> np.ndarray:
    """
    Convert a synthetic clinical state into a fixed-size
    numerical feature vector.

    This encoder is intended for software experiments
    with synthetic data, not real clinical decisions.
    """

    features: list[float] = []

    # --------------------------------------------------
    # 1. Severity
    # --------------------------------------------------

    features.append(
        _number(state.get("severity_index"))
    )

    # --------------------------------------------------
    # 2. Vital signs
    # --------------------------------------------------

    vitals = state.get("vitals") or {}

    # Broad scales for numerical stability.
    # These are engineering scales, not clinical thresholds.
    vital_scales = {
        "heart_rate": 200.0,
        "systolic_bp": 250.0,
        "diastolic_bp": 150.0,
        "respiratory_rate": 60.0,
        "temperature_c": 45.0,
        "spo2": 100.0,
        "gcs": 15.0,
    }

    for key in VITAL_KEYS:
        value = vitals.get(key)

        # Value plus an indicator showing whether it exists.
        features.append(
            _number(value) / vital_scales[key]
        )
        features.append(
            1.0 if value is not None else 0.0
        )

    # --------------------------------------------------
    # 3. Risk scores
    # --------------------------------------------------

    risk_scores = state.get("risk_scores") or {}

    features.append(
        _number(risk_scores.get("qsofa")) / 3.0
    )

    features.append(
        _number(risk_scores.get("news2_approx")) / 20.0
    )

    features.append(
        _number(
            risk_scores.get("sepsis_risk_synthetic")
        )
    )

    # --------------------------------------------------
    # 4. Symptoms
    # --------------------------------------------------

    symptoms = state.get("symptoms") or []

    severe_count = 0
    moderate_count = 0
    mild_count = 0

    for symptom in symptoms:
        severity = str(
            symptom.get("severity", "")
        ).lower()

        if severity == "severe":
            severe_count += 1
        elif severity == "moderate":
            moderate_count += 1
        elif severity == "mild":
            mild_count += 1

    features.extend([
        len(symptoms) / 20.0,
        severe_count / 20.0,
        moderate_count / 20.0,
        mild_count / 20.0,
    ])

    # --------------------------------------------------
    # 5. Medical history
    # --------------------------------------------------

    history = state.get("medical_history") or []

    features.append(
        len(history) / 20.0
    )

    # --------------------------------------------------
    # 6. Laboratory status
    # --------------------------------------------------

    labs_status = str(
        state.get("labs_status", "unknown")
    ).lower()

    features.extend(
        _one_hot(
            labs_status,
            LAB_STATUS_OPTIONS,
        )
    )

    # Laboratory results can be null or a dictionary.
    lab_results = state.get("laboratory_results")

    features.append(
        1.0 if lab_results is not None else 0.0
    )

    # Number of critical flags.
    critical_flags = state.get("critical_lab_flags") or []

    features.append(
        len(critical_flags) / 20.0
    )

    # --------------------------------------------------
    # 7. Diagnostic status
    # --------------------------------------------------

    diagnostics_status = str(
        state.get("diagnostics_status", "unknown")
    ).lower()

    features.extend(
        _one_hot(
            diagnostics_status,
            DIAGNOSTICS_STATUS_OPTIONS,
        )
    )

    # --------------------------------------------------
    # 8. Review flags and pending orders
    # --------------------------------------------------

    features.append(
        float(bool(state.get("labs_reviewed", False)))
    )

    features.append(
        float(bool(state.get("medication_reviewed", False)))
    )

    pending_orders = state.get("pending_orders") or []

    features.append(
        len(pending_orders) / 20.0
    )

    # --------------------------------------------------
    # 9. Agent-derived features
    # --------------------------------------------------

    agent_features = state.get("agent_features") or {}

    symptoms_agent = agent_features.get(
        "SymptomsAgent", {}
    )

    history_agent = agent_features.get(
        "HistoryAgent", {}
    )

    features.append(
        _number(
            symptoms_agent.get("symptom_count")
        ) / 20.0
    )

    features.append(
        _number(
            symptoms_agent.get("severe_symptom_count")
        ) / 20.0
    )

    features.append(
        _number(
            history_agent.get("comorbidity_count")
        ) / 20.0
    )

    # --------------------------------------------------
    # 10. Action mask
    # --------------------------------------------------

    # This represents which actions are permitted by
    # the synthetic dataset at the current state.
    action_mask = state.get("action_mask")

    if action_mask is None:
        action_mask = [True] * NUM_ACTIONS

    if len(action_mask) != NUM_ACTIONS:
        raise ValueError(
            f"Expected {NUM_ACTIONS} action-mask entries, "
            f"received {len(action_mask)}"
        )

    features.extend(
        float(bool(allowed))
        for allowed in action_mask
    )

    result = np.asarray(features, dtype=np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "State encoding produced NaN or infinite values"
        )

    return result


def encode_action_mask(
    state: dict[str, Any],
) -> np.ndarray:
    """Return the permitted-action mask as a Boolean array."""

    mask = state.get("action_mask")

    if mask is None:
        mask = [True] * NUM_ACTIONS

    if len(mask) != NUM_ACTIONS:
        raise ValueError(
            f"Expected {NUM_ACTIONS} action-mask entries, "
            f"received {len(mask)}"
        )

    return np.asarray(mask, dtype=bool)