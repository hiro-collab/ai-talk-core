"""Synthetic AEC comparison helpers with no live-audio authority."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


AEC_OWNER_CLASSES = (
    "windows_voice_capture_dsp",
    "webrtc_apm_aec3",
)
PROCESSING_INVENTORY_CLASSES = (
    "known_no_owner",
    "known_single_owner",
    "unknown",
    "double_owner",
)
MIN_ECHO_CONVERGENCE_DB = 6.0
MIN_NEAR_END_PRESERVATION_RATIO = 0.85
MIN_SELECTION_MARGIN_DB = 1.0
_MIN_VECTOR_LENGTH = 4
_MAX_VECTOR_LENGTH = 48_000
_EPSILON = 1e-9


class AecReferenceError(ValueError):
    """Expose only a fixed failure class for invalid synthetic inputs."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


def evaluate_synthetic_aec_candidate(
    *,
    owner_class: str,
    reference_samples: Sequence[float],
    near_end_samples: Sequence[float],
    microphone_samples: Sequence[float],
    processed_samples: Sequence[float],
) -> dict[str, object]:
    """Return class/metric-only evidence for one synthetic AEC candidate."""

    if owner_class not in AEC_OWNER_CLASSES:
        raise AecReferenceError("aec_owner_class_invalid")

    vectors = tuple(
        _normalize_vector(value)
        for value in (
            reference_samples,
            near_end_samples,
            microphone_samples,
            processed_samples,
        )
    )
    reference, near_end, microphone, processed = vectors
    lengths = {len(value) for value in vectors}
    if len(lengths) != 1:
        raise AecReferenceError("synthetic_aec_vector_length_mismatch")

    reference_energy = _dot(reference, reference)
    near_end_energy = _dot(near_end, near_end)
    if reference_energy <= _EPSILON or near_end_energy <= _EPSILON:
        raise AecReferenceError("synthetic_aec_fixture_energy_invalid")

    fixture_correlation = abs(_dot(reference, near_end)) / math.sqrt(
        reference_energy * near_end_energy
    )
    if fixture_correlation > 0.05:
        raise AecReferenceError("synthetic_aec_fixture_not_separable")

    input_echo_gain = abs(_dot(microphone, reference) / reference_energy)
    residual_echo_gain = abs(_dot(processed, reference) / reference_energy)
    if input_echo_gain <= _EPSILON:
        raise AecReferenceError("synthetic_aec_input_echo_missing")

    convergence_db = 20.0 * math.log10(
        input_echo_gain / max(residual_echo_gain, _EPSILON)
    )
    near_end_gain = abs(_dot(processed, near_end) / near_end_energy)
    preservation_ratio = (
        min(near_end_gain, 1.0 / near_end_gain)
        if near_end_gain > _EPSILON
        else 0.0
    )

    return {
        "owner_class": owner_class,
        "echo_convergence_db": _round_metric(convergence_db),
        "near_end_preservation_ratio": _round_metric(preservation_ratio),
        "sample_count": len(reference),
        "source_class": "synthetic_dual_channel_fixture",
        "raw_audio_persisted": False,
        "turn_input_authority": False,
    }


def select_synthetic_aec_owner(
    *,
    processing_inventory_class: str,
    active_owner_classes: Iterable[str],
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    """Select at most one owner or fail closed to observation-only."""

    inventory = (
        processing_inventory_class
        if processing_inventory_class in PROCESSING_INVENTORY_CLASSES
        else "unknown"
    )
    try:
        active = tuple(active_owner_classes)
    except Exception:
        active = ("invalid",)
    active_valid = all(value in AEC_OWNER_CLASSES for value in active)
    try:
        candidate_values = tuple(candidates)
    except Exception:
        return _selection_result(
            "synthetic_aec_candidate_invalid_observation_only",
            inventory,
            0,
        )
    normalized = tuple(
        value
        for value in (_normalize_candidate(candidate) for candidate in candidate_values)
        if value is not None
    )

    if len(normalized) != len(candidate_values):
        return _selection_result(
            "synthetic_aec_candidate_invalid_observation_only",
            inventory,
            len(candidate_values),
        )

    if inventory == "double_owner" or len(active) > 1:
        return _selection_result(
            "aec_double_owner_rejected",
            inventory,
            len(candidate_values),
        )
    if (
        inventory == "unknown"
        or not active_valid
        or (inventory == "known_single_owner" and len(active) != 1)
        or (inventory == "known_no_owner" and len(active) != 0)
    ):
        return _selection_result(
            "aec_processing_unknown_observation_only",
            "unknown",
            len(candidate_values),
        )

    if len({value["owner_class"] for value in normalized}) != len(normalized):
        return _selection_result(
            "synthetic_aec_owner_ambiguous",
            inventory,
            len(candidate_values),
        )

    eligible = sorted(
        (
            value
            for value in normalized
            if value["echo_convergence_db"] >= MIN_ECHO_CONVERGENCE_DB
            and value["near_end_preservation_ratio"]
            >= MIN_NEAR_END_PRESERVATION_RATIO
        ),
        key=lambda value: (
            value["echo_convergence_db"],
            value["near_end_preservation_ratio"],
        ),
        reverse=True,
    )
    if not eligible:
        return _selection_result(
            "synthetic_aec_candidate_unqualified",
            inventory,
            len(candidate_values),
        )
    if (
        len(eligible) > 1
        and eligible[0]["echo_convergence_db"]
        - eligible[1]["echo_convergence_db"]
        < MIN_SELECTION_MARGIN_DB
    ):
        return _selection_result(
            "synthetic_aec_owner_ambiguous",
            inventory,
            len(candidate_values),
        )

    selected = eligible[0]
    if active and active[0] != selected["owner_class"]:
        return _selection_result(
            "aec_active_owner_mismatch_observation_only",
            inventory,
            len(candidate_values),
        )
    return _selection_result(
        "synthetic_aec_owner_selected",
        inventory,
        len(candidate_values),
        selected,
    )


def _normalize_vector(value: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise AecReferenceError("synthetic_aec_vector_invalid")
    try:
        normalized = tuple(float(sample) for sample in value)
    except Exception:
        raise AecReferenceError("synthetic_aec_vector_invalid") from None
    if not _MIN_VECTOR_LENGTH <= len(normalized) <= _MAX_VECTOR_LENGTH:
        raise AecReferenceError("synthetic_aec_vector_bounds_invalid")
    if any(not math.isfinite(sample) or abs(sample) > 1.0 for sample in normalized):
        raise AecReferenceError("synthetic_aec_vector_invalid")
    return normalized


def _normalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return None
    try:
        owner_class = candidate.get("owner_class")
        convergence = candidate.get("echo_convergence_db")
        preservation = candidate.get("near_end_preservation_ratio")
    except Exception:
        return None
    if owner_class not in AEC_OWNER_CLASSES:
        return None
    if (
        isinstance(convergence, bool)
        or not isinstance(convergence, (int, float))
        or not math.isfinite(convergence)
    ):
        return None
    if (
        isinstance(preservation, bool)
        or not isinstance(preservation, (int, float))
        or not math.isfinite(preservation)
    ):
        return None
    if preservation < 0 or preservation > 1:
        return None
    return {
        "owner_class": owner_class,
        "echo_convergence_db": float(convergence),
        "near_end_preservation_ratio": float(preservation),
    }


def _selection_result(
    result_class: str,
    inventory_class: str,
    candidate_count: int,
    selected: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    selected_owner = selected["owner_class"] if selected else None
    return {
        "schema_version": "synthetic_aec_owner_selection.v0",
        "proof_ceiling": "synthetic_aec_owner_selection_only",
        "result_class": result_class,
        "processing_inventory_class": inventory_class,
        "candidate_count": candidate_count,
        "selected_owner_class": selected_owner,
        "selected_echo_convergence_db": (
            _round_metric(selected["echo_convergence_db"]) if selected else None
        ),
        "selected_near_end_preservation_ratio": (
            _round_metric(selected["near_end_preservation_ratio"])
            if selected
            else None
        ),
        "exactly_one_aec_owner": selected_owner is not None,
        "observation_only": selected_owner is None,
        "render_reference_may_create_turn_input": False,
        "raw_audio_persisted": False,
        "live_audio_used": False,
    }


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _round_metric(value: float) -> float:
    scaled = value * 1000.0
    rounded = math.floor(scaled + 0.5) if value >= 0 else math.ceil(scaled - 0.5)
    return rounded / 1000.0
