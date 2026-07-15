"""Synthetic AEC comparison helpers with no live-audio authority."""

from __future__ import annotations

import math
import ctypes
import hashlib
import hmac
import json
import os
import platform
import secrets
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.input_gate import (
    NEAR_END_DISTINGUISHED_CLASS,
    SELF_OUTPUT_OR_AMBIGUOUS_CLASS,
)


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
LIVE_AEC_OWNER_CLASS = "windows_voice_capture_dsp"
LIVE_AEC_SELECTION_CLASS = "synthetic_aec_owner_selected"
LIVE_CAPTURE_MODE_AEC = "windows_voice_capture_dsp_aec"
LIVE_CAPTURE_MODE_NS_AGC = "windows_voice_capture_dsp_ns_agc"
LIVE_CAPTURE_MODE_CLASSES = frozenset(
    {LIVE_CAPTURE_MODE_AEC, LIVE_CAPTURE_MODE_NS_AGC}
)
LIVE_AEC_FRAME_BYTES = 320
LIVE_AEC_SAMPLE_RATE = 16_000
LIVE_AEC_MAX_CAPTURE_BYTES = LIVE_AEC_SAMPLE_RATE * 2 * 5
LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES = frozenset(
    {
        "processed_pcm_pipe_lease_invalid",
        "processed_pcm_pipe_owner_unavailable",
        "processed_pcm_pipe_lease_missing",
        "processed_pcm_pipe_lease_expired",
        "processed_pcm_pipe_server_identity_mismatch",
        "processed_pcm_pipe_private_input_timeout",
        "processed_pcm_pipe_connect_failed",
        "processed_pcm_pipe_connect_timeout",
        "processed_pcm_pipe_handshake_failed",
        "processed_pcm_pipe_write_failed",
        "live_aec_backend_or_sink_missing",
        "live_aec_bounds_invalid",
        "live_aec_processing_mode_invalid",
        "live_aec_processed_packet_invalid",
        "live_aec_deadline_exceeded",
        "live_aec_cleanup_failed",
        "live_aec_quality_metrics_cleanup_failed",
        "live_aec_quality_metrics_invariant_failed",
        "live_aec_lifecycle_invariant_failed",
        "voice_capture_dsp_activation_failed",
        "voice_capture_dsp_configuration_failed",
        "voice_capture_dsp_output_format_failed",
        "voice_capture_dsp_start_failed",
        "voice_capture_dsp_not_started",
        "voice_capture_dsp_process_output_failed",
        "voice_capture_dsp_stop_failed",
        "live_aec_observer_failed",
    }
)
_LIVE_AEC_PIPE_PREFIX = "sword-aec-"
_LIVE_AEC_ACK = b"\xa1"
_LIVE_AEC_MAX_LEASE_SECONDS = 15
_DOTNET_FILETIME_OFFSET_TICKS = 504_911_232_000_000_000
_USED_NONCE_DIGEST_LIMIT = 1024
_USED_NONCE_DIGESTS: list[bytes] = []
_ACTIVE_NONCE_DIGESTS: set[bytes] = set()
_USED_NONCE_LOCK = threading.Lock()
_HELPER_TOP_LEVEL_FIELDS = {
    "schema_version",
    "proof_ceiling",
    "result_class",
    "capability_class",
    "owner_class",
    "source_class",
    "observation",
    "lifecycle",
    "privacy",
    "authority",
    "does_not_prove",
}
_HELPER_OBSERVATION_FIELDS = {
    "window_ms",
    "packet_count",
    "processed_byte_count",
    "near_end_discrimination_class",
    "quality_metrics_attempt_count",
    "quality_metrics_valid_count",
    "quality_metrics_trusted_count",
    "quality_metrics_ambiguous_count",
    "quality_metrics_cleanup_failure_count",
    "live_capture_used",
}
_HELPER_LIFECYCLE_FIELDS = {
    "backend_activate_count",
    "capture_start_count",
    "capture_stop_attempt_count",
    "capture_stop_count",
    "backend_resource_release_count",
    "sink_connect_count",
    "sink_write_count",
    "sink_release_count",
    "cancel_count",
    "cleanup_class",
    "owned_process_residue_count",
    "pipe_residue_count",
    "temporary_file_residue_count",
}
_HELPER_PRIVACY_FIELDS = {
    "render_reference_published",
    "raw_pcm_published",
    "raw_audio_persisted",
    "transcript_observed",
    "pipe_name_published",
    "process_or_device_identity_published",
    "private_path_published",
    "payload_published",
}
_HELPER_AUTHORITY_FIELDS = {
    "exactly_one_aec_owner",
    "render_reference_turn_input_authority",
    "processed_near_end_turn_input_authority",
    "thought_core_turn_input_authority",
    "user_heard_authority",
    "readiness_authority",
}
_HELPER_DOES_NOT_PROVE = [
    "live_barge_in",
    "self_output_turn_input_blocking",
    "genuine_user_speech_acceptance",
    "aec_effectiveness",
    "subjective_audio_quality",
    "user_heard_audio",
    "release_readiness",
]


def _build_live_near_end_evidence_boundary() -> tuple[Any, Any]:
    owner = object()

    class _LiveNearEndDiscriminationEvidence:
        __slots__ = (
            "_owner",
            "_identity",
            "_classification",
            "_source_epoch_binding",
            "_consumed",
        )

        def __new__(
            cls,
            classification: str,
            source_epoch_binding: object,
            mint_token: object,
        ) -> object:
            if mint_token is not owner:
                raise TypeError("live near-end evidence is producer-owned")
            return super().__new__(cls)

        def __init__(
            self,
            classification: str,
            source_epoch_binding: object,
            mint_token: object,
        ) -> None:
            if mint_token is not owner:
                raise TypeError("live near-end evidence is producer-owned")
            self._owner = owner
            self._identity = object()
            self._classification = classification
            self._source_epoch_binding = source_epoch_binding
            self._consumed = False

        def __repr__(self) -> str:
            return "<live-near-end-discrimination-evidence private>"

        __str__ = __repr__

        def __bool__(self) -> bool:
            raise TypeError("live near-end evidence has no boolean representation")

        def __copy__(self) -> object:
            raise TypeError("live near-end evidence cannot be copied")

        def __deepcopy__(self, memo: object) -> object:
            del memo
            raise TypeError("live near-end evidence cannot be copied")

        def __reduce__(self) -> object:
            raise TypeError("live near-end evidence cannot be serialized")

        def __reduce_ex__(self, protocol: int) -> object:
            del protocol
            raise TypeError("live near-end evidence cannot be serialized")

    def mint(classification: str, source_epoch_binding: object) -> object:
        if classification not in {
            NEAR_END_DISTINGUISHED_CLASS,
            SELF_OUTPUT_OR_AMBIGUOUS_CLASS,
        }:
            raise LiveAecCaptureError("live_aec_helper_result_invalid")
        return _LiveNearEndDiscriminationEvidence(
            classification,
            source_epoch_binding,
            owner,
        )

    def consume(evidence: object, source_epoch_binding: object) -> str | None:
        if type(evidence) is not _LiveNearEndDiscriminationEvidence:
            return None
        if (
            evidence._owner is not owner
            or evidence._consumed
            or evidence._source_epoch_binding is not source_epoch_binding
        ):
            return None
        evidence._consumed = True
        classification = evidence._classification
        evidence._classification = SELF_OUTPUT_OR_AMBIGUOUS_CLASS
        evidence._source_epoch_binding = None
        return classification

    return mint, consume


(
    _new_live_near_end_discrimination_evidence,
    _consume_live_near_end_discrimination_evidence,
) = _build_live_near_end_evidence_boundary()


class AecReferenceError(ValueError):
    """Expose only a fixed failure class for invalid synthetic inputs."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


class LiveAecCaptureError(RuntimeError):
    """Expose only a fixed class for the live AEC transport boundary."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


@dataclass
class LiveAecProcessedCapture:
    """Hold processed near-end PCM only until its in-memory consumer clears it."""

    pcm16: bytearray
    packet_count: int
    sample_rate: int = LIVE_AEC_SAMPLE_RATE
    storage_class: str = "in_memory_ephemeral"
    turn_input_authority: bool = False
    turn_input_authority_class: str = "processed_near_end_observation_only"
    near_end_discrimination_evidence: object | None = None

    @property
    def processed_byte_count(self) -> int:
        return len(self.pcm16)

    def clear(self) -> None:
        _clear_bytearray(self.pcm16)


def validate_live_aec_owner_selection(
    selection: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Require the exact adopted Phase 2 single-owner selection."""

    if not isinstance(selection, Mapping):
        raise LiveAecCaptureError("live_aec_owner_selection_missing")
    try:
        result_class = selection.get("result_class")
        selected_owner = selection.get("selected_owner_class")
        exactly_one = selection.get("exactly_one_aec_owner")
        observation_only = selection.get("observation_only")
        raw_audio_persisted = selection.get("raw_audio_persisted")
        live_audio_used = selection.get("live_audio_used")
    except Exception:
        raise LiveAecCaptureError("live_aec_owner_selection_invalid") from None
    if (
        result_class != LIVE_AEC_SELECTION_CLASS
        or selected_owner != LIVE_AEC_OWNER_CLASS
        or exactly_one is not True
        or observation_only is not False
        or raw_audio_persisted is not False
        or live_audio_used is not False
    ):
        raise LiveAecCaptureError("live_aec_owner_selection_invalid")
    return {
        "result_class": LIVE_AEC_SELECTION_CLASS,
        "selected_owner_class": LIVE_AEC_OWNER_CLASS,
    }


def get_adopted_live_aec_owner_selection() -> dict[str, object]:
    """Return the fixed adopted Phase 2 transport selection.

    This selects only the live DSP transport. It grants no candidate,
    transcription, or TurnInput authority.
    """

    return {
        "result_class": LIVE_AEC_SELECTION_CLASS,
        "selected_owner_class": LIVE_AEC_OWNER_CLASS,
        "exactly_one_aec_owner": True,
        "observation_only": False,
        "raw_audio_persisted": False,
        "live_audio_used": False,
    }


def capture_live_aec_processed_pcm(
    *,
    owner_selection: Mapping[str, Any] | None,
    window_ms: int,
    deadline_ms: int,
    processing_mode_class: str = LIVE_CAPTURE_MODE_AEC,
    source_epoch_binding: object | None = None,
    helper_path: Path | None = None,
    popen_factory: Any = subprocess.Popen,
    server_factory: Any = None,
) -> LiveAecProcessedCapture:
    """Capture one bounded processed-PCM window through a private pipe lease."""

    validated = validate_live_aec_owner_selection(owner_selection)
    if processing_mode_class not in LIVE_CAPTURE_MODE_CLASSES:
        raise LiveAecCaptureError("live_aec_processing_mode_invalid")
    if window_ms < 100 or window_ms > 5000:
        raise LiveAecCaptureError("live_aec_bounds_invalid")
    if deadline_ms < window_ms + 200 or deadline_ms > 10_000:
        raise LiveAecCaptureError("live_aec_bounds_invalid")
    resolved_helper = helper_path or _default_live_aec_helper_path()
    if not resolved_helper.is_file():
        raise LiveAecCaptureError("live_aec_helper_unavailable")

    nonce = bytearray(secrets.token_bytes(32))
    nonce_digest = hashlib.sha256(nonce).digest()
    _register_one_time_nonce(nonce_digest)
    pipe_name = _LIVE_AEC_PIPE_PREFIX + secrets.token_hex(16)
    server = None
    process = None
    lease_bytes = bytearray()
    stdout_bytes = bytearray()
    stderr_bytes = bytearray()
    capture: LiveAecProcessedCapture | None = None
    server_pcm: bytearray | None = None
    cleanup_failed = False
    try:
        server_pid = os.getpid()
        server_creation_ticks = _current_process_creation_utc_ticks()
        expires_ticks = _utc_now_dotnet_ticks() + min(
            deadline_ms + 1000,
            _LIVE_AEC_MAX_LEASE_SECONDS * 1000,
        ) * 10_000
        server = (
            server_factory(pipe_name, nonce, deadline_ms, expires_ticks)
            if server_factory is not None
            else _WindowsProcessedPcmPipeServer(
                pipe_name,
                nonce,
                deadline_ms,
                expires_ticks,
            )
        )
        lease_bytes.extend(
            _encode_private_lease_packet(
                pipe_name=pipe_name,
                nonce=nonce,
                server_process_id=server_pid,
                server_creation_utc_ticks=server_creation_ticks,
                expires_utc_ticks=expires_ticks,
                selection=validated,
                processing_mode_class=processing_mode_class,
            )
        )
        command = [
            _resolve_powershell_executable(),
            "-NoProfile",
            "-File",
            str(resolved_helper),
            "-Mode",
            "live_source",
            "-WindowMs",
            str(window_ms),
            "-DeadlineMs",
            str(deadline_ms),
            "-Compact",
        ]
        process = popen_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise LiveAecCaptureError("live_aec_private_input_unavailable")
        process.stdin.write(lease_bytes)
        process.stdin.write(b"\n")
        process.stdin.flush()
        process.stdin.close()
        process.stdin = None
        server.start(expected_client_process_id=process.pid)
        timeout_seconds = deadline_ms / 1000 + 1.0
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            raise LiveAecCaptureError("live_aec_deadline_exceeded") from None
        stdout_bytes.extend(stdout or b"")
        stderr_bytes.extend(stderr or b"")
        helper_result = _parse_helper_class_only_result(stdout_bytes)
        helper_result_class = helper_result.get("result_class")
        if process.returncode != 0:
            if helper_result_class in LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES:
                raise LiveAecCaptureError(str(helper_result_class))
            raise LiveAecCaptureError("live_aec_helper_failed")
        if helper_result_class in LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES:
            raise LiveAecCaptureError(str(helper_result_class))
        if helper_result_class not in {
            "processed_near_end_pcm_observed",
            "processed_near_end_silence_observed",
        }:
            raise LiveAecCaptureError("live_aec_helper_failed")
        server_result = server.finish(timeout_seconds=1.0)
        server_pcm = server_result.get("pcm16")
        if not isinstance(server_pcm, bytearray):
            raise LiveAecCaptureError("live_aec_pipe_result_invalid")
        raw_packet_count = server_result.get("packet_count")
        if (
            isinstance(raw_packet_count, bool)
            or not isinstance(raw_packet_count, int)
            or raw_packet_count < 0
            or len(server_pcm) != raw_packet_count * LIVE_AEC_FRAME_BYTES
        ):
            raise LiveAecCaptureError("live_aec_pipe_result_invalid")
        if raw_packet_count == 0:
            _clear_bytearray(server_pcm)
            raise LiveAecCaptureError("live_aec_processed_packet_invalid")
        pcm16 = server_pcm
        packet_count = raw_packet_count
        helper_packet_count = helper_result["observation"]["packet_count"]
        helper_byte_count = helper_result["observation"]["processed_byte_count"]
        helper_window_ms = helper_result["observation"]["window_ms"]
        helper_live_capture_used = helper_result["observation"][
            "live_capture_used"
        ]
        if helper_packet_count != packet_count:
            _clear_bytearray(pcm16)
            raise LiveAecCaptureError("live_aec_count_mismatch")
        if helper_byte_count != len(pcm16):
            _clear_bytearray(pcm16)
            raise LiveAecCaptureError("live_aec_count_mismatch")
        if helper_window_ms != window_ms or helper_live_capture_used is not True:
            _clear_bytearray(pcm16)
            raise LiveAecCaptureError("live_aec_helper_result_invalid")
        capture = LiveAecProcessedCapture(
            pcm16=pcm16,
            packet_count=packet_count,
            near_end_discrimination_evidence=(
                _new_live_near_end_discrimination_evidence(
                    helper_result["observation"][
                        "near_end_discrimination_class"
                    ],
                    source_epoch_binding,
                )
            ),
        )
        server_pcm = None
        return capture
    except LiveAecCaptureError:
        raise
    except Exception:
        raise LiveAecCaptureError("live_aec_capture_failed") from None
    finally:
        try:
            if process is not None and process.poll() is None:
                _stop_owned_process(process)
        except Exception:
            cleanup_failed = True
        if server is not None:
            try:
                server.close()
            except Exception:
                cleanup_failed = True
        _clear_bytearray(nonce)
        _clear_bytearray(lease_bytes)
        _clear_bytearray(stdout_bytes)
        _clear_bytearray(stderr_bytes)
        if server_pcm is not None:
            _clear_bytearray(server_pcm)
        try:
            _retire_one_time_nonce(nonce_digest)
        except Exception:
            cleanup_failed = True
        if cleanup_failed:
            if capture is not None:
                capture.clear()
            raise LiveAecCaptureError("live_aec_cleanup_failed")


def _default_live_aec_helper_path() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "runtime"
        / "audio-awareness"
        / "windows"
        / "invoke-voice-capture-dsp-aec.ps1"
    )


def _resolve_powershell_executable() -> str:
    import shutil

    executable = shutil.which("pwsh")
    if executable is None:
        raise LiveAecCaptureError("live_aec_powershell_unavailable")
    return executable


def _encode_private_lease_packet(
    *,
    pipe_name: str,
    nonce: bytearray,
    server_process_id: int,
    server_creation_utc_ticks: int,
    expires_utc_ticks: int,
    selection: Mapping[str, object],
    processing_mode_class: str,
) -> bytes:
    packet = {
        "pipe_name": pipe_name,
        "nonce": bytes(nonce).hex(),
        "server_process_id": server_process_id,
        "server_creation_utc_ticks": server_creation_utc_ticks,
        "expires_utc_ticks": expires_utc_ticks,
        "aec_owner_selection_class": selection["result_class"],
        "selected_owner_class": selection["selected_owner_class"],
        "processing_mode_class": processing_mode_class,
    }
    return json.dumps(packet, separators=(",", ":")).encode("utf-8")


def _parse_helper_class_only_result(raw: bytearray) -> dict[str, Any]:
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except Exception:
        raise LiveAecCaptureError("live_aec_helper_result_invalid") from None
    if not isinstance(payload, dict) or set(payload) != _HELPER_TOP_LEVEL_FIELDS:
        raise LiveAecCaptureError("live_aec_helper_result_invalid")
    observation = payload.get("observation")
    lifecycle = payload.get("lifecycle")
    privacy = payload.get("privacy")
    authority = payload.get("authority")
    if (
        payload.get("schema_version")
        != "voice_capture_dsp_aec_observation.v0"
        or not isinstance(observation, dict)
        or set(observation) != _HELPER_OBSERVATION_FIELDS
        or not isinstance(lifecycle, dict)
        or set(lifecycle) != _HELPER_LIFECYCLE_FIELDS
        or not isinstance(privacy, dict)
        or set(privacy) != _HELPER_PRIVACY_FIELDS
        or not isinstance(authority, dict)
        or set(authority) != _HELPER_AUTHORITY_FIELDS
        or payload.get("does_not_prove") != _HELPER_DOES_NOT_PROVE
    ):
        raise LiveAecCaptureError("live_aec_helper_result_invalid")
    result_class = payload.get("result_class")
    capability_class = payload.get("capability_class")
    live_capture_used = observation.get("live_capture_used")
    if (
        not isinstance(result_class, str)
        or not 1 <= len(result_class) <= 96
        or any(
            not (character.islower() or character.isdigit() or character == "_")
            for character in result_class
        )
        or capability_class
        not in {
            "not_checked",
            "voice_capture_dsp_capability_available",
            "voice_capture_dsp_unsupported_platform",
        }
        or payload.get("owner_class") != "windows_voice_capture_dsp"
        or payload.get("source_class")
        != "windows_voice_capture_dsp_source_mode"
        or not isinstance(live_capture_used, bool)
        or payload.get("proof_ceiling")
        != (
            "local_windows_voice_capture_dsp_reachability_only"
            if live_capture_used
            else "source_static_live_aec_adapter_contract"
        )
        or any(value is not False for value in privacy.values())
        or not isinstance(authority.get("exactly_one_aec_owner"), bool)
        or any(
            authority[field] is not False
            for field in _HELPER_AUTHORITY_FIELDS
            if field != "exactly_one_aec_owner"
        )
    ):
        raise LiveAecCaptureError("live_aec_helper_result_invalid")
    packet_count = observation.get("packet_count")
    byte_count = observation.get("processed_byte_count")
    window_ms = observation.get("window_ms")
    classification = observation.get("near_end_discrimination_class")
    attempt_count = observation.get("quality_metrics_attempt_count")
    valid_count = observation.get("quality_metrics_valid_count")
    trusted_count = observation.get("quality_metrics_trusted_count")
    ambiguous_count = observation.get("quality_metrics_ambiguous_count")
    cleanup_failure_count = observation.get(
        "quality_metrics_cleanup_failure_count"
    )
    observation_counts = (
        window_ms,
        packet_count,
        byte_count,
        attempt_count,
        valid_count,
        trusted_count,
        ambiguous_count,
        cleanup_failure_count,
    )
    lifecycle_count_fields = _HELPER_LIFECYCLE_FIELDS - {
        "cleanup_class",
        "owned_process_residue_count",
        "pipe_residue_count",
        "temporary_file_residue_count",
    }
    lifecycle_counts = tuple(lifecycle[field] for field in lifecycle_count_fields)
    residue = tuple(
        lifecycle[field]
        for field in (
            "owned_process_residue_count",
            "pipe_residue_count",
            "temporary_file_residue_count",
        )
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in observation_counts + lifecycle_counts
        )
        or byte_count != packet_count * LIVE_AEC_FRAME_BYTES
        or classification
        not in {
            NEAR_END_DISTINGUISHED_CLASS,
            SELF_OUTPUT_OR_AMBIGUOUS_CLASS,
        }
        or lifecycle.get("cleanup_class")
        not in {
            "route_owned_cleanup_clear",
            "no_runtime_started",
            "cleanup_not_proven",
        }
        or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != 0
            )
            for value in residue
        )
        or (
            lifecycle.get("cleanup_class") == "cleanup_not_proven"
            and any(value is not None for value in residue)
        )
        or (
            lifecycle.get("cleanup_class") != "cleanup_not_proven"
            and any(value != 0 for value in residue)
        )
    ):
        raise LiveAecCaptureError("live_aec_helper_result_invalid")
    if result_class in {
        "processed_near_end_pcm_observed",
        "processed_near_end_silence_observed",
    }:
        if (
            not 100 <= window_ms <= 5000
            or attempt_count != packet_count
            or valid_count > attempt_count
            or trusted_count > valid_count
            or trusted_count + ambiguous_count != attempt_count
            or cleanup_failure_count != 0
            or live_capture_used is not True
            or capability_class != "voice_capture_dsp_capability_available"
            or authority.get("exactly_one_aec_owner") is not True
            or lifecycle.get("cleanup_class") != "route_owned_cleanup_clear"
            or any(value != 0 for value in residue)
        ):
            raise LiveAecCaptureError("live_aec_helper_result_invalid")
        expected_classification = (
            NEAR_END_DISTINGUISHED_CLASS
            if packet_count > 0
            and valid_count == attempt_count
            and trusted_count == attempt_count
            and ambiguous_count == 0
            else SELF_OUTPUT_OR_AMBIGUOUS_CLASS
        )
        if classification != expected_classification:
            raise LiveAecCaptureError("live_aec_helper_result_invalid")
    elif (
        live_capture_used is not False
        or window_ms != 0
        or packet_count != 0
        or byte_count != 0
        or classification != SELF_OUTPUT_OR_AMBIGUOUS_CLASS
        or any(
            value != 0
            for value in (
                attempt_count,
                valid_count,
                trusted_count,
                ambiguous_count,
                cleanup_failure_count,
            )
        )
        or authority.get("exactly_one_aec_owner") is not False
    ):
        raise LiveAecCaptureError("live_aec_helper_result_invalid")
    return payload


def _register_one_time_nonce(digest: bytes) -> None:
    with _USED_NONCE_LOCK:
        if digest in _USED_NONCE_DIGESTS or digest in _ACTIVE_NONCE_DIGESTS:
            raise LiveAecCaptureError("live_aec_nonce_reuse_rejected")
        _ACTIVE_NONCE_DIGESTS.add(digest)


def _retire_one_time_nonce(digest: bytes) -> None:
    with _USED_NONCE_LOCK:
        _ACTIVE_NONCE_DIGESTS.discard(digest)
        if digest not in _USED_NONCE_DIGESTS:
            _USED_NONCE_DIGESTS.append(digest)
        if len(_USED_NONCE_DIGESTS) > _USED_NONCE_DIGEST_LIMIT:
            del _USED_NONCE_DIGESTS[:-_USED_NONCE_DIGEST_LIMIT]


def _clear_bytearray(value: bytearray) -> None:
    value[:] = b"\x00" * len(value)


def _utc_now_dotnet_ticks() -> int:
    return time.time_ns() // 100 + 621_355_968_000_000_000


def _current_process_creation_utc_ticks() -> int:
    if platform.system() != "Windows":
        raise LiveAecCaptureError("live_aec_platform_unsupported")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    creation = ctypes.c_ulonglong()
    exit_time = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    if not kernel32.GetProcessTimes(
        kernel32.GetCurrentProcess(),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise LiveAecCaptureError("live_aec_server_identity_unavailable")
    return int(creation.value) + _DOTNET_FILETIME_OFFSET_TICKS


def _stop_owned_process(process: Any) -> None:
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=1.0)
        except Exception:
            raise LiveAecCaptureError("live_aec_process_cleanup_failed") from None


def _read_exact_from_chunk_reader(
    target: bytearray,
    read_chunk: Any,
    *,
    allow_eof: bool = False,
) -> int:
    """Fill one mutable target across fragmented byte-mode reads."""

    total = 0
    while total < len(target):
        chunk = read_chunk(len(target) - total)
        if chunk is None:
            if allow_eof:
                return total
            raise LiveAecCaptureError("live_aec_pipe_read_failed")
        if (
            not isinstance(chunk, bytearray)
            or len(chunk) == 0
            or len(chunk) > len(target) - total
        ):
            if isinstance(chunk, bytearray):
                _clear_bytearray(chunk)
            raise LiveAecCaptureError("live_aec_pipe_read_failed")
        try:
            target[total : total + len(chunk)] = chunk
            total += len(chunk)
        finally:
            _clear_bytearray(chunk)
    return total


def _read_processed_pcm_frames(read_exact: Any) -> dict[str, Any]:
    """Read exact framed PCM while clearing all partial state on failure."""

    pcm = bytearray()
    packet_count = 0
    try:
        while True:
            prefix = bytearray(4)
            try:
                received = read_exact(prefix, allow_eof=True)
                if received == 0:
                    break
                if received != 4:
                    raise LiveAecCaptureError(
                        "live_aec_processed_packet_invalid"
                    )
                frame_length = int.from_bytes(prefix, "little", signed=True)
                if frame_length != LIVE_AEC_FRAME_BYTES:
                    raise LiveAecCaptureError(
                        "live_aec_processed_packet_invalid"
                    )
                frame = bytearray(frame_length)
                try:
                    try:
                        read_exact(frame)
                    except LiveAecCaptureError:
                        raise LiveAecCaptureError(
                            "live_aec_processed_packet_invalid"
                        ) from None
                    if len(pcm) + len(frame) > LIVE_AEC_MAX_CAPTURE_BYTES:
                        raise LiveAecCaptureError(
                            "live_aec_capture_bounds_exceeded"
                        )
                    pcm.extend(frame)
                    packet_count += 1
                finally:
                    _clear_bytearray(frame)
            finally:
                _clear_bytearray(prefix)
        return {"pcm16": pcm, "packet_count": packet_count}
    except Exception:
        _clear_bytearray(pcm)
        raise


class _WindowsProcessedPcmPipeServer:
    """One-shot current-user local pipe server for processed PCM only."""

    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _PIPE_ACCESS_DUPLEX = 0x00000003
    _FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
    _PIPE_TYPE_BYTE = 0x00000000
    _PIPE_READMODE_BYTE = 0x00000000
    _PIPE_WAIT = 0x00000000
    _PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
    _ERROR_PIPE_CONNECTED = 535
    _ERROR_BROKEN_PIPE = 109
    _ERROR_NO_DATA = 232

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", ctypes.c_uint32),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", ctypes.c_int),
        ]

    def __init__(
        self,
        pipe_name: str,
        nonce: bytearray,
        deadline_ms: int,
        expires_utc_ticks: int,
    ) -> None:
        if platform.system() != "Windows":
            raise LiveAecCaptureError("live_aec_platform_unsupported")
        self._pipe_name = pipe_name
        self._nonce = nonce
        self._deadline_ms = deadline_ms
        self._expires_utc_ticks = expires_utc_ticks
        self._handle: int | None = None
        self._security_descriptor = ctypes.c_void_p()
        self._thread: threading.Thread | None = None
        self._expected_client_pid = 0
        self._result: dict[str, Any] | None = None
        self._failure: str | None = None
        self._create()

    def _create(self) -> None:
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
            ctypes.c_int
        )
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            "D:P(A;;GA;;;OW)",
            1,
            ctypes.byref(self._security_descriptor),
            None,
        ):
            raise LiveAecCaptureError("live_aec_pipe_acl_failed")
        attributes = self._SecurityAttributes(
            ctypes.sizeof(self._SecurityAttributes),
            self._security_descriptor,
            0,
        )
        kernel32.CreateNamedPipeW.restype = ctypes.c_void_p
        kernel32.CreateNamedPipeW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(self._SecurityAttributes),
        ]
        handle = kernel32.CreateNamedPipeW(
            "\\\\.\\pipe\\" + self._pipe_name,
            self._PIPE_ACCESS_DUPLEX | self._FILE_FLAG_FIRST_PIPE_INSTANCE,
            self._PIPE_TYPE_BYTE
            | self._PIPE_READMODE_BYTE
            | self._PIPE_WAIT
            | self._PIPE_REJECT_REMOTE_CLIENTS,
            1,
            LIVE_AEC_FRAME_BYTES + 4,
            LIVE_AEC_FRAME_BYTES + 4,
            self._deadline_ms,
            ctypes.byref(attributes),
        )
        if handle == self._INVALID_HANDLE_VALUE:
            self.close()
            raise LiveAecCaptureError("live_aec_pipe_create_failed")
        self._handle = int(handle)

    def start(self, *, expected_client_process_id: int) -> None:
        self._expected_client_pid = expected_client_process_id
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def finish(self, *, timeout_seconds: float) -> dict[str, Any]:
        if self._thread is None:
            raise LiveAecCaptureError("live_aec_pipe_not_started")
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise LiveAecCaptureError("live_aec_pipe_deadline_exceeded")
        if self._failure is not None:
            raise LiveAecCaptureError(self._failure)
        if self._result is None:
            raise LiveAecCaptureError("live_aec_pipe_result_missing")
        return self._result

    def _run(self) -> None:
        pcm = bytearray()
        received_nonce = bytearray(32)
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            connected = kernel32.ConnectNamedPipe(
                ctypes.c_void_p(self._handle),
                None,
            )
            if not connected and ctypes.get_last_error() != self._ERROR_PIPE_CONNECTED:
                raise LiveAecCaptureError("live_aec_pipe_connect_failed")
            client_pid = ctypes.c_uint32()
            if not kernel32.GetNamedPipeClientProcessId(
                ctypes.c_void_p(self._handle),
                ctypes.byref(client_pid),
            ) or int(client_pid.value) != self._expected_client_pid:
                raise LiveAecCaptureError("live_aec_pipe_client_identity_mismatch")
            if _utc_now_dotnet_ticks() >= self._expires_utc_ticks:
                raise LiveAecCaptureError("live_aec_pipe_lease_expired")
            self._read_exact(received_nonce)
            if not hmac.compare_digest(received_nonce, self._nonce):
                raise LiveAecCaptureError("live_aec_pipe_nonce_mismatch")
            if _utc_now_dotnet_ticks() >= self._expires_utc_ticks:
                raise LiveAecCaptureError("live_aec_pipe_lease_expired")
            self._write_all(_LIVE_AEC_ACK)
            self._result = _read_processed_pcm_frames(self._read_exact)
        except LiveAecCaptureError as exc:
            _clear_bytearray(pcm)
            self._failure = exc.failure_class
        except Exception:
            _clear_bytearray(pcm)
            self._failure = "live_aec_pipe_failed"
        finally:
            _clear_bytearray(received_nonce)

    def _read_exact(self, target: bytearray, *, allow_eof: bool = False) -> int:
        return _read_exact_from_chunk_reader(
            target,
            self._read_native_chunk,
            allow_eof=allow_eof,
        )

    def _read_native_chunk(self, maximum_bytes: int) -> bytearray | None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        target = bytearray(maximum_bytes)
        read = ctypes.c_uint32()
        try:
            view = (ctypes.c_ubyte * maximum_bytes).from_buffer(target)
            ok = kernel32.ReadFile(
                ctypes.c_void_p(self._handle),
                view,
                len(view),
                ctypes.byref(read),
                None,
            )
            if not ok:
                error = ctypes.get_last_error()
                if error in {self._ERROR_BROKEN_PIPE, self._ERROR_NO_DATA}:
                    return None
                raise LiveAecCaptureError("live_aec_pipe_read_failed")
            if read.value == 0:
                return None
            result = bytearray(target[: int(read.value)])
            return result
        finally:
            _clear_bytearray(target)

    def _write_all(self, value: bytes) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        written = ctypes.c_uint32()
        buffer = ctypes.create_string_buffer(value)
        if not kernel32.WriteFile(
            ctypes.c_void_p(self._handle),
            buffer,
            len(value),
            ctypes.byref(written),
            None,
        ) or written.value != len(value):
            raise LiveAecCaptureError("live_aec_pipe_ack_failed")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
                ctypes.c_void_p(handle)
            )
        if self._security_descriptor:
            ctypes.WinDLL("kernel32", use_last_error=True).LocalFree(
                self._security_descriptor
            )
            self._security_descriptor = ctypes.c_void_p()
        if (
            self._thread is not None
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(1.0)
            if self._thread.is_alive():
                raise LiveAecCaptureError("live_aec_pipe_cleanup_failed")


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
