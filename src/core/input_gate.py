"""Backend-neutral input gating for voice capture sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
import secrets
import threading
from typing import Any, Mapping


class InputGateError(ValueError):
    """Raised when an input-gate update payload is invalid."""


@dataclass(frozen=True)
class InputGateState:
    """Current decision for whether voice input should be accepted."""

    enabled: bool = True
    reason: str = "default"
    source: str = "local"
    timestamp: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reason",
            _normalize_gate_class(self.reason, "default", "reason"),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_gate_class(self.source, "local", "source"),
        )
        object.__setattr__(self, "timestamp", _normalize_timestamp(self.timestamp))

    def to_payload(self) -> dict[str, bool | float | str | None]:
        """Return a stable protocol payload for app or adapter boundaries."""
        return {
            "type": "input_gate_state",
            "input_enabled": self.enabled,
            "reason": self.reason,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class InputGateEvent:
    """One external or local request to update the input gate."""

    input_enabled: bool
    reason: str = "external"
    source: str = "external"
    timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class UserSpeechCandidateEvidence:
    """Private, text-free evidence for one possible live user-speech turn."""

    candidate_id: str
    source_kind: str
    near_end_evidence_class: str
    window_ms: int
    packet_count: int
    processed_byte_count: int
    frame_bytes: int
    storage_class: str
    aec_or_vad_turn_input_authority: bool
    observed_system_speech_session_id: str
    observed_generation: int
    active_system_speech_session_id: str
    active_generation: int
    playback_event_ref: str
    self_output_observation_ref: str
    self_output_observation_schema_version: str
    session_join_status: str
    post_compare_session_status: str
    self_output_correlation_class: str
    active_session_exclusion_status: str
    cooldown_status: str
    opaque_refs_non_dereferenceable: bool
    decision_owner: str
    acceptance_status: str
    may_materialize_thought_core_turninput: bool

    def __repr__(self) -> str:
        return "<user-speech-candidate-evidence private>"

    __str__ = __repr__

    def __copy__(self) -> object:
        raise TypeError("user-speech candidate evidence cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("user-speech candidate evidence cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("user-speech candidate evidence cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("user-speech candidate evidence cannot be serialized")


@dataclass(frozen=True, slots=True)
class _SystemSpeechLifecycleObservation:
    system_speech_session_id: str
    speech_session_generation: int
    playback_event_ref: str
    lifecycle_state: str


@dataclass(frozen=True, slots=True)
class _SelfOutputObservation:
    self_output_observation_ref: str
    system_speech_session_id: str
    speech_session_generation: int
    playback_event_ref: str


@dataclass(slots=True)
class _PendingTurnInputCapability:
    capability: _TurnInputCapability
    capability_generation: int
    candidate: UserSpeechCandidateEvidence
    candidate_identity: object
    lifecycle: _SystemSpeechLifecycleObservation
    self_output: _SelfOutputObservation


class _TurnInputCapability:
    """Nonserializable process-local authority owned only by one InputGate."""

    __slots__ = ("_owner", "_generation", "_candidate_identity", "_nonce")

    def __init__(
        self,
        *,
        owner: object,
        generation: int,
        candidate_identity: object,
    ) -> None:
        self._owner = owner
        self._generation = generation
        self._candidate_identity = candidate_identity
        self._nonce = object()

    def __repr__(self) -> str:
        return "<turn-input-capability private>"

    __str__ = __repr__

    def __bool__(self) -> bool:
        raise TypeError("turn-input capability has no boolean representation")

    def __copy__(self) -> object:
        raise TypeError("turn-input capability cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("turn-input capability cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("turn-input capability cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("turn-input capability cannot be serialized")


_SYSTEM_SPEECH_SESSION_PATTERN = re.compile(
    r"^system-speech-session:sss_[a-f0-9]{32}$"
)
_PLAYBACK_EVENT_PATTERN = re.compile(r"^playback-event:pe_[a-f0-9]{32}$")
_SELF_OUTPUT_OBSERVATION_PATTERN = re.compile(
    r"^self-output-observation:aso_[a-f0-9]{32}$"
)
_PRIVATE_CANDIDATE_PATTERN = re.compile(r"^ausc_live:cid_[a-f0-9]{32}$")
_GATE_REASON_CLASSES = {
    "capture-only": "capture_only",
    "capture_only": "capture_only",
    "default": "default",
    "external": "external",
    "input_disabled": "input_disabled",
    "manual": "manual",
    "sword_sign": "sword_sign",
}
_GATE_SOURCE_CLASSES = {
    "cli": "cli",
    "external": "external",
    "gesture_bridge": "gesture_bridge",
    "local": "local",
    "sword_voice_agent": "sword_voice_agent",
    "test": "test",
    "web-ui": "web-ui",
    "web_ui": "web-ui",
}
_SYSTEM_SPEECH_LIFECYCLE_FIELDS = {
    "schema_version",
    "system_speech_session_id",
    "speech_session_generation",
    "playback_event_ref",
    "lifecycle_state",
    "queue_handoff_status",
    "queue_completion_status",
    "playback_observation_status",
    "suppression_status",
    "cooldown_status",
    "cooldown_ms",
    "compare_and_release_required",
    "may_start_user_turn",
    "turn_adoption_authority",
    "raw_text_published",
    "text_hash_published",
    "provider_payload_published",
    "path_published",
    "url_published",
    "raw_audio_published",
    "device_identity_published",
    "private_data_published",
}
_SELF_OUTPUT_OBSERVATION_FIELDS = {
    "schema_version",
    "self_output_observation_ref",
    "system_speech_session_id",
    "speech_session_generation",
    "playback_event_ref",
    "observation_status",
    "observation_owner",
    "may_start_user_turn",
    "turn_adoption_authority",
    "raw_private_publication_flags",
}
_LIFECYCLE_TRANSITIONS = {
    "handoff_accepted": {"handoff_accepted", "cooldown"},
    "cooldown": {"cooldown", "released"},
    "released": {"released"},
}


class InputGate:
    """Track whether voice capture should currently accept microphone input."""

    def __init__(
        self,
        initially_enabled: bool = True,
        *,
        reason: str = "default",
        source: str = "local",
        timestamp: float | None = None,
    ) -> None:
        self._state = InputGateState(
            enabled=_expect_bool(initially_enabled, "initially_enabled"),
            reason=_normalize_gate_class(reason, "default", "reason"),
            source=_normalize_gate_class(source, "local", "source"),
            timestamp=_normalize_timestamp(timestamp),
        )
        self._lock = threading.RLock()
        self._capability_owner = object()
        self._capability_generation = 0
        self._lifecycle: _SystemSpeechLifecycleObservation | None = None
        self._self_output: _SelfOutputObservation | None = None
        self._pending: dict[int, _PendingTurnInputCapability] = {}
        self._used_candidate_ids: set[str] = set()
        self._consumed_candidates: dict[str, UserSpeechCandidateEvidence] = {}

    @property
    def state(self) -> InputGateState:
        """Return the current immutable state snapshot."""
        with self._lock:
            return self._state

    def is_enabled(self) -> bool:
        """Return whether input should currently be accepted."""
        with self._lock:
            return self._state.enabled

    def set_input_enabled(
        self,
        enabled: bool,
        *,
        reason: str = "manual",
        source: str = "local",
        timestamp: float | None = None,
    ) -> InputGateState:
        """Set the gate directly and return the resulting state."""
        with self._lock:
            self._state = InputGateState(
                enabled=_expect_bool(enabled, "enabled"),
                reason=_normalize_gate_class(reason, "manual", "reason"),
                source=_normalize_gate_class(source, "local", "source"),
                timestamp=_normalize_timestamp(timestamp),
            )
            return self._state

    def update(self, event: InputGateEvent) -> InputGateState:
        """Apply an input-gate event and return the resulting state."""
        return self.set_input_enabled(
            event.input_enabled,
            reason=event.reason,
            source=event.source,
            timestamp=event.timestamp,
        )

    def update_from_payload(self, payload: Mapping[str, Any]) -> InputGateState:
        """Parse and apply a backend-neutral input-gate payload."""
        return self.update(parse_input_gate_payload(payload))

    def observe_system_speech_lifecycle(self, payload: Mapping[str, Any]) -> None:
        """Ingest one bounded AIT lifecycle observation without granting authority."""
        observation = _parse_system_speech_lifecycle(payload)
        with self._lock:
            current = self._lifecycle
            if current is not None:
                if observation.speech_session_generation < current.speech_session_generation:
                    raise InputGateError("system speech lifecycle is stale")
                if observation.speech_session_generation == current.speech_session_generation:
                    if (
                        observation.system_speech_session_id
                        != current.system_speech_session_id
                        or observation.playback_event_ref != current.playback_event_ref
                        or observation.lifecycle_state
                        not in _LIFECYCLE_TRANSITIONS[current.lifecycle_state]
                    ):
                        raise InputGateError("system speech lifecycle changed incompatibly")
                elif observation.lifecycle_state != "handoff_accepted":
                    raise InputGateError("new system speech generation must start at handoff")
            self._lifecycle = observation

    def observe_self_output_observation(self, payload: Mapping[str, Any]) -> None:
        """Ingest one bounded process-observer correlation without granting authority."""
        observation = _parse_self_output_observation(payload)
        with self._lock:
            lifecycle = self._lifecycle
            if lifecycle is None or not _same_observation_context(
                lifecycle,
                observation,
            ):
                raise InputGateError("self-output observation does not match current lifecycle")
            self._self_output = observation

    def issue_turn_input_capability(
        self,
        candidate: object,
    ) -> object | None:
        """Issue one private capability only after the complete current join."""
        if type(candidate) is not UserSpeechCandidateEvidence:
            return None
        with self._lock:
            lifecycle = self._lifecycle
            self_output = self._self_output
            if (
                lifecycle is None
                or self_output is None
                or not _candidate_matches_current_join(
                    candidate,
                    lifecycle,
                    self_output,
                )
                or candidate.candidate_id in self._used_candidate_ids
                or bool(self._pending)
                or len(self._used_candidate_ids) >= 4096
            ):
                return None
            self._capability_generation += 1
            candidate_identity = object()
            capability = _TurnInputCapability(
                owner=self._capability_owner,
                generation=self._capability_generation,
                candidate_identity=candidate_identity,
            )
            self._pending[id(capability)] = _PendingTurnInputCapability(
                capability=capability,
                capability_generation=self._capability_generation,
                candidate=candidate,
                candidate_identity=candidate_identity,
                lifecycle=lifecycle,
                self_output=self_output,
            )
            return capability

    def consume_turn_input_capability(
        self,
        capability: object,
        candidate: object,
    ) -> bool:
        """Atomically consume an exact pending capability once, or fail closed."""
        if type(capability) is not _TurnInputCapability:
            return False
        if type(candidate) is not UserSpeechCandidateEvidence:
            return False
        with self._lock:
            pending = self._pending.get(id(capability))
            if (
                pending is None
                or pending.capability is not capability
                or pending.candidate is not candidate
                or capability._owner is not self._capability_owner
                or capability._generation != pending.capability_generation
                or capability._candidate_identity is not pending.candidate_identity
            ):
                return False
            del self._pending[id(capability)]
            self._used_candidate_ids.add(candidate.candidate_id)
            lifecycle = self._lifecycle
            self_output = self._self_output
            accepted = bool(
                lifecycle is not None
                and self_output is not None
                and lifecycle == pending.lifecycle
                and self_output == pending.self_output
                and _candidate_matches_current_join(
                    candidate,
                    lifecycle,
                    self_output,
                )
            )
            if accepted:
                self._consumed_candidates[candidate.candidate_id] = candidate
            return accepted

    def cancel_turn_input_capability(
        self,
        capability: object,
        candidate: object,
    ) -> bool:
        """Retire one exact issued-but-unconsumed capability without authority."""
        if type(capability) is not _TurnInputCapability:
            return False
        if type(candidate) is not UserSpeechCandidateEvidence:
            return False
        with self._lock:
            pending = self._pending.get(id(capability))
            if (
                pending is None
                or pending.capability is not capability
                or pending.candidate is not candidate
                or capability._owner is not self._capability_owner
                or capability._generation != pending.capability_generation
                or capability._candidate_identity is not pending.candidate_identity
            ):
                return False
            del self._pending[id(capability)]
            self._used_candidate_ids.add(candidate.candidate_id)
            return True

    def discard_consumed_candidate(self, candidate: object) -> bool:
        """Drop exact post-consume private evidence without producing an audit."""
        if type(candidate) is not UserSpeechCandidateEvidence:
            return False
        with self._lock:
            if self._consumed_candidates.get(candidate.candidate_id) is not candidate:
                return False
            del self._consumed_candidates[candidate.candidate_id]
            return True

    def private_authority_residue_count(self) -> int:
        """Return a class-only count of pending or retained private authority."""
        with self._lock:
            return len(self._pending) + len(self._consumed_candidates)

    def evaluate_live_processed_candidate(
        self,
        *,
        has_speech: bool,
        window_ms: int,
        packet_count: int,
        processed_byte_count: int,
        frame_bytes: int,
    ) -> UserSpeechCandidateEvidence | None:
        """Create private candidate evidence from gate-owned current state.

        VAD and AEC facts can establish only a candidate. They do not issue a
        capability and cannot independently grant TurnInput authority.
        """
        if not isinstance(has_speech, bool) or not has_speech:
            return None
        with self._lock:
            lifecycle = self._lifecycle
            self_output = self._self_output
            if (
                not self._state.enabled
                or lifecycle is None
                or self_output is None
                or lifecycle.lifecycle_state != "released"
                or not _same_observation_context(lifecycle, self_output)
                or bool(self._pending)
                or len(self._used_candidate_ids) >= 4096
            ):
                return None
            candidate = UserSpeechCandidateEvidence(
                candidate_id=f"ausc_live:cid_{secrets.token_hex(16)}",
                source_kind="user_speech_candidate",
                near_end_evidence_class="bounded_processed_near_end_candidate",
                window_ms=window_ms,
                packet_count=packet_count,
                processed_byte_count=processed_byte_count,
                frame_bytes=frame_bytes,
                storage_class="in_memory_ephemeral",
                aec_or_vad_turn_input_authority=False,
                observed_system_speech_session_id=(
                    lifecycle.system_speech_session_id
                ),
                observed_generation=lifecycle.speech_session_generation,
                active_system_speech_session_id=(
                    lifecycle.system_speech_session_id
                ),
                active_generation=lifecycle.speech_session_generation,
                playback_event_ref=lifecycle.playback_event_ref,
                self_output_observation_ref=(
                    self_output.self_output_observation_ref
                ),
                self_output_observation_schema_version=(
                    "audio_self_output_observation.v0"
                ),
                session_join_status="current_match",
                post_compare_session_status="current_unchanged",
                self_output_correlation_class="not_self_output",
                active_session_exclusion_status=(
                    "explicitly_excluded_from_candidate"
                ),
                cooldown_status="clear",
                opaque_refs_non_dereferenceable=True,
                decision_owner="ai_talk_core_input_gate",
                acceptance_status="accepted_user_speech_candidate",
                may_materialize_thought_core_turninput=True,
            )
            if not _candidate_matches_current_join(
                candidate,
                lifecycle,
                self_output,
            ):
                return None
            return candidate

    def build_consumed_candidate_audit(
        self,
        candidate: object,
    ) -> dict[str, object] | None:
        """Build a canonical text-free audit only after one-use consumption."""
        if type(candidate) is not UserSpeechCandidateEvidence:
            return None
        with self._lock:
            if self._consumed_candidates.get(candidate.candidate_id) is not candidate:
                return None
            del self._consumed_candidates[candidate.candidate_id]
        candidate_suffix = candidate.candidate_id.rsplit("_", 1)[-1]
        return {
            "schema_version": "accepted_user_speech_candidate_input_gate.v0",
            "candidate_id": candidate.candidate_id,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "proof_layer": "private_live_input_gate_decision",
            "route_id": (
                "A-SELF-OUTPUT-AWARENESS-CANONICAL-INPUT-GATE-SESSION-JOIN-01"
            ),
            "candidate_route": "private_user_speech_input_gate",
            "source_kind": "user_speech_candidate",
            "speaker_role": "user_candidate",
            "recognition_summary": {
                "source_label": "speech.user_candidate_001",
                "recognition_summary_ref": (
                    f"user-speech-candidate-summary:rsc_{candidate_suffix}"
                ),
                "recognition_summary_class": "redacted_local_offline_candidate",
                "recognized_text_class": "present_redacted",
                "recognized_text_length_bucket": "short",
                "language_bucket": "ja",
                "confidence_bucket": "unknown",
                "raw_text_in_shared_artifact": False,
            },
            "text_publication": {
                "text_publication_policy": "text_redacted_or_absent",
                "text_provenance_class": "redacted_or_absent",
                "expected_sample_text": None,
                "recognized_text": None,
                "content_match_text": None,
                "non_sample_or_live_text_policy": "protected_or_redacted",
            },
            "self_output_context": {
                "self_output_correlation_class": "not_self_output",
                "session_join_class": (
                    "current_active_session_explicitly_excluded"
                ),
                "bubble_tts_parity_class": "not_authority_for_user_speech",
                "bubble_or_tts_as_user_speech_authority": False,
            },
            "input_gate": {
                "input_gate_decision_owner": "ai_talk_core_input_gate",
                "input_gate_decision_class": "accepted_user_speech_candidate",
                "normal_turn_block_reason": None,
            },
            "acceptance_decision": {
                "acceptance_status": "accepted_user_speech_candidate",
                "may_materialize_thought_core_turninput": True,
                "private_text_handoff_required": True,
                "shared_artifact_contains_text": False,
                "thought_core_turninput_materialized": False,
                "thought_core_turninput_count": 0,
                "thought_core_turninput_ref": None,
                "turn_materialization_route": (
                    "separate_private_runtime_handoff_required"
                ),
            },
            "runtime_session_join": {
                "near_end_evidence": {
                    "evidence_class": candidate.near_end_evidence_class,
                    "window_ms": candidate.window_ms,
                    "packet_count": candidate.packet_count,
                    "processed_byte_count": candidate.processed_byte_count,
                    "frame_bytes": candidate.frame_bytes,
                    "storage_class": candidate.storage_class,
                    "aec_or_vad_turn_input_authority": False,
                },
                "system_speech_session_join": {
                    "observed_system_speech_session_id": (
                        candidate.observed_system_speech_session_id
                    ),
                    "observed_generation": candidate.observed_generation,
                    "active_system_speech_session_id": (
                        candidate.active_system_speech_session_id
                    ),
                    "active_generation": candidate.active_generation,
                    "playback_event_ref": candidate.playback_event_ref,
                    "self_output_observation_ref": (
                        candidate.self_output_observation_ref
                    ),
                    "self_output_observation_schema_version": (
                        candidate.self_output_observation_schema_version
                    ),
                    "session_join_status": candidate.session_join_status,
                    "post_compare_session_status": (
                        candidate.post_compare_session_status
                    ),
                    "self_output_correlation_class": (
                        candidate.self_output_correlation_class
                    ),
                    "active_session_exclusion_status": (
                        candidate.active_session_exclusion_status
                    ),
                    "cooldown_status": candidate.cooldown_status,
                    "opaque_refs_non_dereferenceable": True,
                },
                "one_use_gate": {
                    "decision_owner": "ai_talk_core_input_gate",
                    "compared_candidate_id": candidate.candidate_id,
                    "candidate_identity_compare_status": "matched",
                    "session_identity_compare_status": "matched",
                    "generation_compare_status": "matched",
                    "candidate_use_state": "unused",
                    "compare_and_release_status": "succeeded",
                    "one_use_consume_status": "consumed",
                    "acceptance_status": "accepted_user_speech_candidate",
                    "may_materialize_thought_core_turninput": True,
                    "turninput_materialization_limit": 1,
                },
            },
            "post_decision_audit": {
                "serialized_contract_role": "post_decision_audit_only",
                "process_local_capability_owner": "ai_talk_core_input_gate",
                "materialization_executor": (
                    "later_ai_talk_core_process_local_code"
                ),
                "process_local_one_use_capability_serialized": False,
                "capability_mint_authority": False,
                "capability_grant_authority": False,
                "capability_transport_authority": False,
                "internal_atomic_compare_completed": True,
                "output_status": (
                    "eligible_for_later_process_local_materialization"
                ),
                "may_materialize_thought_core_turninput": True,
                "thought_core_turninput_materialized": False,
                "thought_core_turninput_count": 0,
            },
            "redaction_guards": {
                "raw_audio_included": False,
                "raw_media_included": False,
                "raw_transcript_included": False,
                "raw_recognized_text_included": False,
                "private_path_included": False,
                "provider_payload_included": False,
                "browser_storage_included": False,
                "token_or_secret_included": False,
                "home_control_action_authority_included": False,
            },
            "non_claims": [
                "not_runtime_turn",
                "not_audio_capture",
                "not_stt_execution",
                "not_user_heard_proof",
                "not_home_control_action",
                "not_touchdesigner_action",
                "not_source_git_adoption",
                "not_readiness_or_rr003_pass",
            ],
            "raw_private_publication_flags": False,
        }


def parse_input_gate_payload(payload: Mapping[str, Any]) -> InputGateEvent:
    """Parse an input-gate control payload.

    The accepted control keys are intentionally generic so an integration app can
    map gesture, keyboard, network, or UI state into this protocol without this
    project importing any gesture-specific package.
    """
    if not isinstance(payload, Mapping):
        raise InputGateError("input gate payload must be a mapping")

    enabled_field = _find_enabled_field(payload)
    if enabled_field is None:
        raise InputGateError(
            "input gate payload must include input_enabled, mic_enabled, or enabled"
        )
    enabled_key, enabled_value = enabled_field
    reason = _normalize_gate_class(payload.get("reason"), "external", "reason")
    source = _normalize_gate_class(payload.get("source"), "external", "source")
    timestamp = _normalize_timestamp(payload.get("timestamp"))
    return InputGateEvent(
        input_enabled=_expect_bool(enabled_value, enabled_key),
        reason=reason,
        source=source,
        timestamp=timestamp,
    )


def _parse_system_speech_lifecycle(
    payload: Mapping[str, Any],
) -> _SystemSpeechLifecycleObservation:
    if not isinstance(payload, Mapping):
        raise InputGateError("system speech lifecycle must be a mapping")
    if set(payload) != _SYSTEM_SPEECH_LIFECYCLE_FIELDS:
        raise InputGateError("system speech lifecycle fields are invalid")
    session_id = _expect_pattern(
        payload.get("system_speech_session_id"),
        _SYSTEM_SPEECH_SESSION_PATTERN,
        "system_speech_session_id",
    )
    generation = _expect_positive_int(
        payload.get("speech_session_generation"),
        "speech_session_generation",
    )
    playback_ref = _expect_pattern(
        payload.get("playback_event_ref"),
        _PLAYBACK_EVENT_PATTERN,
        "playback_event_ref",
    )
    lifecycle_state = payload.get("lifecycle_state")
    if lifecycle_state not in _LIFECYCLE_TRANSITIONS:
        raise InputGateError("lifecycle_state is invalid")
    required = {
        "schema_version": "ait_system_speech_lifecycle.v0",
        "queue_handoff_status": "accepted",
        "playback_observation_status": "not_observed",
        "compare_and_release_required": True,
        "may_start_user_turn": False,
        "turn_adoption_authority": False,
        "raw_text_published": False,
        "text_hash_published": False,
        "provider_payload_published": False,
        "path_published": False,
        "url_published": False,
        "raw_audio_published": False,
        "device_identity_published": False,
        "private_data_published": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise InputGateError("system speech lifecycle guards are invalid")
    if lifecycle_state == "handoff_accepted":
        expected = ("pending", "active", "clear")
    elif lifecycle_state == "cooldown":
        expected = ("callback_observed", "active", "active")
    else:
        expected = ("callback_observed", "released", "elapsed")
    actual = (
        payload.get("queue_completion_status"),
        payload.get("suppression_status"),
        payload.get("cooldown_status"),
    )
    if actual != expected:
        raise InputGateError("system speech lifecycle state guards are invalid")
    cooldown_ms = payload.get("cooldown_ms")
    if (
        isinstance(cooldown_ms, bool)
        or not isinstance(cooldown_ms, int)
        or not 0 <= cooldown_ms <= 2000
    ):
        raise InputGateError("cooldown_ms is invalid")
    return _SystemSpeechLifecycleObservation(
        system_speech_session_id=session_id,
        speech_session_generation=generation,
        playback_event_ref=playback_ref,
        lifecycle_state=str(lifecycle_state),
    )


def _parse_self_output_observation(
    payload: Mapping[str, Any],
) -> _SelfOutputObservation:
    if not isinstance(payload, Mapping):
        raise InputGateError("self-output observation must be a mapping")
    if set(payload) != _SELF_OUTPUT_OBSERVATION_FIELDS:
        raise InputGateError("self-output observation fields are invalid")
    required = {
        "schema_version": "audio_self_output_observation.v0",
        "observation_status": "current",
        "observation_owner": "leased_tts_process_observer",
        "may_start_user_turn": False,
        "turn_adoption_authority": False,
        "raw_private_publication_flags": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise InputGateError("self-output observation guards are invalid")
    return _SelfOutputObservation(
        self_output_observation_ref=_expect_pattern(
            payload.get("self_output_observation_ref"),
            _SELF_OUTPUT_OBSERVATION_PATTERN,
            "self_output_observation_ref",
        ),
        system_speech_session_id=_expect_pattern(
            payload.get("system_speech_session_id"),
            _SYSTEM_SPEECH_SESSION_PATTERN,
            "system_speech_session_id",
        ),
        speech_session_generation=_expect_positive_int(
            payload.get("speech_session_generation"),
            "speech_session_generation",
        ),
        playback_event_ref=_expect_pattern(
            payload.get("playback_event_ref"),
            _PLAYBACK_EVENT_PATTERN,
            "playback_event_ref",
        ),
    )


def _same_observation_context(
    lifecycle: _SystemSpeechLifecycleObservation,
    self_output: _SelfOutputObservation,
) -> bool:
    return (
        lifecycle.system_speech_session_id == self_output.system_speech_session_id
        and lifecycle.speech_session_generation
        == self_output.speech_session_generation
        and lifecycle.playback_event_ref == self_output.playback_event_ref
    )


def _candidate_matches_current_join(
    candidate: UserSpeechCandidateEvidence,
    lifecycle: _SystemSpeechLifecycleObservation,
    self_output: _SelfOutputObservation,
) -> bool:
    if lifecycle.lifecycle_state != "released":
        return False
    if not _same_observation_context(lifecycle, self_output):
        return False
    if not _PRIVATE_CANDIDATE_PATTERN.fullmatch(candidate.candidate_id):
        return False
    if (
        candidate.source_kind != "user_speech_candidate"
        or candidate.near_end_evidence_class
        != "bounded_processed_near_end_candidate"
        or isinstance(candidate.window_ms, bool)
        or not 1 <= candidate.window_ms <= 3000
        or isinstance(candidate.packet_count, bool)
        or not 1 <= candidate.packet_count <= 10_000
        or isinstance(candidate.processed_byte_count, bool)
        or not 1 <= candidate.processed_byte_count <= 10 * 1024 * 1024
        or isinstance(candidate.frame_bytes, bool)
        or not 1 <= candidate.frame_bytes <= 4096
        or candidate.processed_byte_count < candidate.frame_bytes
        or candidate.storage_class != "in_memory_ephemeral"
        or candidate.aec_or_vad_turn_input_authority is not False
    ):
        return False
    if (
        candidate.observed_system_speech_session_id
        != lifecycle.system_speech_session_id
        or candidate.observed_generation != lifecycle.speech_session_generation
        or candidate.active_system_speech_session_id
        != lifecycle.system_speech_session_id
        or candidate.active_generation != lifecycle.speech_session_generation
        or candidate.playback_event_ref != lifecycle.playback_event_ref
        or candidate.self_output_observation_ref
        != self_output.self_output_observation_ref
        or candidate.self_output_observation_schema_version
        != "audio_self_output_observation.v0"
    ):
        return False
    return (
        candidate.session_join_status == "current_match"
        and candidate.post_compare_session_status == "current_unchanged"
        and candidate.self_output_correlation_class == "not_self_output"
        and candidate.active_session_exclusion_status
        == "explicitly_excluded_from_candidate"
        and candidate.cooldown_status == "clear"
        and candidate.opaque_refs_non_dereferenceable is True
        and candidate.decision_owner == "ai_talk_core_input_gate"
        and candidate.acceptance_status == "accepted_user_speech_candidate"
        and candidate.may_materialize_thought_core_turninput is True
    )


def _expect_pattern(value: Any, pattern: re.Pattern[str], field_name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise InputGateError(f"{field_name} is invalid")
    return value


def _expect_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputGateError(f"{field_name} must be a positive integer")
    return value


def _find_enabled_field(payload: Mapping[str, Any]) -> tuple[str, Any] | None:
    for key in ("input_enabled", "mic_enabled", "enabled"):
        if key in payload:
            return key, payload[key]
    return None


def _expect_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise InputGateError(f"{field_name} must be a boolean")


def _normalize_gate_class(value: Any, fallback: str, field_name: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise InputGateError(f"{field_name} must be a string")
    normalized = value.strip().lower()
    if field_name == "reason":
        return _GATE_REASON_CLASSES.get(normalized, fallback)
    if field_name == "source":
        return _GATE_SOURCE_CLASSES.get(normalized, fallback)
    raise InputGateError("input gate class field is invalid")


def _normalize_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputGateError("timestamp must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1_000_000_000_000_000:
        raise InputGateError("timestamp is outside the allowed range")
    return normalized
