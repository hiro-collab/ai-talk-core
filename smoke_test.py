"""Minimal smoke tests for the CLI."""

from __future__ import annotations

from collections.abc import Mapping
import io
import contextlib
import copy
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys
import threading
import time
import unittest
from typing import Any
from unittest import mock

from src.core.handoff_bridge import (
    build_handoff_metadata,
    build_handoff_payload,
    get_default_handoff_output_path,
    get_default_handoff_text_path,
    load_handoff_bundle,
    normalize_handoff_source,
    render_handoff_prompt,
    save_handoff_bundle,
    save_handoff_payload,
)
from src.core.agent_instruction import build_agent_instruction
from src.core.dependency_status import (
    format_dependency_status,
    get_dependency_status,
)
from src.core.finalization import (
    has_stable_duration_for_final,
    maybe_finalize_on_interrupt,
    maybe_finalize_on_silence,
    normalize_transcript_text,
    required_repeat_count_for_final,
    should_mark_result_final,
)
from src.core.input_gate import (
    InputGate,
    InputGateError,
    InputGateEvent,
    LIVE_CAPTURE_MODE_AEC,
    LIVE_CAPTURE_MODE_NS_AGC,
    UserSpeechCandidateEvidence,
    parse_input_gate_payload,
)
from src.core.events import (
    TurnEventBus,
    emit_event,
    read_event_log_events,
    sanitize_event_payload,
    text_payload_facts,
)
from src.core.torch_pin_plan import format_torch_pin_plan, get_torch_pin_plan
from src.main import (
    build_input_gate_data,
    build_doctor_status,
    build_mic_profile_list_data,
    build_mic_tuning_data,
    build_torch_pin_status,
    format_doctor_status,
    format_input_gate_state,
    format_mic_profile_list,
    format_mic_loop_tuning,
    format_runtime_status,
    format_transcription_result,
    print_agent_instruction_only,
    print_runtime_note,
    resolve_mic_loop_tuning,
    validate_final_stable_seconds,
    validate_mic_profile,
)
from src.io.audio import should_retry_model_load_on_cpu
from src.io.audio import AudioInputError
from src.io.audio import AudioEnvironmentError
from src.io.audio import AudioTranscriptionError
from src.io.audio import get_runtime_status
import src.io.aec_reference as aec_reference_module
from src.io.aec_reference import (
    AecReferenceError,
    LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES,
    LiveAecCaptureError,
    LiveAecProcessedCapture,
    capture_live_aec_processed_pcm,
    evaluate_synthetic_aec_candidate,
    get_adopted_live_aec_owner_selection,
    select_synthetic_aec_owner,
    validate_live_aec_owner_selection,
)
from src.io.microphone import (
    LIVE_AEC_MICROPHONE_BACKEND,
    LiveMicrophoneCandidateWindow,
    MICROPHONE_DEVICE_LIST_TIMEOUT_SECONDS,
    capture_live_microphone_candidate_window,
    capture_microphone_chunk,
    classify_live_pcm16_signal,
    get_microphone_runtime_status,
    get_recording_timeout_seconds,
    has_detectable_speech_pcm16,
    list_ffmpeg_dshow_audio_devices,
    record_microphone_audio,
    resolve_microphone_backend,
    validate_vad_aggressiveness,
)
from src.codex_handoff import render_handoff_output
from src.codex_runner import (
    build_template_command,
    normalize_command_args,
    resolve_runner_command,
    validate_runner_command_available,
)
from src.ollama_runner import build_ollama_command
from src.core.pipeline import (
    AudioChunk,
    TranscriptionPipeline,
    TranscriptionResult,
    clear_transcription_pipeline_cache,
    get_cached_transcription_pipeline,
)
from src.core.session import MicLoopSession, MicLoopTuning
from src.drivers import DriverRequest, DriverResponse, DriverResult, dispatch_driver_request
from src.runners.common import emit_driver_result, execute_runner_command
from src.web.app import (
    ENABLE_PROCESS_SHUTDOWN_CONFIG,
    LOCAL_API_TOKEN_ENV,
    WEB_PRESET_CONFIG,
    WEB_MAX_RECORDING_CHUNK_BYTES,
    WEB_RECORDING_CHUNK_RETENTION_SECONDS,
    WEB_MAX_RECORDING_CHUNKS,
    RuntimeStatusWriter,
    build_runtime_status_payload,
    build_input_gate_response,
    create_app,
    format_sse_event,
    get_recording_chunk_dir,
    parse_bearer_token,
    prune_recording_chunk_cache,
    render_page,
)
from src.web.transcription_service import (
    WebTranscriptionRequest,
    WebTranscriptionResponse,
    process_web_transcription,
)


PROJECT_ROOT = Path(__file__).resolve().parent

build_codex_payload = build_handoff_payload
build_codex_instruction = build_agent_instruction
get_default_codex_output_path = get_default_handoff_output_path
get_default_codex_text_path = get_default_handoff_text_path
load_codex_handoff_bundle = load_handoff_bundle
render_codex_prompt = render_handoff_prompt
save_codex_handoff_bundle = save_handoff_bundle
save_codex_payload = save_handoff_payload


def remove_path_with_retry(path: Path, *, attempts: int = 5, delay: float = 0.05) -> None:
    """Remove a test artifact, retrying briefly for transient Windows file locks."""
    for attempt in range(attempts):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI and capture its output."""
    command = [sys.executable, "-m", "src.main", *args]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


SYSTEM_SPEECH_SESSION_ID = (
    "system-speech-session:sss_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)
PLAYBACK_EVENT_REF = "playback-event:pe_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
SELF_OUTPUT_OBSERVATION_REF = (
    "self-output-observation:aso_cccccccccccccccccccccccccccccccc"
)
LIFECYCLE_TRANSPORT_SOURCE = "self-output-awareness-controller"
LIFECYCLE_TRANSPORT_TURN_ID = "web_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def build_system_speech_lifecycle(
    state: str = "released",
    *,
    generation: int = 7,
    session_id: str = SYSTEM_SPEECH_SESSION_ID,
    playback_ref: str = PLAYBACK_EVENT_REF,
) -> dict[str, object]:
    if state == "handoff_accepted":
        completion, suppression, cooldown = "pending", "active", "clear"
    elif state == "cooldown":
        completion, suppression, cooldown = "callback_observed", "active", "active"
    else:
        completion, suppression, cooldown = "callback_observed", "released", "elapsed"
    return {
        "schema_version": "ait_system_speech_lifecycle.v0",
        "system_speech_session_id": session_id,
        "speech_session_generation": generation,
        "playback_event_ref": playback_ref,
        "lifecycle_state": state,
        "queue_handoff_status": "accepted",
        "queue_completion_status": completion,
        "playback_observation_status": "not_observed",
        "suppression_status": suppression,
        "cooldown_status": cooldown,
        "cooldown_ms": 500,
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


def build_self_output_observation(
    *,
    generation: int = 7,
    session_id: str = SYSTEM_SPEECH_SESSION_ID,
    playback_ref: str = PLAYBACK_EVENT_REF,
) -> dict[str, object]:
    return {
        "schema_version": "audio_self_output_observation.v0",
        "self_output_observation_ref": SELF_OUTPUT_OBSERVATION_REF,
        "system_speech_session_id": session_id,
        "speech_session_generation": generation,
        "playback_event_ref": playback_ref,
        "observation_status": "current",
        "observation_owner": "leased_tts_process_observer",
        "may_start_user_turn": False,
        "turn_adoption_authority": False,
        "raw_private_publication_flags": False,
    }


def observe_gate_lifecycle(
    gate: InputGate,
    payload: Mapping[str, object],
    *,
    turn_id: str = LIFECYCLE_TRANSPORT_TURN_ID,
    wall_timestamp: str = "2026-07-13T12:00:00.000Z",
) -> None:
    gate.observe_system_speech_lifecycle(
        payload,
        transport_source=LIFECYCLE_TRANSPORT_SOURCE,
        transport_turn_id=turn_id,
        transport_wall_timestamp=wall_timestamp,
    )


def post_current_lifecycle_events(
    client: Any,
    headers: Mapping[str, str],
) -> None:
    for index, state in enumerate(
        ("handoff_accepted", "cooldown", "released")
    ):
        response = client.post(
            "/api/events/ingest",
            headers=headers,
            json={
                "event": "swordAgentSystemSpeechLifecycleV0",
                "turn_id": "web_abcdef0123456789abcdef0123456789",
                "source": "self-output-awareness-controller",
                "payload": build_system_speech_lifecycle(state),
                "client_timestamp_wall": (
                    f"2026-07-13T12:00:0{index}.000Z"
                ),
                "client_timestamp_monotonic": 12.5 + index,
                "client_performance_now": 12_500.0 + index,
            },
        )
        if response.status_code != 202:
            raise AssertionError("canonical lifecycle setup failed")
    observation = client.post(
        "/api/events/ingest",
        headers=headers,
        json={
            "event": "audioSelfOutputObservationV0",
            "payload": build_self_output_observation(),
        },
    )
    if observation.status_code != 202:
        raise AssertionError("canonical self-output setup failed")


def build_user_speech_candidate(
    **overrides: object,
) -> UserSpeechCandidateEvidence:
    values: dict[str, object] = {
        "candidate_id": "ausc_live:cid_dddddddddddddddddddddddddddddddd",
        "source_kind": "user_speech_candidate",
        "near_end_evidence_class": "bounded_processed_near_end_candidate",
        "window_ms": 1000,
        "packet_count": 50,
        "processed_byte_count": 16000,
        "frame_bytes": 320,
        "storage_class": "in_memory_ephemeral",
        "aec_or_vad_turn_input_authority": False,
        "observed_system_speech_session_id": SYSTEM_SPEECH_SESSION_ID,
        "observed_generation": 7,
        "active_system_speech_session_id": SYSTEM_SPEECH_SESSION_ID,
        "active_generation": 7,
        "playback_event_ref": PLAYBACK_EVENT_REF,
        "self_output_observation_ref": SELF_OUTPUT_OBSERVATION_REF,
        "self_output_observation_schema_version": "audio_self_output_observation.v0",
        "session_join_status": "current_match",
        "post_compare_session_status": "current_unchanged",
        "self_output_correlation_class": "not_self_output",
        "active_session_exclusion_status": "explicitly_excluded_from_candidate",
        "cooldown_status": "clear",
        "opaque_refs_non_dereferenceable": True,
        "decision_owner": "ai_talk_core_input_gate",
        "acceptance_status": "accepted_user_speech_candidate",
        "may_materialize_thought_core_turninput": True,
    }
    values.update(overrides)
    return UserSpeechCandidateEvidence(**values)  # type: ignore[arg-type]


def prepare_current_input_gate() -> InputGate:
    gate = InputGate()
    for state, wall_timestamp in (
        ("handoff_accepted", "2026-07-13T12:00:00.000Z"),
        ("cooldown", "2026-07-13T12:00:01.000Z"),
        ("released", "2026-07-13T12:00:02.000Z"),
    ):
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle(state),
            wall_timestamp=wall_timestamp,
        )
    gate.observe_self_output_observation(build_self_output_observation())
    return gate


def run_handoff_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the Codex handoff CLI and capture its output."""
    command = [sys.executable, "-m", "src.codex_handoff", *args]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_agent_handoff_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the generic agent handoff CLI and capture its output."""
    command = [sys.executable, "-m", "src.agent_handoff", *args]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_runner_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the Codex runner CLI and capture its output."""
    command = [sys.executable, "-m", "src.codex_runner", *args]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def run_agent_runner_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the generic agent runner CLI and capture its output."""
    command = [sys.executable, "-m", "src.agent_runner", *args]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class SmokeTests(unittest.TestCase):
    """Smoke tests for the current CLI behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        cls.client = cls.app.test_client()

    def local_api_headers(self) -> dict[str, str]:
        return {"X-AI-Core-Token": self.app.config["LOCAL_API_TOKEN"]}

    def test_web_app_can_use_configured_local_api_token(self) -> None:
        """External local adapters should be able to use an operator-provided token."""
        with mock.patch.dict(
            "os.environ",
            {LOCAL_API_TOKEN_ENV: "fixed-local-api-token"},
        ):
            app = create_app()
        self.assertEqual(app.config["LOCAL_API_TOKEN"], "fixed-local-api-token")

    def test_sample_audio_succeeds(self) -> None:
        """Sample audio should transcribe successfully."""
        result = run_cli("data/sample_audio.mp3", "--language", "ja")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("こんにちは", result.stdout)

    def test_synthetic_aec_selects_one_owner_without_retaining_audio(self) -> None:
        """Synthetic metrics may select one owner but never gain live authority."""
        reference = [0.4, -0.4, 0.4, -0.4]
        near_end = [0.2, 0.2, -0.2, -0.2]
        microphone = [left + right for left, right in zip(reference, near_end)]
        windows_processed = [
            near + (0.1 * render) for near, render in zip(near_end, reference)
        ]
        webrtc_processed = [
            near + (0.2 * render) for near, render in zip(near_end, reference)
        ]
        candidates = [
            evaluate_synthetic_aec_candidate(
                owner_class="windows_voice_capture_dsp",
                reference_samples=reference,
                near_end_samples=near_end,
                microphone_samples=microphone,
                processed_samples=windows_processed,
            ),
            evaluate_synthetic_aec_candidate(
                owner_class="webrtc_apm_aec3",
                reference_samples=reference,
                near_end_samples=near_end,
                microphone_samples=microphone,
                processed_samples=webrtc_processed,
            ),
        ]
        self.assertEqual(candidates[0]["echo_convergence_db"], 20.0)
        self.assertEqual(candidates[1]["echo_convergence_db"], 13.979)
        self.assertEqual(candidates[0]["near_end_preservation_ratio"], 1.0)
        self.assertEqual(candidates[1]["near_end_preservation_ratio"], 1.0)

        result = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=candidates,
        )

        self.assertEqual(
            result,
            {
                "schema_version": "synthetic_aec_owner_selection.v0",
                "proof_ceiling": "synthetic_aec_owner_selection_only",
                "result_class": "synthetic_aec_owner_selected",
                "processing_inventory_class": "known_no_owner",
                "candidate_count": 2,
                "selected_owner_class": "windows_voice_capture_dsp",
                "selected_echo_convergence_db": 20.0,
                "selected_near_end_preservation_ratio": 1.0,
                "exactly_one_aec_owner": True,
                "observation_only": False,
                "render_reference_may_create_turn_input": False,
                "raw_audio_persisted": False,
                "live_audio_used": False,
            },
        )
        self.assertNotIn("reference_samples", result)
        self.assertNotIn("processed_samples", result)

    def test_synthetic_aec_fails_closed_for_unknown_double_and_ambiguous(self) -> None:
        """Unknown, double-owner, and tied candidates remain observation-only."""
        tied_candidates = [
            {
                "owner_class": "windows_voice_capture_dsp",
                "echo_convergence_db": 12.0,
                "near_end_preservation_ratio": 0.95,
            },
            {
                "owner_class": "webrtc_apm_aec3",
                "echo_convergence_db": 12.0,
                "near_end_preservation_ratio": 0.95,
            },
        ]
        cases = (
            (
                "unknown",
                [],
                "aec_processing_unknown_observation_only",
            ),
            (
                "double_owner",
                ["windows_voice_capture_dsp", "webrtc_apm_aec3"],
                "aec_double_owner_rejected",
            ),
            (
                "known_no_owner",
                [],
                "synthetic_aec_owner_ambiguous",
            ),
        )
        for inventory, active, expected in cases:
            with self.subTest(expected=expected):
                result = select_synthetic_aec_owner(
                    processing_inventory_class=inventory,
                    active_owner_classes=active,
                    candidates=tied_candidates,
                )
                self.assertEqual(result["result_class"], expected)
                self.assertIsNone(result["selected_owner_class"])
                self.assertFalse(result["exactly_one_aec_owner"])
                self.assertTrue(result["observation_only"])
                self.assertFalse(result["render_reference_may_create_turn_input"])

    def test_synthetic_aec_rejects_private_or_nonseparable_inputs(self) -> None:
        """Invalid owner text and coupled fixtures return only fixed failures."""
        with self.assertRaisesRegex(AecReferenceError, "^aec_owner_class_invalid$"):
            evaluate_synthetic_aec_candidate(
                owner_class=r"private C:\\audio\\device",
                reference_samples=[0.4, -0.4, 0.4, -0.4],
                near_end_samples=[0.2, 0.2, -0.2, -0.2],
                microphone_samples=[0.6, -0.2, 0.2, -0.6],
                processed_samples=[0.24, 0.16, -0.16, -0.24],
            )
        with self.assertRaisesRegex(
            AecReferenceError,
            "^synthetic_aec_fixture_not_separable$",
        ):
            evaluate_synthetic_aec_candidate(
                owner_class="windows_voice_capture_dsp",
                reference_samples=[0.4, -0.4, 0.4, -0.4],
                near_end_samples=[0.2, -0.2, 0.2, -0.2],
                microphone_samples=[0.6, -0.6, 0.6, -0.6],
                processed_samples=[0.24, -0.24, 0.24, -0.24],
            )

    def test_synthetic_aec_any_malformed_candidate_blocks_valid_subset(self) -> None:
        """One malformed candidate invalidates the whole comparison set."""
        valid = {
            "owner_class": "windows_voice_capture_dsp",
            "echo_convergence_db": 20.0,
            "near_end_preservation_ratio": 1.0,
        }
        malformed_candidates = (
            {
                "owner_class": "windows_voice_capture_dsp",
                "echo_convergence_db": "private-same-owner-marker",
                "near_end_preservation_ratio": 1.0,
            },
            {
                "owner_class": "webrtc_apm_aec3",
                "echo_convergence_db": 13.979,
                "near_end_preservation_ratio": "private-other-owner-marker",
            },
        )
        for malformed in malformed_candidates:
            with self.subTest(owner=malformed["owner_class"]):
                result = select_synthetic_aec_owner(
                    processing_inventory_class="known_no_owner",
                    active_owner_classes=[],
                    candidates=[valid, malformed],
                )
                self.assertEqual(
                    result["result_class"],
                    "synthetic_aec_candidate_invalid_observation_only",
                )
                self.assertEqual(result["candidate_count"], 2)
                self.assertIsNone(result["selected_owner_class"])
                self.assertTrue(result["observation_only"])
                rendered = repr(result)
                self.assertNotIn("private-same-owner-marker", rendered)
                self.assertNotIn("private-other-owner-marker", rendered)

    def test_synthetic_aec_active_owner_mismatch_is_observation_only(self) -> None:
        """A synthetic winner cannot silently replace the one active owner."""
        result = select_synthetic_aec_owner(
            processing_inventory_class="known_single_owner",
            active_owner_classes=["webrtc_apm_aec3"],
            candidates=[
                {
                    "owner_class": "windows_voice_capture_dsp",
                    "echo_convergence_db": 18.0,
                    "near_end_preservation_ratio": 0.98,
                }
            ],
        )
        self.assertEqual(
            result["result_class"],
            "aec_active_owner_mismatch_observation_only",
        )
        self.assertIsNone(result["selected_owner_class"])
        self.assertTrue(result["observation_only"])

    def test_synthetic_aec_duplicate_owner_is_ambiguous(self) -> None:
        """Two summaries for one owner cannot masquerade as a comparison."""
        result = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=[
                {
                    "owner_class": "windows_voice_capture_dsp",
                    "echo_convergence_db": 20.0,
                    "near_end_preservation_ratio": 1.0,
                },
                {
                    "owner_class": "windows_voice_capture_dsp",
                    "echo_convergence_db": 15.0,
                    "near_end_preservation_ratio": 0.95,
                },
            ],
        )
        self.assertEqual(result["result_class"], "synthetic_aec_owner_ambiguous")
        self.assertIsNone(result["selected_owner_class"])
        self.assertTrue(result["observation_only"])

    def test_synthetic_aec_requires_near_end_preservation(self) -> None:
        """Echo reduction cannot qualify a candidate that suppresses near-end speech."""
        reference = [0.4, -0.4, 0.4, -0.4]
        near_end = [0.2, 0.2, -0.2, -0.2]
        candidate = evaluate_synthetic_aec_candidate(
            owner_class="windows_voice_capture_dsp",
            reference_samples=reference,
            near_end_samples=near_end,
            microphone_samples=[
                render + near for render, near in zip(reference, near_end)
            ],
            processed_samples=[
                (0.02 * render) + (0.7 * near)
                for render, near in zip(reference, near_end)
            ],
        )
        self.assertGreater(candidate["echo_convergence_db"], 20)
        self.assertEqual(candidate["near_end_preservation_ratio"], 0.7)

        result = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=[candidate],
        )
        self.assertEqual(
            result["result_class"], "synthetic_aec_candidate_unqualified"
        )
        self.assertIsNone(result["selected_owner_class"])
        self.assertTrue(result["observation_only"])

    def test_synthetic_aec_selection_threshold_boundaries(self) -> None:
        """Convergence, preservation, and winner margin boundaries are exact."""

        def candidate(owner: str, convergence: float, preservation: float) -> dict:
            return {
                "owner_class": owner,
                "echo_convergence_db": convergence,
                "near_end_preservation_ratio": preservation,
            }

        single_cases = (
            (6.0, 1.0, "synthetic_aec_owner_selected"),
            (5.999, 1.0, "synthetic_aec_candidate_unqualified"),
            (30.0, 0.85, "synthetic_aec_owner_selected"),
            (30.0, 0.849, "synthetic_aec_candidate_unqualified"),
        )
        for convergence, preservation, expected in single_cases:
            with self.subTest(convergence=convergence, preservation=preservation):
                result = select_synthetic_aec_owner(
                    processing_inventory_class="known_no_owner",
                    active_owner_classes=[],
                    candidates=[
                        candidate(
                            "windows_voice_capture_dsp",
                            convergence,
                            preservation,
                        )
                    ],
                )
                self.assertEqual(result["result_class"], expected)

        margin_selected = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=[
                candidate("windows_voice_capture_dsp", 10.0, 1.0),
                candidate("webrtc_apm_aec3", 9.0, 1.0),
            ],
        )
        self.assertEqual(
            margin_selected["result_class"], "synthetic_aec_owner_selected"
        )
        margin_ambiguous = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=[
                candidate("windows_voice_capture_dsp", 10.0, 1.0),
                candidate("webrtc_apm_aec3", 9.001, 1.0),
            ],
        )
        self.assertEqual(
            margin_ambiguous["result_class"], "synthetic_aec_owner_ambiguous"
        )

        half_step = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=[
                candidate("windows_voice_capture_dsp", 6.2345, 0.9005),
            ],
        )
        self.assertEqual(
            half_step,
            {
                "schema_version": "synthetic_aec_owner_selection.v0",
                "proof_ceiling": "synthetic_aec_owner_selection_only",
                "result_class": "synthetic_aec_owner_selected",
                "processing_inventory_class": "known_no_owner",
                "candidate_count": 1,
                "selected_owner_class": "windows_voice_capture_dsp",
                "selected_echo_convergence_db": 6.235,
                "selected_near_end_preservation_ratio": 0.901,
                "exactly_one_aec_owner": True,
                "observation_only": False,
                "render_reference_may_create_turn_input": False,
                "raw_audio_persisted": False,
                "live_audio_used": False,
            },
        )

    def test_synthetic_aec_vector_failures_do_not_echo_private_markers(self) -> None:
        """Conversion, iteration, and overflow failures expose one fixed class."""
        import traceback

        private_marker = "private-vector-marker-do-not-echo"

        class PrivateFloat:
            def __float__(self) -> float:
                raise ValueError(private_marker)

        class PrivateIterable:
            def __iter__(self):
                raise RuntimeError(private_marker)

        invalid_vectors = (
            [PrivateFloat(), 0.0, 0.0, 0.0],
            PrivateIterable(),
            [10**10_000, 0.0, 0.0, 0.0],
        )
        for vector in invalid_vectors:
            with self.subTest(vector_type=type(vector).__name__):
                try:
                    evaluate_synthetic_aec_candidate(
                        owner_class="windows_voice_capture_dsp",
                        reference_samples=vector,
                        near_end_samples=[0.2, 0.2, -0.2, -0.2],
                        microphone_samples=[0.6, -0.2, 0.2, -0.6],
                        processed_samples=[0.24, 0.16, -0.16, -0.24],
                    )
                except AecReferenceError as error:
                    self.assertEqual(
                        error.failure_class,
                        "synthetic_aec_vector_invalid",
                    )
                    formatted = traceback.format_exc()
                    self.assertNotIn(private_marker, str(error))
                    self.assertNotIn(private_marker, formatted)
                else:
                    self.fail("invalid synthetic vector was accepted")

    def test_live_aec_private_lease_and_pcm_are_in_memory_and_cleared(self) -> None:
        """The fake live boundary keeps lease data off argv and clears buffers."""

        selection = select_synthetic_aec_owner(
            processing_inventory_class="known_no_owner",
            active_owner_classes=[],
            candidates=[
                {
                    "owner_class": "windows_voice_capture_dsp",
                    "echo_convergence_db": 20.0,
                    "near_end_preservation_ratio": 1.0,
                }
            ],
        )
        created_servers = []
        created_processes = []
        helper_observation = {"packet_count": 1, "processed_byte_count": 320}

        class FakePrivateStdin:
            def __init__(self) -> None:
                self.buffer = bytearray()

            def write(self, value: bytes | bytearray) -> int:
                self.buffer.extend(value)
                return len(value)

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class FakeProcess:
            def __init__(self, command: list[str], **kwargs) -> None:
                self.command = command
                self.pid = 4321
                self._private_stdin = FakePrivateStdin()
                self.stdin = self._private_stdin
                self.returncode = None
                self.lease_keys: set[str] = set()
                self.processing_mode_class = ""
                self.private_input_cleared = False
                created_processes.append(self)

            def communicate(self, timeout: float):
                del timeout
                payload = json.loads(bytes(self._private_stdin.buffer).decode("utf-8"))
                self.lease_keys = set(payload)
                self.processing_mode_class = str(
                    payload.get("processing_mode_class") or ""
                )
                self._private_stdin.buffer[:] = b"\x00" * len(
                    self._private_stdin.buffer
                )
                self.private_input_cleared = all(
                    value == 0 for value in self._private_stdin.buffer
                )
                self.returncode = 0
                result = {
                    "schema_version": "voice_capture_dsp_aec_observation.v0",
                    "result_class": "processed_near_end_pcm_observed",
                    "observation": dict(helper_observation),
                }
                return json.dumps(result).encode("utf-8"), b""

            def poll(self):
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -1

            def kill(self) -> None:
                self.returncode = -9

            def wait(self, timeout: float):
                del timeout
                return self.returncode

        class FakeServer:
            def __init__(
                self,
                pipe_name,
                nonce,
                deadline_ms,
                expires_utc_ticks,
            ) -> None:
                self.pipe_name = pipe_name
                self.nonce = nonce
                self.deadline_ms = deadline_ms
                self.expires_utc_ticks = expires_utc_ticks
                self.expected_pid = None
                self.close_count = 0
                self.result_pcm = None
                created_servers.append(self)

            def start(self, *, expected_client_process_id: int) -> None:
                self.expected_pid = expected_client_process_id

            def finish(self, *, timeout_seconds: float):
                del timeout_seconds
                self.result_pcm = bytearray([1, 2] * 160)
                return {"pcm16": self.result_pcm, "packet_count": 1}

            def close(self) -> None:
                self.close_count += 1

        helper_path = Path(__file__)
        with (
            mock.patch(
                "src.io.aec_reference._resolve_powershell_executable",
                return_value="pwsh",
            ),
            mock.patch(
                "src.io.aec_reference._current_process_creation_utc_ticks",
                return_value=123,
            ),
            mock.patch(
                "src.io.aec_reference._utc_now_dotnet_ticks",
                return_value=456,
            ),
        ):
            capture = capture_live_aec_processed_pcm(
                owner_selection=selection,
                window_ms=100,
                deadline_ms=1000,
                helper_path=helper_path,
                popen_factory=FakeProcess,
                server_factory=FakeServer,
            )

        process = created_processes[0]
        server = created_servers[0]
        command_text = " ".join(process.command)
        self.assertNotIn("sword-aec-", command_text)
        self.assertNotIn("0123456789abcdef", command_text)
        self.assertEqual(
            process.lease_keys,
            {
                "pipe_name",
                "nonce",
                "server_process_id",
                "server_creation_utc_ticks",
                "expires_utc_ticks",
                "aec_owner_selection_class",
                "selected_owner_class",
                "processing_mode_class",
            },
        )
        self.assertTrue(process.private_input_cleared)
        self.assertEqual(
            process.processing_mode_class,
            LIVE_CAPTURE_MODE_AEC,
        )
        self.assertEqual(server.expected_pid, 4321)
        self.assertGreater(server.expires_utc_ticks, 456)
        self.assertEqual(server.close_count, 1)
        self.assertTrue(all(value == 0 for value in server.nonce))
        self.assertEqual(capture.storage_class, "in_memory_ephemeral")
        self.assertEqual(capture.packet_count, 1)
        self.assertEqual(capture.processed_byte_count, 320)
        self.assertTrue(any(value != 0 for value in capture.pcm16))
        capture.clear()
        self.assertTrue(all(value == 0 for value in capture.pcm16))

        class ZeroPacketProcess(FakeProcess):
            def communicate(self, timeout: float):
                del timeout
                self.returncode = 0
                result = {
                    "schema_version": "voice_capture_dsp_aec_observation.v0",
                    "result_class": "processed_near_end_silence_observed",
                    "observation": {
                        "packet_count": 0,
                        "processed_byte_count": 0,
                    },
                }
                return json.dumps(result).encode("utf-8"), b""

        class ZeroPacketServer(FakeServer):
            def finish(self, *, timeout_seconds: float):
                del timeout_seconds
                self.result_pcm = bytearray()
                return {"pcm16": self.result_pcm, "packet_count": 0}

        with (
            mock.patch("src.io.aec_reference._resolve_powershell_executable", return_value="pwsh"),
            mock.patch("src.io.aec_reference._current_process_creation_utc_ticks", return_value=123),
            mock.patch("src.io.aec_reference._utc_now_dotnet_ticks", return_value=456),
        ):
            with self.assertRaisesRegex(
                LiveAecCaptureError,
                "^live_aec_processed_packet_invalid$",
            ):
                capture_live_aec_processed_pcm(
                    owner_selection=selection,
                    window_ms=100,
                    deadline_ms=1000,
                    helper_path=helper_path,
                    popen_factory=ZeroPacketProcess,
                    server_factory=ZeroPacketServer,
                )
        zero_packet_server = created_servers[-1]
        self.assertEqual(zero_packet_server.close_count, 1)

        with self.assertRaisesRegex(
            LiveAecCaptureError,
            "^live_aec_processing_mode_invalid$",
        ):
            capture_live_aec_processed_pcm(
                owner_selection=selection,
                window_ms=100,
                deadline_ms=1000,
                processing_mode_class="caller_selected_user_intent",
            )

        helper_observation.update(packet_count=2, processed_byte_count=640)
        with (
            mock.patch("src.io.aec_reference._resolve_powershell_executable", return_value="pwsh"),
            mock.patch("src.io.aec_reference._current_process_creation_utc_ticks", return_value=123),
            mock.patch("src.io.aec_reference._utc_now_dotnet_ticks", return_value=456),
        ):
            with self.assertRaisesRegex(
                LiveAecCaptureError,
                "^live_aec_count_mismatch$",
            ):
                capture_live_aec_processed_pcm(
                    owner_selection=selection,
                    window_ms=100,
                    deadline_ms=1000,
                    helper_path=helper_path,
                    popen_factory=FakeProcess,
                    server_factory=FakeServer,
                )
        failed_server = created_servers[-1]
        self.assertEqual(failed_server.close_count, 1)
        self.assertTrue(
            all(value == 0 for value in failed_server.result_pcm or bytearray())
        )

        stuck_processes = []

        class StuckProcess(FakeProcess):
            def __init__(self, command: list[str], **kwargs) -> None:
                super().__init__(command, **kwargs)
                self.terminate_count = 0
                self.kill_count = 0
                self.wait_count = 0
                stuck_processes.append(self)

            def communicate(self, timeout: float):
                raise subprocess.TimeoutExpired(self.command, timeout)

            def poll(self):
                return None

            def terminate(self) -> None:
                self.terminate_count += 1

            def kill(self) -> None:
                self.kill_count += 1
                self._private_stdin.buffer[:] = b"\x00" * len(
                    self._private_stdin.buffer
                )

            def wait(self, timeout: float):
                self.wait_count += 1
                raise subprocess.TimeoutExpired(self.command, timeout)

        fixed_nonce = bytes(range(32))
        fixed_digest = aec_reference_module.hashlib.sha256(fixed_nonce).digest()
        helper_observation.update(packet_count=1, processed_byte_count=320)
        with (
            mock.patch("src.io.aec_reference.secrets.token_bytes", return_value=fixed_nonce),
            mock.patch("src.io.aec_reference._resolve_powershell_executable", return_value="pwsh"),
            mock.patch("src.io.aec_reference._current_process_creation_utc_ticks", return_value=123),
            mock.patch("src.io.aec_reference._utc_now_dotnet_ticks", return_value=456),
        ):
            with self.assertRaisesRegex(
                LiveAecCaptureError,
                "^live_aec_cleanup_failed$",
            ):
                capture_live_aec_processed_pcm(
                    owner_selection=selection,
                    window_ms=100,
                    deadline_ms=1000,
                    helper_path=helper_path,
                    popen_factory=StuckProcess,
                    server_factory=FakeServer,
                )
        stuck = stuck_processes[-1]
        stuck_server = created_servers[-1]
        self.assertEqual(stuck.terminate_count, 1)
        self.assertEqual(stuck.kill_count, 1)
        self.assertEqual(stuck.wait_count, 2)
        self.assertEqual(stuck_server.close_count, 1)
        self.assertTrue(all(value == 0 for value in stuck_server.nonce))
        self.assertTrue(
            all(value == 0 for value in stuck._private_stdin.buffer)
        )
        with aec_reference_module._USED_NONCE_LOCK:
            self.assertNotIn(
                fixed_digest,
                aec_reference_module._ACTIVE_NONCE_DIGESTS,
            )
            while fixed_digest in aec_reference_module._USED_NONCE_DIGESTS:
                aec_reference_module._USED_NONCE_DIGESTS.remove(fixed_digest)

    def test_live_aec_owner_selection_fails_before_transport(self) -> None:
        """Missing, unknown, or double-owner state cannot launch the helper."""

        invalid = (
            None,
            select_synthetic_aec_owner(
                processing_inventory_class="unknown",
                active_owner_classes=[],
                candidates=[],
            ),
            select_synthetic_aec_owner(
                processing_inventory_class="double_owner",
                active_owner_classes=[
                    "windows_voice_capture_dsp",
                    "webrtc_apm_aec3",
                ],
                candidates=[],
            ),
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(LiveAecCaptureError) as raised:
                    validate_live_aec_owner_selection(value)
                self.assertIn(
                    raised.exception.failure_class,
                    {
                        "live_aec_owner_selection_missing",
                        "live_aec_owner_selection_invalid",
                    },
                )

    def test_live_aec_nonce_replay_registry_rejects_active_and_used_digest(self) -> None:
        """The Core owner rejects one nonce while active and after retirement."""

        digest = b"bounded-replay-digest-for-test"
        with aec_reference_module._USED_NONCE_LOCK:
            aec_reference_module._ACTIVE_NONCE_DIGESTS.discard(digest)
            while digest in aec_reference_module._USED_NONCE_DIGESTS:
                aec_reference_module._USED_NONCE_DIGESTS.remove(digest)
        try:
            aec_reference_module._register_one_time_nonce(digest)
            with self.assertRaisesRegex(
                LiveAecCaptureError,
                "^live_aec_nonce_reuse_rejected$",
            ):
                aec_reference_module._register_one_time_nonce(digest)
            aec_reference_module._retire_one_time_nonce(digest)
            with self.assertRaisesRegex(
                LiveAecCaptureError,
                "^live_aec_nonce_reuse_rejected$",
            ):
                aec_reference_module._register_one_time_nonce(digest)
        finally:
            with aec_reference_module._USED_NONCE_LOCK:
                aec_reference_module._ACTIVE_NONCE_DIGESTS.discard(digest)
                while digest in aec_reference_module._USED_NONCE_DIGESTS:
                    aec_reference_module._USED_NONCE_DIGESTS.remove(digest)

    def test_live_aec_source_contract_is_local_private_and_single_instance(self) -> None:
        """The native pipe shape keeps identity, ACL, and lease data bounded."""

        source = Path(aec_reference_module.__file__).read_text(encoding="utf-8")
        self.assertIn('"D:P(A;;GA;;;OW)"', source)
        self.assertIn("_FILE_FLAG_FIRST_PIPE_INSTANCE", source)
        self.assertIn("_PIPE_REJECT_REMOTE_CLIENTS", source)
        self.assertIn("GetNamedPipeClientProcessId", source)
        self.assertIn("hmac.compare_digest", source)
        self.assertIn("_ACTIVE_NONCE_DIGESTS", source)
        self.assertGreaterEqual(
            source.count("_utc_now_dotnet_ticks() >= self._expires_utc_ticks"),
            2,
        )
        command_slice = source[source.index("command = [") : source.index(
            "process = popen_factory"
        )]
        self.assertNotIn("pipe_name", command_slice)
        self.assertNotIn("nonce", command_slice)
        self.assertNotIn("server_process_id", command_slice)

    def test_live_aec_reader_handles_fragmentation_and_rejects_bad_frames(self) -> None:
        """Mutable reader coverage includes split/coalesced and truncated frames."""

        class ChunkReader:
            def __init__(self, payload: bytes, sizes: list[int]) -> None:
                self.buffer = bytearray(payload)
                self.sizes = list(sizes)

            def __call__(self, maximum_bytes: int) -> bytearray | None:
                if not self.buffer:
                    return None
                requested = self.sizes.pop(0) if self.sizes else maximum_bytes
                count = min(maximum_bytes, requested, len(self.buffer))
                result = bytearray(self.buffer[:count])
                self.buffer[:count] = b"\x00" * count
                del self.buffer[:count]
                return result

        def exact_reader(reader: ChunkReader):
            def read_exact(target: bytearray, *, allow_eof: bool = False) -> int:
                return aec_reference_module._read_exact_from_chunk_reader(
                    target,
                    reader,
                    allow_eof=allow_eof,
                )

            return read_exact

        nonce = bytes(range(32))
        nonce_reader = ChunkReader(nonce, [1, 2, 5, 7, 17])
        nonce_target = bytearray(32)
        self.assertEqual(
            aec_reference_module._read_exact_from_chunk_reader(
                nonce_target,
                nonce_reader,
            ),
            32,
        )
        self.assertEqual(bytes(nonce_target), nonce)
        nonce_target[:] = b"\x00" * len(nonce_target)

        frame = bytes([1, 2] * 160)
        framed = (320).to_bytes(4, "little", signed=True) + frame
        coalesced_reader = ChunkReader(framed + framed, [10_000])
        coalesced = aec_reference_module._read_processed_pcm_frames(
            exact_reader(coalesced_reader)
        )
        self.assertEqual(coalesced["packet_count"], 2)
        self.assertEqual(len(coalesced["pcm16"]), 640)
        coalesced["pcm16"][:] = b"\x00" * len(coalesced["pcm16"])

        split_reader = ChunkReader(framed, [1, 1, 2, 3, 5, 7, 11, 293])
        split = aec_reference_module._read_processed_pcm_frames(
            exact_reader(split_reader)
        )
        self.assertEqual(split["packet_count"], 1)
        self.assertEqual(len(split["pcm16"]), 320)
        split["pcm16"][:] = b"\x00" * len(split["pcm16"])

        invalid_payloads = {
            "truncated_prefix": (320).to_bytes(4, "little")[:2],
            "truncated_frame": (320).to_bytes(4, "little") + frame[:100],
            "oversize_length": (322).to_bytes(4, "little") + bytes(322),
            "late_truncated_frame": framed
            + (320).to_bytes(4, "little")
            + frame[:100],
        }
        original_clear = aec_reference_module._clear_bytearray
        for case_name, payload in invalid_payloads.items():
            cleared_lengths: list[int] = []

            def clearing_spy(value: bytearray) -> None:
                length = len(value)
                original_clear(value)
                if all(item == 0 for item in value):
                    cleared_lengths.append(length)

            reader = ChunkReader(payload, [1, 2, 7, 13, 301, 10_000])
            with self.subTest(case=case_name), mock.patch(
                "src.io.aec_reference._clear_bytearray",
                side_effect=clearing_spy,
            ):
                with self.assertRaisesRegex(
                    LiveAecCaptureError,
                    "^live_aec_processed_packet_invalid$",
                ):
                    aec_reference_module._read_processed_pcm_frames(
                        exact_reader(reader)
                    )
                self.assertTrue(cleared_lengths)
                if case_name == "late_truncated_frame":
                    self.assertGreaterEqual(cleared_lengths.count(320), 2)

    def test_live_aec_owned_process_cleanup_converges_or_fails_closed(self) -> None:
        """Only the owned helper is terminated, then killed if it cannot converge."""

        class FakeProcess:
            def __init__(self, fail_waits: int) -> None:
                self.fail_waits = fail_waits
                self.terminate_count = 0
                self.kill_count = 0
                self.wait_count = 0

            def terminate(self) -> None:
                self.terminate_count += 1

            def kill(self) -> None:
                self.kill_count += 1

            def wait(self, timeout: float) -> int:
                self.assert_timeout = timeout
                self.wait_count += 1
                if self.wait_count <= self.fail_waits:
                    raise subprocess.TimeoutExpired("fixed", timeout)
                return 0

        graceful = FakeProcess(fail_waits=0)
        aec_reference_module._stop_owned_process(graceful)
        self.assertEqual(graceful.terminate_count, 1)
        self.assertEqual(graceful.kill_count, 0)
        self.assertEqual(graceful.wait_count, 1)

        escalated = FakeProcess(fail_waits=1)
        aec_reference_module._stop_owned_process(escalated)
        self.assertEqual(escalated.terminate_count, 1)
        self.assertEqual(escalated.kill_count, 1)
        self.assertEqual(escalated.wait_count, 2)

        blocked = FakeProcess(fail_waits=2)
        with self.assertRaisesRegex(
            LiveAecCaptureError,
            "^live_aec_process_cleanup_failed$",
        ):
            aec_reference_module._stop_owned_process(blocked)

    def test_live_aec_microphone_adapter_is_explicit_and_pathless(self) -> None:
        """The live adapter returns one pathless chunk and never changes auto."""

        selection = {
            "result_class": "synthetic_aec_owner_selected",
            "selected_owner_class": "windows_voice_capture_dsp",
            "exactly_one_aec_owner": True,
            "observation_only": False,
            "raw_audio_persisted": False,
            "live_audio_used": False,
        }
        captured = {}

        def fake_capture(**kwargs):
            captured.update(kwargs)
            return LiveAecProcessedCapture(
                pcm16=bytearray([1, 2] * 160),
                packet_count=1,
            )

        chunk = capture_microphone_chunk(
            output_path=Path("private-should-not-exist.wav"),
            duration=1,
            backend=LIVE_AEC_MICROPHONE_BACKEND,
            aec_owner_selection=selection,
            live_aec_capture=fake_capture,
        )
        self.assertIsNone(chunk.path)
        self.assertEqual(chunk.storage_class, "in_memory_ephemeral")
        self.assertEqual(chunk.sample_rate, 16_000)
        self.assertFalse(chunk.turn_input_authority)
        self.assertEqual(
            chunk.turn_input_authority_class,
            "processed_near_end_observation_only",
        )
        self.assertIs(captured["owner_selection"], selection)
        self.assertEqual(captured["window_ms"], 1000)
        self.assertEqual(captured["deadline_ms"], 2000)
        self.assertNotEqual(resolve_microphone_backend("auto"), LIVE_AEC_MICROPHONE_BACKEND)
        with self.assertRaisesRegex(AudioInputError, "in-memory only"):
            record_microphone_audio(
                output_path=Path("private-should-not-exist.wav"),
                duration=1,
                backend=LIVE_AEC_MICROPHONE_BACKEND,
            )
        with self.assertRaisesRegex(
            AudioEnvironmentError,
            "^live_aec_owner_selection_missing$",
        ):
            capture_microphone_chunk(
                output_path=Path("private-should-not-exist.wav"),
                duration=1,
                backend=LIVE_AEC_MICROPHONE_BACKEND,
            )

        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        model = mock.Mock()
        model.transcribe.return_value = {"text": "accepted"}
        pipeline.model = model
        with self.assertRaisesRegex(AudioInputError, "observation-only"):
            pipeline.transcribe_chunk(chunk, language="ja")
        self.assertTrue(all(value == 0 for value in chunk.pcm16 or bytearray()))
        self.assertEqual(model.transcribe.call_count, 0)

        with self.assertRaisesRegex(AudioInputError, "metadata is invalid"):
            AudioChunk(
                path=None,
                source="microphone",
                pcm16=bytearray([1, 2] * 160),
                sample_rate=16_000,
                storage_class="in_memory_ephemeral",
            )


    def test_live_aec_cli_without_private_selection_fails_before_capture(self) -> None:
        """CLI selection cannot manufacture or expose the private owner lease."""

        result = run_cli(
            "--mic",
            "--duration",
            "1",
            "--mic-backend",
            LIVE_AEC_MICROPHONE_BACKEND,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stdout.strip(),
            "Environment error: live_aec_owner_selection_missing",
        )
        self.assertEqual(result.stderr, "")
        self.assertNotIn("sword-aec-", result.stdout)

    def test_missing_file_fails_with_input_error(self) -> None:
        """Missing files should return an input error."""
        result = run_cli("no_such_file.wav")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: audio file not found", result.stdout)

    def test_invalid_model_fails_with_input_error(self) -> None:
        """Invalid model names should return an input error."""
        result = run_cli("data/sample_audio.mp3", "--model", "notamodel")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: invalid Whisper model name", result.stdout)

    def test_command_only_outputs_instruction_text(self) -> None:
        """command-only mode should print only the normalized instruction."""
        result = run_cli("data/sample_audio.mp3", "--language", "ja", "--command-only")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("こんにちは", result.stdout)
        self.assertNotIn("[command]", result.stdout)

    def test_instruction_only_alias_outputs_instruction_text(self) -> None:
        """instruction-only alias should behave like command-only."""
        result = run_cli("data/sample_audio.mp3", "--language", "ja", "--instruction-only")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("こんにちは", result.stdout)
        self.assertNotIn("[command]", result.stdout)

    def test_command_output_writes_payload_json(self) -> None:
        """command-output should save a Codex payload JSON file."""
        output_path = PROJECT_ROOT / ".cache" / "tests" / "command_payload.json"
        text_path = output_path.with_suffix(".txt")
        if output_path.exists():
            remove_path_with_retry(output_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        result = run_cli(
            "data/sample_audio.mp3",
            "--language",
            "ja",
            "--command-output",
            str(output_path),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(output_path.exists())
        self.assertTrue(text_path.exists())
        payload_json = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertIn("こんにちは", payload_json["transcript"])
        self.assertEqual(payload_json["command"], payload_json["transcript"].strip())
        remove_path_with_retry(output_path)
        remove_path_with_retry(text_path)

    def test_handoff_output_alias_writes_payload_json(self) -> None:
        """handoff-output alias should save the same payload bundle."""
        output_path = PROJECT_ROOT / ".cache" / "tests" / "handoff_payload.json"
        text_path = output_path.with_suffix(".txt")
        if output_path.exists():
            remove_path_with_retry(output_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        result = run_cli(
            "data/sample_audio.mp3",
            "--language",
            "ja",
            "--handoff-output",
            str(output_path),
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertTrue(output_path.exists())
        self.assertTrue(text_path.exists())
        remove_path_with_retry(output_path)
        remove_path_with_retry(text_path)

    def test_iterations_requires_mic_loop(self) -> None:
        """Iterations should only be accepted with mic-loop."""
        result = run_cli("--iterations", "2", "data/sample_audio.mp3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: --iterations can only be used with --mic-loop", result.stdout)

    def test_iterations_must_be_positive(self) -> None:
        """Mic-loop iterations must be greater than zero."""
        result = run_cli("--mic-loop", "--duration", "1", "--iterations", "0")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: --iterations must be greater than 0", result.stdout)

    def test_vad_aggressiveness_must_be_in_supported_range(self) -> None:
        """Mic-loop VAD aggressiveness should be validated."""
        result = run_cli("--mic-loop", "--duration", "1", "--vad-aggressiveness", "9")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: VAD aggressiveness must be one of: 0, 1, 2, 3", result.stdout)

    def test_mic_profile_must_be_supported_value(self) -> None:
        """Mic-loop profile should reject unknown values."""
        result = run_cli("--mic-loop", "--duration", "1", "--mic-profile", "fastish")
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "Input error: --mic-profile must be one of: responsive, balanced, strict, low_latency",
            result.stdout,
        )

    def test_list_mic_profiles_prints_available_profiles(self) -> None:
        """Profile listing should print all available tuning presets."""
        result = run_cli("--list-mic-profiles")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Available mic-loop profiles:", result.stdout)
        self.assertIn("responsive", result.stdout)
        self.assertIn("balanced", result.stdout)
        self.assertIn("strict", result.stdout)
        self.assertIn("low_latency", result.stdout)

    def test_list_mic_profiles_can_return_json(self) -> None:
        """Profile listing should support JSON output."""
        result = run_cli("--list-mic-profiles", "--mic-tuning-format", "json")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload[0]["profile"], "responsive")
        self.assertIn("description", payload[0])

    def test_show_mic_tuning_uses_profile_defaults(self) -> None:
        """show-mic-tuning should print the resolved default preset values."""
        result = run_cli("--show-mic-tuning", "--mic-profile", "strict")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "[mic-tuning] profile=strict vad_aggressiveness=3 final_stable_seconds=10",
            result.stdout,
        )

    def test_show_mic_tuning_applies_explicit_overrides(self) -> None:
        """show-mic-tuning should reflect CLI overrides over preset defaults."""
        result = run_cli(
            "--show-mic-tuning",
            "--mic-profile",
            "responsive",
            "--vad-aggressiveness",
            "3",
            "--final-stable-seconds",
            "9",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "[mic-tuning] profile=responsive vad_aggressiveness=3 final_stable_seconds=9",
            result.stdout,
        )

    def test_show_mic_tuning_can_return_json(self) -> None:
        """Resolved tuning should support JSON output."""
        result = run_cli(
            "--show-mic-tuning",
            "--mic-profile",
            "balanced",
            "--mic-tuning-format",
            "json",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["profile"], "balanced")
        self.assertEqual(payload["vad_aggressiveness"], 2)
        self.assertEqual(payload["final_stable_seconds"], 8)

    def test_show_input_gate_can_return_json(self) -> None:
        """Input-gate status should be inspectable without starting audio capture."""
        result = run_cli(
            "--show-input-gate",
            "--input-disabled",
            "--input-gate-reason",
            "sword_sign",
            "--input-gate-format",
            "json",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["input_enabled"])
        self.assertEqual(payload["reason"], "sword_sign")
        self.assertEqual(payload["source"], "cli")

    def test_show_runtime_status_can_return_json(self) -> None:
        """Runtime status should support JSON output."""
        result = run_cli("--show-runtime-status", "--runtime-status-format", "json")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("ffmpeg_available", payload)
        self.assertIn("ffprobe_available", payload)
        self.assertIn("nvidia_smi_available", payload)
        self.assertIn("torch_cuda_available", payload)
        self.assertIn("transcription_device", payload)
        self.assertIn("suggested_action", payload)

    def test_show_dependency_status_can_return_json(self) -> None:
        """Dependency status should support JSON output."""
        result = run_cli("--show-dependency-status", "--dependency-status-format", "json")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("direct_dependencies", payload)
        self.assertIn("installed_versions", payload)
        self.assertIn("torch_direct_dependency", payload)

    def test_doctor_can_return_json(self) -> None:
        """Doctor output should support JSON output."""
        result = run_cli("--doctor", "--doctor-format", "json")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("runtime", payload)
        self.assertIn("microphone", payload)
        self.assertIn("dependencies", payload)

    def test_torch_pin_plan_can_return_json(self) -> None:
        """Torch pin plan output should support JSON output."""
        result = run_cli("--show-torch-pin-plan", "--torch-pin-plan-format", "json")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("steps", payload)
        self.assertIn("command_examples", payload)

    def test_final_stable_seconds_must_be_positive(self) -> None:
        """Mic-loop stable duration threshold should be validated."""
        result = run_cli("--mic-loop", "--duration", "1", "--final-stable-seconds", "0")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: --final-stable-seconds must be greater than 0", result.stdout)

    def test_no_trim_silence_argument_is_accepted(self) -> None:
        """no-trim-silence should parse and follow normal validation flow."""
        result = run_cli("--mic", "--duration", "1", "--no-trim-silence", "data/sample_audio.mp3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error: audio_file cannot be used together with --mic", result.stdout)

    def test_web_index_loads(self) -> None:
        """Web UI index page should load."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("ai_core Web UI", page)
        self.assertIn("upload_instruction_only", page)
        self.assertIn("record_instruction_only", page)
        self.assertIn("record_gate_auto", page)
        self.assertIn("record_device_id", page)
        self.assertIn("record_echo_cancellation", page)
        self.assertIn("record_noise_suppression", page)
        self.assertIn("record_auto_gain_control", page)
        self.assertIn("upload_save_handoff", page)
        self.assertIn("record_save_handoff", page)
        self.assertIn("data-api-doctor", page)
        self.assertIn("data-api-input-gate", page)
        self.assertIn("data-api-recording-chunk", page)
        self.assertIn("data-api-events-ingest", page)
        self.assertIn("data-api-events", page)
        self.assertIn("data-api-token", page)
        self.assertIn("data-web-preset", page)
        self.assertIn("app.css", page)
        self.assertIn("app.js", page)
        self.assertIn("待機中", page)
        self.assertIn("active-microphone", page)
        self.assertIn("開発者向けデバッグ情報", page)
        self.assertIn("diag-input-gate", page)

    def test_web_static_assets_load(self) -> None:
        """Web UI CSS and JS assets should be served separately."""
        css_response = self.client.get("/static/app.css")
        js_response = self.client.get("/static/app.js")
        try:
            self.assertEqual(css_response.status_code, 200)
            self.assertEqual(js_response.status_code, 200)
            self.assertIn("text/css", css_response.content_type)
            self.assertIn("javascript", js_response.content_type)
            js_text = js_response.get_data(as_text=True)
            self.assertNotIn("指示草案:\\\\n", js_text)
            self.assertNotIn('join("\\\\n")', js_text)
            self.assertIn("handleInputGateRecording", js_text)
            self.assertIn("startRecording", js_text)
            self.assertIn("buildAudioConstraints", js_text)
            self.assertIn("getSupportedConstraints", js_text)
            self.assertIn("getSettings", js_text)
            self.assertIn("getUserMedia({ audio: audioConstraints })", js_text)
            self.assertIn("RECORDING_CHUNK_TIMESLICE_MS", js_text)
            self.assertIn("recordingChunk", js_text)
            self.assertIn("eventsIngest", js_text)
            self.assertIn("record_start", js_text)
            self.assertIn("record_stop", js_text)
            self.assertIn("OPTION_PROFILES", js_text)
            self.assertIn("OPTION_PROFILE_ALIASES", js_text)
            self.assertIn("integration", js_text)
            self.assertIn("dify", js_text)
            self.assertIn("record_gate_auto", js_text)
            self.assertIn("WEB_OPTIONS_STORAGE_KEY", js_text)
            self.assertIn("no_persist", js_text)
            self.assertIn("reset_options", js_text)
            self.assertIn("QUERY_OPTION_ALIASES", js_text)
            self.assertIn("options.startup", js_text)
        finally:
            css_response.close()
            js_response.close()

    def test_web_favicon_loads(self) -> None:
        """Web UI should not emit a missing favicon request."""
        response = self.client.get("/favicon.ico")
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/svg+xml", response.content_type)

    def test_api_doctor_returns_runtime_sections(self) -> None:
        """Web UI should expose doctor status for diagnostics display."""
        response = self.client.get("/api/doctor", headers=self.local_api_headers())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertIn("runtime", payload)
        self.assertIn("microphone", payload)
        self.assertIn("dependencies", payload)
        self.assertIn("selected_microphone_device", payload["microphone"])

    def test_api_health_returns_integration_status(self) -> None:
        """Health endpoint should expose generic integration readiness state."""
        response = self.client.get("/api/health", headers=self.local_api_headers())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertIn("server", payload)
        self.assertIn("active_transcriptions", payload["server"])
        self.assertIn("stt", payload)
        self.assertIn("ffmpeg_available", payload["stt"])
        self.assertIn("events", payload)
        self.assertEqual(payload["events"]["stream"], "/api/events")
        self.assertIn("input_gate", payload)
        self.assertIn("latest_handoff", payload)

    def test_health_endpoint_returns_process_contract(self) -> None:
        """Unprefixed health endpoint should expose the supervisor contract."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["module"], "ai_talk_core.web")
        self.assertEqual(payload["pid"], os.getpid())
        self.assertIn("uptime_s", payload)
        self.assertEqual(payload["host"], "127.0.0.1")
        self.assertEqual(payload["port"], 8000)

    def test_api_health_reports_relative_event_log_path(self) -> None:
        """Status output should not expose the operator's absolute workspace root."""
        response = self.client.get("/api/health", headers=self.local_api_headers())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        event_log_path = Path(payload["events"]["log_path"])
        self.assertFalse(event_log_path.is_absolute())
        self.assertEqual(event_log_path.parts[0], ".cache")

    def test_api_status_alias_returns_health_payload(self) -> None:
        """Status endpoint should mirror the health shape."""
        response = self.client.get("/api/status", headers=self.local_api_headers())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertIn("server", payload)
        self.assertIn("stt", payload)

    def test_api_health_requires_local_token(self) -> None:
        """Integration status exposes local state and should require the UI token."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 403)

    def test_api_shutdown_requires_local_token(self) -> None:
        """Shutdown endpoint should not be callable without the local UI token."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        response = client.post("/api/shutdown", json={"reason": "test"})
        self.assertEqual(response.status_code, 403)

    def test_api_shutdown_sets_runtime_state_without_process_exit(self) -> None:
        """Shutdown endpoint should expose graceful shutdown state."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        response = client.post(
            "/api/shutdown",
            json={"reason": "test"},
            headers={"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]},
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["shutdown"]["shutdown_requested"])
        self.assertEqual(payload["shutdown"]["shutdown_reason"], "test")
        status_response = client.get(
            "/api/status",
            headers={"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]},
        )
        status_payload = status_response.get_json()
        self.assertIsNotNone(status_payload)
        self.assertFalse(status_payload["ready"])

    def test_shutdown_endpoint_sets_runtime_state_on_loopback(self) -> None:
        """Unprefixed shutdown should stop cooperatively for local supervisors."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        response = client.post("/shutdown")
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["shutdown"]["shutdown_requested"])
        self.assertEqual(payload["shutdown"]["shutdown_reason"], "shutdown_endpoint")

    def test_non_loopback_shutdown_accepts_sword_agent_token(self) -> None:
        """Non-loopback bind configurations should require an automation token."""
        app = create_app(host="0.0.0.0")
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        token = app.config["LOCAL_API_TOKEN"]
        rejected = client.post("/shutdown")
        self.assertEqual(rejected.status_code, 403)
        accepted = client.post(
            "/shutdown",
            headers={"X-Sword-Agent-Token": token},
        )
        self.assertEqual(accepted.status_code, 202)

    def test_authorization_bearer_token_is_supported(self) -> None:
        """Automation callers should be able to use Authorization: Bearer."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        response = client.get(
            "/api/health",
            headers={"Authorization": f"Bearer {app.config['LOCAL_API_TOKEN']}"},
        )
        self.assertEqual(response.status_code, 200)

    def test_parse_bearer_token_rejects_non_bearer_values(self) -> None:
        """Only Bearer auth should be interpreted as a local API token."""
        self.assertEqual(parse_bearer_token("Bearer abc123"), "abc123")
        self.assertEqual(parse_bearer_token("Basic abc123"), "")

    def test_runtime_status_payload_contains_supervisor_fields(self) -> None:
        """Runtime status JSON should give launch supervisors exact process facts."""
        payload = build_runtime_status_payload(
            state="running",
            host="127.0.0.1",
            port=8000,
            started_at="2026-04-29T00:00:00Z",
            command_line="python -m src.web.app",
        )
        self.assertEqual(payload["module"], "ai_talk_core.web")
        self.assertEqual(payload["state"], "running")
        self.assertEqual(payload["pid"], os.getpid())
        self.assertEqual(payload["parent_pid"], os.getppid())
        self.assertEqual(payload["health_url"], "http://127.0.0.1:8000/health")
        self.assertEqual(payload["shutdown_url"], "http://127.0.0.1:8000/shutdown")
        self.assertEqual(payload["command_line"], "python -m src.web.app")

    def test_runtime_status_writer_updates_json_file(self) -> None:
        """Runtime status writer should leave a stopped status file on shutdown."""
        status_path = PROJECT_ROOT / ".cache" / "tests" / "runtime_status.json"
        if status_path.exists():
            remove_path_with_retry(status_path)
        writer = RuntimeStatusWriter(
            status_path,
            host="127.0.0.1",
            port=8000,
            started_at="2026-04-29T00:00:00Z",
            command_line="python -m src.web.app",
        )
        writer.write("running")
        running = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(running["state"], "running")
        writer.write("stopped")
        stopped = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(stopped["state"], "stopped")
        self.assertIn("stopped_at", stopped)
        remove_path_with_retry(status_path)

    def test_api_input_gate_returns_current_state(self) -> None:
        """Web UI should expose current input-gate state."""
        response = self.client.get("/api/input-gate", headers=self.local_api_headers())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertIn("input_enabled", payload["input_gate"])

    def test_api_input_gate_requires_local_token(self) -> None:
        """Input-gate state controls should not be exposed without the UI token."""
        response = self.client.get("/api/input-gate")
        self.assertEqual(response.status_code, 403)

    def test_api_input_gate_updates_state(self) -> None:
        """Web UI should accept backend-neutral input-gate payloads."""
        response = self.client.post(
            "/api/input-gate",
            json={
                "input_enabled": False,
                "reason": "sword_sign",
                "source": "sword_voice_agent",
                "timestamp": 12.5,
            },
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertFalse(payload["input_gate"]["input_enabled"])
        self.assertEqual(payload["input_gate"]["reason"], "sword_sign")
        self.assertEqual(payload["input_gate"]["source"], "sword_voice_agent")

    def test_api_input_gate_rejects_invalid_payload(self) -> None:
        """Web UI should reject malformed input-gate payloads."""
        response = self.client.post(
            "/api/input-gate",
            json={"input_enabled": "yes"},
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertFalse(payload["ok"])

    def test_api_event_ingest_accepts_client_timing_event(self) -> None:
        """Web UI client events should enter the turn event bus."""
        response = self.client.post(
            "/api/events/ingest",
            json={
                "event": "record_start",
                "turn_id": "web_0123456789abcdef0123456789abcdef",
                "source": "web-ui",
                "client_timestamp_wall": "2026-07-13T12:00:00.000Z",
                "client_timestamp_monotonic": 12.5,
                "client_performance_now": 12_500.0,
                "payload": {
                    "trigger": "manual",
                    "timeslice_ms": 500,
                    "mime_type": "audio/webm",
                },
            },
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["event"]["event"], "record_start")
        self.assertEqual(
            payload["event"]["turn_id"],
            "web_0123456789abcdef0123456789abcdef",
        )
        self.assertIn("timestamp_wall", payload["event"])
        self.assertIn("timestamp_monotonic", payload["event"])

    def test_api_event_ingest_rejects_unexpected_client_payload(self) -> None:
        """Client-origin events should reject arbitrary text-bearing fields."""
        response = self.client.post(
            "/api/events/ingest",
            json={
                "event": "record_start",
                "turn_id": "web_0123456789abcdef0123456789abcdef",
                "source": "../bad source",
                "payload": {
                    "trigger": "manual",
                    "timeslice_ms": 500,
                    "mime_type": "audio/webm",
                    "transcript": "secret transcript",
                    "nested": {"token": "secret"},
                },
            },
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertFalse(payload["ok"])
        self.assertNotIn("secret transcript", response.get_data(as_text=True))

    def test_api_event_ingest_requires_local_token(self) -> None:
        """Event stream inputs expose local timing state and require the UI token."""
        response = self.client.post(
            "/api/events/ingest",
            json={"event": "record_start", "turn_id": "webtestevent"},
        )
        self.assertEqual(response.status_code, 403)

    def test_event_payload_sanitizer_bounds_debug_data(self) -> None:
        """Event projections should avoid absolute paths and oversized values."""
        payload = sanitize_event_payload(
            {
                "path": Path("C:/Example/secret.wav"),
                "long": "x" * 600,
                "items": list(range(20)),
            }
        )
        self.assertEqual(payload["path"], "secret.wav")
        self.assertNotIn("C:/Example", str(payload))
        self.assertTrue(str(payload["long"]).endswith("...[truncated]"))
        self.assertEqual(payload["items"][-1], "...4 more")

    def test_text_payload_facts_do_not_store_content_hash(self) -> None:
        """Latency metadata should not retain transcript fingerprints."""
        payload = text_payload_facts("secret phrase")
        self.assertEqual(payload["text_length"], len("secret phrase"))
        self.assertTrue(payload["text_present"])
        self.assertNotIn("text_sha256", payload)
        self.assertNotIn("secret phrase", str(payload))

    def test_turn_event_bus_rotates_bounded_event_log(self) -> None:
        """The JSONL event projection should not grow without bound."""
        log_path = PROJECT_ROOT / ".cache" / "events_rotation_test.jsonl"
        archive_path = Path(f"{log_path}.1")
        if log_path.exists():
            remove_path_with_retry(log_path)
        if archive_path.exists():
            remove_path_with_retry(archive_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("x" * 64, encoding="utf-8")
        with mock.patch("src.core.events.MAX_EVENT_LOG_BYTES", 32):
            bus = TurnEventBus(log_path=log_path)
            bus.emit("test_event", turn_id="turn1", payload={"value": "ok"})
        self.assertTrue(archive_path.exists())
        self.assertIn('"event": "test_event"', log_path.read_text(encoding="utf-8"))
        remove_path_with_retry(log_path)
        remove_path_with_retry(archive_path)

    def test_read_event_log_events_filters_by_turn_id(self) -> None:
        """One-shot trace readers should be able to filter events by turn id."""
        log_path = PROJECT_ROOT / ".cache" / "tests" / "events_read_test.jsonl"
        if log_path.exists():
            remove_path_with_retry(log_path)
        bus = TurnEventBus(log_path=log_path)
        bus.emit("trace_one", turn_id="trace_a", payload={"value": "a"})
        bus.emit("trace_two", turn_id="trace_b", payload={"value": "b"})
        events = read_event_log_events(log_path=log_path, limit=10, turn_id="trace_b")
        self.assertEqual([event["event"] for event in events], ["trace_two"])
        remove_path_with_retry(log_path)

    def test_api_events_once_returns_json_trace(self) -> None:
        """/api/events?once=1 should expose a bounded JSON trace."""
        turn_id = f"oncetest{int(time.time() * 1000)}"
        emit_event("trace_probe", turn_id=turn_id, source="test", payload={"value": "ok"})
        response = self.client.get(
            f"/api/events?once=1&turn_id={turn_id}&limit=5",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["projection"], "events.jsonl")
        self.assertGreaterEqual(payload["count"], 1)
        self.assertEqual(payload["events"][-1]["event"], "trace_probe")
        self.assertEqual(payload["events"][-1]["turn_id"], turn_id)

    def test_api_browser_recording_emits_server_record_stop(self) -> None:
        """Browser final uploads should create a stable server-side record_stop event."""
        turn_id = f"recordstop{int(time.time() * 1000)}"
        response_payload = WebTranscriptionResponse(
            message="ok",
            transcript="",
            command="",
            command_path="",
            command_text_path="",
            error="",
            status_code=200,
            turn_id=turn_id,
            debug={},
        )
        with mock.patch(
            "src.web.app.process_web_transcription",
            return_value=response_payload,
        ):
            response = self.client.post(
                "/api/transcribe-browser-recording",
                data={
                    "audio_blob": (io.BytesIO(b"fake-audio"), "browser_recording.webm"),
                    "turn_id": turn_id,
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        events = read_event_log_events(limit=20, turn_id=turn_id)
        record_events = [
            event for event in events if event.get("event") == "record_stop"
        ]
        self.assertTrue(record_events)
        record_payload = record_events[-1]["payload"]
        self.assertEqual(record_payload["transport"], "final_upload")
        self.assertEqual(record_payload["filename"], "browser_recording.webm")
        self.assertEqual(record_payload["size_bytes"], len(b"fake-audio"))

    def test_api_recording_chunk_persists_chunk_boundary(self) -> None:
        """Browser recording chunks should have a server-side landing boundary."""
        turn_id = "webtestchunk"
        chunk_path = get_recording_chunk_dir(turn_id) / "chunk_000003.webm"
        if chunk_path.exists():
            remove_path_with_retry(chunk_path)
        response = self.client.post(
            "/api/recording-chunk",
            data={
                "audio_chunk": (io.BytesIO(b"chunk-bytes"), "chunk_000003.webm"),
                "turn_id": turn_id,
                "sequence": "3",
            },
            content_type="multipart/form-data",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["turn_id"], turn_id)
        self.assertEqual(payload["sequence"], 3)
        self.assertTrue(chunk_path.exists())
        self.assertEqual(chunk_path.read_bytes(), b"chunk-bytes")
        remove_path_with_retry(chunk_path)

    def test_api_recording_chunk_rejects_excessive_sequence(self) -> None:
        """Chunk filenames should stay within a bounded sequence range."""
        response = self.client.post(
            "/api/recording-chunk",
            data={
                "audio_chunk": (io.BytesIO(b"chunk-bytes"), "chunk_999999.webm"),
                "turn_id": "webtestchunk",
                "sequence": str(WEB_MAX_RECORDING_CHUNKS + 1),
            },
            content_type="multipart/form-data",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 400)

    def test_api_recording_chunk_rejects_large_chunk(self) -> None:
        """Chunk uploads should have a tighter limit than whole recording uploads."""
        response = self.client.post(
            "/api/recording-chunk",
            data={
                "audio_chunk": (
                    io.BytesIO(b"x" * (WEB_MAX_RECORDING_CHUNK_BYTES + 1)),
                    "chunk_000000.webm",
                ),
                "turn_id": "webtestchunklarge",
                "sequence": "0",
            },
            content_type="multipart/form-data",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 413)

    def test_recording_chunk_cache_prunes_expired_turn_dirs(self) -> None:
        """Chunk cache retention should remove old per-turn directories."""
        cache_dir = PROJECT_ROOT / ".cache" / "web_recording_chunks_prune_test"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        expired_dir = cache_dir / "expired"
        fresh_dir = cache_dir / "fresh"
        expired_dir.mkdir(parents=True)
        fresh_dir.mkdir()
        (expired_dir / "chunk_000000.webm").write_bytes(b"expired")
        (fresh_dir / "chunk_000000.webm").write_bytes(b"fresh")
        old_timestamp = time.time() - WEB_RECORDING_CHUNK_RETENTION_SECONDS - 60
        os.utime(expired_dir, (old_timestamp, old_timestamp))
        try:
            prune_recording_chunk_cache(cache_dir)
            self.assertFalse(expired_dir.exists())
            self.assertTrue(fresh_dir.exists())
        finally:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

    def test_api_recording_chunk_requires_local_token(self) -> None:
        """Chunk upload boundaries should not be exposed without the UI token."""
        response = self.client.post(
            "/api/recording-chunk",
            data={
                "audio_chunk": (io.BytesIO(b"chunk-bytes"), "chunk_000000.webm"),
                "turn_id": "webtestchunk",
                "sequence": "0",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 403)

    def test_build_input_gate_response_wraps_state_payload(self) -> None:
        """Input-gate response helper should return a stable envelope."""
        response = build_input_gate_response(InputGate(initially_enabled=False).state)
        self.assertTrue(response["ok"])
        self.assertFalse(response["input_gate"]["input_enabled"])

    def test_render_page_with_prompt_only_omits_empty_handoff_label(self) -> None:
        """Prompt-only results should not render an empty handoff label."""
        with self.app.test_request_context("/"):
            page = render_page(command_text_path="/tmp/web_latest.txt")
        self.assertIn("プロンプト保存先:\n/tmp/web_latest.txt", page)
        self.assertNotIn("handoff 保存先:\n\nプロンプト保存先", page)

    def test_render_page_can_embed_web_preset(self) -> None:
        """Server-side presets should be available to the browser startup logic."""
        original_preset = self.app.config[WEB_PRESET_CONFIG]
        self.app.config[WEB_PRESET_CONFIG] = "integration"
        try:
            with self.app.test_request_context("/"):
                page = render_page()
        finally:
            self.app.config[WEB_PRESET_CONFIG] = original_preset
        self.assertIn('data-web-preset="integration"', page)

    def test_render_page_places_handoff_paths_after_result_actions(self) -> None:
        """Handoff paths should sit directly after the related action buttons."""
        with self.app.test_request_context("/"):
            page = render_page(
                transcript="hello",
                command="say hello",
                command_path=r"C:\tmp\web_latest.json",
                command_text_path=r"C:\tmp\web_latest.txt",
            )
        self.assertLess(page.index('id="result-actions"'), page.index('id="page-meta"'))
        self.assertIn("handoff 保存先:\nC:\\tmp\\web_latest.json", page)
        self.assertIn("プロンプト保存先:\nC:\\tmp\\web_latest.txt", page)

    def test_webrtcvad_dependency_is_available(self) -> None:
        """webrtcvad should be importable after dependency sync."""
        import importlib

        module = importlib.import_module("_webrtcvad")
        self.assertTrue(hasattr(module, "create"))

    def test_validate_vad_aggressiveness_accepts_supported_values(self) -> None:
        """Supported VAD aggressiveness values should pass validation."""
        for value in (0, 1, 2, 3):
            validate_vad_aggressiveness(value)

    def test_resolve_microphone_backend_uses_os_default(self) -> None:
        """Auto microphone backend should select the OS-specific recorder."""
        with mock.patch("src.io.microphone.platform.system", return_value="Windows"):
            self.assertEqual(resolve_microphone_backend("auto"), "ffmpeg-dshow")
        with mock.patch("src.io.microphone.platform.system", return_value="Linux"):
            self.assertEqual(resolve_microphone_backend("auto"), "arecord")

    def test_list_ffmpeg_dshow_audio_devices_parses_audio_section(self) -> None:
        """DirectShow device parsing should extract only audio device names."""
        dshow_output = "\n".join(
            [
                "[dshow @ 000] DirectShow video devices",
                '[dshow @ 000]  "HD Pro Webcam C920"',
                "[dshow @ 000] DirectShow audio devices",
                '[dshow @ 000]  "Microphone Array (Realtek(R) Audio)"',
                '[dshow @ 000]     Alternative name "@device_cm_{abc}"',
            ]
        )
        with mock.patch(
            "src.io.microphone.platform.system",
            return_value="Windows",
        ), mock.patch(
            "src.io.microphone.ensure_ffmpeg_available",
        ), mock.patch(
            "src.io.microphone.subprocess.run"
        ) as subprocess_run:
            subprocess_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr=dshow_output,
            )
            self.assertEqual(
                list_ffmpeg_dshow_audio_devices(),
                ["Microphone Array (Realtek(R) Audio)"],
            )
            self.assertEqual(
                subprocess_run.call_args.kwargs["timeout"],
                MICROPHONE_DEVICE_LIST_TIMEOUT_SECONDS,
            )

    def test_list_ffmpeg_dshow_audio_devices_parses_typed_lines(self) -> None:
        """DirectShow parsing should support ffmpeg lines marked with (audio)."""
        dshow_output = "\n".join(
            [
                '[dshow @ 000] "OBS Virtual Camera" (video)',
                '[dshow @ 000] "Webcam 4 (NDI Webcam Audio)" (audio)',
                '[dshow @ 000] "HD Pro Webcam C920" (none)',
            ]
        )
        with mock.patch(
            "src.io.microphone.platform.system",
            return_value="Windows",
        ), mock.patch(
            "src.io.microphone.ensure_ffmpeg_available",
        ), mock.patch(
            "src.io.microphone.subprocess.run"
        ) as subprocess_run:
            subprocess_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=dshow_output,
                stderr="",
            )
            self.assertEqual(
                list_ffmpeg_dshow_audio_devices(),
                ["Webcam 4 (NDI Webcam Audio)"],
            )
            self.assertEqual(
                subprocess_run.call_args.kwargs["timeout"],
                MICROPHONE_DEVICE_LIST_TIMEOUT_SECONDS,
            )

    def test_record_microphone_audio_uses_arecord_backend(self) -> None:
        """Linux microphone backend should keep the existing arecord command shape."""
        output_path = PROJECT_ROOT / ".cache" / "tests" / "mic_arecord.wav"
        with mock.patch(
            "src.io.microphone.ensure_arecord_available"
        ), mock.patch(
            "src.io.microphone.get_default_arecord_microphone_device",
            return_value="plughw:1,0",
        ), mock.patch(
            "src.io.microphone.subprocess.run"
        ) as subprocess_run:
            subprocess_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="",
                stderr="",
            )
            result = record_microphone_audio(
                output_path=output_path,
                duration=2,
                backend="arecord",
                trim_silence_enabled=False,
            )
        command = subprocess_run.call_args.args[0]
        self.assertEqual(result, output_path)
        self.assertEqual(command[:2], ["arecord", "-D"])
        self.assertIn("plughw:1,0", command)
        self.assertEqual(
            subprocess_run.call_args.kwargs["timeout"],
            get_recording_timeout_seconds(2),
        )

    def test_record_microphone_audio_uses_ffmpeg_dshow_backend(self) -> None:
        """Windows microphone backend should record through ffmpeg DirectShow."""
        output_path = PROJECT_ROOT / ".cache" / "tests" / "mic_dshow.wav"
        with mock.patch(
            "src.io.microphone.platform.system",
            return_value="Windows",
        ), mock.patch(
            "src.io.microphone.ensure_ffmpeg_available"
        ), mock.patch(
            "src.io.microphone.subprocess.run"
        ) as subprocess_run:
            subprocess_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="",
                stderr="",
            )
            result = record_microphone_audio(
                output_path=output_path,
                duration=2,
                device="Microphone Array (Realtek(R) Audio)",
                backend="ffmpeg-dshow",
                trim_silence_enabled=False,
            )
        command = subprocess_run.call_args.args[0]
        self.assertEqual(result, output_path)
        self.assertEqual(command[:4], ["ffmpeg", "-y", "-f", "dshow"])
        self.assertIn("audio=Microphone Array (Realtek(R) Audio)", command)
        self.assertIn("pcm_s16le", command)
        self.assertEqual(
            subprocess_run.call_args.kwargs["timeout"],
            get_recording_timeout_seconds(2),
        )

    def test_record_microphone_audio_converts_dshow_timeout(self) -> None:
        """Hung DirectShow recording should become a normal environment error."""
        output_path = PROJECT_ROOT / ".cache" / "tests" / "mic_dshow_timeout.wav"
        with mock.patch(
            "src.io.microphone.platform.system",
            return_value="Windows",
        ), mock.patch(
            "src.io.microphone.ensure_ffmpeg_available"
        ), mock.patch(
            "src.io.microphone.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=12),
        ):
            with self.assertRaises(AudioEnvironmentError):
                record_microphone_audio(
                    output_path=output_path,
                    duration=2,
                    device="Microphone Array (Realtek(R) Audio)",
                    backend="ffmpeg-dshow",
                    trim_silence_enabled=False,
                )

    def test_get_microphone_runtime_status_reports_backend_availability(self) -> None:
        """Microphone status should expose OS defaults and backend availability."""
        with mock.patch(
            "src.io.microphone.platform.system",
            return_value="Windows",
        ), mock.patch(
            "src.io.microphone.shutil.which",
            side_effect=lambda name: "C:\\ffmpeg\\bin\\ffmpeg.exe"
            if name == "ffmpeg"
            else None,
        ):
            status = get_microphone_runtime_status()
        self.assertEqual(status["default_microphone_backend"], "ffmpeg-dshow")
        self.assertTrue(status["selected_microphone_backend_available"])
        self.assertIn("ffmpeg-dshow", status["available_microphone_backends"])

    def test_web_upload_transcribes_sample_audio(self) -> None:
        """Web UI upload route should transcribe sample audio."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
        }
        response = self.client.post(
            "/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("こんにちは", response.get_data(as_text=True))

    def test_web_upload_fetch_returns_json(self) -> None:
        """Fetch-style upload should return JSON for partial page updates."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
        }
        response = self.client.post(
            "/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
            headers={"X-Requested-With": "fetch"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertIn("こんにちは", payload_json["transcript"])
        self.assertEqual(payload_json["command"], payload_json["transcript"].strip())

    def test_web_upload_fetch_command_only_returns_command_without_transcript(self) -> None:
        """Fetch-style upload should support command-only responses."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
            "command_only": "true",
        }
        response = self.client.post(
            "/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
            headers={"X-Requested-With": "fetch"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["transcript"], "")
        self.assertIn("こんにちは", payload_json["command"])

    def test_api_upload_returns_json(self) -> None:
        """Dedicated API upload route should return JSON."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
        }
        response = self.client.post(
            "/api/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertIn("こんにちは", payload_json["transcript"])
        self.assertEqual(payload_json["command"], payload_json["transcript"].strip())

    def test_api_upload_command_only_returns_command_without_transcript(self) -> None:
        """Dedicated API upload route should support command-only responses."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
            "command_only": "true",
        }
        response = self.client.post(
            "/api/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["transcript"], "")
        self.assertIn("こんにちは", payload_json["command"])

    def test_api_upload_instruction_only_alias_returns_command_without_transcript(self) -> None:
        """Dedicated API upload route should also accept instruction_only."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
            "instruction_only": "true",
        }
        response = self.client.post(
            "/api/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["transcript"], "")
        self.assertIn("こんにちは", payload_json["command"])

    def test_api_upload_can_save_command_payload(self) -> None:
        """API upload route should optionally save the Codex payload."""
        output_path = get_default_codex_output_path(source="web")
        text_path = get_default_codex_text_path(source="web")
        if output_path.exists():
            remove_path_with_retry(output_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
            "save_command": "true",
        }
        response = self.client.post(
            "/api/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["command_path"], str(output_path))
        self.assertEqual(payload_json["command_text_path"], str(text_path))
        self.assertTrue(output_path.exists())
        self.assertTrue(text_path.exists())
        remove_path_with_retry(output_path)
        remove_path_with_retry(text_path)

    def test_api_upload_can_save_handoff_alias(self) -> None:
        """API upload route should also accept save_handoff."""
        output_path = get_default_codex_output_path(source="web")
        text_path = get_default_codex_text_path(source="web")
        if output_path.exists():
            remove_path_with_retry(output_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_file": (io.BytesIO(sample_path.read_bytes()), "sample_audio.mp3"),
            "model": "small",
            "language": "ja",
            "save_handoff": "true",
        }
        response = self.client.post(
            "/api/transcribe-upload",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["command_path"], str(output_path))
        self.assertEqual(payload_json["command_text_path"], str(text_path))
        self.assertTrue(output_path.exists())
        self.assertTrue(text_path.exists())
        remove_path_with_retry(output_path)
        remove_path_with_retry(text_path)

    def test_api_codex_handoff_latest_returns_saved_bundle(self) -> None:
        """Latest handoff API should return saved prompt bundle contents."""
        output_path = get_default_codex_output_path(source="web")
        text_path = get_default_codex_text_path(source="web")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=output_path,
            text_path=text_path,
        )
        response = self.client.get(
            "/api/codex-handoff-latest?source=web",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 200)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["command"], "依存関係を確認して")
        self.assertIn("Voice transcript:", payload_json["prompt_text"])
        self.assertTrue(payload_json["handoff_id"])
        self.assertTrue(payload_json["updated_at"])
        self.assertTrue(payload_json["metadata"]["exists"])
        remove_path_with_retry(output_path)
        remove_path_with_retry(text_path)

    def test_api_agent_handoff_latest_returns_saved_bundle(self) -> None:
        """Agent handoff API alias should return saved prompt bundle contents."""
        output_path = get_default_codex_output_path(source="web")
        text_path = get_default_codex_text_path(source="web")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=output_path,
            text_path=text_path,
        )
        response = self.client.get(
            "/api/agent-handoff-latest?source=web",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 200)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertEqual(payload_json["command"], "依存関係を確認して")
        self.assertIn("Voice transcript:", payload_json["prompt_text"])
        self.assertTrue(payload_json["handoff_id"])
        self.assertTrue(payload_json["updated_at"])
        self.assertTrue(payload_json["metadata"]["exists"])
        remove_path_with_retry(output_path)
        remove_path_with_retry(text_path)

    def test_api_codex_handoff_latest_returns_404_without_bundle(self) -> None:
        """Latest handoff API should return 404 when no bundle exists."""
        output_path = get_default_codex_output_path(source="missing")
        text_path = get_default_codex_text_path(source="missing")
        if output_path.exists():
            remove_path_with_retry(output_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        response = self.client.get(
            "/api/codex-handoff-latest?source=missing",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 404)

    def test_api_agent_handoff_latest_returns_404_without_bundle(self) -> None:
        """Agent handoff API alias should return 404 when no bundle exists."""
        output_path = get_default_codex_output_path(source="missing")
        text_path = get_default_codex_text_path(source="missing")
        if output_path.exists():
            remove_path_with_retry(output_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        response = self.client.get(
            "/api/agent-handoff-latest?source=missing",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 404)

    def test_api_handoff_latest_requires_local_token(self) -> None:
        """Latest handoff API should require the local per-process token."""
        response = self.client.get("/api/agent-handoff-latest?source=web")
        self.assertEqual(response.status_code, 403)

    def test_api_handoff_latest_rejects_query_token(self) -> None:
        """Local API tokens should not be accepted from URL query parameters."""
        response = self.client.get(
            f"/api/agent-handoff-latest?source=web&api_token={self.app.config['LOCAL_API_TOKEN']}"
        )
        self.assertEqual(response.status_code, 403)

    def test_api_handoff_latest_rejects_invalid_source(self) -> None:
        """Latest handoff API should reject path-like source values."""
        response = self.client.get(
            "/api/agent-handoff-latest?source=../web",
            headers=self.local_api_headers(),
        )
        self.assertEqual(response.status_code, 400)

    def test_handoff_metadata_reports_latest_saved_bundle(self) -> None:
        """Handoff metadata should give watchers an id and update timestamp."""
        json_path = get_default_codex_output_path(source="metadata_test")
        text_path = get_default_codex_text_path(source="metadata_test")
        if json_path.exists():
            remove_path_with_retry(json_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        missing = build_handoff_metadata(source="metadata_test")
        self.assertFalse(missing["exists"])
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        metadata = build_handoff_metadata(source="metadata_test")
        self.assertTrue(metadata["exists"])
        self.assertTrue(metadata["handoff_id"])
        self.assertTrue(metadata["updated_at"])
        self.assertGreater(metadata["json_size_bytes"], 0)
        self.assertGreater(metadata["text_size_bytes"], 0)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_handoff_cli_reads_latest_prompt(self) -> None:
        """Handoff CLI should print the saved prompt text."""
        json_path = get_default_codex_output_path(source="cli_test")
        text_path = get_default_codex_text_path(source="cli_test")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_handoff_cli("--source", "cli_test", "--format", "prompt")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Voice transcript:", result.stdout)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_handoff_source_rejects_path_segments(self) -> None:
        """Handoff source labels should not be usable as path components."""
        with self.assertRaises(ValueError):
            normalize_handoff_source("../web")
        with self.assertRaises(ValueError):
            get_default_codex_output_path(source="../web")

    def test_handoff_cli_rejects_invalid_source(self) -> None:
        """Handoff CLI should return a normal input error for invalid sources."""
        result = run_agent_handoff_cli("--source", "../web", "--format", "prompt")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Input error:", result.stdout)

    def test_agent_handoff_cli_reads_latest_prompt(self) -> None:
        """Generic agent handoff CLI should print the saved prompt text."""
        json_path = get_default_codex_output_path(source="agent_cli_test")
        text_path = get_default_codex_text_path(source="agent_cli_test")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_agent_handoff_cli("--source", "agent_cli_test", "--format", "prompt")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Voice transcript:", result.stdout)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_handoff_cli_reads_latest_command(self) -> None:
        """Handoff CLI should print the saved command text."""
        json_path = get_default_codex_output_path(source="cli_command")
        text_path = get_default_codex_text_path(source="cli_command")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_handoff_cli("--source", "cli_command", "--format", "command")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "依存関係を確認して")
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_agent_handoff_cli_reads_latest_command(self) -> None:
        """Generic agent handoff CLI should print the saved command text."""
        json_path = get_default_codex_output_path(source="agent_cli_command")
        text_path = get_default_codex_text_path(source="agent_cli_command")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_agent_handoff_cli("--source", "agent_cli_command", "--format", "command")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "依存関係を確認して")
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_runner_cli_print_only_outputs_prompt(self) -> None:
        """Runner CLI should print the latest prompt in print-only mode."""
        json_path = get_default_codex_output_path(source="runner_print")
        text_path = get_default_codex_text_path(source="runner_print")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_runner_cli("--source", "runner_print", "--print-only")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Voice transcript:", result.stdout)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_agent_runner_cli_print_only_outputs_prompt(self) -> None:
        """Generic agent runner CLI should print the latest prompt in print-only mode."""
        json_path = get_default_codex_output_path(source="agent_runner_print")
        text_path = get_default_codex_text_path(source="agent_runner_print")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_agent_runner_cli("--source", "agent_runner_print", "--print-only")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Voice transcript:", result.stdout)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_runner_cli_pipes_prompt_to_command(self) -> None:
        """Runner CLI should pass the rendered prompt to stdin."""
        json_path = get_default_codex_output_path(source="runner_pipe")
        text_path = get_default_codex_text_path(source="runner_pipe")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_runner_cli(
            "--source",
            "runner_pipe",
            "--",
            "python",
            "-c",
            "import sys; print(sys.stdin.read())",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Requested task:", result.stdout)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_runner_cli_template_cat_outputs_prompt(self) -> None:
        """Runner CLI should support built-in command templates."""
        json_path = get_default_codex_output_path(source="runner_template")
        text_path = get_default_codex_text_path(source="runner_template")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        result = run_runner_cli("--source", "runner_template", "--template", "cat")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Voice transcript:", result.stdout)
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_api_upload_missing_file_returns_400(self) -> None:
        """Dedicated API upload route should validate missing files."""
        response = self.client.post("/api/transcribe-upload", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertIn("音声ファイルを選択してください", payload_json["error"])

    def test_api_upload_rejects_cross_origin_post(self) -> None:
        """Local Web API should reject browser posts from another origin."""
        response = self.client.post(
            "/api/transcribe-upload",
            data={},
            content_type="multipart/form-data",
            headers={"Origin": "http://example.com"},
        )
        self.assertEqual(response.status_code, 403)

    def test_local_policy_rejects_non_loopback_remote_without_token_leak(self) -> None:
        """Host headers should not be enough when the TCP peer is not loopback."""
        response = self.client.get(
            "/",
            base_url="http://127.0.0.1:8000",
            environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
        )
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 403)
        self.assertIn("許可されていない接続元", body)
        self.assertNotIn(self.app.config["LOCAL_API_TOKEN"], body)

    def test_local_policy_rejects_bad_host_without_token_leak(self) -> None:
        """Policy denials should not render the token-bearing Web UI."""
        response = self.client.get("/", headers={"Host": "example.com"})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 403)
        self.assertIn("許可されていない Host", body)
        self.assertNotIn(self.app.config["LOCAL_API_TOKEN"], body)

    def test_api_upload_rejects_oversized_request(self) -> None:
        """Flask should reject uploads that exceed the configured byte limit."""
        original_limit = self.app.config["MAX_CONTENT_LENGTH"]
        self.app.config["MAX_CONTENT_LENGTH"] = 128
        try:
            response = self.client.post(
                "/api/transcribe-upload",
                data={
                    "audio_file": (io.BytesIO(b"x" * 1024), "sample.wav"),
                },
                content_type="multipart/form-data",
            )
        finally:
            self.app.config["MAX_CONTENT_LENGTH"] = original_limit
        self.assertEqual(response.status_code, 413)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertIn("大きすぎます", payload_json["error"])

    def test_api_browser_recording_returns_json(self) -> None:
        """Dedicated browser-recording API route should return JSON."""
        sample_path = PROJECT_ROOT / "data" / "sample_audio.mp3"
        payload = {
            "audio_blob": (io.BytesIO(sample_path.read_bytes()), "browser_recording.mp3"),
            "model": "small",
            "language": "ja",
        }
        response = self.client.post(
            "/api/transcribe-browser-recording",
            data=payload,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)
        payload_json = response.get_json()
        self.assertIsNotNone(payload_json)
        self.assertIn("こんにちは", payload_json["transcript"])
        self.assertEqual(payload_json["command"], payload_json["transcript"].strip())

    def test_repeat_transcript_marks_result_final(self) -> None:
        """Three consecutive matching transcripts should be treated as final."""
        result = TranscriptionResult(
            source="microphone",
            text="こんにちは",
            is_final=False,
            chunk_count=2,
        )
        self.assertTrue(should_mark_result_final(result, 3, False, 3, 8))

    def test_blank_transcript_does_not_mark_result_final(self) -> None:
        """Blank transcripts should not become final unless loop ends."""
        result = TranscriptionResult(
            source="microphone",
            text="   ",
            is_final=False,
            chunk_count=2,
        )
        self.assertFalse(should_mark_result_final(result, 2, False, 3, 8))

    def test_short_transcript_does_not_mark_result_final(self) -> None:
        """Very short repeated transcripts should remain partial."""
        result = TranscriptionResult(
            source="microphone",
            text="はい",
            is_final=False,
            chunk_count=2,
        )
        self.assertFalse(should_mark_result_final(result, 3, False, 3, 8))

    def test_single_repeat_does_not_mark_result_final(self) -> None:
        """Two consecutive matching transcripts should still remain partial."""
        result = TranscriptionResult(
            source="microphone",
            text="こんにちは",
            is_final=False,
            chunk_count=2,
        )
        self.assertFalse(should_mark_result_final(result, 2, False, 3, 8))

    def test_low_latency_threshold_can_mark_first_short_utterance_final(self) -> None:
        """Low-latency tuning may finalize a short utterance after one chunk."""
        result = TranscriptionResult(
            source="microphone",
            text="こんにちは",
            is_final=False,
            chunk_count=1,
        )
        self.assertTrue(should_mark_result_final(result, 1, False, 1, 1))

    def test_long_transcript_marks_result_final_with_two_repeats(self) -> None:
        """Longer stable transcripts may finalize after two repeats."""
        result = TranscriptionResult(
            source="microphone",
            text="依存関係を確認してから進めてください",
            is_final=False,
            chunk_count=2,
        )
        self.assertTrue(should_mark_result_final(result, 2, False, 3, 8))

    def test_normalize_transcript_text_collapses_whitespace(self) -> None:
        """Transcript normalization should collapse redundant whitespace."""
        self.assertEqual(normalize_transcript_text("  こんにちは   世界 "), "こんにちは 世界")

    def test_required_repeat_count_for_final_relaxes_for_long_text(self) -> None:
        """Longer transcripts should require fewer repeats."""
        self.assertEqual(required_repeat_count_for_final("依存関係を確認してから進めてください"), 2)
        self.assertEqual(required_repeat_count_for_final("こんにちは"), 3)

    def test_stable_duration_for_final_accepts_medium_text_after_longer_time(self) -> None:
        """Time stability should help medium-length transcripts become final."""
        self.assertTrue(has_stable_duration_for_final("依存関係を確認して", 2, 4, 8))

    def test_stable_duration_for_final_ignores_short_text(self) -> None:
        """Very short text should not finalize only from elapsed time."""
        self.assertFalse(has_stable_duration_for_final("はい", 4, 3, 8))

    def test_stable_duration_for_final_uses_configured_threshold(self) -> None:
        """Stable-duration finalization should respect the configured threshold."""
        self.assertFalse(has_stable_duration_for_final("依存関係を確認して", 2, 3, 8))
        self.assertTrue(has_stable_duration_for_final("依存関係を確認して", 2, 3, 6))

    def test_stable_duration_for_final_requires_more_than_one_repeat(self) -> None:
        """A single long chunk should not finalize only from chunk duration."""
        self.assertFalse(has_stable_duration_for_final("依存関係を確認して", 1, 8, 8))

    def test_validate_final_stable_seconds_accepts_positive_values(self) -> None:
        """Positive stable-duration thresholds should pass validation."""
        validate_final_stable_seconds(1)
        validate_final_stable_seconds(8)

    def test_validate_mic_profile_accepts_supported_values(self) -> None:
        """Supported mic profiles should pass validation."""
        for value in ("responsive", "balanced", "strict", "low_latency"):
            validate_mic_profile(value)

    def test_resolve_mic_loop_tuning_uses_profile_defaults(self) -> None:
        """Mic profile should resolve default VAD and final thresholds."""
        self.assertEqual(resolve_mic_loop_tuning("responsive", None, None), (1, 5))
        self.assertEqual(resolve_mic_loop_tuning("balanced", None, None), (2, 8))
        self.assertEqual(resolve_mic_loop_tuning("strict", None, None), (3, 10))
        self.assertEqual(resolve_mic_loop_tuning("low_latency", None, None), (1, 1))

    def test_resolve_mic_loop_tuning_preserves_explicit_overrides(self) -> None:
        """Explicit CLI overrides should win over profile defaults."""
        self.assertEqual(resolve_mic_loop_tuning("strict", 0, None), (0, 10))
        self.assertEqual(resolve_mic_loop_tuning("responsive", None, 12), (1, 12))
        self.assertEqual(resolve_mic_loop_tuning("balanced", 3, 6), (3, 6))

    def test_format_mic_loop_tuning_reports_resolved_values(self) -> None:
        """Mic-loop tuning formatter should expose the active settings."""
        self.assertEqual(
            format_mic_loop_tuning("balanced", 2, 8),
            "[mic-tuning] profile=balanced vad_aggressiveness=2 final_stable_seconds=8",
        )

    def test_format_mic_profile_list_mentions_profile_details(self) -> None:
        """Profile list formatter should describe the preset values."""
        listing = format_mic_profile_list()
        self.assertIn("responsive", listing)
        self.assertIn("low_latency", listing)
        self.assertIn("vad_aggressiveness=1", listing)
        self.assertIn("final_stable_seconds=10", listing)

    def test_build_mic_profile_list_data_returns_structured_profiles(self) -> None:
        """Structured profile listing should expose all expected keys."""
        payload = build_mic_profile_list_data()
        self.assertEqual(payload[0]["profile"], "responsive")
        self.assertEqual(payload[1]["vad_aggressiveness"], 2)
        self.assertIn("description", payload[2])

    def test_build_mic_tuning_data_returns_structured_values(self) -> None:
        """Structured tuning data should match the resolved values."""
        self.assertEqual(
            build_mic_tuning_data("strict", 3, 10),
            {
                "profile": "strict",
                "vad_aggressiveness": 3,
                "final_stable_seconds": 10,
            },
        )

    def test_format_runtime_status_mentions_core_fields(self) -> None:
        """Runtime status formatter should expose core runtime keys."""
        text = format_runtime_status(
            {
                "ffmpeg_available": True,
                "ffprobe_available": True,
                "nvidia_smi_available": True,
                "nvidia_driver_version": "535.288.01",
                "nvidia_gpu_name": "NVIDIA GeForce RTX 3070",
                "torch_version": "2.10.0+cu128",
                "torch_cuda_version": "12.8",
                "torch_cuda_available": False,
                "transcription_device": "cpu",
                "whisper_version": "20250625",
                "runtime_note": (
                    "Torch CUDA build is present but unavailable; transcription will use CPU "
                    "fallback. nvidia-smi is available, so a Torch/driver CUDA mismatch or "
                    "local CUDA initialization problem is likely."
                ),
                "suggested_action": (
                    "Inspect the uv-managed Torch version and pin a driver-compatible "
                    "build inside .venv before changing system drivers."
                ),
            }
        )
        self.assertIn("Runtime status:", text)
        self.assertIn("torch_cuda_available: False", text)
        self.assertIn("nvidia_driver_version: 535.288.01", text)
        self.assertIn("transcription_device: cpu", text)
        self.assertIn("ffmpeg_available: True", text)
        self.assertIn("Torch/driver CUDA mismatch", text)
        self.assertIn("uv-managed Torch version", text)

    def test_get_runtime_status_returns_expected_keys(self) -> None:
        """Runtime status helper should return the expected status fields."""
        status = get_runtime_status()
        self.assertIn("ffmpeg_available", status)
        self.assertIn("ffprobe_available", status)
        self.assertIn("nvidia_smi_available", status)
        self.assertIn("nvidia_driver_version", status)
        self.assertIn("nvidia_gpu_name", status)
        self.assertIn("torch_version", status)
        self.assertIn("torch_cuda_build", status)
        self.assertIn("torch_cuda_available", status)
        self.assertIn("transcription_device", status)
        self.assertIn("runtime_note", status)
        self.assertIn("suggested_action", status)

    def test_get_runtime_status_notes_cpu_torch_when_nvidia_is_visible(self) -> None:
        """Runtime status should explain CPU-only Torch when an NVIDIA GPU is visible."""
        def fake_which(name: str) -> str | None:
            if name in {"ffmpeg", "ffprobe", "nvidia-smi"}:
                return name
            return None

        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="596.21, NVIDIA GeForce RTX 3070\n",
            stderr="",
        )
        with (
            mock.patch("src.io.audio.shutil.which", side_effect=fake_which),
            mock.patch("src.io.audio.subprocess.run", return_value=completed),
            mock.patch("src.io.audio.torch.cuda.is_available", return_value=False),
            mock.patch("src.io.audio.torch.version.cuda", None),
            mock.patch("src.io.audio.torch.__version__", "2.10.0+cpu"),
        ):
            status = get_runtime_status()

        self.assertTrue(status["nvidia_smi_available"])
        self.assertFalse(status["torch_cuda_build"])
        self.assertIn("CPU-only", str(status["runtime_note"]))
        self.assertIn(".venv", str(status["suggested_action"]))

    def test_format_dependency_status_mentions_torch_source(self) -> None:
        """Dependency formatter should explain how torch is resolved."""
        text = format_dependency_status(
            {
                "pyproject_path": "/tmp/pyproject.toml",
                "direct_dependencies": ["flask>=3.1.3", "openai-whisper>=20250625"],
                "direct_dependency_names": ["flask", "openai-whisper"],
                "torch_direct_dependency": False,
                "installed_versions": {
                    "flask": "3.1.3",
                    "openai-whisper": "20250625",
                    "setuptools": "82.0.1",
                    "torch": "2.10.0+cu128",
                    "webrtcvad": "2.0.10",
                },
                "dependency_note": (
                    "torch is currently resolved transitively via openai-whisper unless it "
                    "is added explicitly to pyproject.toml."
                ),
            }
        )
        self.assertIn("Dependency status:", text)
        self.assertIn("torch_direct_dependency: False", text)
        self.assertIn("openai-whisper>=20250625", text)
        self.assertIn("transitively via openai-whisper", text)

    def test_get_dependency_status_returns_expected_keys(self) -> None:
        """Dependency status helper should expose direct and installed package state."""
        status = get_dependency_status()
        self.assertIn("direct_dependencies", status)
        self.assertIn("direct_dependency_names", status)
        self.assertIn("torch_direct_dependency", status)
        self.assertIn("installed_versions", status)
        self.assertIn("dependency_note", status)

    def test_format_doctor_status_includes_runtime_and_dependency_sections(self) -> None:
        """Doctor formatter should combine runtime and dependency summaries."""
        text = format_doctor_status(
            {
                "runtime": {
                    "ffmpeg_available": True,
                    "ffprobe_available": True,
                    "nvidia_smi_available": True,
                    "nvidia_driver_version": "535.288.01",
                    "nvidia_gpu_name": "NVIDIA GeForce RTX 3070",
                    "torch_version": "2.10.0+cu128",
                    "torch_cuda_version": "12.8",
                    "torch_cuda_available": False,
                    "transcription_device": "cpu",
                    "whisper_version": "20250625",
                    "runtime_note": "cpu fallback",
                    "suggested_action": "pin torch locally",
                },
                "microphone": {
                    "platform_system": "Windows",
                    "default_microphone_backend": "ffmpeg-dshow",
                    "selected_microphone_backend": "ffmpeg-dshow",
                    "selected_microphone_backend_available": True,
                    "available_microphone_backends": ["ffmpeg-dshow"],
                    "arecord_available": False,
                    "ffmpeg_dshow_available": True,
                    "microphone_note": None,
                },
                "dependencies": {
                    "pyproject_path": "/tmp/pyproject.toml",
                    "direct_dependencies": ["flask>=3.1.3", "openai-whisper>=20250625"],
                    "direct_dependency_names": ["flask", "openai-whisper"],
                    "torch_direct_dependency": False,
                    "installed_versions": {
                        "flask": "3.1.3",
                        "openai-whisper": "20250625",
                        "setuptools": "82.0.1",
                        "torch": "2.10.0+cu128",
                        "webrtcvad": "2.0.10",
                    },
                    "dependency_note": "torch is transitive",
                },
            }
        )
        self.assertIn("Doctor summary:", text)
        self.assertIn("Runtime status:", text)
        self.assertIn("Microphone status:", text)
        self.assertIn("default_microphone_backend: ffmpeg-dshow", text)
        self.assertIn("Dependency status:", text)
        self.assertIn("torch_direct_dependency: False", text)

    def test_build_doctor_status_returns_expected_sections(self) -> None:
        """Doctor status helper should include runtime and dependency sections."""
        status = build_doctor_status()
        self.assertIn("runtime", status)
        self.assertIn("microphone", status)
        self.assertIn("dependencies", status)

    def test_format_torch_pin_plan_includes_steps_and_commands(self) -> None:
        """Torch pin formatter should render plan details."""
        text = format_torch_pin_plan(
            {
                "torch_direct_dependency": False,
                "current_torch_version": "2.10.0+cu128",
                "current_torch_base_version": "2.10.0",
                "current_torch_build_suffix": "cu128",
                "current_torch_cuda_version": "12.8",
                "current_driver_version": "535.288.01",
                "recommended_torch_spec": "torch==2.10.0",
                "recommended_cuda_family": "cu121",
                "pytorch_index_url": "https://download.pytorch.org/whl/cu121",
                "uv_pip_install_command": (
                    "uv pip install --upgrade torch "
                    "--index-url https://download.pytorch.org/whl/cu121"
                ),
                "setup_script_command": ".\\setup_gpu_windows.ps1 -Cuda cu121",
                "explicit_build_selection_needed": True,
                "pyproject_dependency_entry": "torch==2.10.0",
                "uv_add_command": "uv add 'torch==2.10.0'",
                "steps": ["step one", "step two"],
                "command_examples": ["uv add 'torch==<base-version>'", "uv lock"],
                "plan_note": "project-local only",
            }
        )
        self.assertIn("Torch pin plan:", text)
        self.assertIn("recommended_cuda_family: cu121", text)
        self.assertIn("setup_gpu_windows.ps1", text)
        self.assertIn("pytorch_index_url: https://download.pytorch.org/whl/cu121", text)
        self.assertIn("uv lock", text)
        self.assertIn("explicit_build_selection_needed: True", text)
        self.assertIn("uv_add_command: uv add 'torch==2.10.0'", text)

    def test_build_torch_pin_status_returns_expected_keys(self) -> None:
        """Torch pin status helper should include planning details."""
        status = build_torch_pin_status()
        self.assertIn("torch_direct_dependency", status)
        self.assertIn("current_torch_version", status)
        self.assertIn("current_torch_build_suffix", status)
        self.assertIn("steps", status)
        self.assertIn("command_examples", status)
        self.assertIn("uv_add_command", status)
        self.assertIn("setup_script_command", status)
        self.assertIn("uv_pip_install_command", status)
        self.assertIn("venv_doctor_command", status)
        self.assertIn("uv_run_no_sync_doctor_command", status)

    def test_get_torch_pin_plan_recommends_project_local_steps(self) -> None:
        """Torch pin plan should emphasize a project-local adjustment path."""
        plan = get_torch_pin_plan()
        self.assertIn("steps", plan)
        self.assertIn("command_examples", plan)
        self.assertIn(".venv", str(plan["plan_note"]))

    def test_get_torch_pin_plan_marks_explicit_build_selection_when_suffix_exists(self) -> None:
        """Torch pin plan should flag version-only pinning as insufficient when a local CUDA suffix exists."""
        plan = get_torch_pin_plan()
        self.assertIn("explicit_build_selection_needed", plan)

    def test_windows_helper_scripts_exist(self) -> None:
        """Windows startup and GPU helpers should be present at the repo root."""
        self.assertTrue((PROJECT_ROOT / "start_web.ps1").is_file())
        self.assertTrue((PROJECT_ROOT / "setup_gpu_windows.ps1").is_file())

    def test_windows_gpu_helper_avoids_uv_run_resync_after_torch_install(self) -> None:
        """GPU helper should verify with venv Python so uv does not restore CPU Torch."""
        script = (PROJECT_ROOT / "setup_gpu_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("$projectPython", script)
        self.assertIn("& $projectPython -c", script)
        self.assertIn("& $projectPython -m src.main --doctor", script)
        self.assertIn("setuptools>=82.0.1", script)
        self.assertNotIn("& uv run python -c", script)
        self.assertNotIn("& uv run python -m src.main --doctor", script)

    def test_windows_startup_uses_venv_python_to_preserve_gpu_torch(self) -> None:
        """Startup helper should not trigger uv sync through uv run after setup."""
        script = (PROJECT_ROOT / "start_web.ps1").read_text(encoding="utf-8")
        self.assertIn("$projectPython", script)
        self.assertIn("& $projectPython -m src.main --doctor", script)
        self.assertIn('"src.web.app"', script)
        self.assertIn("--runtime-status-file", script)
        self.assertIn("@webArgs", script)
        self.assertIn("[string]$Preset", script)
        self.assertIn("[string]$RuntimeStatusFile", script)
        self.assertIn("AI_TALK_CORE_WEB_PRESET", script)
        self.assertIn("profile=", script)
        self.assertNotIn("& uv run python -m src.main --doctor", script)
        self.assertNotIn("& uv run python -m src.web.app", script)

    def test_last_iteration_marks_blank_result_final(self) -> None:
        """Last mic-loop iteration should still become final."""
        result = TranscriptionResult(
            source="microphone",
            text="",
            is_final=False,
            chunk_count=3,
        )
        self.assertTrue(should_mark_result_final(result, 0, True, 3, 8))

    def test_format_transcription_result_marks_silence(self) -> None:
        """Silence results should be labeled explicitly."""
        result = TranscriptionResult(
            source="microphone",
            text="",
            is_final=False,
            chunk_count=4,
            is_silence=True,
        )
        self.assertEqual(format_transcription_result(result), "[silence 4] silence detected")

    def test_format_transcription_result_marks_input_disabled(self) -> None:
        """Input-gated results should not be presented as silence."""
        result = TranscriptionResult(
            source="microphone",
            text="",
            is_final=False,
            chunk_count=0,
            input_enabled=False,
            input_gate_reason="sword_sign",
        )
        self.assertEqual(
            format_transcription_result(result),
            "[disabled] input disabled: sword_sign",
        )

    def test_parse_input_gate_payload_accepts_mic_enabled_alias(self) -> None:
        """Input-gate protocol should accept mic_enabled for integration adapters."""
        event = parse_input_gate_payload(
            {
                "type": "input_gate_state",
                "mic_enabled": True,
                "reason": "sword_sign",
                "source": "gesture_bridge",
                "timestamp": 1710000000.0,
            }
        )
        self.assertTrue(event.input_enabled)
        self.assertEqual(event.reason, "sword_sign")
        self.assertEqual(event.source, "gesture_bridge")
        self.assertEqual(event.timestamp, 1710000000.0)

    def test_parse_input_gate_payload_rejects_non_boolean_enabled_value(self) -> None:
        """Input-gate protocol should reject ambiguous string booleans."""
        with self.assertRaises(InputGateError):
            parse_input_gate_payload({"mic_enabled": "true"})

    def test_input_gate_state_formats_and_serializes(self) -> None:
        """Input-gate state should have stable text and JSON-friendly views."""
        gate = InputGate(initially_enabled=False, reason="sword_sign", source="test")
        self.assertEqual(
            format_input_gate_state(gate.state),
            "[input-gate] input_enabled=False reason=sword_sign source=test",
        )
        payload = build_input_gate_data(gate.state)
        self.assertEqual(payload["type"], "input_gate_state")
        self.assertFalse(payload["input_enabled"])

    def test_current_independent_user_candidate_consumes_exactly_once(self) -> None:
        """Only the current complete not-self-output join should mint once."""
        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        self.assertIsNotNone(capability)
        self.assertTrue(gate.consume_turn_input_capability(capability, candidate))
        self.assertFalse(gate.consume_turn_input_capability(capability, candidate))
        self.assertIsNone(gate.issue_turn_input_capability(candidate))

    def test_private_pcm_session_consumes_before_one_transcription(self) -> None:
        """MicLoopSession should consume once immediately before private PCM STT."""
        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        pipeline = mock.Mock()

        def transcribe_private(buffer: object, **_: object) -> TranscriptionResult:
            chunk = buffer.latest_chunk()
            chunk.clear()
            return TranscriptionResult(
                source="microphone",
                text="private result",
                is_final=False,
                chunk_count=1,
            )

        pipeline._transcribe_private_buffer_result.side_effect = transcribe_private
        session = MicLoopSession(
            pipeline=pipeline,
            tuning=MicLoopTuning(vad_aggressiveness=2, final_stable_seconds=8),
            input_gate=gate,
        )
        pcm = bytearray(b"\x01\x00" * 160)
        chunk = AudioChunk(
            path=None,
            source="microphone",
            pcm16=pcm,
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        result = session.process_chunk(
            chunk,
            has_speech=True,
            language="ja",
            chunk_duration=1,
            is_last_iteration=False,
            candidate_evidence=candidate,
            turn_input_capability=capability,
        )
        self.assertEqual(result.text, "private result")
        pipeline._transcribe_private_buffer_result.assert_called_once()
        self.assertEqual(pcm, bytearray(len(pcm)))

        replay = AudioChunk(
            path=None,
            source="microphone",
            pcm16=bytearray(b"\x01\x00" * 160),
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        with self.assertRaises(AudioInputError):
            session.process_chunk(
                replay,
                has_speech=True,
                language="ja",
                chunk_duration=1,
                is_last_iteration=False,
                candidate_evidence=candidate,
                turn_input_capability=capability,
            )
        pipeline._transcribe_private_buffer_result.assert_called_once()
        self.assertEqual(replay.pcm16, bytearray(len(replay.pcm16 or b"")))

    def test_private_pcm_pipeline_transcribes_in_memory_and_clears_samples(self) -> None:
        """The private pipeline should use no path and clear PCM after conversion."""
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline.model_name = "test"
        pipeline.model = mock.Mock()
        pipeline.model.transcribe.return_value = {"text": " private result "}
        pcm = bytearray(b"\x01\x00" * 160)
        chunk = AudioChunk(
            path=None,
            source="microphone",
            pcm16=pcm,
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        text = pipeline._transcribe_private_pcm_chunk(chunk, language="ja")
        self.assertEqual(text, "private result")
        pipeline.model.transcribe.assert_called_once()
        model_audio = pipeline.model.transcribe.call_args.args[0]
        self.assertEqual(float(model_audio.sum()), 0.0)
        self.assertEqual(pcm, bytearray(len(pcm)))

    def test_private_pcm_chunk_blocks_representation_copy_and_pickle(self) -> None:
        """Private PCM should not cross representation or object-copy boundaries."""
        private_marker = b"private-pcm-marker-do-not-echo"
        pcm = bytearray(private_marker)
        chunk = AudioChunk(
            path=None,
            source="microphone",
            pcm16=pcm,
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        self.assertEqual(repr(chunk), "<audio-chunk private-pcm>")
        self.assertEqual(str(chunk), repr(chunk))
        self.assertNotIn(private_marker.decode(), repr(chunk))
        with self.assertRaises(TypeError):
            copy.copy(chunk)
        with self.assertRaises(TypeError):
            copy.deepcopy(chunk)
        with self.assertRaises(TypeError):
            pickle.dumps(chunk)

        file_chunk = AudioChunk(path=Path("prepared.wav"), source="file")
        for clone in (
            copy.copy(file_chunk),
            copy.deepcopy(file_chunk),
            pickle.loads(pickle.dumps(file_chunk)),
        ):
            self.assertEqual(clone, file_chunk)
            self.assertIsNone(clone.pcm16)

    def test_private_pcm_conversion_and_transcription_failures_clear_storage(self) -> None:
        """Conversion and model failures should zero every mutable audio buffer."""
        conversion_marker = "private-conversion-marker-do-not-echo"
        conversion_pcm = bytearray(b"\x01\x00" * 160)
        conversion_chunk = AudioChunk(
            path=None,
            source="microphone",
            pcm16=conversion_pcm,
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        fake_numpy = mock.Mock()
        fake_numpy.int16 = object()
        fake_numpy.float32 = object()
        fake_view = mock.Mock()
        fake_view.astype.side_effect = ValueError(conversion_marker)
        fake_numpy.frombuffer.return_value = fake_view
        pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
        pipeline.model_name = "test"
        pipeline.model = mock.Mock()
        with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
            with self.assertRaises(AudioTranscriptionError) as conversion_error:
                pipeline._transcribe_private_pcm_chunk(conversion_chunk)
        self.assertNotIn(conversion_marker, str(conversion_error.exception))
        self.assertEqual(conversion_pcm, bytearray(len(conversion_pcm)))
        fake_view.fill.assert_called_once_with(0)
        pipeline.model.transcribe.assert_not_called()

        transcription_marker = "private-transcription-marker-do-not-echo"
        transcription_pcm = bytearray(b"\x01\x00" * 160)
        transcription_chunk = AudioChunk(
            path=None,
            source="microphone",
            pcm16=transcription_pcm,
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        retained_samples: list[object] = []

        def fail_transcription(samples: object, **_: object) -> object:
            retained_samples.append(samples)
            raise RuntimeError(transcription_marker)

        pipeline.model.transcribe.side_effect = fail_transcription
        with self.assertRaises(AudioTranscriptionError) as transcription_error:
            pipeline._transcribe_private_pcm_chunk(transcription_chunk)
        self.assertNotIn(transcription_marker, str(transcription_error.exception))
        self.assertEqual(transcription_pcm, bytearray(len(transcription_pcm)))
        self.assertEqual(len(retained_samples), 1)
        self.assertEqual(float(retained_samples[0].sum()), 0.0)

    def test_concurrent_capability_consumption_has_one_winner(self) -> None:
        """The pending capability compare-and-consume should be atomic."""
        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        barrier = threading.Barrier(8)
        results: list[bool] = []
        result_lock = threading.Lock()

        def consume() -> None:
            barrier.wait()
            result = gate.consume_turn_input_capability(capability, candidate)
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=consume) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 7)

    def test_candidate_join_mismatch_active_cooldown_and_ambiguity_fail_closed(self) -> None:
        """Every incomplete or non-independent join should issue no capability."""
        mismatch_cases = {
            "candidate_id": {"candidate_id": "not-a-private-candidate"},
            "observed_generation": {"observed_generation": 6},
            "active_generation": {"active_generation": 8},
            "session_id": {
                "observed_system_speech_session_id": (
                    "system-speech-session:sss_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                )
            },
            "playback": {
                "playback_event_ref": (
                    "playback-event:pe_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                )
            },
            "self_output_ref": {
                "self_output_observation_ref": (
                    "self-output-observation:aso_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                )
            },
            "ambiguous": {"self_output_correlation_class": "ambiguous"},
            "self_output": {"self_output_correlation_class": "self_output"},
            "cooldown": {"cooldown_status": "active"},
            "aec_authority": {"aec_or_vad_turn_input_authority": True},
            "decision_owner": {"decision_owner": "caller"},
            "acceptance": {"acceptance_status": "accepted"},
        }
        for name, overrides in mismatch_cases.items():
            with self.subTest(name=name):
                gate = prepare_current_input_gate()
                self.assertIsNone(
                    gate.issue_turn_input_capability(
                        build_user_speech_candidate(**overrides)
                    )
                )
        for state in ("handoff_accepted", "cooldown"):
            with self.subTest(lifecycle_state=state):
                gate = InputGate()
                observe_gate_lifecycle(
                    gate,
                    build_system_speech_lifecycle("handoff_accepted"),
                )
                if state == "cooldown":
                    observe_gate_lifecycle(
                        gate,
                        build_system_speech_lifecycle("cooldown"),
                        wall_timestamp="2026-07-13T12:00:01.000Z",
                    )
                gate.observe_self_output_observation(
                    build_self_output_observation()
                )
                self.assertIsNone(
                    gate.issue_turn_input_capability(build_user_speech_candidate())
                )

    def test_missing_provider_stale_and_toctou_fail_closed(self) -> None:
        """Provider absence, stale evidence, or post-issue change must block."""
        missing = InputGate()
        self.assertIsNone(
            missing.issue_turn_input_capability(build_user_speech_candidate())
        )
        for state, wall_timestamp in (
            ("handoff_accepted", "2026-07-13T12:00:00.000Z"),
            ("cooldown", "2026-07-13T12:00:01.000Z"),
            ("released", "2026-07-13T12:00:02.000Z"),
        ):
            observe_gate_lifecycle(
                missing,
                build_system_speech_lifecycle(state),
                wall_timestamp=wall_timestamp,
            )
        self.assertIsNone(
            missing.issue_turn_input_capability(build_user_speech_candidate())
        )
        with self.assertRaises(InputGateError):
            missing.observe_self_output_observation(
                build_self_output_observation(
                    playback_ref=(
                        "playback-event:pe_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                    )
                )
            )
        with self.assertRaises(InputGateError):
            missing.observe_system_speech_lifecycle(
                build_system_speech_lifecycle(generation=6)
            )

        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle(
                "handoff_accepted",
                generation=8,
                session_id=(
                    "system-speech-session:sss_ffffffffffffffffffffffffffffffff"
                ),
                playback_ref=(
                    "playback-event:pe_ffffffffffffffffffffffffffffffff"
                ),
            ),
            turn_id="web_ffffffffffffffffffffffffffffffff",
            wall_timestamp="2026-07-13T12:01:00.000Z",
        )
        self.assertFalse(gate.consume_turn_input_capability(capability, candidate))
        self.assertFalse(gate.consume_turn_input_capability(capability, candidate))

    def test_foreign_copy_serialized_and_scalar_capabilities_fail(self) -> None:
        """Only the exact process-local object and candidate identity may consume."""
        gate = prepare_current_input_gate()
        private_marker = "private-candidate-marker-do-not-echo"
        candidate = build_user_speech_candidate(
            active_system_speech_session_id=private_marker
        )
        self.assertEqual(
            repr(candidate),
            "<user-speech-candidate-evidence private>",
        )
        self.assertEqual(str(candidate), repr(candidate))
        self.assertNotIn(private_marker, repr(candidate))
        with self.assertRaises(TypeError):
            copy.copy(candidate)
        with self.assertRaises(TypeError):
            copy.deepcopy(candidate)
        with self.assertRaises(TypeError):
            pickle.dumps(candidate)

        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        self.assertIsNotNone(capability)
        self.assertEqual(repr(capability), "<turn-input-capability private>")
        with self.assertRaises(TypeError):
            bool(capability)
        with self.assertRaises(TypeError):
            copy.copy(capability)
        with self.assertRaises(TypeError):
            copy.deepcopy(capability)
        with self.assertRaises(TypeError):
            pickle.dumps(capability)
        with self.assertRaises(TypeError):
            json.dumps(capability)
        with self.assertRaises(TypeError):
            vars(capability)

        foreign_gate = prepare_current_input_gate()
        self.assertFalse(
            foreign_gate.consume_turn_input_capability(capability, candidate)
        )
        copied_candidate = build_user_speech_candidate()
        self.assertIsNot(copied_candidate, candidate)
        self.assertFalse(
            gate.consume_turn_input_capability(capability, copied_candidate)
        )
        for substitute in (
            None,
            True,
            False,
            1,
            "accepted_user_speech_candidate",
            {},
            {"capability": repr(capability)},
            object(),
        ):
            with self.subTest(substitute=type(substitute).__name__):
                self.assertFalse(
                    gate.consume_turn_input_capability(substitute, candidate)
                )
        self.assertTrue(gate.consume_turn_input_capability(capability, candidate))
        serialized_surfaces = json.dumps(
            {
                "gate": gate.state.to_payload(),
                "health": build_input_gate_response(gate.state),
            }
        )
        self.assertNotIn("capability", serialized_surfaces)
        self.assertNotIn("candidate_id", serialized_surfaces)

    def test_audiochunk_aec_vad_and_helper_values_cannot_promote_pcm(self) -> None:
        """PCM metadata and helper values should never substitute gate authority."""
        with self.assertRaises(AudioInputError):
            AudioChunk(
                path=None,
                source="microphone",
                pcm16=bytearray(b"\x01\x00"),
                sample_rate=16_000,
                storage_class="in_memory_ephemeral",
                turn_input_authority=True,
                turn_input_authority_class="accepted_user_speech_candidate",
            )
        pipeline = mock.Mock()
        session = MicLoopSession(
            pipeline=pipeline,
            tuning=MicLoopTuning(vad_aggressiveness=2, final_stable_seconds=8),
            input_gate=InputGate(initially_enabled=True),
        )
        for substitute in (None, True, "aec_selected", {"has_speech": True}):
            pcm = bytearray(b"\x01\x00" * 160)
            chunk = AudioChunk(
                path=None,
                source="microphone",
                pcm16=pcm,
                sample_rate=16_000,
                storage_class="in_memory_ephemeral",
                turn_input_authority=False,
                turn_input_authority_class="processed_near_end_observation_only",
            )
            with self.subTest(substitute=substitute):
                with self.assertRaises(AudioInputError):
                    session.process_chunk(
                        chunk,
                        has_speech=True,
                        language="ja",
                        chunk_duration=1,
                        is_last_iteration=False,
                        candidate_evidence=build_user_speech_candidate(),
                        turn_input_capability=substitute,
                    )
                self.assertEqual(pcm, bytearray(len(pcm)))
        pipeline._transcribe_private_buffer_result.assert_not_called()

        silence_pcm = bytearray(b"\x01\x00" * 160)
        silence = session.process_chunk(
            AudioChunk(
                path=None,
                source="microphone",
                pcm16=silence_pcm,
                sample_rate=16_000,
                storage_class="in_memory_ephemeral",
                turn_input_authority=False,
                turn_input_authority_class="processed_near_end_observation_only",
            ),
            has_speech=False,
            language="ja",
            chunk_duration=1,
            is_last_iteration=False,
        )
        self.assertTrue(silence.is_silence)
        self.assertEqual(silence_pcm, bytearray(len(silence_pcm)))

    def test_http_intake_is_observation_only_and_gate_update_is_capture_only(self) -> None:
        """JSON intake may update observations or capture state, never capability."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        for index, state in enumerate(
            ("handoff_accepted", "cooldown", "released")
        ):
            response = client.post(
                "/api/events/ingest",
                headers=headers,
                json={
                    "event": "swordAgentSystemSpeechLifecycleV0",
                    "turn_id": "web_abcdef0123456789abcdef0123456789",
                    "source": "self-output-awareness-controller",
                    "payload": build_system_speech_lifecycle(state),
                    "client_timestamp_wall": (
                        f"2026-07-13T12:00:0{index}.000Z"
                    ),
                    "client_timestamp_monotonic": 12.5 + index,
                    "client_performance_now": 12_500.0 + index,
                },
            )
            self.assertEqual(response.status_code, 202)
            response_text = response.get_data(as_text=True)
            self.assertNotIn(SYSTEM_SPEECH_SESSION_ID, response_text)
            self.assertNotIn(PLAYBACK_EVENT_REF, response_text)
            self.assertNotIn("capability", response_text)
        response = client.post(
            "/api/events/ingest",
            headers=headers,
            json={
                "event": "audioSelfOutputObservationV0",
                "payload": build_self_output_observation(),
            },
        )
        self.assertEqual(response.status_code, 202)
        self.assertNotIn(SYSTEM_SPEECH_SESSION_ID, response.get_data(as_text=True))
        self.assertNotIn(PLAYBACK_EVENT_REF, response.get_data(as_text=True))
        self.assertNotIn("capability", response.get_data(as_text=True))

        response = client.post(
            "/api/input-gate",
            headers=headers,
            json={
                "input_enabled": True,
                "reason": "capture-only",
                "candidate_id": "ausc_live:cid_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "may_materialize_thought_core_turninput": True,
                "capability": {"accepted": True},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            set(payload["input_gate"]),
            {"type", "input_enabled", "reason", "source", "timestamp"},
        )
        serialized = json.dumps(payload)
        self.assertNotIn("capability", serialized)
        self.assertNotIn("candidate_id", serialized)
        for endpoint in ("/api/health", "/api/status"):
            status = client.get(endpoint, headers=headers)
            self.assertEqual(status.status_code, 200)
            self.assertNotIn("capability", status.get_data(as_text=True))
            self.assertNotIn("candidate_id", status.get_data(as_text=True))

    def test_http_lifecycle_intake_forwards_validated_transport_context(self) -> None:
        """Core restart ordering uses only the validated controller envelope."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        turn_id = "web_abcdef0123456789abcdef0123456789"
        wall = "2026-07-13T12:00:00.000Z"
        with mock.patch.object(
            InputGate,
            "observe_system_speech_lifecycle",
            autospec=True,
        ) as observe:
            missing_timing = client.post(
                "/api/events/ingest",
                headers=headers,
                json={
                    "event": "swordAgentSystemSpeechLifecycleV0",
                    "turn_id": turn_id,
                    "source": "self-output-awareness-controller",
                    "payload": build_system_speech_lifecycle(
                        "handoff_accepted"
                    ),
                },
            )
            self.assertEqual(missing_timing.status_code, 400)
            self.assertEqual(observe.call_count, 0)
            complete_handoff = {
                "event": "swordAgentSystemSpeechLifecycleV0",
                "turn_id": turn_id,
                "source": "self-output-awareness-controller",
                "payload": build_system_speech_lifecycle(
                    "handoff_accepted"
                ),
                "client_timestamp_wall": wall,
                "client_timestamp_monotonic": 12.5,
                "client_performance_now": 12_500.0,
            }
            for missing_field in ("source", "turn_id"):
                with self.subTest(missing_transport_field=missing_field):
                    incomplete_handoff = dict(complete_handoff)
                    del incomplete_handoff[missing_field]
                    rejected = client.post(
                        "/api/events/ingest",
                        headers=headers,
                        json=incomplete_handoff,
                    )
                    self.assertEqual(rejected.status_code, 400)
                    self.assertEqual(observe.call_count, 0)
            response = client.post(
                "/api/events/ingest",
                headers=headers,
                json={
                    "event": "swordAgentSystemSpeechLifecycleV0",
                    "turn_id": turn_id,
                    "source": "self-output-awareness-controller",
                    "payload": build_system_speech_lifecycle(
                        "handoff_accepted"
                    ),
                    "client_timestamp_wall": wall,
                    "client_timestamp_monotonic": 12.5,
                    "client_performance_now": 12_500.0,
                },
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(observe.call_count, 1)
        self.assertEqual(
            observe.call_args.kwargs,
            {
                "transport_source": "self-output-awareness-controller",
                "transport_turn_id": turn_id,
                "transport_wall_timestamp": wall,
            },
        )

        rejected_initial_released = client.post(
            "/api/events/ingest",
            headers=headers,
            json={
                "event": "swordAgentSystemSpeechLifecycleV0",
                "turn_id": "web_11111111111111111111111111111111",
                "source": "self-output-awareness-controller",
                "payload": build_system_speech_lifecycle("released"),
                "client_timestamp_wall": "2026-07-13T12:01:00.000Z",
                "client_timestamp_monotonic": 13.5,
                "client_performance_now": 13_500.0,
            },
        )
        self.assertEqual(rejected_initial_released.status_code, 400)

    def test_web_event_and_gate_fields_never_echo_private_markers(self) -> None:
        """Strict event/gate parsing should reject or classify every marker."""
        marker = "private-web-marker-do-not-echo"
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        valid_timing: dict[str, object] = {
            "client_timestamp_wall": "2026-07-13T12:00:00.000Z",
            "client_timestamp_monotonic": 12.5,
            "client_performance_now": 12_500.0,
        }
        valid_record_start: dict[str, object] = {
            "event": "record_start",
            "turn_id": "web_abcdef0123456789abcdef0123456789",
            "source": "web-ui",
            "payload": {
                "trigger": "manual",
                "timeslice_ms": 500,
                "mime_type": "audio/webm",
            },
            **valid_timing,
        }

        def assert_rejected(body: dict[str, object]) -> None:
            response = client.post(
                "/api/events/ingest",
                headers=headers,
                json=body,
            )
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(marker, response.get_data(as_text=True))

        for field in ("event", "turn_id", "source", "payload"):
            with self.subTest(event_top_level_field=field):
                body = copy.deepcopy(valid_record_start)
                body[field] = marker
                assert_rejected(body)
        for field in (
            "client_timestamp_wall",
            "client_timestamp_monotonic",
            "client_performance_now",
        ):
            with self.subTest(event_timing_field=field):
                body = copy.deepcopy(valid_record_start)
                body[field] = marker
                assert_rejected(body)
        unexpected = copy.deepcopy(valid_record_start)
        unexpected["capability"] = marker
        assert_rejected(unexpected)

        for field in ("trigger", "timeslice_ms", "mime_type"):
            with self.subTest(record_start_field=field):
                body = copy.deepcopy(valid_record_start)
                body["payload"][field] = marker
                assert_rejected(body)

        valid_record_stop: dict[str, object] = {
            "event": "record_stop",
            "turn_id": "web_abcdef0123456789abcdef0123456789",
            "source": "web-ui",
            "payload": {
                "chunk_count": 1,
                "chunk_sequence": 1,
                "blob_size_bytes": 640,
            },
            **valid_timing,
        }
        for field in ("chunk_count", "chunk_sequence", "blob_size_bytes"):
            with self.subTest(record_stop_field=field):
                body = copy.deepcopy(valid_record_stop)
                body["payload"][field] = marker
                assert_rejected(body)

        lifecycle = build_system_speech_lifecycle("handoff_accepted")
        for field in lifecycle:
            with self.subTest(lifecycle_field=field):
                mutated = dict(lifecycle)
                mutated[field] = marker
                assert_rejected(
                    {
                        "event": "swordAgentSystemSpeechLifecycleV0",
                        "payload": mutated,
                    }
                )
        accepted_lifecycle = client.post(
            "/api/events/ingest",
            headers=headers,
            json={
                "event": "swordAgentSystemSpeechLifecycleV0",
                "turn_id": "web_abcdef0123456789abcdef0123456789",
                "source": "self-output-awareness-controller",
                "payload": lifecycle,
                **valid_timing,
            },
        )
        self.assertEqual(accepted_lifecycle.status_code, 202)
        observation = build_self_output_observation()
        for field in observation:
            with self.subTest(self_output_field=field):
                mutated = dict(observation)
                mutated[field] = marker
                assert_rejected(
                    {
                        "event": "audioSelfOutputObservationV0",
                        "payload": mutated,
                    }
                )
        for extra_field in ("candidate", "capability", "pcm16"):
            with self.subTest(observation_extra_field=extra_field):
                mutated = dict(observation)
                mutated[extra_field] = marker
                assert_rejected(
                    {
                        "event": "audioSelfOutputObservationV0",
                        "payload": mutated,
                    }
                )

        gate_cases = (
            {"input_enabled": marker},
            {"mic_enabled": marker},
            {"enabled": marker},
            {"input_enabled": True, "timestamp": marker},
            {"input_enabled": True, "reason": marker},
            {"input_enabled": True, "source": marker},
            {"input_enabled": True, "candidate_id": marker},
            {"input_enabled": True, "capability": {"value": marker}},
            {"input_enabled": True, "pcm16": marker},
        )
        for gate_body in gate_cases:
            with self.subTest(gate_fields=tuple(gate_body)):
                response = client.post(
                    "/api/input-gate",
                    headers=headers,
                    json=gate_body,
                )
                self.assertIn(response.status_code, {200, 400})
                self.assertNotIn(marker, response.get_data(as_text=True))

        for endpoint in ("/api/health", "/api/status", "/api/events?once=1"):
            response = client.get(endpoint, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(marker, response.get_data(as_text=True))
        for event in read_event_log_events(limit=100):
            self.assertNotIn(marker, json.dumps(event))
            self.assertNotIn(
                marker,
                format_sse_event(event, str(event.get("event", "message"))),
            )

    def test_maybe_finalize_on_silence_returns_final_result(self) -> None:
        """A silence chunk after repeated speech should finalize the last speech."""
        silence_result = TranscriptionResult(
            source="microphone",
            text="",
            is_final=False,
            chunk_count=5,
            is_silence=True,
        )
        last_spoken_result = TranscriptionResult(
            source="microphone",
            text="依存関係を確認して",
            is_final=False,
            chunk_count=4,
        )
        final_result = maybe_finalize_on_silence(
            result=silence_result,
            last_spoken_result=last_spoken_result,
            repeat_count=2,
            finalized_text=None,
        )
        self.assertTrue(final_result.is_final)
        self.assertFalse(final_result.is_silence)
        self.assertEqual(final_result.text, "依存関係を確認して")

    def test_maybe_finalize_on_silence_keeps_silence_without_repeat(self) -> None:
        """Silence should remain silence when speech was not yet stable."""
        silence_result = TranscriptionResult(
            source="microphone",
            text="",
            is_final=False,
            chunk_count=5,
            is_silence=True,
        )
        last_spoken_result = TranscriptionResult(
            source="microphone",
            text="依存関係を確認して",
            is_final=False,
            chunk_count=4,
        )
        final_result = maybe_finalize_on_silence(
            result=silence_result,
            last_spoken_result=last_spoken_result,
            repeat_count=1,
            finalized_text=None,
        )
        self.assertTrue(final_result.is_silence)

    def test_maybe_finalize_on_interrupt_returns_final_result(self) -> None:
        """Interrupt should flush the latest spoken result when not yet finalized."""
        last_spoken_result = TranscriptionResult(
            source="microphone",
            text="依存関係を確認して",
            is_final=False,
            chunk_count=4,
        )
        final_result = maybe_finalize_on_interrupt(
            last_spoken_result=last_spoken_result,
            finalized_text=None,
            chunk_count=5,
        )
        self.assertIsNotNone(final_result)
        assert final_result is not None
        self.assertTrue(final_result.is_final)
        self.assertEqual(final_result.chunk_count, 5)
        self.assertEqual(final_result.text, "依存関係を確認して")

    def test_maybe_finalize_on_interrupt_skips_already_finalized_text(self) -> None:
        """Interrupt should not re-emit already finalized speech."""
        last_spoken_result = TranscriptionResult(
            source="microphone",
            text="依存関係を確認して",
            is_final=False,
            chunk_count=4,
        )
        final_result = maybe_finalize_on_interrupt(
            last_spoken_result=last_spoken_result,
            finalized_text="依存関係を確認して",
            chunk_count=5,
        )
        self.assertIsNone(final_result)

    def test_mic_loop_session_tracks_repeat_state_and_finalizes(self) -> None:
        """Mic-loop session should promote repeated speech to final."""
        pipeline = mock.Mock()
        pipeline.transcribe_buffer_result.side_effect = [
            TranscriptionResult(
                source="microphone",
                text="依存関係を確認して",
                is_final=False,
                chunk_count=1,
            ),
            TranscriptionResult(
                source="microphone",
                text="依存関係を確認して",
                is_final=False,
                chunk_count=2,
            ),
        ]
        session = MicLoopSession(
            pipeline=pipeline,
            tuning=MicLoopTuning(vad_aggressiveness=2, final_stable_seconds=8),
        )
        first = session.process_chunk(
            AudioChunk(path=Path("chunk1.wav"), source="microphone"),
            has_speech=True,
            language="ja",
            chunk_duration=4,
            is_last_iteration=False,
        )
        second = session.process_chunk(
            AudioChunk(path=Path("chunk2.wav"), source="microphone"),
            has_speech=True,
            language="ja",
            chunk_duration=4,
            is_last_iteration=False,
        )
        self.assertFalse(first.is_final)
        self.assertTrue(second.is_final)
        self.assertEqual(session.state.repeat_count, 2)
        self.assertEqual(session.state.finalized_text, "依存関係を確認して")

    def test_mic_loop_session_exposes_input_gate_decision(self) -> None:
        """Mic-loop session should expose input gating without gesture details."""
        pipeline = mock.Mock()
        session = MicLoopSession(
            pipeline=pipeline,
            tuning=MicLoopTuning(vad_aggressiveness=2, final_stable_seconds=8),
            input_gate=InputGate(initially_enabled=False, reason="sword_sign"),
        )
        self.assertFalse(session.should_accept_input())
        disabled_result = session.process_input_disabled()
        self.assertFalse(disabled_result.input_enabled)
        self.assertEqual(disabled_result.input_gate_reason, "sword_sign")
        blocked_result = session.process_chunk(
            AudioChunk(path=Path("blocked.wav"), source="microphone"),
            has_speech=True,
            language="ja",
            chunk_duration=3,
            is_last_iteration=False,
        )
        self.assertFalse(blocked_result.input_enabled)
        pipeline.transcribe_buffer_result.assert_not_called()
        session.update_input_gate(
            InputGateEvent(
                input_enabled=True,
                reason="sword_sign",
                source="gesture_bridge",
            )
        )
        self.assertTrue(session.should_accept_input())
        self.assertEqual(session.input_gate_state().source, "gesture_bridge")
        pipeline.transcribe_buffer_result.return_value = TranscriptionResult(
            source="microphone",
            text="依存関係を確認して",
            is_final=False,
            chunk_count=1,
        )
        result = session.process_chunk(
            AudioChunk(path=Path("chunk1.wav"), source="microphone"),
            has_speech=True,
            language="ja",
            chunk_duration=3,
            is_last_iteration=True,
        )
        self.assertTrue(result.input_enabled)
        self.assertEqual(result.text, "依存関係を確認して")

    def test_mic_loop_session_finalize_on_interrupt_uses_internal_state(self) -> None:
        """Interrupt finalization should use the session's tracked last speech."""
        pipeline = mock.Mock()
        pipeline.transcribe_buffer_result.return_value = TranscriptionResult(
            source="microphone",
            text="依存関係を確認して",
            is_final=False,
            chunk_count=1,
        )
        session = MicLoopSession(
            pipeline=pipeline,
            tuning=MicLoopTuning(vad_aggressiveness=2, final_stable_seconds=8),
        )
        session.process_chunk(
            AudioChunk(path=Path("chunk1.wav"), source="microphone"),
            has_speech=True,
            language="ja",
            chunk_duration=3,
            is_last_iteration=False,
        )
        final_result = session.finalize_on_interrupt()
        self.assertIsNotNone(final_result)
        assert final_result is not None
        self.assertTrue(final_result.is_final)
        self.assertEqual(final_result.chunk_count, 1)

    def test_process_web_transcription_supports_command_only(self) -> None:
        """Web transcription service should hide transcript in command-only mode."""
        with mock.patch(
            "src.web.transcription_service.get_cached_transcription_pipeline"
        ) as pipeline_cls, mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            return_value=2.0,
        ):
            pipeline_cls.return_value.transcribe_chunk.return_value = "依存関係を確認して"
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-audio",
                    filename="sample.wav",
                    model_name="small",
                    language="ja",
                    command_only=True,
                )
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.transcript, "")
        self.assertEqual(response.command, "依存関係を確認して")

    def test_process_web_transcription_can_save_handoff_paths(self) -> None:
        """Web transcription service should return saved handoff paths."""
        saved_paths = mock.Mock(
            json_path=Path("/tmp/web_latest.json"),
            text_path=Path("/tmp/web_latest.txt"),
        )
        with mock.patch(
            "src.web.transcription_service.get_cached_transcription_pipeline"
        ) as pipeline_cls, mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            return_value=2.0,
        ), mock.patch(
            "src.web.transcription_service.save_handoff_bundle",
            return_value=saved_paths,
        ):
            pipeline_cls.return_value.transcribe_chunk.return_value = "依存関係を確認して"
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-audio",
                    filename="sample.wav",
                    turn_id="handofftest",
                    model_name="small",
                    save_handoff=True,
                )
            )
        self.assertEqual(response.command_path, str(saved_paths.json_path))
        self.assertEqual(response.command_text_path, str(saved_paths.text_path))
        events = read_event_log_events(limit=10, turn_id="handofftest")
        handoff_events = [
            event for event in events if event.get("event") == "handoff_saved"
        ]
        self.assertTrue(handoff_events)
        handoff_payload = handoff_events[-1]["payload"]
        self.assertEqual(handoff_payload["json_filename"], "web_latest.json")
        self.assertEqual(handoff_payload["text_filename"], "web_latest.txt")
        self.assertIn("transcript", handoff_payload)
        self.assertNotIn("依存関係", str(handoff_payload))

    def test_process_web_transcription_skips_short_audio_before_whisper(self) -> None:
        """Very short recordings should not be sent to Whisper."""
        with mock.patch(
            "src.web.transcription_service.get_cached_transcription_pipeline"
        ) as pipeline_cls, mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            return_value=0.1,
        ):
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-audio",
                    filename="short.wav",
                    model_name="small",
                    language="ja",
                )
            )
        pipeline_cls.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.message, "音声を認識できませんでした。")
        self.assertEqual(response.transcript, "")
        self.assertEqual(response.debug["skip_reason"], "duration_below_minimum")
        self.assertFalse(response.debug["whisper_invoked"])
        self.assertTrue(response.debug["whisper_skipped"])
        self.assertEqual(response.debug["model"], "small")
        self.assertEqual(response.debug["language"], "ja")

    def test_process_web_transcription_skips_vad_no_speech_before_whisper(self) -> None:
        """Recordings with no VAD-detectable speech should not be sent to Whisper."""
        with mock.patch(
            "src.web.transcription_service.get_cached_transcription_pipeline"
        ) as pipeline_cls, mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            return_value=2.0,
        ), mock.patch(
            "src.web.transcription_service.has_detectable_speech",
            return_value=False,
        ):
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-audio",
                    filename="silent.wav",
                    model_name="small",
                    language="ja",
                )
            )
        pipeline_cls.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.message, "音声を認識できませんでした。")
        self.assertEqual(response.debug["skip_reason"], "vad_no_speech")
        self.assertEqual(response.debug["vad"]["reason"], "no_speech_detected")
        self.assertFalse(response.debug["whisper_invoked"])

    def test_process_web_transcription_records_webm_normalization_debug(self) -> None:
        """WebM recordings should expose normalized file facts in debug output."""
        def fake_normalize(input_path: Path, output_path: Path, timeout_seconds: int) -> Path:
            output_path.write_bytes(b"fake-normalized-wav")
            return output_path

        def fake_validate(audio_path: Path) -> float:
            self.assertEqual(audio_path.suffix, ".wav")
            return 1.9

        with mock.patch(
            "src.web.transcription_service.get_cached_transcription_pipeline"
        ) as pipeline_cls, mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            side_effect=fake_validate,
        ), mock.patch(
            "src.web.transcription_service.normalize_audio_for_transcription",
            side_effect=fake_normalize,
        ), mock.patch(
            "src.web.transcription_service.has_detectable_speech",
            return_value=False,
        ):
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-webm",
                    filename="browser_recording.webm",
                    model_name="small",
                    language="ja",
                )
            )
        pipeline_cls.assert_not_called()
        self.assertTrue(response.debug["webm_normalized"])
        self.assertEqual(response.debug["normalized_audio"]["suffix"], ".wav")
        self.assertGreater(response.debug["normalized_audio"]["size_bytes"], 0)
        self.assertEqual(response.debug["normalized_audio"]["duration_seconds"], 1.9)
        self.assertEqual(response.debug["skip_reason"], "vad_no_speech")

    def test_webm_transcription_skips_original_duration_probe(self) -> None:
        """Browser WebM uploads should normalize before duration probing."""
        def fake_normalize(input_path: Path, output_path: Path, timeout_seconds: int) -> Path:
            output_path.write_bytes(b"fake-normalized-wav")
            return output_path

        with mock.patch(
            "src.web.transcription_service.get_cached_transcription_pipeline"
        ) as pipeline_cls, mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.normalize_audio_for_transcription",
            side_effect=fake_normalize,
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            return_value=1.0,
        ) as validate_content, mock.patch(
            "src.web.transcription_service.has_detectable_speech",
            return_value=False,
        ):
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-webm",
                    filename="browser_recording.webm",
                    model_name="small",
                    language="ja",
                )
            )
        pipeline_cls.assert_not_called()
        self.assertEqual(validate_content.call_count, 1)
        validated_path = validate_content.call_args.args[0]
        self.assertEqual(validated_path.suffix, ".wav")
        self.assertTrue(response.debug["webm_normalized"])

    def test_web_transcription_debug_redacts_audio_tool_paths(self) -> None:
        """Debug error details should help diagnosis without leaking local paths."""
        with mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available"
        ), mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.validate_uploaded_audio_content",
            side_effect=AudioInputError(
                r"uploaded file is not readable audio: C:\Example\secret\bad.webm invalid data"
            ),
        ):
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-audio",
                    filename="sample.wav",
                    model_name="small",
                )
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not readable audio", response.debug["error_detail"])
        self.assertNotIn("C:\\Example", response.debug["error_detail"])

    def test_process_web_transcription_rejects_unsupported_extension(self) -> None:
        """Web transcription service should reject non-audio upload extensions."""
        response = process_web_transcription(
            WebTranscriptionRequest(
                raw_bytes=b"fake-audio",
                filename="notes.txt",
                model_name="small",
            )
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("ファイル形式", response.error)

    def test_process_web_transcription_hides_environment_details(self) -> None:
        """Web transcription errors should not expose internal exception details."""
        with mock.patch(
            "src.web.transcription_service.validate_model_name"
        ), mock.patch(
            "src.web.transcription_service.ensure_ffmpeg_available",
            side_effect=AudioEnvironmentError(r"secret path C:\internal\ffmpeg"),
        ), mock.patch(
            "src.web.transcription_service.LOGGER.exception"
        ):
            response = process_web_transcription(
                WebTranscriptionRequest(
                    raw_bytes=b"fake-audio",
                    filename="sample.wav",
                    model_name="small",
                )
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("サーバー側", response.error)
        self.assertNotIn("secret path", response.error)
        self.assertNotIn("C:\\internal", response.error)

    def test_build_codex_instruction_returns_none_for_blank(self) -> None:
        """Blank transcripts should not produce instruction drafts."""
        self.assertIsNone(build_codex_instruction("   "))

    def test_build_codex_instruction_normalizes_whitespace(self) -> None:
        """Instruction drafts should normalize whitespace."""
        draft = build_codex_instruction("  依存関係を   確認して ")
        self.assertIsNotNone(draft)
        assert draft is not None
        self.assertEqual(draft.instruction, "依存関係を 確認して")

    def test_build_codex_payload_returns_none_for_blank(self) -> None:
        """Blank transcripts should not produce Codex payloads."""
        self.assertIsNone(build_codex_payload("   "))

    def test_render_codex_prompt_includes_transcript_and_task(self) -> None:
        """Prompt text should include both transcript and requested task."""
        prompt = render_codex_prompt("  依存関係を   確認して ")
        self.assertEqual(
            prompt,
            "Voice transcript:\n依存関係を 確認して\n\nRequested task:\n依存関係を 確認して\n",
        )

    def test_save_codex_payload_writes_json(self) -> None:
        """Codex payload helper should save normalized JSON output."""
        output_path = PROJECT_ROOT / ".cache" / "tests" / "payload_helper.json"
        if output_path.exists():
            remove_path_with_retry(output_path)
        saved_path = save_codex_payload("  依存関係を   確認して ", output_path)
        self.assertEqual(saved_path, output_path)
        payload_json = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            payload_json,
            {
                "transcript": "依存関係を 確認して",
                "command": "依存関係を 確認して",
            },
        )
        remove_path_with_retry(output_path)

    def test_save_codex_handoff_bundle_writes_json_and_text(self) -> None:
        """Codex handoff helper should save both JSON and text outputs."""
        json_path = PROJECT_ROOT / ".cache" / "tests" / "handoff_bundle.json"
        text_path = PROJECT_ROOT / ".cache" / "tests" / "handoff_bundle.txt"
        if json_path.exists():
            remove_path_with_retry(json_path)
        if text_path.exists():
            remove_path_with_retry(text_path)
        saved_paths = save_codex_handoff_bundle(
            "  依存関係を   確認して ",
            json_path=json_path,
            text_path=text_path,
        )
        self.assertIsNotNone(saved_paths)
        assert saved_paths is not None
        self.assertEqual(saved_paths.json_path, json_path)
        self.assertEqual(saved_paths.text_path, text_path)
        self.assertEqual(
            text_path.read_text(encoding="utf-8"),
            "Voice transcript:\n依存関係を 確認して\n\nRequested task:\n依存関係を 確認して\n",
        )
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_load_codex_handoff_bundle_returns_saved_contents(self) -> None:
        """Handoff loader should return saved JSON and prompt text."""
        json_path = get_default_codex_output_path(source="loader_test")
        text_path = get_default_codex_text_path(source="loader_test")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        handoff = load_codex_handoff_bundle(source="loader_test")
        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual(handoff.command, "依存関係を確認して")
        self.assertIn("Requested task:", handoff.prompt_text)
        self.assertTrue(handoff.metadata["exists"])
        self.assertTrue(handoff.metadata["handoff_id"])
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_render_handoff_output_returns_json(self) -> None:
        """Handoff renderer should support JSON output."""
        json_path = get_default_codex_output_path(source="render_json")
        text_path = get_default_codex_text_path(source="render_json")
        save_codex_handoff_bundle(
            "依存関係を確認して",
            json_path=json_path,
            text_path=text_path,
        )
        payload_json = json.loads(render_handoff_output("render_json", "json"))
        self.assertEqual(payload_json["command"], "依存関係を確認して")
        remove_path_with_retry(json_path)
        remove_path_with_retry(text_path)

    def test_normalize_command_args_strips_separator(self) -> None:
        """Runner CLI should strip a leading '--' from command args."""
        self.assertEqual(
            normalize_command_args(["--", "python", "-c", "print('ok')"]),
            ["python", "-c", "print('ok')"],
        )

    def test_resolve_runner_command_prefers_template(self) -> None:
        """Runner command resolution should prefer templates when requested."""
        self.assertEqual(
            resolve_runner_command(
                "cat",
                ["--", "python", "-c", "print('ignored')"],
                PROJECT_ROOT,
            ),
            build_template_command("cat", PROJECT_ROOT),
        )

    def test_build_template_command_supports_codex_exec(self) -> None:
        """Runner templates should include a Codex exec bridge."""
        self.assertEqual(
            build_template_command("codex-exec", PROJECT_ROOT),
            ["codex", "exec", "-C", str(PROJECT_ROOT), "-"],
        )

    def test_drivers_package_exports_public_contract(self) -> None:
        """Drivers package should expose the public driver contract surface."""
        response = DriverResponse(
            backend_name="agent",
            command_name="codex",
            command_line="codex exec -",
            returncode=0,
            status="ok",
            succeeded=True,
            has_output=True,
            stdout_text="ok\n",
            stderr_text="",
            stream="stdout",
            text="ok\n",
        )

        self.assertEqual(response.command_name, "codex")
        self.assertEqual(response.status, "ok")

    def test_validate_runner_command_available_accepts_existing_path_command(self) -> None:
        """Absolute path commands should pass when they exist."""
        validate_runner_command_available([sys.executable, "--version"])

    def test_validate_runner_command_available_rejects_missing_path_command(self) -> None:
        """Missing absolute path commands should fail early."""
        with self.assertRaisesRegex(AudioInputError, "runner command not found"):
            validate_runner_command_available([str(PROJECT_ROOT / "missing-command")])

    def test_validate_runner_command_available_rejects_missing_path_entry(self) -> None:
        """PATH lookups should fail early for missing commands."""
        with mock.patch("src.drivers.base.shutil.which", return_value=None):
            with self.assertRaisesRegex(AudioInputError, "runner command not found in PATH: codex"):
                validate_runner_command_available(["codex", "exec"])

    def test_dispatch_driver_request_returns_normalized_result(self) -> None:
        """Driver dispatch should return backend metadata and subprocess output."""
        with mock.patch(
            "src.drivers.base.validate_driver_command_available"
        ), mock.patch(
            "src.drivers.base.subprocess.run"
        ) as subprocess_run:
            subprocess_run.return_value = subprocess.CompletedProcess(
                args=["cat"],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )
            result = dispatch_driver_request(
                DriverRequest(
                    backend_name="agent",
                    command=["cat"],
                    payload="hello",
                )
            )
        self.assertEqual(result.backend_name, "agent")
        self.assertEqual(result.command, ["cat"])
        self.assertEqual(result.payload, "hello")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ok\n")

    def test_dispatch_driver_request_wraps_missing_command(self) -> None:
        """Driver dispatch should surface missing commands as input errors."""
        with mock.patch(
            "src.drivers.base.validate_driver_command_available"
        ), mock.patch(
            "src.drivers.base.subprocess.run",
            side_effect=FileNotFoundError(2, "No such file or directory", "codex"),
        ):
            with self.assertRaisesRegex(AudioInputError, "runner command not found: codex"):
                dispatch_driver_request(
                    DriverRequest(
                        backend_name="agent",
                        command=["codex", "exec"],
                        payload="hello",
                    )
                )

    def test_dispatch_driver_request_returns_timeout_result(self) -> None:
        """Driver dispatch should bound external runner execution time."""
        with mock.patch(
            "src.drivers.base.validate_driver_command_available"
        ), mock.patch(
            "src.drivers.base.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["codex", "exec"], timeout=1),
        ):
            result = dispatch_driver_request(
                DriverRequest(
                    backend_name="agent",
                    command=["codex", "exec"],
                    payload="hello",
                    timeout_seconds=1,
                )
            )
        self.assertEqual(result.returncode, 124)
        self.assertIn("timed out", result.stderr)

    def test_driver_result_response_returns_backend_neutral_view(self) -> None:
        """Driver results should expose a backend-neutral response view."""
        result = DriverResult(
            backend_name="agent",
            command=["codex", "exec", "-"],
            payload="hello",
            returncode=0,
            stdout="ok\n",
            stderr="warn\n",
            command_name="codex",
        )

        response = result.response
        self.assertEqual(response.backend_name, "agent")
        self.assertEqual(response.command_name, "codex")
        self.assertEqual(response.command_line, "codex exec -")
        quoted = DriverResult(
            backend_name="agent",
            command=["python", "-c", "print('hello world')"],
            payload="",
            returncode=0,
            stdout="",
            stderr="",
            command_name="python",
        )
        self.assertEqual(quoted.command_line, shlex.join(quoted.command))
        self.assertEqual(response.returncode, 0)
        self.assertEqual(response.status, "ok")
        self.assertTrue(response.succeeded)
        self.assertTrue(response.has_output)
        self.assertEqual(response.stdout_text, "ok\n")
        self.assertEqual(response.stderr_text, "warn\n")
        self.assertEqual(response.stream, "stdout")
        self.assertEqual(response.text, "ok\n")

    def test_driver_result_status_distinguishes_output_cases(self) -> None:
        """Driver status labels should distinguish output/no-output success and failure."""
        success_no_output = DriverResult(
            backend_name="agent",
            command=["true"],
            payload="",
            returncode=0,
            stdout="",
            stderr="",
            command_name="true",
        )
        failure_no_output = DriverResult(
            backend_name="agent",
            command=["false"],
            payload="",
            returncode=1,
            stdout="",
            stderr="",
            command_name="false",
        )

        self.assertEqual(success_no_output.status, "ok_no_output")
        self.assertEqual(success_no_output.response.status, "ok_no_output")
        self.assertEqual(failure_no_output.status, "error_no_output")
        self.assertEqual(failure_no_output.response.status, "error_no_output")

    def test_execute_runner_command_dispatches_normalized_request(self) -> None:
        """Runner helpers should build driver requests consistently."""
        expected = DriverResult(
            backend_name="agent",
            command=["cat"],
            payload="hello",
            returncode=0,
            stdout="ok\n",
            stderr="",
            command_name="cat",
        )
        with mock.patch("src.runners.common.dispatch_driver_request", return_value=expected) as dispatch:
            result = execute_runner_command("agent", ["cat"], "hello")

        dispatch.assert_called_once()
        request = dispatch.call_args.args[0]
        self.assertEqual(request.backend_name, "agent")
        self.assertEqual(request.command, ["cat"])
        self.assertEqual(request.payload, "hello")
        self.assertIs(result, expected)

    def test_emit_driver_result_uses_response_stream(self) -> None:
        """Runner output emission should respect the normalized response stream."""
        result = DriverResult(
            backend_name="ollama",
            command=["ollama", "run", "llama3"],
            payload="hello",
            returncode=1,
            stdout="partial\n",
            stderr="failed\n",
            command_name="ollama",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = emit_driver_result(result)

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "partial\n")
        self.assertEqual(stderr.getvalue(), "failed\n")

    def test_build_ollama_command_normalizes_model_name(self) -> None:
        """Ollama runner should trim the model name."""
        self.assertEqual(build_ollama_command(" llama3 "), ["ollama", "run", "llama3"])

    def test_build_ollama_command_rejects_blank_model_name(self) -> None:
        """Ollama runner should reject blank model names."""
        with self.assertRaisesRegex(AudioInputError, "Ollama model name must not be blank"):
            build_ollama_command("   ")

    def test_retry_model_load_on_cpu_matches_busy_cuda_error(self) -> None:
        """CUDA busy errors should trigger a CPU retry."""
        exc = RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
        self.assertTrue(should_retry_model_load_on_cpu(exc))

    def test_retry_model_load_on_cpu_ignores_unrelated_errors(self) -> None:
        """Non-CUDA model load errors should not trigger a CPU retry."""
        exc = RuntimeError("unknown Whisper load failure")
        self.assertFalse(should_retry_model_load_on_cpu(exc))

    def test_transcription_pipeline_cache_reuses_model_by_name(self) -> None:
        """Web transcription should be able to reuse process-local Whisper pipelines."""
        clear_transcription_pipeline_cache()
        try:
            with mock.patch("src.core.pipeline.load_transcription_model") as load_model:
                load_model.side_effect = [object(), object()]
                first = get_cached_transcription_pipeline("small")
                second = get_cached_transcription_pipeline("small")
                third = get_cached_transcription_pipeline("base")
            self.assertIs(first, second)
            self.assertIsNot(first, third)
            self.assertEqual(load_model.call_count, 2)
        finally:
            clear_transcription_pipeline_cache()

    def test_print_agent_instruction_only_handles_blank(self) -> None:
        """command-only printer should handle blank transcripts."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_agent_instruction_only("   ")
        self.assertEqual(buffer.getvalue().strip(), "no instruction draft available")

    def test_print_runtime_note_writes_to_stderr(self) -> None:
        """Operational notes should go to stderr."""
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            print_runtime_note("[mic-tuning] profile=balanced")
        self.assertEqual(buffer.getvalue().strip(), "[mic-tuning] profile=balanced")

    def test_live_aec_preserves_only_allowlisted_child_failure_class(self) -> None:
        """A child failure should survive without stdout, stderr, or marker echo."""
        private_marker = "private-live-aec-child-marker-do-not-echo"

        class FakeStdin:
            def write(self, value: object) -> None:
                del value

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        class FakeServer:
            def __init__(self, *_: object) -> None:
                self.finish_count = 0

            def start(self, *, expected_client_process_id: int) -> None:
                del expected_client_process_id

            def finish(self, *, timeout_seconds: float) -> object:
                del timeout_seconds
                self.finish_count += 1
                raise AssertionError("failure result must not wait for private PCM")

            def close(self) -> None:
                return None

        def make_process(result_class: str):
            class FakeProcess:
                def __init__(self, command: object, **_: object) -> None:
                    del command
                    self.pid = 1234
                    self.stdin = FakeStdin()
                    self.returncode = 1

                def communicate(self, timeout: float) -> tuple[bytes, bytes]:
                    del timeout
                    result = {
                        "schema_version": "voice_capture_dsp_aec_observation.v0",
                        "result_class": result_class,
                        "observation": {
                            "packet_count": 0,
                            "processed_byte_count": 0,
                        },
                    }
                    return json.dumps(result).encode(), private_marker.encode()

                def poll(self) -> int:
                    return self.returncode

            return FakeProcess

        self.assertIn(
            "voice_capture_dsp_start_failed",
            LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES,
        )
        for result_class, expected in (
            ("voice_capture_dsp_start_failed", "voice_capture_dsp_start_failed"),
            (private_marker, "live_aec_helper_failed"),
        ):
            with self.subTest(result_class=result_class):
                with (
                    mock.patch(
                        "src.io.aec_reference._resolve_powershell_executable",
                        return_value="pwsh",
                    ),
                    mock.patch(
                        "src.io.aec_reference._current_process_creation_utc_ticks",
                        return_value=123,
                    ),
                    mock.patch(
                        "src.io.aec_reference._utc_now_dotnet_ticks",
                        return_value=456,
                    ),
                ):
                    with self.assertRaises(LiveAecCaptureError) as raised:
                        capture_live_aec_processed_pcm(
                            owner_selection=get_adopted_live_aec_owner_selection(),
                            window_ms=100,
                            deadline_ms=1000,
                            helper_path=Path(__file__),
                            popen_factory=make_process(result_class),
                            server_factory=FakeServer,
                        )
                self.assertEqual(raised.exception.failure_class, expected)
                self.assertNotIn(private_marker, str(raised.exception))

    def test_live_pcm_signal_class_is_fixed_and_noncontent(self) -> None:
        """Signal diagnostics should expose only fixed coarse classes."""
        self.assertEqual(
            classify_live_pcm16_signal(bytearray(320)),
            "all_zero",
        )
        self.assertEqual(
            classify_live_pcm16_signal(bytearray(b"\x20\x00" * 160)),
            "low_signal",
        )
        self.assertEqual(
            classify_live_pcm16_signal(bytearray(b"\x20\x00\x21\x00" * 80)),
            "low_signal",
        )
        self.assertEqual(
            classify_live_pcm16_signal(bytearray(b"\x21\x00" * 160)),
            "signal_above_floor",
        )

    def test_live_pcm_vad_clears_mutable_temporary_frames(self) -> None:
        """VAD should retain no immutable or uncleared frame copy."""
        retained_frames: list[bytearray] = []

        class FakeVad:
            def __init__(self, aggressiveness: int) -> None:
                self.aggressiveness = aggressiveness

            def is_speech(self, frame: bytearray, sample_rate: int) -> bool:
                self.assertions = (sample_rate, len(frame))
                retained_frames.append(frame)
                return len(retained_frames) == 2

        pcm = bytearray(b"\x01\x00" * 320)
        self.assertTrue(
            has_detectable_speech_pcm16(
                pcm,
                vad_factory=FakeVad,
            )
        )
        self.assertEqual(len(retained_frames), 2)
        self.assertTrue(
            all(all(value == 0 for value in frame) for frame in retained_frames)
        )
        self.assertTrue(any(value != 0 for value in pcm))

    def test_live_candidate_window_uses_fixed_owner_and_retains_counts(self) -> None:
        """The endpoint capture seam should choose the owner internally."""
        observed: dict[str, object] = {}
        pcm = bytearray(b"\x01\x00" * 160)

        def fake_capture(**kwargs: object) -> LiveAecProcessedCapture:
            observed.update(kwargs)
            return LiveAecProcessedCapture(pcm16=pcm, packet_count=1)

        window = capture_live_microphone_candidate_window(
            window_ms=100,
            deadline_ms=1000,
            live_aec_capture=fake_capture,
        )
        self.assertEqual(
            observed["owner_selection"],
            get_adopted_live_aec_owner_selection(),
        )
        self.assertEqual(
            observed["processing_mode_class"],
            LIVE_CAPTURE_MODE_AEC,
        )
        self.assertEqual(window.packet_count, 1)
        self.assertEqual(window.processed_byte_count, 320)
        self.assertEqual(repr(window), "<live-microphone-candidate-window private-pcm>")
        window.clear()
        self.assertEqual(pcm, bytearray(len(pcm)))

    def test_live_capture_mode_follows_only_gate_owned_lifecycle(self) -> None:
        """AEC is conservative; released joined input uses no-render NS/AGC."""
        gate = InputGate()
        self.assertEqual(
            LIVE_CAPTURE_MODE_AEC,
            aec_reference_module.LIVE_CAPTURE_MODE_AEC,
        )
        self.assertEqual(
            LIVE_CAPTURE_MODE_NS_AGC,
            aec_reference_module.LIVE_CAPTURE_MODE_NS_AGC,
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle("handoff_accepted"),
        )
        gate.observe_self_output_observation(build_self_output_observation())
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle("cooldown"),
            wall_timestamp="2026-07-13T12:00:01.000Z",
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle("released"),
            wall_timestamp="2026-07-13T12:00:02.000Z",
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_NS_AGC,
        )

        next_session_id = (
            "system-speech-session:sss_dddddddddddddddddddddddddddddddd"
        )
        next_playback_ref = (
            "playback-event:pe_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        )
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle(
                "handoff_accepted",
                generation=8,
                session_id=next_session_id,
                playback_ref=next_playback_ref,
            ),
            turn_id="web_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            wall_timestamp="2026-07-13T12:01:00.000Z",
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle(
                "cooldown",
                generation=8,
                session_id=next_session_id,
                playback_ref=next_playback_ref,
            ),
            turn_id="web_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            wall_timestamp="2026-07-13T12:01:01.000Z",
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        observe_gate_lifecycle(
            gate,
            build_system_speech_lifecycle(
                "released",
                generation=8,
                session_id=next_session_id,
                playback_ref=next_playback_ref,
            ),
            turn_id="web_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            wall_timestamp="2026-07-13T12:01:02.000Z",
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        gate.observe_self_output_observation(
            build_self_output_observation(
                generation=8,
                session_id=next_session_id,
                playback_ref=next_playback_ref,
            )
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_NS_AGC,
        )

    def test_lower_generation_restart_requires_new_ordered_transport_join(
        self,
    ) -> None:
        """A trusted page reload invalidates the old join before mode changes."""
        missing_context_gate = InputGate()
        with self.assertRaisesRegex(InputGateError, "requires trusted handoff"):
            missing_context_gate.observe_system_speech_lifecycle(
                build_system_speech_lifecycle("handoff_accepted")
            )
        self.assertEqual(
            missing_context_gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        initial_released_gate = InputGate()
        with self.assertRaisesRegex(InputGateError, "requires trusted handoff"):
            observe_gate_lifecycle(
                initial_released_gate,
                build_system_speech_lifecycle("released"),
            )
        self.assertEqual(
            initial_released_gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )

        gate = InputGate()
        first_context = {
            "transport_source": "self-output-awareness-controller",
            "transport_turn_id": "web_11111111111111111111111111111111",
            "transport_wall_timestamp": "2026-07-13T12:00:00.000Z",
        }
        for state, timestamp in (
            ("handoff_accepted", "2026-07-13T12:00:00.000Z"),
            ("cooldown", "2026-07-13T12:00:01.000Z"),
            ("released", "2026-07-13T12:00:02.000Z"),
        ):
            gate.observe_system_speech_lifecycle(
                build_system_speech_lifecycle(state),
                **{
                    **first_context,
                    "transport_wall_timestamp": timestamp,
                },
            )
        gate.observe_self_output_observation(build_self_output_observation())
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_NS_AGC,
        )

        restarted_session_id = (
            "system-speech-session:sss_44444444444444444444444444444444"
        )
        restarted_playback_ref = (
            "playback-event:pe_55555555555555555555555555555555"
        )
        restarted_handoff = build_system_speech_lifecycle(
            "handoff_accepted",
            generation=1,
            session_id=restarted_session_id,
            playback_ref=restarted_playback_ref,
        )
        with self.assertRaisesRegex(InputGateError, "lifecycle is stale"):
            gate.observe_system_speech_lifecycle(restarted_handoff)
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )

        with self.assertRaisesRegex(InputGateError, "lifecycle is stale"):
            gate.observe_system_speech_lifecycle(
                restarted_handoff,
                transport_source="self-output-awareness-controller",
                transport_turn_id="web_11111111111111111111111111111111",
                transport_wall_timestamp="2026-07-13T12:00:03.000Z",
            )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        with self.assertRaisesRegex(InputGateError, "lifecycle is stale"):
            gate.observe_system_speech_lifecycle(
                restarted_handoff,
                transport_source="self-output-awareness-controller",
                transport_turn_id="web_99999999999999999999999999999999",
                transport_wall_timestamp="2026-07-13T12:00:02.000Z",
            )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )

        restart_context = {
            "transport_source": "self-output-awareness-controller",
            "transport_turn_id": "web_22222222222222222222222222222222",
            "transport_wall_timestamp": "2026-07-13T12:01:00.000Z",
        }
        gate.observe_system_speech_lifecycle(
            restarted_handoff,
            **restart_context,
        )
        with self.assertRaisesRegex(InputGateError, "does not match"):
            gate.observe_self_output_observation(build_self_output_observation())
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        for state, timestamp in (
            ("cooldown", "2026-07-13T12:01:01.000Z"),
            ("released", "2026-07-13T12:01:02.000Z"),
        ):
            gate.observe_system_speech_lifecycle(
                build_system_speech_lifecycle(
                    state,
                    generation=1,
                    session_id=restarted_session_id,
                    playback_ref=restarted_playback_ref,
                ),
                **{
                    **restart_context,
                    "transport_wall_timestamp": timestamp,
                },
            )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )
        gate.observe_self_output_observation(
            build_self_output_observation(
                generation=1,
                session_id=restarted_session_id,
                playback_ref=restarted_playback_ref,
            )
        )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_NS_AGC,
        )

        with self.assertRaisesRegex(InputGateError, "must start at handoff"):
            gate.observe_system_speech_lifecycle(
                build_system_speech_lifecycle("handoff_accepted"),
                transport_source="self-output-awareness-controller",
                transport_turn_id="web_33333333333333333333333333333333",
                transport_wall_timestamp="2026-07-13T12:02:00.000Z",
            )
        self.assertEqual(
            gate.live_capture_processing_mode_class(),
            LIVE_CAPTURE_MODE_AEC,
        )

    def test_consumed_candidate_audit_requires_exact_private_identity(self) -> None:
        """A same-value candidate copy must not inherit a consumed capability."""
        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        self.assertTrue(
            gate.consume_turn_input_capability(capability, candidate)
        )
        self.assertIsNotNone(gate.build_consumed_candidate_audit(candidate))
        self.assertIsNone(gate.build_consumed_candidate_audit(candidate))
        same_value_copy = build_user_speech_candidate()
        self.assertEqual(same_value_copy, candidate)
        self.assertIsNone(
            gate.build_consumed_candidate_audit(same_value_copy)
        )
        self.assertEqual(gate.private_authority_residue_count(), 0)

    def test_cancel_pending_capability_is_exact_and_grants_no_audit(self) -> None:
        """Cancellation should retire only one exact pending private authority."""
        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        same_value_copy = build_user_speech_candidate()
        self.assertFalse(
            gate.cancel_turn_input_capability(capability, same_value_copy)
        )
        self.assertEqual(gate.private_authority_residue_count(), 1)
        self.assertTrue(
            gate.cancel_turn_input_capability(capability, candidate)
        )
        self.assertFalse(
            gate.cancel_turn_input_capability(capability, candidate)
        )
        self.assertFalse(
            gate.consume_turn_input_capability(capability, candidate)
        )
        self.assertIsNone(gate.build_consumed_candidate_audit(candidate))
        self.assertIsNone(gate.issue_turn_input_capability(candidate))
        self.assertEqual(gate.private_authority_residue_count(), 0)

    def test_gate_disable_before_consume_cancels_without_authority(self) -> None:
        """A gate state change must leave no pending or consumed authority."""
        gate = prepare_current_input_gate()
        candidate = build_user_speech_candidate()
        capability = gate.issue_turn_input_capability(candidate)
        gate.set_input_enabled(False, reason="system_speaking")
        session = MicLoopSession(
            pipeline=mock.Mock(),
            tuning=MicLoopTuning(
                vad_aggressiveness=2,
                final_stable_seconds=8,
            ),
            input_gate=gate,
        )
        chunk = AudioChunk(
            path=None,
            source="microphone",
            pcm16=bytearray(b"\x01\x00" * 160),
            sample_rate=16_000,
            storage_class="in_memory_ephemeral",
            turn_input_authority=False,
            turn_input_authority_class="processed_near_end_observation_only",
        )
        result = session.process_chunk(
            chunk,
            has_speech=True,
            language="ja",
            chunk_duration=1,
            is_last_iteration=True,
            candidate_evidence=candidate,
            turn_input_capability=capability,
        )
        self.assertFalse(result.input_enabled)
        self.assertTrue(
            gate.cancel_turn_input_capability(capability, candidate)
        )
        self.assertIsNone(gate.build_consumed_candidate_audit(candidate))
        self.assertEqual(gate.private_authority_residue_count(), 0)

    def test_pipeline_setup_failure_releases_capability_for_next_window(self) -> None:
        """One setup failure must not block the next valid candidate window."""
        private_marker = "private-recovery-transcript-do-not-echo"
        sink = mock.Mock(
            return_value={
                "result_class": "thought_core_turninput_accepted",
                "submission_count": 1,
                "thought_core_turninput_count": 1,
            }
        )
        app = create_app(private_turn_sink=sink)
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        post_current_lifecycle_events(client, headers)

        pcm_values = [bytearray(b"\x01\x00" * 160) for _ in range(2)]

        def make_window(pcm: bytearray) -> LiveMicrophoneCandidateWindow:
            return LiveMicrophoneCandidateWindow(
                chunk=AudioChunk(
                    path=None,
                    source="microphone",
                    pcm16=pcm,
                    sample_rate=16_000,
                    storage_class="in_memory_ephemeral",
                    turn_input_authority=False,
                    turn_input_authority_class=(
                        "processed_near_end_observation_only"
                    ),
                ),
                packet_count=1,
                window_ms=100,
            )

        pipeline = mock.Mock()
        pipeline._transcribe_private_buffer_result.return_value = (
            TranscriptionResult(
                source="microphone",
                text=private_marker,
                is_final=False,
                chunk_count=1,
            )
        )
        request_payload = {
            "scenario": "independent_current_session_user_speech",
            "window_ms": 100,
            "deadline_ms": 1000,
        }
        with (
            mock.patch(
                "src.web.app.capture_live_microphone_candidate_window",
                side_effect=[make_window(value) for value in pcm_values],
            ),
            mock.patch(
                "src.web.app.has_detectable_speech_pcm16",
                return_value=True,
            ),
            mock.patch(
                "src.web.app.get_cached_transcription_pipeline",
                side_effect=[RuntimeError(private_marker), pipeline],
            ),
        ):
            failed = client.post(
                "/api/live-input-gate/candidate-window",
                headers=headers,
                json=request_payload,
            )
            recovered = client.post(
                "/api/live-input-gate/candidate-window",
                headers=headers,
                json=request_payload,
            )

        self.assertEqual(failed.status_code, 503)
        self.assertEqual(
            failed.get_json()["result_class"],
            "live_candidate_window_failed",
        )
        self.assertEqual(failed.get_json()["private_authority_residue_count"], 0)
        self.assertNotIn(private_marker, failed.get_data(as_text=True))
        self.assertEqual(recovered.status_code, 200)
        self.assertEqual(
            recovered.get_json()["result_class"],
            "independent_user_speech_turninput_accepted",
        )
        self.assertEqual(recovered.get_json()["thought_core_turninput_count"], 1)
        self.assertEqual(
            recovered.get_json()["private_authority_residue_count"],
            0,
        )
        sink.assert_called_once()
        self.assertTrue(
            all(pcm == bytearray(len(pcm)) for pcm in pcm_values)
        )

    def test_live_candidate_endpoint_accepts_one_private_turn_without_echo(self) -> None:
        """One actual accepted candidate should transcribe and invoke the sink once."""
        private_marker = "private-live-transcript-marker-do-not-echo"
        sink_calls: list[tuple[Mapping[str, object], str]] = []

        def sink(
            candidate: Mapping[str, object],
            transcript: str,
        ) -> Mapping[str, object]:
            sink_calls.append((candidate, transcript))
            return {
                "result_class": "thought_core_turninput_accepted",
                "submission_count": 1,
                "thought_core_turninput_count": 1,
            }

        app = create_app(private_turn_sink=sink)
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        post_current_lifecycle_events(client, headers)
        pcm = bytearray(b"\x21\x00" * 160)
        capture_window = LiveMicrophoneCandidateWindow(
            chunk=AudioChunk(
                path=None,
                source="microphone",
                pcm16=pcm,
                sample_rate=16_000,
                storage_class="in_memory_ephemeral",
                turn_input_authority=False,
                turn_input_authority_class="processed_near_end_observation_only",
            ),
            packet_count=1,
            window_ms=100,
        )
        pipeline = mock.Mock()
        pipeline._transcribe_private_buffer_result.return_value = TranscriptionResult(
            source="microphone",
            text=private_marker,
            is_final=False,
            chunk_count=1,
        )
        with (
            mock.patch(
                "src.web.app.capture_live_microphone_candidate_window",
                return_value=capture_window,
            ) as capture_mock,
            mock.patch(
                "src.web.app.has_detectable_speech_pcm16",
                return_value=True,
            ),
            mock.patch(
                "src.web.app.get_cached_transcription_pipeline",
                return_value=pipeline,
            ),
        ):
            response = client.post(
                "/api/live-input-gate/candidate-window",
                headers=headers,
                json={
                    "scenario": "independent_current_session_user_speech",
                    "window_ms": 100,
                    "deadline_ms": 1000,
                },
            )
        self.assertEqual(response.status_code, 200)
        capture_mock.assert_called_once_with(
            window_ms=100,
            deadline_ms=1000,
            processing_mode_class=LIVE_CAPTURE_MODE_NS_AGC,
        )
        payload = response.get_json()
        self.assertEqual(
            payload["result_class"],
            "independent_user_speech_turninput_accepted",
        )
        self.assertEqual(payload["transcription_count"], 1)
        self.assertEqual(payload["submission_count"], 1)
        self.assertEqual(payload["thought_core_turninput_count"], 1)
        self.assertEqual(payload["signal_class"], "signal_above_floor")
        self.assertEqual(payload["vad_decision_class"], "speech_detected")
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "result_class",
                "expectation_class",
                "capture_packet_count",
                "capture_byte_count",
                "signal_class",
                "vad_decision_class",
                "transcription_count",
                "submission_count",
                "thought_core_turninput_count",
                "elapsed_ms",
                "pcm_cleanup_count",
                "private_authority_residue_count",
                "raw_private_publication_flags",
            },
        )
        for forbidden_key in (
            "pcm16",
            "rms_dbfs",
            "peak_dbfs",
            "device_id",
            "transcript",
        ):
            self.assertNotIn(forbidden_key, payload)
        self.assertEqual(payload["pcm_cleanup_count"], 1)
        self.assertEqual(payload["private_authority_residue_count"], 0)
        self.assertEqual(pcm, bytearray(len(pcm)))
        self.assertEqual(len(sink_calls), 1)
        candidate_audit, transcript = sink_calls[0]
        self.assertEqual(transcript, private_marker)
        self.assertEqual(
            candidate_audit["schema_version"],
            "accepted_user_speech_candidate_input_gate.v0",
        )
        self.assertFalse(candidate_audit["raw_private_publication_flags"])
        serialized_response = response.get_data(as_text=True)
        self.assertNotIn(private_marker, serialized_response)
        self.assertNotIn(SYSTEM_SPEECH_SESSION_ID, serialized_response)
        self.assertNotIn(PLAYBACK_EVENT_REF, serialized_response)
        self.assertNotIn("candidate_id", serialized_response)

    def test_live_candidate_scenario_is_expectation_only(self) -> None:
        """Scenario labels must never force acceptance or classification."""
        for scenario, actual_speech in (
            ("independent_current_session_user_speech", False),
            ("self_output_or_ambiguous", True),
        ):
            with self.subTest(scenario=scenario):
                sink = mock.Mock()
                app = create_app(private_turn_sink=sink)
                app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
                client = app.test_client()
                headers = {
                    "X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]
                }
                post_current_lifecycle_events(client, headers)
                pcm = bytearray(
                    (b"\x00\x02" if not actual_speech else b"\x01\x00")
                    * 160
                )
                capture_window = LiveMicrophoneCandidateWindow(
                    chunk=AudioChunk(
                        path=None,
                        source="microphone",
                        pcm16=pcm,
                        sample_rate=16_000,
                        storage_class="in_memory_ephemeral",
                        turn_input_authority=False,
                        turn_input_authority_class=(
                            "processed_near_end_observation_only"
                        ),
                    ),
                    packet_count=1,
                    window_ms=100,
                )
                with (
                    mock.patch(
                        "src.web.app.capture_live_microphone_candidate_window",
                        return_value=capture_window,
                    ),
                    mock.patch(
                        "src.web.app.has_detectable_speech_pcm16",
                        return_value=actual_speech,
                    ),
                ):
                    response = client.post(
                        "/api/live-input-gate/candidate-window",
                        headers=headers,
                        json={
                            "scenario": scenario,
                            "window_ms": 100,
                            "deadline_ms": 1000,
                        },
                    )
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(
                    payload["result_class"],
                    "scenario_expectation_not_met",
                )
                self.assertEqual(payload["transcription_count"], 0)
                self.assertEqual(payload["submission_count"], 0)
                self.assertEqual(payload["thought_core_turninput_count"], 0)
                self.assertEqual(
                    payload["signal_class"],
                    "signal_above_floor" if not actual_speech else "low_signal",
                )
                self.assertEqual(
                    payload["vad_decision_class"],
                    "speech_detected" if actual_speech else "speech_not_detected",
                )
                self.assertEqual(pcm, bytearray(len(pcm)))
                self.assertEqual(payload["private_authority_residue_count"], 0)
                sink.assert_not_called()

    def test_live_candidate_request_rejects_authority_fields_before_capture(self) -> None:
        """Caller classification and authority values should fail before capture."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        marker = "private-request-marker-do-not-echo"
        with mock.patch(
            "src.web.app.capture_live_microphone_candidate_window"
        ) as capture:
            response = client.post(
                "/api/live-input-gate/candidate-window",
                headers=headers,
                json={
                    "scenario": "independent_current_session_user_speech",
                    "window_ms": 100,
                    "deadline_ms": 1000,
                    "candidate": marker,
                    "may_materialize_thought_core_turninput": True,
                    "processing_mode_class": LIVE_CAPTURE_MODE_NS_AGC,
                },
            )
        self.assertEqual(response.status_code, 400)
        capture.assert_not_called()
        serialized = response.get_data(as_text=True)
        self.assertNotIn(marker, serialized)
        self.assertNotIn("may_materialize_thought_core_turninput", serialized)
        self.assertEqual(response.get_json()["submission_count"], 0)

    def test_live_candidate_fixed_capture_failure_has_zero_private_residue(self) -> None:
        """A fixed child failure should be class-only with zero turn counts."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        client = app.test_client()
        headers = {"X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]}
        with mock.patch(
            "src.web.app.capture_live_microphone_candidate_window",
            side_effect=AudioEnvironmentError("voice_capture_dsp_start_failed"),
        ):
            response = client.post(
                "/api/live-input-gate/candidate-window",
                headers=headers,
                json={
                    "scenario": "self_output_or_ambiguous",
                    "window_ms": 100,
                    "deadline_ms": 1000,
                },
            )
        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertEqual(payload["result_class"], "voice_capture_dsp_start_failed")
        self.assertEqual(payload["transcription_count"], 0)
        self.assertEqual(payload["submission_count"], 0)
        self.assertEqual(payload["thought_core_turninput_count"], 0)
        self.assertEqual(payload["signal_class"], "not_evaluated")
        self.assertEqual(payload["vad_decision_class"], "not_evaluated")
        self.assertEqual(payload["pcm_cleanup_count"], 0)
        self.assertEqual(payload["private_authority_residue_count"], 0)

    def test_live_candidate_sink_absent_or_failed_submits_zero(self) -> None:
        """A missing or failed private sink must not claim a Thought Core turn."""
        private_marker = "private-sink-failure-marker-do-not-echo"

        def fail_sink(*_: object) -> object:
            raise RuntimeError(private_marker)

        for sink, expected_class in (
            (None, "private_turn_sink_unavailable"),
            (fail_sink, "private_turn_sink_failed"),
        ):
            with self.subTest(expected_class=expected_class):
                app = create_app(private_turn_sink=sink)
                app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
                client = app.test_client()
                headers = {
                    "X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]
                }
                post_current_lifecycle_events(client, headers)
                pcm = bytearray(b"\x01\x00" * 160)
                capture_window = LiveMicrophoneCandidateWindow(
                    chunk=AudioChunk(
                        path=None,
                        source="microphone",
                        pcm16=pcm,
                        sample_rate=16_000,
                        storage_class="in_memory_ephemeral",
                        turn_input_authority=False,
                        turn_input_authority_class=(
                            "processed_near_end_observation_only"
                        ),
                    ),
                    packet_count=1,
                    window_ms=100,
                )
                pipeline = mock.Mock()
                pipeline._transcribe_private_buffer_result.return_value = (
                    TranscriptionResult(
                        source="microphone",
                        text=private_marker,
                        is_final=False,
                        chunk_count=1,
                    )
                )
                with (
                    mock.patch(
                        "src.web.app.capture_live_microphone_candidate_window",
                        return_value=capture_window,
                    ),
                    mock.patch(
                        "src.web.app.has_detectable_speech_pcm16",
                        return_value=True,
                    ),
                    mock.patch(
                        "src.web.app.get_cached_transcription_pipeline",
                        return_value=pipeline,
                    ),
                ):
                    response = client.post(
                        "/api/live-input-gate/candidate-window",
                        headers=headers,
                        json={
                            "scenario": (
                                "independent_current_session_user_speech"
                            ),
                            "window_ms": 100,
                            "deadline_ms": 1000,
                        },
                    )
                self.assertEqual(response.status_code, 503)
                payload = response.get_json()
                self.assertEqual(payload["result_class"], expected_class)
                self.assertEqual(payload["transcription_count"], 1)
                self.assertEqual(payload["submission_count"], 0)
                self.assertEqual(payload["thought_core_turninput_count"], 0)
                self.assertEqual(payload["pcm_cleanup_count"], 1)
                self.assertEqual(
                    payload["private_authority_residue_count"],
                    0,
                )
                self.assertEqual(pcm, bytearray(len(pcm)))
                self.assertNotIn(private_marker, response.get_data(as_text=True))

    def test_live_candidate_endpoint_token_and_single_window_guards(self) -> None:
        """The live endpoint should require a token and reject overlap."""
        app = create_app()
        app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = False
        request_payload = {
            "scenario": "self_output_or_ambiguous",
            "window_ms": 100,
            "deadline_ms": 1000,
        }
        with mock.patch(
            "src.web.app.capture_live_microphone_candidate_window"
        ) as capture:
            unauthenticated = app.test_client().post(
                "/api/live-input-gate/candidate-window",
                json=request_payload,
            )
        self.assertEqual(unauthenticated.status_code, 403)
        capture.assert_not_called()

        entered = threading.Event()
        release = threading.Event()
        first_result: list[object] = []
        capture_calls = 0

        def blocking_capture(**_: object) -> LiveMicrophoneCandidateWindow:
            nonlocal capture_calls
            capture_calls += 1
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return LiveMicrophoneCandidateWindow(
                chunk=AudioChunk(
                    path=None,
                    source="microphone",
                    pcm16=bytearray(320),
                    sample_rate=16_000,
                    storage_class="in_memory_ephemeral",
                    turn_input_authority=False,
                    turn_input_authority_class=(
                        "processed_near_end_observation_only"
                    ),
                ),
                packet_count=1,
                window_ms=100,
            )

        def run_first() -> None:
            client = app.test_client()
            first_result.append(
                client.post(
                    "/api/live-input-gate/candidate-window",
                    headers={
                        "X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]
                    },
                    json=request_payload,
                )
            )

        with (
            mock.patch(
                "src.web.app.capture_live_microphone_candidate_window",
                side_effect=blocking_capture,
            ),
            mock.patch(
                "src.web.app.has_detectable_speech_pcm16",
                return_value=False,
            ),
        ):
            thread = threading.Thread(target=run_first)
            thread.start()
            self.assertTrue(entered.wait(timeout=2))
            overlap = app.test_client().post(
                "/api/live-input-gate/candidate-window",
                headers={
                    "X-AI-Core-Token": app.config["LOCAL_API_TOKEN"]
                },
                json=request_payload,
            )
            release.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(capture_calls, 1)
        self.assertEqual(overlap.status_code, 409)
        overlap_payload = overlap.get_json()
        self.assertEqual(
            overlap_payload["result_class"],
            "live_candidate_window_busy",
        )
        self.assertEqual(overlap_payload["submission_count"], 0)
        self.assertEqual(
            overlap_payload["private_authority_residue_count"],
            0,
        )
        self.assertEqual(len(first_result), 1)
        self.assertEqual(first_result[0].status_code, 200)


if __name__ == "__main__":
    unittest.main()
    has_detectable_speech_pcm16,
