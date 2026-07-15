"""Local web UI for audio transcription."""

from __future__ import annotations

import _thread
import argparse
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import hmac
from ipaddress import ip_address
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from flask import (
    Flask,
    Response,
    current_app,
    has_app_context,
    jsonify,
    render_template,
    request,
)
from werkzeug.exceptions import RequestEntityTooLarge

from src.core.events import (
    MAX_EVENT_LOG_BYTES,
    emit_event,
    get_event_bus,
    get_event_log_path,
    new_turn_id,
    read_event_log_events,
)
from src.core.input_gate import (
    InputGate,
    InputGateError,
    InputGateState,
    UserSpeechCandidateEvidence,
)
from src.core.pipeline import TranscriptionResult, get_cached_transcription_pipeline
from src.core.session import MicLoopSession, MicLoopTuning
from src.core.status_report import build_doctor_status
from src.core.handoff_bridge import build_handoff_metadata, load_handoff_bundle
from src.io.aec_reference import LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES
from src.io.audio import AudioEnvironmentError, AudioInputError, AudioTranscriptionError
from src.io.microphone import (
    LiveMicrophoneCandidateWindow,
    capture_live_microphone_candidate_window,
    classify_live_pcm16_signal,
    find_last_vad_speech_frame_offset_ms,
    has_detectable_speech_pcm16,
)
from src.web.transcription_service import (
    WEB_MAX_UPLOAD_BYTES,
    WebTranscriptionRequest,
    process_web_transcription,
)


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" rx="8" fill="#0f766e"/>
<path d="M9 20.5h14v3H9zM11 8.5h10a4 4 0 0 1 0 8H11z" fill="#f8fafc"/>
<circle cx="21" cy="12.5" r="1.6" fill="#0f766e"/>
</svg>"""
LOCAL_API_TOKEN_HEADER = "X-AI-Core-Token"
LOCAL_API_TOKEN_CONFIG = "LOCAL_API_TOKEN"
LOCAL_API_TOKEN_ENV = "AI_TALK_CORE_WEB_TOKEN"
ENABLE_PROCESS_SHUTDOWN_CONFIG = "ENABLE_PROCESS_SHUTDOWN"
WEB_PRESET_CONFIG = "WEB_PRESET"
WEB_PRESET_ENV = "AI_TALK_CORE_WEB_PRESET"
WEB_RUNTIME_STATE_CONFIG = "WEB_RUNTIME_STATE"
WEB_BIND_HOST_CONFIG = "WEB_BIND_HOST"
WEB_BIND_PORT_CONFIG = "WEB_BIND_PORT"
WEB_STARTED_AT_CONFIG = "WEB_STARTED_AT"
RUNTIME_STATUS_WRITER_CONFIG = "RUNTIME_STATUS_WRITER"
PRIVATE_TURN_SINK_CONFIG = "PRIVATE_TURN_SINK"
WEB_MODULE_NAME = "ai_talk_core.web"
SWORD_AGENT_TOKEN_HEADER = "X-Sword-Agent-Token"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
TOKEN_PROTECTED_ENDPOINTS = {
    "api_doctor",
    "api_event_ingest",
    "api_events",
    "api_health",
    "api_input_gate_body_state_get",
    "api_input_gate_get",
    "api_input_gate_post",
    "api_live_input_gate_candidate_window",
    "api_recording_chunk",
    "api_status",
    "api_agent_handoff_latest",
    "api_codex_handoff_latest",
    "api_shutdown",
}
WEB_RECORDING_CHUNK_TIMESLICE_MS = 500
WEB_MAX_RECORDING_CHUNK_BYTES = 1 * 1024 * 1024
WEB_MAX_RECORDING_CHUNKS = 720
WEB_MAX_RECORDING_TURN_BYTES = WEB_MAX_UPLOAD_BYTES
WEB_MAX_RECORDING_CHUNK_CACHE_BYTES = 100 * 1024 * 1024
WEB_MAX_RECORDING_CHUNK_CACHE_TURNS = 20
WEB_RECORDING_CHUNK_RETENTION_SECONDS = 24 * 60 * 60
WEB_EVENT_TRACE_DEFAULT_LIMIT = 100
CLIENT_EVENT_PAYLOAD_KEYS = {
    "blob_size_bytes",
    "chunk_count",
    "chunk_sequence",
    "mime_type",
    "timeslice_ms",
    "trigger",
}
CLIENT_EVENT_TIMING_KEYS = {
    "client_timestamp_wall",
    "client_timestamp_monotonic",
    "client_performance_now",
}
CLIENT_EVENT_NAMES = {
    "record_start",
    "record_stop",
    "swordAgentSystemSpeechLifecycleV0",
    "audioSelfOutputObservationV0",
}
CLIENT_EVENT_TOP_LEVEL_KEYS = {
    "event",
    "turn_id",
    "source",
    "payload",
    *CLIENT_EVENT_TIMING_KEYS,
}
CLIENT_EVENT_SOURCE_BY_NAME = {
    "record_start": "web-ui",
    "record_stop": "web-ui",
    "swordAgentSystemSpeechLifecycleV0": "self-output-awareness-controller",
    "audioSelfOutputObservationV0": "self-output-awareness-controller",
}
RECORD_START_PAYLOAD_KEYS = {"trigger", "timeslice_ms", "mime_type"}
RECORD_STOP_PAYLOAD_KEYS = {"chunk_count", "chunk_sequence", "blob_size_bytes"}
CLIENT_RECORDING_MIME_TYPES = {"", "audio/webm", "audio/webm;codecs=opus"}
CLIENT_TURN_ID_PATTERN = re.compile(r"^web_[a-f0-9]{32}$")
CLIENT_WALL_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
LIVE_CANDIDATE_SCENARIOS = {
    "self_output_or_ambiguous",
    "independent_current_session_user_speech",
}
LIVE_CANDIDATE_REQUEST_FIELDS = {"scenario", "window_ms", "deadline_ms"}
LIVE_CANDIDATE_SINK_RESULT_FIELDS = {
    "result_class",
    "submission_count",
    "thought_core_turninput_count",
    "presentation_class",
    "assistant_event_id",
    "thought_core_first_event_elapsed_ms",
    "raw_private_publication_flags",
}
LIVE_CANDIDATE_PRESENTATION_CLASSES = {
    "aituber_presentation_forwarded",
    "aituber_presentation_forwarded_timing_unavailable",
    "aituber_presentation_not_forwarded",
}
LIVE_CANDIDATE_ASSISTANT_EVENT_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)
LIVE_CANDIDATE_PRIVATE_EVENT_ID_TOKEN_PATTERN = re.compile(
    r"(?:^|[._:-])(?:private|raw|secret|token|transcript|prompt|path|url)(?:$|[._:-])",
    flags=re.IGNORECASE,
)


class WebRuntimeState:
    """Track in-flight Web transcription work and shutdown requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_transcriptions = 0
        self._shutdown_requested = False
        self._shutdown_reason = ""
        self._shutdown_scheduled = False

    def begin_transcription(self) -> bool:
        """Reserve one transcription slot unless shutdown has started."""
        with self._lock:
            if self._shutdown_requested:
                return False
            self._active_transcriptions += 1
            return True

    def end_transcription(self) -> bool:
        """Release one transcription slot and report whether shutdown can run."""
        with self._lock:
            if self._active_transcriptions > 0:
                self._active_transcriptions -= 1
            return (
                self._shutdown_requested
                and self._active_transcriptions == 0
                and not self._shutdown_scheduled
            )

    def request_shutdown(
        self,
        reason: str = "",
        *,
        force: bool = False,
    ) -> tuple[dict[str, object], bool]:
        """Request shutdown and return a state snapshot plus scheduling decision."""
        with self._lock:
            self._shutdown_requested = True
            self._shutdown_reason = (reason or "api_request").strip() or "api_request"
            should_schedule = (
                force or self._active_transcriptions == 0
            ) and not self._shutdown_scheduled
            return self._snapshot_unlocked(), should_schedule

    def mark_shutdown_scheduled(self) -> dict[str, object]:
        """Mark that the process shutdown timer has been scheduled."""
        with self._lock:
            self._shutdown_scheduled = True
            return self._snapshot_unlocked()

    def snapshot(self) -> dict[str, object]:
        """Return a thread-safe runtime state snapshot."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, object]:
        return {
            "active_transcriptions": self._active_transcriptions,
            "shutdown_requested": self._shutdown_requested,
            "shutdown_reason": self._shutdown_reason,
            "shutdown_scheduled": self._shutdown_scheduled,
        }


class RuntimeStatusWriter:
    """Write integration-friendly process status JSON for the Web server."""

    def __init__(
        self,
        path: Path | None,
        *,
        host: str,
        port: int,
        started_at: str,
        command_line: str | None = None,
    ) -> None:
        self.path = path
        self.host = host
        self.port = port
        self.started_at = started_at
        self.command_line = command_line or subprocess.list2cmdline(sys.argv)

    def write(self, state: str, **extra: object) -> None:
        """Write the current runtime status when a status path is configured."""
        if self.path is None:
            return
        payload = build_runtime_status_payload(
            state=state,
            host=self.host,
            port=self.port,
            started_at=self.started_at,
            command_line=self.command_line,
            **extra,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def create_app(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    runtime_status_writer: RuntimeStatusWriter | None = None,
    started_at: str | None = None,
    private_turn_sink: (
        Callable[[Mapping[str, object], str], Mapping[str, object]] | None
    ) = None,
) -> Flask:
    """Create the local Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = WEB_MAX_UPLOAD_BYTES
    app.config[LOCAL_API_TOKEN_CONFIG] = (
        os.environ.get(LOCAL_API_TOKEN_ENV, "").strip() or secrets.token_urlsafe(32)
    )
    app.config[ENABLE_PROCESS_SHUTDOWN_CONFIG] = True
    app.config[WEB_PRESET_CONFIG] = os.environ.get(WEB_PRESET_ENV, "").strip()
    app.config[WEB_RUNTIME_STATE_CONFIG] = WebRuntimeState()
    app.config[WEB_BIND_HOST_CONFIG] = host
    app.config[WEB_BIND_PORT_CONFIG] = port
    app.config[WEB_STARTED_AT_CONFIG] = (
        started_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    app.config[RUNTIME_STATUS_WRITER_CONFIG] = runtime_status_writer
    app.config[PRIVATE_TURN_SINK_CONFIG] = private_turn_sink
    input_gate = InputGate()
    live_candidate_window_lock = threading.Lock()

    @app.before_request
    def enforce_local_request_policy() -> tuple[object, int] | None:
        if not is_allowed_local_remote(request.remote_addr):
            return build_policy_error_response(
                "このローカル UI では許可されていない接続元です。",
                403,
            )
        if not is_allowed_local_host(request.host):
            return build_policy_error_response(
                "このローカル UI では許可されていない Host です。",
                403,
            )
        if request.method in UNSAFE_METHODS and not has_trusted_origin():
            return build_policy_error_response(
                "このローカル UI では許可されていない送信元です。",
                403,
            )
        if (
            request.endpoint in TOKEN_PROTECTED_ENDPOINTS
            and not has_valid_local_api_token()
        ):
            return build_policy_error_response("local API token が不正です。", 403)
        return None

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_: RequestEntityTooLarge) -> tuple[object, int]:
        return build_error_response(
            "音声ファイルが大きすぎます。25MB 以下のファイルを指定してください。",
            413,
        )

    @app.get("/")
    def index() -> str:
        return render_page()

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(FAVICON_SVG, mimetype="image/svg+xml")

    @app.get("/health")
    def health() -> tuple[object, int]:
        return jsonify(build_process_health_response()), 200

    @app.post("/shutdown")
    def shutdown() -> tuple[object, int]:
        if shutdown_requires_token() and not has_valid_local_api_token():
            return jsonify({"ok": False, "error": "shutdown token is required"}), 403
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            payload = {}
        return build_shutdown_response(
            reason=str(payload.get("reason") or request.args.get("reason") or "shutdown_endpoint"),
            force=parse_boolish(payload.get("force", request.args.get("force"))) is True,
        )

    @app.post("/transcribe-upload")
    def transcribe_upload() -> str | Response:
        file_storage = request.files.get("audio_file")
        if file_storage is None or file_storage.filename == "":
            return render_page(error="音声ファイルを選択してください。")
        return handle_transcription(file_storage.read(), file_storage.filename)

    @app.post("/api/transcribe-upload")
    def api_transcribe_upload() -> tuple[object, int]:
        file_storage = request.files.get("audio_file")
        if file_storage is None or file_storage.filename == "":
            return jsonify(
                {
                    "message": "",
                    "transcript": "",
                    "error": "音声ファイルを選択してください。",
                }
            ), 400
        return handle_transcription_api(file_storage.read(), file_storage.filename)

    @app.post("/transcribe-browser-recording")
    def transcribe_browser_recording() -> str | Response:
        file_storage = request.files.get("audio_blob")
        if file_storage is None or file_storage.filename == "":
            return render_page(error="録音データを受け取れませんでした。")
        return handle_transcription(
            file_storage.read(),
            file_storage.filename,
            message="ブラウザ録音を処理しました。",
        )

    @app.post("/api/transcribe-browser-recording")
    def api_transcribe_browser_recording() -> tuple[object, int]:
        file_storage = request.files.get("audio_blob")
        if file_storage is None or file_storage.filename == "":
            return jsonify(
                {
                    "message": "",
                    "transcript": "",
                    "error": "録音データを受け取れませんでした。",
                }
            ), 400
        return handle_transcription_api(
            file_storage.read(),
            file_storage.filename,
            message="ブラウザ録音を処理しました。",
        )

    @app.post("/api/recording-chunk")
    def api_recording_chunk() -> tuple[object, int]:
        file_storage = request.files.get("audio_chunk")
        turn_id = normalize_turn_id(
            request.form.get("turn_id", "").strip()
        ) or new_turn_id("web")
        sequence = parse_nonnegative_int(request.form.get("sequence"))
        if file_storage is None or file_storage.filename == "":
            return jsonify({"ok": False, "error": "audio_chunk is required"}), 400
        if sequence is None:
            return jsonify({"ok": False, "error": "sequence must be a non-negative integer"}), 400
        if sequence > WEB_MAX_RECORDING_CHUNKS:
            return jsonify({"ok": False, "error": "sequence exceeds recording chunk limit"}), 400
        raw_bytes = file_storage.read()
        if not raw_bytes:
            return jsonify({"ok": False, "error": "audio_chunk is empty"}), 400
        if len(raw_bytes) > WEB_MAX_RECORDING_CHUNK_BYTES:
            return jsonify({"ok": False, "error": "audio_chunk is too large"}), 413
        chunk_dir = get_recording_chunk_dir(turn_id)
        chunk_path = get_recording_chunk_path(chunk_dir, sequence)
        if (
            get_recording_chunk_total_bytes(chunk_dir, exclude_path=chunk_path)
            + len(raw_bytes)
            > WEB_MAX_RECORDING_TURN_BYTES
        ):
            return jsonify({"ok": False, "error": "recording turn is too large"}), 413
        chunk_path.write_bytes(raw_bytes)
        is_final = parse_boolish(request.form.get("is_final")) is True
        event = emit_event(
            "record_chunk",
            turn_id=turn_id,
            source="web",
            payload={
                "sequence": sequence,
                "size_bytes": len(raw_bytes),
                "is_final": is_final,
                "filename": chunk_path.name,
                "mime_type": file_storage.mimetype,
            },
        )
        return jsonify(
            {
                "ok": True,
                "turn_id": turn_id,
                "sequence": sequence,
                "size_bytes": len(raw_bytes),
                "event": event.to_payload(),
            }
        ), 202

    @app.post("/api/events/ingest")
    def api_event_ingest() -> tuple[object, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
        try:
            event_name = validate_client_event_name(payload.get("event"))
            validate_client_event_top_level(payload)
            turn_id = validate_client_turn_id(payload.get("turn_id"))
            source = validate_client_event_source(payload.get("source"), event_name)
            event_payload = validate_client_event_payload(
                event_name,
                payload.get("payload"),
            )
            timing_payload = validate_client_event_timing(payload)
            if event_name == "swordAgentSystemSpeechLifecycleV0":
                if (
                    "source" not in payload
                    or "turn_id" not in payload
                    or not timing_payload
                ):
                    raise InputGateError(
                        "system speech transport context is incomplete"
                    )
                input_gate.observe_system_speech_lifecycle(
                    event_payload,
                    transport_source=source,
                    transport_turn_id=turn_id,
                    transport_wall_timestamp=str(
                        timing_payload["client_timestamp_wall"]
                    ),
                )
            elif event_name == "audioSelfOutputObservationV0":
                input_gate.observe_self_output_observation(event_payload)
        except InputGateError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if event_name in {
            "swordAgentSystemSpeechLifecycleV0",
            "audioSelfOutputObservationV0",
        }:
            event_payload = {}
        event_payload.update(timing_payload)
        event = emit_event(
            event_name,
            turn_id=turn_id,
            source=source,
            payload=event_payload,
        )
        return jsonify({"ok": True, "event": event.to_payload()}), 202

    @app.get("/api/events")
    def api_events() -> Response | tuple[object, int]:
        if parse_boolish(request.args.get("once")) is True:
            turn_id = normalize_turn_id(request.args.get("turn_id"))
            events = read_event_log_events(
                limit=resolve_event_trace_limit(request.args.get("limit")),
                turn_id=turn_id or None,
            )
            return jsonify(
                {
                    "ok": True,
                    "count": len(events),
                    "events": events,
                    "projection": "events.jsonl",
                }
            ), 200

        def stream_events() -> object:
            yield ": connected\n\n"
            with get_event_bus().subscribe() as subscriber:
                while True:
                    try:
                        event = subscriber.get(timeout=15)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield format_sse_event(event.to_payload(), event.event)

        return Response(
            stream_events(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def build_handoff_response() -> tuple[object, int]:
        """Return the latest saved handoff bundle as JSON."""
        source = request.args.get("source", "web").strip() or "web"
        try:
            handoff = load_handoff_bundle(source=source)
        except ValueError:
            return jsonify({"error": "invalid handoff source"}), 400
        if handoff is None:
            return jsonify({"error": f"handoff not found for source: {source}"}), 404
        return jsonify(
            {
                "transcript": handoff.transcript,
                "command": handoff.command,
                "prompt_text": handoff.prompt_text,
                "command_path": str(handoff.json_path),
                "command_text_path": str(handoff.text_path),
                "source": source,
                "handoff_id": handoff.metadata.get("handoff_id", ""),
                "updated_at": handoff.metadata.get("updated_at", ""),
                "metadata": handoff.metadata,
            }
        ), 200

    @app.get("/api/codex-handoff-latest")
    def api_codex_handoff_latest() -> tuple[object, int]:
        return build_handoff_response()

    @app.get("/api/agent-handoff-latest")
    def api_agent_handoff_latest() -> tuple[object, int]:
        return build_handoff_response()

    @app.get("/api/doctor")
    def api_doctor() -> tuple[object, int]:
        return jsonify(build_doctor_status()), 200

    @app.get("/api/input-gate")
    def api_input_gate_get() -> tuple[object, int]:
        return jsonify(build_input_gate_response(input_gate.state)), 200

    @app.post("/api/input-gate")
    def api_input_gate_post() -> tuple[object, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
        try:
            state = input_gate.update_from_payload(payload)
        except InputGateError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(build_input_gate_response(state)), 200

    @app.get("/api/input-gate/body-state")
    def api_input_gate_body_state_get() -> tuple[object, int]:
        return jsonify(input_gate.body_state_projection()), 200

    @app.post("/api/live-input-gate/candidate-window")
    def api_live_input_gate_candidate_window() -> tuple[object, int]:
        payload = request.get_json(silent=True)
        try:
            scenario, window_ms, deadline_ms = validate_live_candidate_request(
                payload
            )
        except InputGateError:
            return jsonify(
                build_live_candidate_window_response(
                    result_class="live_candidate_request_invalid",
                    expectation_class="not_evaluated",
                )
            ), 400
        finally:
            if isinstance(payload, dict):
                payload.clear()
        if not live_candidate_window_lock.acquire(blocking=False):
            return jsonify(
                build_live_candidate_window_response(
                    result_class="live_candidate_window_busy",
                    expectation_class="not_evaluated",
                )
            ), 409
        try:
            response = execute_live_candidate_window(
                input_gate=input_gate,
                scenario=scenario,
                window_ms=window_ms,
                deadline_ms=deadline_ms,
                private_turn_sink=app.config.get(PRIVATE_TURN_SINK_CONFIG),
            )
        finally:
            live_candidate_window_lock.release()
        status = 200 if response["result_class"] in {
            "self_output_or_ambiguous_confirmed",
            "independent_user_speech_turninput_accepted",
            "scenario_expectation_not_met",
        } else 503
        return jsonify(response), status

    @app.get("/api/health")
    def api_health() -> tuple[object, int]:
        return jsonify(
            build_health_response(
                input_gate_state=input_gate.state,
                runtime_state=get_web_runtime_state(),
                source=request.args.get("source", "web"),
            )
        ), 200

    @app.get("/api/status")
    def api_status() -> tuple[object, int]:
        return api_health()

    @app.post("/api/shutdown")
    def api_shutdown() -> tuple[object, int]:
        runtime_state = get_web_runtime_state()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            payload = {}
        reason = str(
            payload.get("reason") or request.args.get("reason") or "api_request"
        )
        force_value = (
            payload["force"] if "force" in payload else request.args.get("force")
        )
        return build_shutdown_response(
            reason=reason,
            force=parse_boolish(force_value) is True,
        )

    return app


def validate_live_candidate_request(
    payload: object,
) -> tuple[str, int, int]:
    """Validate the bounded expectation-only request surface."""
    if not isinstance(payload, dict) or set(payload) != LIVE_CANDIDATE_REQUEST_FIELDS:
        raise InputGateError("live candidate request fields are invalid")
    scenario = payload.get("scenario")
    window_ms = payload.get("window_ms")
    deadline_ms = payload.get("deadline_ms")
    if scenario not in LIVE_CANDIDATE_SCENARIOS:
        raise InputGateError("live candidate scenario is invalid")
    if (
        isinstance(window_ms, bool)
        or not isinstance(window_ms, int)
        or not 100 <= window_ms <= 3000
    ):
        raise InputGateError("live candidate window is invalid")
    if (
        isinstance(deadline_ms, bool)
        or not isinstance(deadline_ms, int)
        or not window_ms + 200 <= deadline_ms <= 10_000
    ):
        raise InputGateError("live candidate deadline is invalid")
    return str(scenario), window_ms, deadline_ms


def build_live_candidate_window_response(
    *,
    result_class: str,
    expectation_class: str,
    capture_packet_count: int = 0,
    capture_byte_count: int = 0,
    signal_class: str = "not_evaluated",
    vad_decision_class: str = "not_evaluated",
    transcription_count: int = 0,
    submission_count: int = 0,
    thought_core_turninput_count: int = 0,
    elapsed_ms: int = 0,
    last_vad_speech_frame_offset_ms: int | None = None,
    utterance_end_to_candidate_result_ms: int | None = None,
    utterance_end_timing_class: str = "not_evaluated",
    stt_start_offset_ms: int | None = None,
    stt_end_offset_ms: int | None = None,
    canonical_accept_offset_ms: int | None = None,
    canonical_accept_latency_class: str = "not_evaluated",
    inflight_sink_cancellation_class: str = "not_applicable",
    presentation_class: str = "presentation_not_attempted",
    assistant_event_id: str | None = None,
    thought_core_first_event_elapsed_ms: int | None = None,
    pcm_cleanup_count: int = 0,
    private_authority_residue_count: int = 0,
) -> dict[str, object]:
    """Return only bounded class/count/timing/cleanup evidence."""
    return {
        "schema_version": "ai_talk_core.live_input_gate_candidate_window.v0",
        "result_class": result_class,
        "expectation_class": expectation_class,
        "capture_packet_count": capture_packet_count,
        "capture_byte_count": capture_byte_count,
        "signal_class": signal_class,
        "vad_decision_class": vad_decision_class,
        "transcription_count": transcription_count,
        "submission_count": submission_count,
        "thought_core_turninput_count": thought_core_turninput_count,
        "elapsed_ms": elapsed_ms,
        "last_vad_speech_frame_offset_ms": last_vad_speech_frame_offset_ms,
        "utterance_end_to_candidate_result_ms": (
            utterance_end_to_candidate_result_ms
        ),
        "utterance_end_timing_class": utterance_end_timing_class,
        "stt_start_offset_ms": stt_start_offset_ms,
        "stt_end_offset_ms": stt_end_offset_ms,
        "canonical_accept_offset_ms": canonical_accept_offset_ms,
        "canonical_accept_latency_class": canonical_accept_latency_class,
        "inflight_sink_cancellation_class": inflight_sink_cancellation_class,
        "presentation_class": presentation_class,
        "assistant_event_id": assistant_event_id,
        "thought_core_first_event_elapsed_ms": (
            thought_core_first_event_elapsed_ms
        ),
        "pcm_cleanup_count": pcm_cleanup_count,
        "private_authority_residue_count": private_authority_residue_count,
        "raw_private_publication_flags": False,
    }


def classify_canonical_accept_latency(offset_ms: int | None) -> str:
    """Classify one accepted-turn offset without turning timing into authority."""
    if offset_ms is None:
        return "not_evaluated"
    if offset_ms <= 2000:
        return "within_2s_target"
    if offset_ms <= 10_000:
        return "over_2s_within_10s"
    return "over_10s"


def _offset_from_last_vad_frame_ms(
    *,
    capture_started_monotonic: float,
    last_vad_speech_frame_offset_ms: int | None,
    observed_monotonic: float,
) -> int | None:
    if last_vad_speech_frame_offset_ms is None:
        return None
    observed_from_capture_ms = max(
        last_vad_speech_frame_offset_ms,
        int((observed_monotonic - capture_started_monotonic) * 1000),
    )
    return observed_from_capture_ms - last_vad_speech_frame_offset_ms


def execute_live_candidate_window(
    *,
    input_gate: InputGate,
    scenario: str,
    window_ms: int,
    deadline_ms: int,
    private_turn_sink: object,
) -> dict[str, object]:
    """Own one private capture, gate decision, transcription and sink handoff."""
    started = time.monotonic()
    route_deadline_monotonic = started + (deadline_ms / 1000)
    capture_started_monotonic = started
    capture_window: LiveMicrophoneCandidateWindow | None = None
    transcript = ""
    packet_count = 0
    byte_count = 0
    signal_class = "not_evaluated"
    vad_decision_class = "not_evaluated"
    transcription_count = 0
    submission_count = 0
    turninput_count = 0
    presentation_class = "presentation_not_attempted"
    assistant_event_id: str | None = None
    thought_core_first_event_elapsed_ms: int | None = None
    last_vad_speech_frame_offset_ms: int | None = None
    utterance_end_to_candidate_result_ms: int | None = None
    utterance_end_timing_class = "not_evaluated"
    stt_start_offset_ms: int | None = None
    stt_end_offset_ms: int | None = None
    canonical_accept_offset_ms: int | None = None
    canonical_accept_latency_class = "not_evaluated"
    inflight_sink_cancellation_class = "not_applicable"
    result_class = "live_candidate_window_failed"
    expectation_class = "not_evaluated"
    session: MicLoopSession | None = None
    transcription: TranscriptionResult | None = None
    candidate_audit: dict[str, object] | None = None
    sink_result: object | None = None
    candidate: UserSpeechCandidateEvidence | None = None
    capability: object | None = None
    source_epoch: object | None = None
    try:
        source_epoch = input_gate.begin_live_input_source_epoch(
            capture_started_monotonic=capture_started_monotonic,
            deadline_ms=deadline_ms,
        )
        if source_epoch is None:
            raise InputGateError("input_source_epoch_unavailable")
        processing_mode_class = input_gate.live_input_source_epoch_processing_mode(
            source_epoch
        )
        if processing_mode_class is None:
            raise InputGateError("input_source_epoch_unavailable")
        capture_window = capture_live_microphone_candidate_window(
            window_ms=window_ms,
            deadline_ms=deadline_ms,
            processing_mode_class=processing_mode_class,
        )
        packet_count = capture_window.packet_count
        byte_count = capture_window.processed_byte_count
        pcm16 = capture_window.chunk.pcm16
        if pcm16 is None:
            raise AudioInputError("live candidate PCM is unavailable")
        signal_class = classify_live_pcm16_signal(pcm16)
        has_speech = has_detectable_speech_pcm16(pcm16)
        if has_speech:
            last_vad_speech_frame_offset_ms = (
                find_last_vad_speech_frame_offset_ms(pcm16)
            )
            utterance_end_timing_class = (
                "vad_speech_frame_available"
                if last_vad_speech_frame_offset_ms is not None
                else "vad_speech_frame_inconsistent"
            )
        else:
            utterance_end_timing_class = "no_vad_speech_frame"
        vad_decision_class = (
            "speech_detected" if has_speech else "speech_not_detected"
        )
        candidate = input_gate.evaluate_live_processed_candidate(
            has_speech=has_speech,
            window_ms=window_ms,
            packet_count=packet_count,
            processed_byte_count=byte_count,
            frame_bytes=capture_window.frame_bytes,
            source_epoch=source_epoch,
            speech_end_monotonic=(
                capture_started_monotonic
                + (last_vad_speech_frame_offset_ms / 1000)
                if last_vad_speech_frame_offset_ms is not None
                else None
            ),
        )
        expected_acceptance = (
            scenario == "independent_current_session_user_speech"
        )
        actual_acceptance = candidate is not None
        if has_speech and last_vad_speech_frame_offset_ms is None:
            result_class = "speech_timing_observation_missing"
            expectation_class = "not_evaluated"
        elif expected_acceptance != actual_acceptance:
            result_class = "scenario_expectation_not_met"
            expectation_class = "mismatch"
        elif not expected_acceptance:
            result_class = "self_output_or_ambiguous_confirmed"
            expectation_class = "matched"
        else:
            expectation_class = "matched"
            capability = input_gate.issue_turn_input_capability(candidate)
            if capability is None:
                result_class = "input_gate_capability_unavailable"
            else:
                stt_start_offset_ms = _offset_from_last_vad_frame_ms(
                    capture_started_monotonic=capture_started_monotonic,
                    last_vad_speech_frame_offset_ms=(
                        last_vad_speech_frame_offset_ms
                    ),
                    observed_monotonic=time.monotonic(),
                )
                pipeline = get_cached_transcription_pipeline()
                if time.monotonic() > route_deadline_monotonic:
                    raise InputGateError("live candidate deadline exceeded")
                session = MicLoopSession(
                    pipeline=pipeline,
                    tuning=MicLoopTuning(
                        vad_aggressiveness=2,
                        final_stable_seconds=8,
                    ),
                    input_gate=input_gate,
                )
                transcription_count = 1
                transcription = session.process_chunk(
                    capture_window.chunk,
                    has_speech=True,
                    language="ja",
                    chunk_duration=max(1, math.ceil(window_ms / 1000)),
                    is_last_iteration=True,
                    candidate_evidence=candidate,
                    turn_input_capability=capability,
                )
                stt_completed_monotonic = time.monotonic()
                stt_end_offset_ms = _offset_from_last_vad_frame_ms(
                    capture_started_monotonic=capture_started_monotonic,
                    last_vad_speech_frame_offset_ms=(
                        last_vad_speech_frame_offset_ms
                    ),
                    observed_monotonic=stt_completed_monotonic,
                )
                route_deadline_expired = (
                    stt_completed_monotonic > route_deadline_monotonic
                )
                transcript = transcription.text.strip()
                candidate_audit = (
                    input_gate.build_consumed_candidate_audit(candidate)
                    if transcript
                    else None
                )
                if not transcript or candidate_audit is None:
                    result_class = "private_transcription_not_accepted"
                elif (
                    stt_end_offset_ms is None
                    or stt_start_offset_ms is None
                ):
                    result_class = "speech_timing_observation_missing"
                elif stt_end_offset_ms > 10_000:
                    result_class = "voice_response_latency_over_10s"
                    canonical_accept_latency_class = "over_10s"
                    inflight_sink_cancellation_class = (
                        "pre_sink_over_10s_suppressed"
                    )
                elif route_deadline_expired:
                    result_class = "live_candidate_window_failed"
                elif not callable(private_turn_sink):
                    result_class = "private_turn_sink_unavailable"
                else:
                    try:
                        sink_result = private_turn_sink(candidate_audit, transcript)
                    except Exception:
                        sink_result = None
                    validated_sink_result = _validate_live_candidate_sink_result(
                        sink_result
                    )
                    if validated_sink_result is not None:
                        canonical_accept_offset_ms = (
                            _offset_from_last_vad_frame_ms(
                                capture_started_monotonic=(
                                    capture_started_monotonic
                                ),
                                last_vad_speech_frame_offset_ms=(
                                    last_vad_speech_frame_offset_ms
                                ),
                                observed_monotonic=time.monotonic(),
                            )
                        )
                        canonical_accept_latency_class = (
                            classify_canonical_accept_latency(
                                canonical_accept_offset_ms
                            )
                        )
                        inflight_sink_cancellation_class = (
                            "inflight_sink_cancellation_not_proven"
                            if canonical_accept_latency_class == "over_10s"
                            else "not_required_within_10s"
                        )
                        submission_count = 1
                        turninput_count = 1
                        presentation_class = str(
                            validated_sink_result["presentation_class"]
                        )
                        assistant_event_id = validated_sink_result[
                            "assistant_event_id"
                        ]
                        thought_core_first_event_elapsed_ms = (
                            validated_sink_result[
                                "thought_core_first_event_elapsed_ms"
                            ]
                        )
                        result_class = (
                            "independent_user_speech_turninput_accepted"
                        )
                    else:
                        result_class = "private_turn_sink_failed"
    except InputGateError as exc:
        failure_class = str(exc)
        result_class = (
            failure_class
            if failure_class == "input_source_epoch_unavailable"
            else "live_candidate_window_failed"
        )
    except AudioEnvironmentError as exc:
        failure_class = str(exc)
        result_class = (
            failure_class
            if failure_class in LIVE_AEC_FIXED_CHILD_FAILURE_CLASSES
            else "live_candidate_environment_unavailable"
        )
    except (AudioInputError, AudioTranscriptionError):
        result_class = "live_candidate_processing_failed"
    except Exception:
        result_class = "live_candidate_window_failed"
    finally:
        if capability is not None and candidate is not None:
            input_gate.cancel_turn_input_capability(capability, candidate)
        if candidate is not None:
            input_gate.discard_consumed_candidate(candidate)
            input_gate.discard_candidate(candidate)
        private_authority_residue_count = (
            input_gate.private_authority_residue_count()
        )
        transcript = ""
        candidate_audit = None
        sink_result = None
        transcription = None
        if session is not None:
            session.state.previous_text = None
            session.state.last_spoken_result = None
            session.state.finalized_text = None
            for buffered_chunk in session.state.buffer.chunks:
                buffered_chunk.clear()
            session.state.buffer.chunks.clear()
        pcm_cleanup_count = 0
        if capture_window is not None:
            capture_window.clear()
            pcm16 = capture_window.chunk.pcm16
            if pcm16 is not None and all(value == 0 for value in pcm16):
                pcm_cleanup_count = 1
        elapsed_ms = min(
            20_000,
            max(0, int((time.monotonic() - started) * 1000)),
        )
        if last_vad_speech_frame_offset_ms is not None:
            utterance_end_to_candidate_result_ms = max(
                0,
                elapsed_ms - last_vad_speech_frame_offset_ms,
            )
    return build_live_candidate_window_response(
        result_class=result_class,
        expectation_class=expectation_class,
        capture_packet_count=packet_count,
        capture_byte_count=byte_count,
        signal_class=signal_class,
        vad_decision_class=vad_decision_class,
        transcription_count=transcription_count,
        submission_count=submission_count,
        thought_core_turninput_count=turninput_count,
        elapsed_ms=elapsed_ms,
        last_vad_speech_frame_offset_ms=last_vad_speech_frame_offset_ms,
        utterance_end_to_candidate_result_ms=(
            utterance_end_to_candidate_result_ms
        ),
        utterance_end_timing_class=utterance_end_timing_class,
        stt_start_offset_ms=stt_start_offset_ms,
        stt_end_offset_ms=stt_end_offset_ms,
        canonical_accept_offset_ms=canonical_accept_offset_ms,
        canonical_accept_latency_class=canonical_accept_latency_class,
        inflight_sink_cancellation_class=(
            inflight_sink_cancellation_class
        ),
        presentation_class=presentation_class,
        assistant_event_id=assistant_event_id,
        thought_core_first_event_elapsed_ms=(
            thought_core_first_event_elapsed_ms
        ),
        pcm_cleanup_count=pcm_cleanup_count,
        private_authority_residue_count=private_authority_residue_count,
    )


def _validate_live_candidate_sink_result(
    value: object,
) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or set(value) != LIVE_CANDIDATE_SINK_RESULT_FIELDS:
        return None
    if (
        value.get("result_class") != "thought_core_turninput_accepted"
        or value.get("submission_count") != 1
        or isinstance(value.get("submission_count"), bool)
        or value.get("thought_core_turninput_count") != 1
        or isinstance(value.get("thought_core_turninput_count"), bool)
        or value.get("raw_private_publication_flags") is not False
    ):
        return None

    presentation_class = value.get("presentation_class")
    assistant_event_id = value.get("assistant_event_id")
    first_event_elapsed_ms = value.get("thought_core_first_event_elapsed_ms")
    if presentation_class not in LIVE_CANDIDATE_PRESENTATION_CLASSES:
        return None
    if presentation_class == "aituber_presentation_not_forwarded":
        if assistant_event_id is not None or first_event_elapsed_ms is not None:
            return None
    else:
        if (
            not isinstance(assistant_event_id, str)
            or LIVE_CANDIDATE_ASSISTANT_EVENT_ID_PATTERN.fullmatch(
                assistant_event_id
            )
            is None
            or LIVE_CANDIDATE_PRIVATE_EVENT_ID_TOKEN_PATTERN.search(
                assistant_event_id
            )
            is not None
        ):
            return None
        if presentation_class == "aituber_presentation_forwarded":
            if (
                isinstance(first_event_elapsed_ms, bool)
                or not isinstance(first_event_elapsed_ms, int)
                or not 0 <= first_event_elapsed_ms <= 20_000
            ):
                return None
        elif first_event_elapsed_ms is not None:
            return None

    return {
        "presentation_class": presentation_class,
        "assistant_event_id": assistant_event_id,
        "thought_core_first_event_elapsed_ms": first_event_elapsed_ms,
    }


def build_shutdown_response(reason: str, *, force: bool = False) -> tuple[object, int]:
    """Request cooperative server shutdown and return the updated state."""
    runtime_state = get_web_runtime_state()
    shutdown_state, should_schedule = runtime_state.request_shutdown(
        reason=reason,
        force=force,
    )
    status_writer = get_runtime_status_writer()
    if status_writer is not None:
        status_writer.write("stopping", shutdown=shutdown_state)
    if should_schedule and current_app.config.get(
        ENABLE_PROCESS_SHUTDOWN_CONFIG,
        True,
    ):
        shutdown_state = runtime_state.mark_shutdown_scheduled()
        if status_writer is not None:
            status_writer.write("stopping", shutdown=shutdown_state)
        schedule_server_shutdown(request.environ.get("werkzeug.server.shutdown"))
    return jsonify(
        {
            "ok": True,
            "shutdown": shutdown_state,
            "message": (
                "shutdown scheduled"
                if shutdown_state["shutdown_scheduled"]
                else "shutdown requested"
            ),
        }
    ), 202


def shutdown_requires_token() -> bool:
    """Return whether the unprefixed shutdown endpoint requires a token."""
    return not is_loopback_host(get_bind_host())


def get_bind_host() -> str:
    """Return the configured server bind host."""
    if has_app_context():
        return str(current_app.config.get(WEB_BIND_HOST_CONFIG, "127.0.0.1"))
    return "127.0.0.1"


def get_bind_port() -> int:
    """Return the configured server bind port."""
    if has_app_context():
        try:
            return int(current_app.config.get(WEB_BIND_PORT_CONFIG, 8000))
        except (TypeError, ValueError):
            return 8000
    return 8000


def get_runtime_status_writer() -> RuntimeStatusWriter | None:
    """Return the configured runtime status writer, if any."""
    writer = current_app.config.get(RUNTIME_STATUS_WRITER_CONFIG)
    return writer if isinstance(writer, RuntimeStatusWriter) else None


def build_process_health_response() -> dict[str, object]:
    """Return the compact health contract expected by launch supervisors."""
    host = get_bind_host()
    port = get_bind_port()
    return {
        "ok": True,
        "module": WEB_MODULE_NAME,
        "pid": os.getpid(),
        "uptime_s": round(get_process_uptime_seconds(), 3),
        "host": host,
        "port": port,
    }


def get_process_uptime_seconds() -> float:
    """Return process uptime based on the Web app startup timestamp."""
    started_at = str(current_app.config.get(WEB_STARTED_AT_CONFIG, "")).strip()
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(UTC) - started).total_seconds())


def build_runtime_status_payload(
    *,
    state: str,
    host: str,
    port: int,
    started_at: str,
    command_line: str,
    **extra: object,
) -> dict[str, object]:
    """Build the runtime status JSON written for integration supervisors."""
    payload: dict[str, object] = {
        "module": WEB_MODULE_NAME,
        "state": state,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "started_at": started_at,
        "host": host,
        "port": port,
        "health_url": build_local_url(host, port, "/health"),
        "shutdown_url": build_local_url(host, port, "/shutdown"),
        "command_line": command_line,
    }
    if state == "stopped":
        payload["stopped_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def build_local_url(host: str, port: int, path: str) -> str:
    """Return a usable local URL for status files."""
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in url_host and not url_host.startswith("["):
        url_host = f"[{url_host}]"
    return f"http://{url_host}:{port}{path}"


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is loopback-only."""
    normalized = (host or "").strip().lower()
    if normalized in {"localhost"}:
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def install_shutdown_signal_handlers(status_writer: RuntimeStatusWriter) -> None:
    """Install cooperative process signal handlers for the Web server."""
    def handle_signal(signum: int, _frame: object) -> None:
        status_writer.write("stopping", signal=signal.Signals(signum).name)
        print(f"ai_core Web UI received {signal.Signals(signum).name}; stopping.")
        raise KeyboardInterrupt

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, handle_signal)


def build_error_response(message: str, status_code: int) -> tuple[object, int]:
    """Return a local-policy error in the response shape expected by the caller."""
    if wants_json_response():
        return jsonify({"message": "", "transcript": "", "error": message}), status_code
    return render_page(error=message), status_code


def build_policy_error_response(message: str, status_code: int) -> tuple[object, int]:
    """Return an access-control error without rendering token-bearing HTML."""
    if wants_json_response():
        return jsonify({"message": "", "transcript": "", "error": message}), status_code
    return Response(message, status=status_code, mimetype="text/plain; charset=utf-8"), status_code


def wants_json_response() -> bool:
    """Return whether the current request expects a JSON-style API response."""
    return (
        request.path.startswith("/api/")
        or request.headers.get("X-Requested-With") == "fetch"
    )


def get_local_api_token() -> str:
    """Return the per-process token used by the local Web UI."""
    if not has_app_context():
        return ""
    return str(current_app.config.get(LOCAL_API_TOKEN_CONFIG, ""))


def get_web_preset() -> str:
    """Return the configured Web UI startup preset name."""
    if has_app_context():
        return str(current_app.config.get(WEB_PRESET_CONFIG, "")).strip()
    return os.environ.get(WEB_PRESET_ENV, "").strip()


def get_web_runtime_state() -> WebRuntimeState:
    """Return the per-process Web runtime state."""
    state = current_app.config.get(WEB_RUNTIME_STATE_CONFIG)
    if not isinstance(state, WebRuntimeState):
        state = WebRuntimeState()
        current_app.config[WEB_RUNTIME_STATE_CONFIG] = state
    return state


def has_valid_local_api_token() -> bool:
    """Return whether the request carries the per-process local API token header."""
    expected = get_local_api_token()
    provided = (
        request.headers.get(LOCAL_API_TOKEN_HEADER, "")
        or request.headers.get(SWORD_AGENT_TOKEN_HEADER, "")
        or parse_bearer_token(request.headers.get("Authorization", ""))
    )
    return bool(expected and provided and hmac.compare_digest(provided, expected))


def parse_bearer_token(value: str) -> str:
    """Parse an Authorization: Bearer token header."""
    prefix = "bearer "
    normalized = (value or "").strip()
    if normalized.lower().startswith(prefix):
        return normalized[len(prefix):].strip()
    return ""


def is_allowed_local_host(host_header: str) -> bool:
    """Return whether the Host header targets a loopback browser origin."""
    hostname = parse_hostname(host_header)
    return hostname in LOCAL_HOSTS


def is_allowed_local_remote(remote_addr: str | None) -> bool:
    """Return whether the TCP peer is loopback, independent of Host spoofing."""
    if not remote_addr:
        return False
    try:
        return ip_address(remote_addr).is_loopback
    except ValueError:
        return False


def has_trusted_origin() -> bool:
    """Validate Origin/Referer when a browser sends one for a local request."""
    origin = request.headers.get("Origin")
    if origin:
        return origin_matches_request(origin)
    referer = request.headers.get("Referer")
    if referer:
        return origin_matches_request(referer)
    return True


def origin_matches_request(origin: str) -> bool:
    """Return whether an Origin/Referer value matches the current local origin."""
    parsed_origin = urlsplit(origin)
    parsed_request = urlsplit(request.host_url)
    return (
        parsed_origin.scheme == parsed_request.scheme
        and parsed_origin.hostname == parsed_request.hostname
        and parsed_origin.hostname in LOCAL_HOSTS
        and parsed_origin.port == parsed_request.port
    )


def parse_hostname(host_value: str) -> str:
    """Parse a Host or Origin host component into a lowercase hostname."""
    try:
        return (urlsplit(f"//{host_value}").hostname or "").lower()
    except ValueError:
        return ""


def build_input_gate_response(state: InputGateState) -> dict[str, object]:
    return {
        "ok": True,
        "input_gate": state.to_payload(),
    }


def build_health_response(
    input_gate_state: InputGateState,
    runtime_state: WebRuntimeState,
    source: str = "web",
) -> dict[str, object]:
    """Return a generic status payload for integration supervisors."""
    safe_source = (source or "web").strip() or "web"
    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffprobe_available = shutil.which("ffprobe") is not None
    runtime = runtime_state.snapshot()
    try:
        latest_handoff = build_handoff_metadata(source=safe_source)
    except ValueError:
        latest_handoff = {
            "source": safe_source,
            "exists": False,
            "error": "invalid handoff source",
        }
    return {
        "ok": True,
        "ready": not runtime["shutdown_requested"],
        "server": runtime,
        "stt": {
            "ready": ffmpeg_available and ffprobe_available,
            "ffmpeg_available": ffmpeg_available,
            "ffprobe_available": ffprobe_available,
        },
        "recording": {
            "ready": ffmpeg_available,
            "max_upload_bytes": WEB_MAX_UPLOAD_BYTES,
        },
        "events": {
            "ready": True,
            "stream": "/api/events",
            "log_path": str(project_relative_path(get_event_log_path())),
            "max_log_bytes": MAX_EVENT_LOG_BYTES,
        },
        "input_gate": input_gate_state.to_payload(),
        "latest_handoff": latest_handoff,
        "web_preset": get_web_preset(),
    }


def render_page(
    transcript: str | None = None,
    command: str | None = None,
    command_path: str | None = None,
    command_text_path: str | None = None,
    error: str | None = None,
    message: str | None = None,
) -> str:
    """Render the single-page web UI."""
    return render_template(
        "index.html",
        transcript=transcript,
        command=command,
        command_path=command_path,
        command_text_path=command_text_path,
        error=error,
        message=message,
        api_token=get_local_api_token(),
        web_preset=get_web_preset(),
    )


def process_transcription_request(
    raw_bytes: bytes,
    filename: str,
    message: str | None = None,
) -> tuple[dict[str, object], int]:
    """Run one transcription request and normalize the response payload."""
    runtime_state = get_web_runtime_state()
    if not runtime_state.begin_transcription():
        return {
            "message": "",
            "transcript": "",
            "command": "",
            "command_path": "",
            "command_text_path": "",
            "error": "シャットダウン要求中のため新しい文字起こしは受け付けません。",
            "debug": {
                "server": runtime_state.snapshot(),
                "whisper_invoked": False,
                "whisper_skipped": True,
                "skip_reason": "shutdown_requested",
            },
        }, 503

    def read_bool_flag(*names: str) -> bool:
        for name in names:
            value = request.form.get(name, "").strip().lower()
            if value in {"1", "true", "yes", "on"}:
                return True
        return False

    turn_id = normalize_turn_id(request.form.get("turn_id")) or new_turn_id("web")
    if request.endpoint in {
        "api_transcribe_browser_recording",
        "transcribe_browser_recording",
    }:
        emit_event(
            "record_stop",
            turn_id=turn_id,
            source="web",
            payload={
                "transport": "final_upload",
                "filename": Path(filename).name,
                "size_bytes": len(raw_bytes),
            },
        )
    try:
        response = process_web_transcription(
            WebTranscriptionRequest(
                raw_bytes=raw_bytes,
                filename=filename,
                turn_id=turn_id,
                model_name=request.form.get("model", "small").strip() or "small",
                language=request.form.get("language", "").strip() or None,
                command_only=read_bool_flag("instruction_only", "command_only"),
                save_handoff=read_bool_flag("save_handoff", "save_command"),
                source="web",
                success_message=message,
            )
        )
        return response.to_payload(), response.status_code
    finally:
        should_shutdown = runtime_state.end_transcription()
        if should_shutdown and current_app.config.get(
            ENABLE_PROCESS_SHUTDOWN_CONFIG,
            True,
        ):
            runtime_state.mark_shutdown_scheduled()
            schedule_server_shutdown(request.environ.get("werkzeug.server.shutdown"))


def handle_transcription(
    raw_bytes: bytes,
    filename: str,
    message: str | None = None,
) -> str | Response:
    """Persist uploaded audio temporarily and transcribe it."""
    payload, status_code = process_transcription_request(
        raw_bytes,
        filename,
        message=message,
    )
    if request.headers.get("X-Requested-With") == "fetch":
        response = jsonify(payload)
        response.status_code = status_code
        return response
    if payload["error"]:
        return render_page(error=payload["error"])
    return render_page(
        transcript=payload["transcript"],
        command=payload["command"],
        command_path=payload["command_path"],
        command_text_path=payload["command_text_path"],
        message=payload["message"],
    )


def handle_transcription_api(
    raw_bytes: bytes,
    filename: str,
    message: str | None = None,
) -> tuple[object, int]:
    """Handle uploaded audio and return a dedicated JSON API response."""
    payload, status_code = process_transcription_request(
        raw_bytes,
        filename,
        message=message,
    )
    return jsonify(payload), status_code


def parse_boolish(value: object) -> bool | None:
    """Parse common bool-like request values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def normalize_turn_id(value: object) -> str:
    """Return a path-safe turn id or an empty string."""
    if value is None:
        return ""
    normalized = str(value).strip()
    if not normalized or len(normalized) > 128:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(character not in allowed for character in normalized):
        return ""
    return normalized


def normalize_event_name(value: object) -> str:
    """Return a compact event name accepted by the local event ingest API."""
    if value is None:
        return ""
    normalized = str(value).strip()
    if not normalized or len(normalized) > 80:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    if any(character not in allowed for character in normalized):
        return ""
    return normalized


def parse_nonnegative_int(value: object) -> int | None:
    """Parse a non-negative integer from request data."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def resolve_event_trace_limit(value: object) -> int:
    """Return a bounded event trace read limit."""
    parsed = parse_nonnegative_int(value)
    if parsed is None or parsed <= 0:
        return WEB_EVENT_TRACE_DEFAULT_LIMIT
    return parsed


def validate_client_event_name(value: object) -> str:
    """Return one fixed event name accepted from the local client boundary."""
    if not isinstance(value, str) or value not in CLIENT_EVENT_NAMES:
        raise InputGateError("client event name is invalid")
    return value


def validate_client_event_top_level(payload: dict[str, object]) -> None:
    """Reject unexpected client fields instead of filtering private markers."""
    if "event" not in payload or "payload" not in payload:
        raise InputGateError("client event fields are invalid")
    if not set(payload).issubset(CLIENT_EVENT_TOP_LEVEL_KEYS):
        raise InputGateError("client event fields are invalid")


def validate_client_turn_id(value: object) -> str:
    """Accept only browser-generated opaque turn ids, or create one locally."""
    if value is None:
        return new_turn_id("web")
    if not isinstance(value, str) or CLIENT_TURN_ID_PATTERN.fullmatch(value) is None:
        raise InputGateError("client turn id is invalid")
    return value


def validate_client_event_source(value: object, event_name: str) -> str:
    """Return the fixed source class for one allowed local event."""
    expected = CLIENT_EVENT_SOURCE_BY_NAME[event_name]
    if value is None:
        return expected
    if not isinstance(value, str) or value != expected:
        raise InputGateError("client event source is invalid")
    return expected


def validate_client_event_payload(
    event_name: str,
    value: object,
) -> dict[str, object]:
    """Validate exact class/count payloads without retaining arbitrary values."""
    if not isinstance(value, dict):
        raise InputGateError("client event payload is invalid")
    if event_name == "record_start":
        if set(value) != RECORD_START_PAYLOAD_KEYS:
            raise InputGateError("record-start payload fields are invalid")
        trigger = value.get("trigger")
        timeslice_ms = value.get("timeslice_ms")
        mime_type = value.get("mime_type")
        if trigger not in {"manual", "input_gate"}:
            raise InputGateError("record-start trigger is invalid")
        if type(timeslice_ms) is not int or timeslice_ms != WEB_RECORDING_CHUNK_TIMESLICE_MS:
            raise InputGateError("record-start timeslice is invalid")
        if not isinstance(mime_type, str) or mime_type not in CLIENT_RECORDING_MIME_TYPES:
            raise InputGateError("record-start media class is invalid")
        return {
            "trigger": trigger,
            "timeslice_ms": timeslice_ms,
            "mime_type": mime_type,
        }
    if event_name == "record_stop":
        if set(value) != RECORD_STOP_PAYLOAD_KEYS:
            raise InputGateError("record-stop payload fields are invalid")
        chunk_count = value.get("chunk_count")
        chunk_sequence = value.get("chunk_sequence")
        blob_size_bytes = value.get("blob_size_bytes")
        if type(chunk_count) is not int or not 0 <= chunk_count <= WEB_MAX_RECORDING_CHUNKS:
            raise InputGateError("record-stop chunk count is invalid")
        if type(chunk_sequence) is not int or not 0 <= chunk_sequence <= WEB_MAX_RECORDING_CHUNKS:
            raise InputGateError("record-stop chunk sequence is invalid")
        if type(blob_size_bytes) is not int or not 0 <= blob_size_bytes <= WEB_MAX_UPLOAD_BYTES:
            raise InputGateError("record-stop byte count is invalid")
        return {
            "chunk_count": chunk_count,
            "chunk_sequence": chunk_sequence,
            "blob_size_bytes": blob_size_bytes,
        }
    return value


def validate_client_event_timing(payload: dict[str, object]) -> dict[str, object]:
    """Accept either no client timing or one complete, bounded timing tuple."""
    present = set(payload).intersection(CLIENT_EVENT_TIMING_KEYS)
    if not present:
        return {}
    if present != CLIENT_EVENT_TIMING_KEYS:
        raise InputGateError("client event timing fields are incomplete")
    wall = payload.get("client_timestamp_wall")
    monotonic = payload.get("client_timestamp_monotonic")
    performance_now = payload.get("client_performance_now")
    if (
        not isinstance(wall, str)
        or len(wall) > 27
        or CLIENT_WALL_TIMESTAMP_PATTERN.fullmatch(wall) is None
    ):
        raise InputGateError("client wall timestamp is invalid")
    try:
        parsed_wall = datetime.fromisoformat(wall.replace("Z", "+00:00"))
    except ValueError:
        raise InputGateError("client wall timestamp is invalid") from None
    if parsed_wall.tzinfo != UTC or not 2020 <= parsed_wall.year <= 2100:
        raise InputGateError("client wall timestamp is outside the allowed range")
    for field_name, field_value in (
        ("client monotonic timestamp", monotonic),
        ("client performance timestamp", performance_now),
    ):
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(float(field_value))
            or not 0 <= float(field_value) <= 1_000_000_000_000
        ):
            raise InputGateError(f"{field_name} is invalid")
    return {
        "client_timestamp_wall": wall,
        "client_timestamp_monotonic": float(monotonic),
        "client_performance_now": float(performance_now),
    }


def get_recording_chunk_dir(turn_id: str) -> Path:
    """Return the server-side chunk landing directory for one turn."""
    safe_turn_id = normalize_turn_id(turn_id)
    if not safe_turn_id:
        raise ValueError("invalid turn_id")
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / ".cache" / "web_recording_chunks"
    chunk_dir = cache_dir / safe_turn_id
    if not chunk_dir.exists():
        prune_recording_chunk_cache(cache_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir


def get_recording_chunk_path(chunk_dir: Path, sequence: int) -> Path:
    """Return a bounded chunk path for one sequence number."""
    if sequence < 0 or sequence > WEB_MAX_RECORDING_CHUNKS:
        raise ValueError("recording chunk sequence is out of range")
    safe_dir = chunk_dir.resolve()
    chunk_path = (safe_dir / f"chunk_{sequence:06d}.webm").resolve()
    if not chunk_path.is_relative_to(safe_dir):
        raise ValueError("recording chunk path escaped the cache directory")
    return chunk_path


def get_recording_chunk_total_bytes(
    chunk_dir: Path,
    *,
    exclude_path: Path | None = None,
) -> int:
    """Return the current byte total for one turn's server-side chunks."""
    if not chunk_dir.exists():
        return 0
    safe_exclude = exclude_path.resolve() if exclude_path is not None else None
    total = 0
    for path in chunk_dir.glob("chunk_*.webm"):
        try:
            if safe_exclude is not None and path.resolve() == safe_exclude:
                continue
            total += path.stat().st_size
        except OSError:
            continue
    return total


def prune_recording_chunk_cache(cache_dir: Path) -> None:
    """Remove old browser-recording chunk directories from the local cache."""
    if not cache_dir.exists():
        return
    safe_cache_dir = cache_dir.resolve()
    now = time.time()
    try:
        turn_dirs = [
            path
            for path in cache_dir.iterdir()
            if path.is_dir() and path.resolve().is_relative_to(safe_cache_dir)
        ]
    except OSError:
        return
    for turn_dir in list(turn_dirs):
        try:
            if now - turn_dir.stat().st_mtime > WEB_RECORDING_CHUNK_RETENTION_SECONDS:
                remove_recording_chunk_dir(safe_cache_dir, turn_dir)
                turn_dirs.remove(turn_dir)
        except OSError:
            continue

    turn_dirs = sorted(
        [path for path in turn_dirs if path.exists()],
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for turn_dir in turn_dirs[WEB_MAX_RECORDING_CHUNK_CACHE_TURNS:]:
        remove_recording_chunk_dir(safe_cache_dir, turn_dir)

    remaining_dirs = [
        path
        for path in turn_dirs[:WEB_MAX_RECORDING_CHUNK_CACHE_TURNS]
        if path.exists()
    ]
    while (
        get_recording_chunk_cache_total_bytes(remaining_dirs)
        > WEB_MAX_RECORDING_CHUNK_CACHE_BYTES
    ):
        if not remaining_dirs:
            break
        oldest = min(
            remaining_dirs,
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
        )
        remove_recording_chunk_dir(safe_cache_dir, oldest)
        remaining_dirs = [path for path in remaining_dirs if path.exists()]


def get_recording_chunk_cache_total_bytes(turn_dirs: list[Path]) -> int:
    """Return recursive size for selected recording chunk turn directories."""
    total = 0
    for turn_dir in turn_dirs:
        try:
            total += sum(
                path.stat().st_size
                for path in turn_dir.rglob("chunk_*.webm")
                if path.is_file()
            )
        except OSError:
            continue
    return total


def remove_recording_chunk_dir(cache_dir: Path, turn_dir: Path) -> None:
    """Remove one cache child only after confirming it is under the cache root."""
    try:
        safe_turn_dir = turn_dir.resolve()
    except OSError:
        return
    if not safe_turn_dir.is_relative_to(cache_dir):
        return
    shutil.rmtree(safe_turn_dir, ignore_errors=True)


def project_relative_path(path: Path) -> Path:
    """Return a project-relative path where possible to avoid leaking local roots."""
    project_root = Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return Path(path.name)


def format_sse_event(payload: dict[str, object], event_name: str) -> str:
    """Format one event as a Server-Sent Events frame."""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


def schedule_server_shutdown(werkzeug_shutdown: object | None = None) -> None:
    """Schedule local development server shutdown outside the request thread."""
    def shutdown() -> None:
        if callable(werkzeug_shutdown):
            werkzeug_shutdown()
            return
        _thread.interrupt_main()

    timer = threading.Timer(0.2, shutdown)
    timer.daemon = True
    timer.start()


def build_parser() -> argparse.ArgumentParser:
    """Build the Web server argument parser."""
    parser = argparse.ArgumentParser(description="Run the local ai_core Web UI.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the Web UI. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port for the Web UI. Default: 8000",
    )
    parser.add_argument(
        "--runtime-status-file",
        default=None,
        help="Optional JSON status file for integration supervisors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the local Flask development server."""
    args = build_parser().parse_args(argv)
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    status_path = (
        Path(args.runtime_status_file).expanduser().resolve()
        if args.runtime_status_file
        else None
    )
    status_writer = RuntimeStatusWriter(
        status_path,
        host=args.host,
        port=args.port,
        started_at=started_at,
    )
    app = create_app(
        host=args.host,
        port=args.port,
        runtime_status_writer=status_writer,
        started_at=started_at,
    )
    install_shutdown_signal_handlers(status_writer)
    status_writer.write("running")
    final_state = "stopped"
    final_extra: dict[str, object] = {}
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        return 0
    except BaseException as exc:
        final_state = "error"
        final_extra = {
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        raise
    finally:
        status_writer.write(final_state, **final_extra)
        print("ai_core Web UI stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
