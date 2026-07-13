"""CLI entrypoint for local audio transcription."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.core.agent_instruction import build_agent_instruction
from src.core.events import (
    emit_event,
    new_turn_id,
    normalize_event_turn_id,
    text_payload_facts,
)
from src.core.handoff_bridge import save_handoff_bundle
from src.core.input_gate import InputGate, InputGateState
from src.core.pipeline import AudioChunk, TranscriptionPipeline, TranscriptionResult
from src.core.session import MicLoopSession, MicLoopTuning
from src.core.status_report import (
    build_doctor_status,
    build_torch_pin_status,
    format_doctor_status,
    format_runtime_status,
    print_status_command,
)
from src.io.audio import (
    AudioEnvironmentError,
    AudioInputError,
    AudioTranscriptionError,
    ensure_ffmpeg_available,
    validate_audio_file,
    validate_model_name,
)
from src.io.microphone import (
    SUPPORTED_MICROPHONE_BACKENDS,
    capture_microphone_chunk,
    get_temp_recording_path,
    has_detectable_speech,
    validate_vad_aggressiveness,
)


MIC_LOOP_PROFILES: dict[str, tuple[int, int, str]] = {
    "responsive": (1, 5, "反応を優先し、早めに partial/final を出す"),
    "balanced": (2, 8, "既定のバランス設定"),
    "strict": (3, 10, "無音と誤認識を減らす代わりに確定を遅くする"),
    "low_latency": (1, 1, "短い発話を1回の final として試験的に早期確定する"),
}


def format_transcription_result(result: TranscriptionResult) -> str:
    """Format a realtime transcription result for terminal output."""
    if not result.input_enabled:
        reason = f": {result.input_gate_reason}" if result.input_gate_reason else ""
        return f"[disabled] input disabled{reason}"
    if result.is_silence:
        chunk_count = result.chunk_count
        result_type = "final" if result.is_final else "silence"
        return f"[{result_type} {chunk_count}] silence detected"
    result_type = "final" if result.is_final else "partial"
    chunk_count = result.chunk_count
    text = result.text
    return f"[{result_type} {chunk_count}] {text}"


def print_agent_instruction_if_requested(text: str, emit_command: bool) -> None:
    """Print an agent-ready instruction draft when requested."""
    if not emit_command:
        return
    draft = build_agent_instruction(text)
    if draft is None:
        print("[command] no instruction draft available")
        return
    print(f"[command] {draft.instruction}")


def print_agent_instruction_only(text: str) -> None:
    """Print only the agent-ready instruction draft."""
    draft = build_agent_instruction(text)
    if draft is None:
        print("no instruction draft available")
        return
    print(draft.instruction)


def save_handoff_if_requested(
    text: str,
    output_path: str | None,
    *,
    turn_id: str | None = None,
    source: str = "cli",
) -> None:
    """Save a handoff bundle to disk when requested."""
    if not output_path:
        return
    json_path = Path(output_path).expanduser().resolve()
    text_path = json_path.with_suffix(".txt")
    saved_paths = save_handoff_bundle(text, json_path=json_path, text_path=text_path)
    if saved_paths is None:
        print(f"[command-file] no instruction draft available for {output_path}")
        return
    if turn_id:
        emit_event(
            "handoff_saved",
            turn_id=turn_id,
            source=source,
            payload={
                "json_filename": saved_paths.json_path.name,
                "text_filename": saved_paths.text_path.name,
                "transcript": text_payload_facts(text),
            },
        )
    print(f"[command-file] {saved_paths.json_path}")
    print(f"[command-text] {saved_paths.text_path}")


def print_runtime_note(message: str) -> None:
    """Print operational notes to stderr without polluting stdout payloads."""
    print(message, file=sys.stderr)


def build_input_gate_data(state: InputGateState) -> dict[str, bool | float | str | None]:
    """Return the input-gate state as a stable JSON-friendly payload."""
    return state.to_payload()


def format_input_gate_state(state: InputGateState) -> str:
    """Format the current input-gate state for terminal output."""
    return (
        "[input-gate] "
        f"input_enabled={state.enabled} "
        f"reason={state.reason} "
        f"source={state.source}"
    )


def format_mic_loop_tuning(
    profile: str,
    vad_aggressiveness: int,
    final_stable_seconds: int,
) -> str:
    """Format the resolved mic-loop tuning values for terminal output."""
    return (
        "[mic-tuning] "
        f"profile={profile} "
        f"vad_aggressiveness={vad_aggressiveness} "
        f"final_stable_seconds={final_stable_seconds}"
    )


def run_mic_loop(
    duration: int,
    mic_device: str,
    mic_backend: str,
    model_name: str,
    language: str | None,
    iterations: int | None,
    trim_silence_enabled: bool,
    emit_command: bool,
    command_only: bool,
    command_output: str | None,
    mic_profile: str,
    vad_aggressiveness: int,
    final_stable_seconds: int,
    input_enabled: bool = True,
    input_gate_reason: str = "manual",
    aec_owner_selection: dict[str, object] | None = None,
) -> int:
    """Record and transcribe microphone chunks until interrupted."""
    input_gate = InputGate(
        initially_enabled=input_enabled,
        reason=input_gate_reason,
        source="cli",
    )
    session: MicLoopSession | None = None
    completed_iterations = 0

    def get_session() -> MicLoopSession:
        nonlocal session
        if session is None:
            session = MicLoopSession(
                pipeline=TranscriptionPipeline(model_name=model_name),
                tuning=MicLoopTuning(
                    vad_aggressiveness=vad_aggressiveness,
                    final_stable_seconds=final_stable_seconds,
                ),
                input_gate=input_gate,
            )
        return session

    print_runtime_note(
        format_mic_loop_tuning(
            profile=mic_profile,
            vad_aggressiveness=vad_aggressiveness,
            final_stable_seconds=final_stable_seconds,
        )
    )
    print_runtime_note(format_input_gate_state(input_gate.state))

    try:
        while iterations is None or completed_iterations < iterations:
            next_iteration = completed_iterations + 1
            is_last_iteration = iterations is not None and next_iteration == iterations
            if not input_gate.is_enabled():
                if session is None:
                    result = TranscriptionResult(
                        source="microphone",
                        text="",
                        is_final=is_last_iteration,
                        chunk_count=0,
                        input_enabled=False,
                        input_gate_reason=input_gate.state.reason,
                    )
                else:
                    result = session.process_input_disabled(
                        is_last_iteration=is_last_iteration
                    )
                print_runtime_note(format_transcription_result(result))
                completed_iterations += 1
                if iterations is None or completed_iterations < iterations:
                    time.sleep(duration)
                continue

            chunk = capture_microphone_chunk(
                output_path=get_temp_recording_path(),
                duration=duration,
                device=mic_device,
                backend=mic_backend,
                trim_silence_enabled=trim_silence_enabled,
                aec_owner_selection=aec_owner_selection,
            )
            active_session = get_session()
            result = active_session.process_chunk(
                chunk,
                has_speech=has_detectable_speech(
                    chunk.path,
                    aggressiveness=vad_aggressiveness,
                ),
                language=language,
                chunk_duration=duration,
                is_last_iteration=is_last_iteration,
            )
            if command_only:
                print_agent_instruction_only(result.text)
            else:
                print(format_transcription_result(result))
                print_agent_instruction_if_requested(result.text, emit_command=emit_command)
            save_handoff_if_requested(result.text, command_output)
            completed_iterations += 1
    except KeyboardInterrupt:
        final_result = None if session is None else session.finalize_on_interrupt()
        if final_result is not None:
            if command_only:
                print_agent_instruction_only(final_result.text)
            else:
                print(format_transcription_result(final_result))
                print_agent_instruction_if_requested(
                    final_result.text,
                    emit_command=emit_command,
                )
            save_handoff_if_requested(final_result.text, command_output)
        print_runtime_note("Stopped microphone loop.")
        return 0

    return 0


def validate_iterations(iterations: int | None) -> None:
    """Validate optional mic-loop iteration count."""
    if iterations is not None and iterations <= 0:
        raise AudioInputError("--iterations must be greater than 0")


def validate_mic_loop_options(vad_aggressiveness: int) -> None:
    """Validate mic-loop specific options."""
    validate_vad_aggressiveness(vad_aggressiveness)


def validate_mic_profile(profile: str) -> None:
    """Validate the selected mic-loop tuning profile."""
    if profile not in MIC_LOOP_PROFILES:
        raise AudioInputError(
            "--mic-profile must be one of: "
            + ", ".join(MIC_LOOP_PROFILES)
        )


def validate_final_stable_seconds(final_stable_seconds: int) -> None:
    """Validate the stable-duration threshold for finalization."""
    if final_stable_seconds <= 0:
        raise AudioInputError("--final-stable-seconds must be greater than 0")


def resolve_mic_loop_tuning(
    profile: str,
    vad_aggressiveness: int | None,
    final_stable_seconds: int | None,
) -> tuple[int, int]:
    """Resolve mic-loop tuning from a named profile and optional overrides."""
    preset_vad, preset_final_seconds, _description = MIC_LOOP_PROFILES[profile]
    resolved_vad = preset_vad if vad_aggressiveness is None else vad_aggressiveness
    resolved_final_seconds = (
        preset_final_seconds
        if final_stable_seconds is None
        else final_stable_seconds
    )
    return resolved_vad, resolved_final_seconds


def format_mic_profile_list() -> str:
    """Render the available mic-loop tuning profiles."""
    lines = ["Available mic-loop profiles:"]
    for profile, (vad, stable_seconds, description) in MIC_LOOP_PROFILES.items():
        lines.append(
            f"- {profile}: vad_aggressiveness={vad}, "
            f"final_stable_seconds={stable_seconds} ({description})"
        )
    return "\n".join(lines)


def build_mic_profile_list_data() -> list[dict[str, str | int]]:
    """Return the available mic-loop profiles as structured data."""
    return [
        {
            "profile": profile,
            "vad_aggressiveness": vad,
            "final_stable_seconds": stable_seconds,
            "description": description,
        }
        for profile, (vad, stable_seconds, description) in MIC_LOOP_PROFILES.items()
    ]


def build_mic_tuning_data(
    profile: str,
    vad_aggressiveness: int,
    final_stable_seconds: int,
) -> dict[str, str | int]:
    """Return the resolved mic-loop tuning as structured data."""
    return {
        "profile": profile,
        "vad_aggressiveness": vad_aggressiveness,
        "final_stable_seconds": final_stable_seconds,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe a local audio file with Whisper."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to the local audio file.",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Record from the microphone for a fixed duration.",
    )
    parser.add_argument(
        "--mic-loop",
        action="store_true",
        help="Repeat fixed-duration microphone recording until interrupted.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds for --mic or --mic-loop. Default: 5",
    )
    parser.add_argument(
        "--mic-device",
        default="default",
        help="Microphone device for the active recording backend. Default: default",
    )
    parser.add_argument(
        "--mic-backend",
        choices=SUPPORTED_MICROPHONE_BACKENDS,
        default="auto",
        help=(
            "Microphone recording backend: auto, arecord, or ffmpeg-dshow. "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name to use. Default: small",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code such as ja or en.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional iteration count for --mic-loop. Default: run until Ctrl+C",
    )
    parser.add_argument(
        "--no-trim-silence",
        action="store_true",
        help="Disable ffmpeg-based silence trimming for microphone recordings.",
    )
    parser.add_argument(
        "--mic-profile",
        default="balanced",
        help="Mic-loop tuning profile: responsive, balanced, strict, or low_latency. Default: balanced",
    )
    parser.add_argument(
        "--list-mic-profiles",
        action="store_true",
        help="Print available mic-loop tuning profiles and exit.",
    )
    parser.add_argument(
        "--show-mic-tuning",
        action="store_true",
        help="Print the resolved mic-loop tuning values and exit.",
    )
    parser.add_argument(
        "--mic-tuning-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --list-mic-profiles or --show-mic-tuning. Default: text",
    )
    parser.add_argument(
        "--show-input-gate",
        action="store_true",
        help="Print the initial input-gate state and exit.",
    )
    parser.add_argument(
        "--input-gate-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --show-input-gate. Default: text",
    )
    parser.add_argument(
        "--input-disabled",
        action="store_true",
        help="Start mic-loop with microphone input gated off.",
    )
    parser.add_argument(
        "--input-gate-reason",
        default="manual",
        help="Reason label used when initializing the input gate. Default: manual",
    )
    parser.add_argument(
        "--show-runtime-status",
        action="store_true",
        help="Print the local Whisper / Torch / ffmpeg runtime status and exit.",
    )
    parser.add_argument(
        "--show-dependency-status",
        action="store_true",
        help="Print direct dependency and installed package status and exit.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print a combined runtime and dependency diagnosis and exit.",
    )
    parser.add_argument(
        "--show-torch-pin-plan",
        action="store_true",
        help="Print a project-local Torch pin plan and exit.",
    )
    parser.add_argument(
        "--runtime-status-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --show-runtime-status. Default: text",
    )
    parser.add_argument(
        "--dependency-status-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --show-dependency-status. Default: text",
    )
    parser.add_argument(
        "--doctor-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --doctor. Default: text",
    )
    parser.add_argument(
        "--torch-pin-plan-format",
        choices=("text", "json"),
        default="text",
        help="Output format for --show-torch-pin-plan. Default: text",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=None,
        help="WebRTC VAD aggressiveness for --mic-loop (0-3). Overrides --mic-profile.",
    )
    parser.add_argument(
        "--final-stable-seconds",
        type=int,
        default=None,
        help="Stable duration threshold in seconds for mic-loop finalization. Overrides --mic-profile.",
    )
    parser.add_argument(
        "--emit-command",
        "--emit-instruction",
        dest="emit_command",
        action="store_true",
        help="Print an agent-ready instruction draft from the transcript.",
    )
    parser.add_argument(
        "--command-only",
        "--instruction-only",
        dest="command_only",
        action="store_true",
        help="Print only the agent-ready instruction draft.",
    )
    parser.add_argument(
        "--command-output",
        "--handoff-output",
        dest="command_output",
        default=None,
        help="Optional path to save a handoff payload JSON file.",
    )
    parser.add_argument(
        "--turn-id",
        default=None,
        help="Optional turn id for emitted latency events.",
    )
    return parser


def main() -> int:
    """Run the transcription CLI."""
    args = build_parser().parse_args()
    turn_id = (
        normalize_event_turn_id(args.turn_id)
        if args.turn_id
        else new_turn_id("cli")
    )

    try:
        if args.list_mic_profiles:
            if args.mic_tuning_format == "json":
                print(json.dumps(build_mic_profile_list_data(), ensure_ascii=False, indent=2))
            else:
                print(format_mic_profile_list())
            return 0
        if args.show_mic_tuning:
            validate_mic_profile(args.mic_profile)
            resolved_vad_aggressiveness, resolved_final_stable_seconds = (
                resolve_mic_loop_tuning(
                    profile=args.mic_profile,
                    vad_aggressiveness=args.vad_aggressiveness,
                    final_stable_seconds=args.final_stable_seconds,
                )
            )
            validate_mic_loop_options(resolved_vad_aggressiveness)
            validate_final_stable_seconds(resolved_final_stable_seconds)
            if args.mic_tuning_format == "json":
                print(
                    json.dumps(
                        build_mic_tuning_data(
                            profile=args.mic_profile,
                            vad_aggressiveness=resolved_vad_aggressiveness,
                            final_stable_seconds=resolved_final_stable_seconds,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(
                    format_mic_loop_tuning(
                        profile=args.mic_profile,
                        vad_aggressiveness=resolved_vad_aggressiveness,
                        final_stable_seconds=resolved_final_stable_seconds,
                    )
                )
            return 0
        if args.show_input_gate:
            input_gate = InputGate(
                initially_enabled=not args.input_disabled,
                reason=args.input_gate_reason,
                source="cli",
            )
            if args.input_gate_format == "json":
                print(
                    json.dumps(
                        build_input_gate_data(input_gate.state),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(format_input_gate_state(input_gate.state))
            return 0
        if print_status_command(args):
            return 0
        validate_model_name(args.model)
        ensure_ffmpeg_available()
        if args.command_only:
            args.emit_command = True
        if args.mic and args.mic_loop:
            raise AudioInputError("--mic and --mic-loop cannot be used together")
        if args.input_disabled and not args.mic_loop:
            raise AudioInputError(
                "--input-disabled can only be used with --mic-loop or --show-input-gate"
            )
        validate_iterations(args.iterations)
        if args.iterations is not None and not args.mic_loop:
            raise AudioInputError("--iterations can only be used with --mic-loop")
        if args.mic_loop:
            validate_mic_profile(args.mic_profile)
            resolved_vad_aggressiveness, resolved_final_stable_seconds = (
                resolve_mic_loop_tuning(
                    profile=args.mic_profile,
                    vad_aggressiveness=args.vad_aggressiveness,
                    final_stable_seconds=args.final_stable_seconds,
                )
            )
            validate_mic_loop_options(resolved_vad_aggressiveness)
            validate_final_stable_seconds(resolved_final_stable_seconds)
            if args.audio_file is not None:
                raise AudioInputError("audio_file cannot be used together with --mic-loop")
            return run_mic_loop(
                duration=args.duration,
                mic_device=args.mic_device,
                mic_backend=args.mic_backend,
                model_name=args.model,
                language=args.language,
                iterations=args.iterations,
                trim_silence_enabled=not args.no_trim_silence,
                emit_command=args.emit_command,
                command_only=args.command_only,
                command_output=args.command_output,
                mic_profile=args.mic_profile,
                vad_aggressiveness=resolved_vad_aggressiveness,
                final_stable_seconds=resolved_final_stable_seconds,
                input_enabled=not args.input_disabled,
                input_gate_reason=args.input_gate_reason,
            )
        chunk: AudioChunk
        if args.mic:
            if args.audio_file is not None:
                raise AudioInputError("audio_file cannot be used together with --mic")
            chunk = capture_microphone_chunk(
                output_path=get_temp_recording_path(),
                duration=args.duration,
                device=args.mic_device,
                backend=args.mic_backend,
                trim_silence_enabled=not args.no_trim_silence,
            )
            record_payload: dict[str, object] = {
                "transport": "cli_microphone",
                "duration_seconds": args.duration,
                "backend": args.mic_backend,
                "storage_class": chunk.storage_class,
            }
            if chunk.path is not None:
                record_payload["filename"] = chunk.path.name
            emit_event(
                "record_stop",
                turn_id=turn_id,
                source="cli",
                payload=record_payload,
            )
        else:
            if not args.audio_file:
                raise AudioInputError(
                    "audio_file is required unless --mic or --mic-loop is used"
                )
            audio_path = Path(args.audio_file).expanduser().resolve()
            validate_audio_file(audio_path)
            chunk = AudioChunk(path=audio_path, source="file")
        source = "microphone" if args.mic else "file"
        stt_payload: dict[str, object] = {
            "model": args.model,
            "language": args.language or "",
            "source": source,
            "storage_class": chunk.storage_class,
        }
        if chunk.path is not None:
            stt_payload["filename"] = chunk.path.name
        emit_event(
            "stt_start",
            turn_id=turn_id,
            source="cli",
            payload=stt_payload,
        )
        try:
            pipeline = TranscriptionPipeline(model_name=args.model)
            text = pipeline.transcribe_chunk(chunk, language=args.language)
        except Exception as exc:
            emit_event(
                "stt_done",
                turn_id=turn_id,
                source="cli",
                payload={
                    "model": args.model,
                    "status": "error",
                    "error_type": exc.__class__.__name__,
                },
            )
            raise
        emit_event(
            "stt_done",
            turn_id=turn_id,
            source="cli",
            payload={
                "model": args.model,
                "status": "ok",
                **text_payload_facts(text),
            },
        )
        emit_event(
            "stt_final",
            turn_id=turn_id,
            source="cli",
            payload=text_payload_facts(text),
        )
    except AudioInputError as exc:
        print(f"Input error: {exc}")
        return 1
    except AudioEnvironmentError as exc:
        print(f"Environment error: {exc}")
        return 1
    except AudioTranscriptionError as exc:
        print(f"Transcription error: {exc}")
        return 1

    if args.command_only:
        print_agent_instruction_only(text)
    else:
        print(text)
        print_agent_instruction_if_requested(text, emit_command=args.emit_command)
    save_handoff_if_requested(text, args.command_output, turn_id=turn_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
