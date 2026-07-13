"""Microphone recording helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
import platform
import re
import shutil
import subprocess
import wave
from collections.abc import Mapping
from typing import Any, Callable

from src.core.pipeline import AudioChunk
from src.io.aec_reference import (
    LiveAecCaptureError,
    LiveAecProcessedCapture,
    capture_live_aec_processed_pcm,
)
from src.io.audio import (
    AudioEnvironmentError,
    AudioInputError,
    ensure_ffmpeg_available,
)


LIVE_AEC_MICROPHONE_BACKEND = "windows-voice-capture-dsp-aec"
SUPPORTED_MICROPHONE_BACKENDS = (
    "auto",
    "arecord",
    "ffmpeg-dshow",
    LIVE_AEC_MICROPHONE_BACKEND,
)
RECORDING_MICROPHONE_BACKENDS = ("arecord", "ffmpeg-dshow")
MICROPHONE_DEVICE_LIST_TIMEOUT_SECONDS = 10
MICROPHONE_TRIM_TIMEOUT_SECONDS = 30
MICROPHONE_RECORDING_EXTRA_TIMEOUT_SECONDS = 10


class WebRtcVadAdapter:
    """Minimal wrapper around the native _webrtcvad module."""

    def __init__(self, aggressiveness: int = 2) -> None:
        try:
            module = importlib.import_module("_webrtcvad")
        except ModuleNotFoundError as exc:
            raise AudioEnvironmentError("webrtcvad native module is not available") from exc
        self._module = module
        self._vad = module.create()
        module.init(self._vad)
        module.set_mode(self._vad, aggressiveness)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        """Return whether a single PCM frame contains speech."""
        return self._module.process(self._vad, sample_rate, frame, int(len(frame) / 2))


def get_temp_recording_path() -> Path:
    """Return the temporary wav path used for microphone recordings."""
    project_root = Path(__file__).resolve().parents[2]
    temp_dir = project_root / ".cache" / "recordings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / "mic_input.wav"


def get_trimmed_recording_path() -> Path:
    """Return the temporary wav path used for trimmed microphone recordings."""
    project_root = Path(__file__).resolve().parents[2]
    temp_dir = project_root / ".cache" / "recordings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / "mic_input_trimmed.wav"


def normalize_microphone_backend(backend: str = "auto") -> str:
    """Normalize and validate a microphone backend name."""
    normalized = (backend or "auto").strip().lower()
    if normalized not in SUPPORTED_MICROPHONE_BACKENDS:
        raise AudioInputError(
            "--mic-backend must be one of: "
            f"{', '.join(SUPPORTED_MICROPHONE_BACKENDS)}"
        )
    return normalized


def get_platform_default_microphone_backend() -> str | None:
    """Return the preferred microphone backend for the current OS."""
    system_name = platform.system().lower()
    if system_name == "windows":
        return "ffmpeg-dshow"
    if system_name == "linux":
        return "arecord"
    return None


def resolve_microphone_backend(backend: str = "auto") -> str:
    """Resolve an explicit or automatic microphone backend name."""
    normalized = normalize_microphone_backend(backend)
    if normalized != "auto":
        return normalized
    default_backend = get_platform_default_microphone_backend()
    if default_backend is None:
        raise AudioEnvironmentError(
            "microphone recording is not supported on this OS with --mic-backend auto"
        )
    return default_backend


def ensure_arecord_available() -> None:
    """Ensure arecord is available in the local environment."""
    if shutil.which("arecord") is None:
        raise AudioEnvironmentError("arecord is not installed or not found in PATH")


def ensure_ffmpeg_dshow_available() -> None:
    """Ensure the Windows ffmpeg DirectShow microphone backend is usable."""
    if platform.system().lower() != "windows":
        raise AudioEnvironmentError(
            "ffmpeg-dshow microphone backend is only supported on Windows"
        )
    ensure_ffmpeg_available()


def get_default_arecord_microphone_device() -> str:
    """Return a preferred arecord device string from detected capture devices."""
    ensure_arecord_available()

    try:
        result = subprocess.run(
            ["arecord", "-l"],
            check=True,
            capture_output=True,
            text=True,
            timeout=MICROPHONE_DEVICE_LIST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioEnvironmentError("microphone device listing timed out") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise AudioEnvironmentError(f"failed to list microphone devices: {message}") from exc

    for line in result.stdout.splitlines():
        match = re.search(r"card\s+(\d+).+device\s+(\d+)", line)
        if match:
            card_index, device_index = match.groups()
            return f"plughw:{card_index},{device_index}"

    return "default"


def list_ffmpeg_dshow_audio_devices() -> list[str]:
    """Return DirectShow audio device names reported by ffmpeg."""
    ensure_ffmpeg_dshow_available()
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            text=True,
            check=False,
            timeout=MICROPHONE_DEVICE_LIST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioEnvironmentError("DirectShow audio device listing timed out") from exc
    output = "\n".join(part for part in (result.stderr, result.stdout) if part)
    devices: list[str] = []
    in_audio_section = False
    for line in output.splitlines():
        typed_match = re.search(r'"([^"]+)"\s+\(audio\)', line)
        if typed_match:
            devices.append(typed_match.group(1))
            continue
        if "DirectShow audio devices" in line:
            in_audio_section = True
            continue
        if "DirectShow video devices" in line:
            in_audio_section = False
            continue
        if not in_audio_section or "Alternative name" in line:
            continue
        match = re.search(r'"([^"]+)"', line)
        if match:
            devices.append(match.group(1))
    return devices


def get_default_ffmpeg_dshow_microphone_device() -> str:
    """Return the first DirectShow audio input reported by ffmpeg."""
    devices = list_ffmpeg_dshow_audio_devices()
    if devices:
        return devices[0]
    raise AudioEnvironmentError(
        "no DirectShow audio capture device found. "
        "Run `ffmpeg -hide_banner -list_devices true -f dshow -i dummy` "
        "and pass a device name with --mic-device."
    )


def get_default_microphone_device(backend: str = "auto") -> str:
    """Return a default device string for the selected microphone backend."""
    resolved_backend = resolve_microphone_backend(backend)
    if resolved_backend == "arecord":
        return get_default_arecord_microphone_device()
    if resolved_backend == "ffmpeg-dshow":
        return get_default_ffmpeg_dshow_microphone_device()
    raise AudioEnvironmentError(f"unsupported microphone backend: {resolved_backend}")


def get_microphone_runtime_status() -> dict[str, object]:
    """Return a compact view of microphone backend support."""
    platform_system = platform.system() or "Unknown"
    default_backend = get_platform_default_microphone_backend()
    available_backends = [
        backend
        for backend in RECORDING_MICROPHONE_BACKENDS
        if _is_microphone_backend_available(backend)
    ]
    selected_backend = default_backend or "unsupported"
    selected_available = selected_backend in available_backends
    selected_device: str | None = None
    note: str | None = None
    if default_backend is None:
        note = "No automatic microphone backend is defined for this OS."
    elif not selected_available:
        note = (
            f"Default microphone backend '{default_backend}' is not currently available."
        )
    else:
        try:
            selected_device = get_default_microphone_device(selected_backend)
        except AudioEnvironmentError as exc:
            note = f"{note} {exc}" if note else str(exc)

    return {
        "platform_system": platform_system,
        "default_microphone_backend": default_backend,
        "selected_microphone_backend": selected_backend,
        "selected_microphone_device": selected_device,
        "selected_microphone_backend_available": selected_available,
        "available_microphone_backends": available_backends,
        "arecord_available": shutil.which("arecord") is not None,
        "ffmpeg_dshow_available": _is_microphone_backend_available("ffmpeg-dshow"),
        "microphone_note": note,
    }


def _is_microphone_backend_available(backend: str) -> bool:
    """Return whether a microphone backend has its local command prerequisites."""
    if backend == "arecord":
        return shutil.which("arecord") is not None
    if backend == "ffmpeg-dshow":
        return (
            platform.system().lower() == "windows"
            and shutil.which("ffmpeg") is not None
        )
    return False


def validate_duration(duration: int) -> None:
    """Validate microphone recording duration."""
    if duration <= 0:
        raise AudioInputError("microphone duration must be greater than 0 seconds")


def validate_vad_aggressiveness(aggressiveness: int) -> None:
    """Validate the supported WebRTC VAD aggressiveness range."""
    if aggressiveness not in {0, 1, 2, 3}:
        raise AudioInputError("VAD aggressiveness must be one of: 0, 1, 2, 3")


def trim_silence(
    input_path: Path,
    output_path: Path,
    silence_duration: float = 0.3,
    silence_threshold_db: float = -40.0,
) -> Path:
    """Trim leading and trailing silence from a wav file with ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_value = (
        "silenceremove="
        f"start_periods=1:start_duration={silence_duration}:start_threshold={silence_threshold_db}dB:"
        f"stop_periods=1:stop_duration={silence_duration}:stop_threshold={silence_threshold_db}dB"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_value,
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=MICROPHONE_TRIM_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioEnvironmentError("silence trimming timed out") from exc
    except subprocess.CalledProcessError as exc:
        message = _subprocess_error_message(exc)
        raise AudioEnvironmentError(f"silence trimming failed: {message}") from exc

    if not output_path.exists() or output_path.stat().st_size <= 1024:
        return input_path

    return output_path


def iter_vad_frames(audio_bytes: bytes, sample_rate: int, frame_ms: int = 30) -> list[bytes]:
    """Split PCM bytes into fixed-size frames for VAD."""
    bytes_per_sample = 2
    frame_size = int(sample_rate * frame_ms / 1000) * bytes_per_sample
    if frame_size <= 0:
        return []
    return [
        audio_bytes[index:index + frame_size]
        for index in range(0, len(audio_bytes), frame_size)
        if len(audio_bytes[index:index + frame_size]) == frame_size
    ]


def has_detectable_speech(
    audio_path: Path,
    aggressiveness: int = 2,
) -> bool:
    """Return whether WebRTC VAD detects speech in a wav clip."""
    validate_vad_aggressiveness(aggressiveness)
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            audio_bytes = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError) as exc:
        raise AudioEnvironmentError(f"speech detection failed: {exc}") from exc

    if sample_width != 2:
        raise AudioEnvironmentError(
            f"speech detection expects 16-bit PCM wav, got sample width {sample_width}"
        )
    if channels != 1:
        raise AudioEnvironmentError(
            f"speech detection expects mono wav, got {channels} channels"
        )
    if sample_rate not in {8000, 16000, 32000, 48000}:
        raise AudioEnvironmentError(
            f"speech detection does not support sample rate {sample_rate}"
        )

    vad = WebRtcVadAdapter(aggressiveness=aggressiveness)
    for frame in iter_vad_frames(audio_bytes, sample_rate=sample_rate):
        if vad.is_speech(frame, sample_rate):
            return True
    return False


def record_microphone_audio(
    output_path: Path,
    duration: int,
    device: str = "default",
    sample_rate: int = 16000,
    channels: int = 1,
    trim_silence_enabled: bool = True,
    backend: str = "auto",
) -> Path:
    """Record a fixed-duration wav file from the microphone."""
    validate_duration(duration)
    resolved_backend = resolve_microphone_backend(backend)
    if resolved_backend == LIVE_AEC_MICROPHONE_BACKEND:
        raise AudioInputError(
            "live AEC capture is in-memory only and cannot use the file recorder"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_backend == "arecord":
        _record_arecord_audio(
            output_path=output_path,
            duration=duration,
            device=device,
            sample_rate=sample_rate,
            channels=channels,
        )
    elif resolved_backend == "ffmpeg-dshow":
        _record_ffmpeg_dshow_audio(
            output_path=output_path,
            duration=duration,
            device=device,
            sample_rate=sample_rate,
            channels=channels,
        )
    else:
        raise AudioEnvironmentError(f"unsupported microphone backend: {resolved_backend}")

    if trim_silence_enabled:
        return trim_silence(
            input_path=output_path,
            output_path=get_trimmed_recording_path(),
        )

    return output_path


def _record_arecord_audio(
    output_path: Path,
    duration: int,
    device: str,
    sample_rate: int,
    channels: int,
) -> None:
    """Record a wav file using Linux arecord."""
    ensure_arecord_available()
    resolved_device = (
        get_default_arecord_microphone_device() if device == "default" else device
    )
    command = [
        "arecord",
        "-D",
        resolved_device,
        "-f",
        "S16_LE",
        "-c",
        str(channels),
        "-r",
        str(sample_rate),
        "-d",
        str(duration),
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=get_recording_timeout_seconds(duration),
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioEnvironmentError("microphone recording timed out") from exc
    except subprocess.CalledProcessError as exc:
        message = _subprocess_error_message(exc)
        raise AudioEnvironmentError(f"microphone recording failed: {message}") from exc


def _record_ffmpeg_dshow_audio(
    output_path: Path,
    duration: int,
    device: str,
    sample_rate: int,
    channels: int,
) -> None:
    """Record a wav file using Windows ffmpeg DirectShow."""
    ensure_ffmpeg_dshow_available()
    resolved_device = (
        get_default_ffmpeg_dshow_microphone_device()
        if device == "default"
        else device
    )
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "dshow",
        "-t",
        str(duration),
        "-i",
        f"audio={resolved_device}",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            text=True,
            timeout=get_recording_timeout_seconds(duration),
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioEnvironmentError(
            f"microphone recording timed out with ffmpeg-dshow device '{resolved_device}'"
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = _subprocess_error_message(exc)
        raise AudioEnvironmentError(
            f"microphone recording failed with ffmpeg-dshow device "
            f"'{resolved_device}': {message}"
        ) from exc


def capture_microphone_chunk(
    output_path: Path,
    duration: int,
    device: str = "default",
    sample_rate: int = 16000,
    channels: int = 1,
    trim_silence_enabled: bool = True,
    backend: str = "auto",
    aec_owner_selection: Mapping[str, Any] | None = None,
    live_aec_deadline_ms: int | None = None,
    live_aec_capture: Callable[..., LiveAecProcessedCapture] = (
        capture_live_aec_processed_pcm
    ),
) -> AudioChunk:
    """Capture one microphone chunk and wrap it for the pipeline."""
    resolved_backend = resolve_microphone_backend(backend)
    if resolved_backend == LIVE_AEC_MICROPHONE_BACKEND:
        validate_duration(duration)
        window_ms = duration * 1000
        deadline_ms = live_aec_deadline_ms or min(window_ms + 1000, 10_000)
        try:
            capture = live_aec_capture(
                owner_selection=aec_owner_selection,
                window_ms=window_ms,
                deadline_ms=deadline_ms,
            )
        except LiveAecCaptureError as exc:
            raise AudioEnvironmentError(exc.failure_class) from None
        return AudioChunk(
            path=None,
            source="microphone",
            pcm16=capture.pcm16,
            sample_rate=capture.sample_rate,
            storage_class=capture.storage_class,
            turn_input_authority=capture.turn_input_authority,
            turn_input_authority_class=capture.turn_input_authority_class,
        )
    audio_path = record_microphone_audio(
        output_path=output_path,
        duration=duration,
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        trim_silence_enabled=trim_silence_enabled,
        backend=resolved_backend,
    )
    return AudioChunk(path=audio_path, source="microphone")


def get_recording_timeout_seconds(duration: int) -> int:
    """Return a bounded subprocess timeout for one fixed-duration recording."""
    return duration + MICROPHONE_RECORDING_EXTRA_TIMEOUT_SECONDS


def _subprocess_error_message(exc: subprocess.CalledProcessError) -> str:
    """Return a safe message from a failed subprocess call."""
    stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    stdout = exc.stdout if isinstance(exc.stdout, str) else ""
    return stderr.strip() or stdout.strip() or str(exc)
