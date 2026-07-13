"""Shared transcription pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from src.io.audio import (
    AudioInputError,
    load_transcription_model,
    transcribe_file,
)


@dataclass(frozen=True)
class AudioChunk:
    """A captured audio chunk ready for transcription."""

    path: Path | None
    source: str
    pcm16: bytearray | None = None
    sample_rate: int | None = None
    storage_class: str = "file"
    turn_input_authority: bool = True
    turn_input_authority_class: str = "file_input"

    def __post_init__(self) -> None:
        has_path = self.path is not None
        has_pcm = self.pcm16 is not None
        if has_path == has_pcm:
            raise AudioInputError("audio chunk must contain exactly one input source")
        if has_pcm and (
            self.sample_rate != 16_000
            or self.storage_class != "in_memory_ephemeral"
            or (
                self.turn_input_authority,
                self.turn_input_authority_class,
            )
            != (False, "processed_near_end_observation_only")
        ):
            raise AudioInputError("in-memory audio chunk metadata is invalid")
        if has_path and self.turn_input_authority_class != "file_input":
            raise AudioInputError("file audio chunk authority metadata is invalid")

    def clear(self) -> None:
        """Clear any ephemeral PCM owned by this chunk."""
        if self.pcm16 is not None:
            self.pcm16[:] = b"\x00" * len(self.pcm16)


@dataclass(frozen=True)
class TranscriptionResult:
    """A normalized transcription result for realtime-style flows."""

    source: str
    text: str
    is_final: bool
    chunk_count: int
    is_silence: bool = False
    input_enabled: bool = True
    input_gate_reason: str = ""


@dataclass
class AudioBuffer:
    """A simple ordered buffer of captured audio chunks."""

    source: str
    chunks: list[AudioChunk] = field(default_factory=list)

    def append(self, chunk: AudioChunk) -> None:
        """Append a chunk to the buffer."""
        if chunk.source != self.source:
            raise AudioInputError(
                f"audio chunk source mismatch: expected {self.source}, got {chunk.source}"
            )
        self.chunks.append(chunk)

    def latest_chunk(self) -> AudioChunk:
        """Return the latest chunk in the buffer."""
        if not self.chunks:
            raise AudioInputError("audio buffer is empty")
        return self.chunks[-1]


class TranscriptionPipeline:
    """Keep a loaded Whisper model and transcribe audio chunks."""

    def __init__(self, model_name: str = "small") -> None:
        self.model_name = model_name
        self.model: Any = load_transcription_model(model_name=model_name)

    def transcribe_chunk(self, chunk: AudioChunk, language: str | None = None) -> str:
        """Transcribe a captured audio chunk."""
        if chunk.pcm16 is not None:
            chunk.clear()
            raise AudioInputError(
                "live AEC in-memory PCM is observation-only"
            )
        if chunk.path is None:
            raise AudioInputError("audio chunk source is unavailable")
        return transcribe_file(audio_path=chunk.path, model=self.model, language=language)

    def transcribe_buffer(self, buffer: AudioBuffer, language: str | None = None) -> str:
        """Transcribe the latest chunk from a buffer."""
        return self.transcribe_chunk(buffer.latest_chunk(), language=language)

    def transcribe_buffer_result(
        self,
        buffer: AudioBuffer,
        language: str | None = None,
        is_final: bool = False,
    ) -> TranscriptionResult:
        """Transcribe the latest chunk and return a realtime-style result."""
        text = self.transcribe_buffer(buffer, language=language)
        return TranscriptionResult(
            source=buffer.source,
            text=text,
            is_final=is_final,
            chunk_count=len(buffer.chunks),
        )


_PIPELINE_CACHE: dict[str, TranscriptionPipeline] = {}
_PIPELINE_CACHE_LOCK = threading.Lock()


def get_cached_transcription_pipeline(model_name: str = "small") -> TranscriptionPipeline:
    """Return one process-local transcription pipeline per Whisper model name."""
    with _PIPELINE_CACHE_LOCK:
        pipeline = _PIPELINE_CACHE.get(model_name)
        if pipeline is None:
            pipeline = TranscriptionPipeline(model_name=model_name)
            _PIPELINE_CACHE[model_name] = pipeline
        return pipeline


def clear_transcription_pipeline_cache() -> None:
    """Clear cached pipelines, primarily for tests."""
    with _PIPELINE_CACHE_LOCK:
        _PIPELINE_CACHE.clear()
