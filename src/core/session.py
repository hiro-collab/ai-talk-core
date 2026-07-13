"""Session helpers for realtime-style microphone transcription flows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.core.finalization import (
    maybe_finalize_on_interrupt,
    maybe_finalize_on_silence,
    normalize_transcript_text,
    should_mark_result_final,
)
from src.core.input_gate import (
    InputGate,
    InputGateEvent,
    InputGateState,
    UserSpeechCandidateEvidence,
)
from src.io.audio import AudioInputError
from src.core.pipeline import AudioBuffer, AudioChunk, TranscriptionPipeline, TranscriptionResult


@dataclass(frozen=True)
class MicLoopTuning:
    """Resolved tuning values for one microphone loop session."""

    vad_aggressiveness: int
    final_stable_seconds: int


@dataclass
class MicLoopState:
    """Mutable state for one microphone loop session."""

    buffer: AudioBuffer = field(default_factory=lambda: AudioBuffer(source="microphone"))
    previous_text: str | None = None
    repeat_count: int = 0
    last_spoken_result: TranscriptionResult | None = None
    finalized_text: str | None = None
    input_disabled_count: int = 0


class MicLoopSession:
    """Track mic-loop state separately from CLI or Web presentation concerns."""

    def __init__(
        self,
        pipeline: TranscriptionPipeline,
        tuning: MicLoopTuning,
        source: str = "microphone",
        input_gate: InputGate | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.tuning = tuning
        self.state = MicLoopState(buffer=AudioBuffer(source=source))
        self.input_gate = input_gate or InputGate()

    def input_gate_state(self) -> InputGateState:
        """Return the current input-gate state."""
        return self.input_gate.state

    def should_accept_input(self) -> bool:
        """Return whether the caller should capture and submit microphone input."""
        return self.input_gate.is_enabled()

    def set_input_enabled(
        self,
        enabled: bool,
        *,
        reason: str = "manual",
        source: str = "session",
    ) -> InputGateState:
        """Update whether this session should accept input."""
        return self.input_gate.set_input_enabled(
            enabled,
            reason=reason,
            source=source,
        )

    def update_input_gate(self, event: InputGateEvent) -> InputGateState:
        """Apply an input-gate event to this session."""
        return self.input_gate.update(event)

    def process_input_disabled(self, *, is_last_iteration: bool = False) -> TranscriptionResult:
        """Return a result for a loop tick where input capture was intentionally gated."""
        self.state.input_disabled_count += 1
        return TranscriptionResult(
            source=self.state.buffer.source,
            text="",
            is_final=is_last_iteration,
            chunk_count=len(self.state.buffer.chunks),
            input_enabled=False,
            input_gate_reason=self.input_gate.state.reason,
        )

    def process_chunk(
        self,
        chunk: AudioChunk,
        *,
        has_speech: bool,
        language: str | None,
        chunk_duration: int,
        is_last_iteration: bool,
        candidate_evidence: UserSpeechCandidateEvidence | None = None,
        turn_input_capability: object | None = None,
    ) -> TranscriptionResult:
        """Consume one captured chunk and return the current mic-loop result."""
        if not self.should_accept_input():
            chunk.clear()
            return self.process_input_disabled(is_last_iteration=is_last_iteration)
        if chunk.pcm16 is not None and not has_speech:
            if candidate_evidence is not None and turn_input_capability is not None:
                self.input_gate.consume_turn_input_capability(
                    turn_input_capability,
                    candidate_evidence,
                )
            chunk.clear()
            result = TranscriptionResult(
                source=self.state.buffer.source,
                text="",
                is_final=False,
                chunk_count=len(self.state.buffer.chunks),
                is_silence=True,
            )
        else:
            self.state.buffer.append(chunk)
        if has_speech:
            if chunk.pcm16 is not None:
                if not self.input_gate.consume_turn_input_capability(
                    turn_input_capability,
                    candidate_evidence,
                ):
                    self.state.buffer.chunks.pop()
                    chunk.clear()
                    raise AudioInputError(
                        "private live candidate lacks InputGate authority"
                    )
                result = self.pipeline._transcribe_private_buffer_result(
                    self.state.buffer,
                    language=language,
                    is_final=False,
                )
            else:
                result = self.pipeline.transcribe_buffer_result(
                    self.state.buffer,
                    language=language,
                    is_final=False,
                )
        elif chunk.pcm16 is None:
            result = TranscriptionResult(
                source=self.state.buffer.source,
                text="",
                is_final=False,
                chunk_count=len(self.state.buffer.chunks),
                is_silence=True,
            )
        result = maybe_finalize_on_silence(
            result=result,
            last_spoken_result=self.state.last_spoken_result,
            repeat_count=self.state.repeat_count,
            finalized_text=self.state.finalized_text,
        )
        normalized_text = normalize_transcript_text(result.text)
        if normalized_text and not result.is_silence:
            if normalized_text == self.state.previous_text:
                self.state.repeat_count += 1
            else:
                self.state.repeat_count = 1
            self.state.previous_text = normalized_text
            self.state.last_spoken_result = result
        else:
            self.state.repeat_count = 0
        if should_mark_result_final(
            result,
            self.state.repeat_count,
            is_last_iteration,
            chunk_duration,
            self.tuning.final_stable_seconds,
        ):
            result = replace(result, is_final=True)
        if result.is_final and not result.is_silence and normalized_text:
            self.state.finalized_text = normalized_text
        return result

    def finalize_on_interrupt(self) -> TranscriptionResult | None:
        """Return one final flush result when the mic-loop is interrupted."""
        return maybe_finalize_on_interrupt(
            last_spoken_result=self.state.last_spoken_result,
            finalized_text=self.state.finalized_text,
            chunk_count=len(self.state.buffer.chunks),
        )
