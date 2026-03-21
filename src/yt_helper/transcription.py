import gc
import logging
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import pipeline

DEFAULT_ASR_MODEL = "openai/whisper-large-v3"
DEFAULT_SEGMENT_MAX_CHARS = 36
SEGMENT_BREAK_PUNCTUATION = (
    "\u3002",
    "\uff01",
    "\uff1f",
    "\uff1b",
    "\uff1a",
    ",",
    ".",
    "!",
    "?",
    ";",
)

_ASR_PIPELINE: Any | None = None
_ASR_DEVICE_LABEL: str | None = None
_SUPPRESSED_TRANSFORMERS_WARNINGS = (
    "A custom logits processor of type <class "
    "'transformers.generation.logits_process.SuppressTokensLogitsProcessor'>",
    "A custom logits processor of type <class "
    "'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'>",
)


class TranscriptionError(RuntimeError):
    """Raised when transcription cannot be completed."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Represents a completed transcription job."""

    output_path: Path
    source: str
    device_label: str | None = None


class _KnownTransformersWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            warning in message for warning in _SUPPRESSED_TRANSFORMERS_WARNINGS
        )


@contextmanager
def _suppress_known_transformers_warnings() -> Any:
    logger = logging.getLogger("transformers.generation.utils")
    warning_filter = _KnownTransformersWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def _resolve_device() -> tuple[int, torch.dtype, str]:
    if torch.cuda.is_available():
        return 0, torch.float16, "CUDA"
    return -1, torch.float32, "CPU"


def _clear_asr_pipeline() -> None:
    global _ASR_PIPELINE, _ASR_DEVICE_LABEL

    _ASR_PIPELINE = None
    _ASR_DEVICE_LABEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_asr_pipeline() -> tuple[Any, str]:
    global _ASR_PIPELINE, _ASR_DEVICE_LABEL

    if _ASR_PIPELINE is not None and _ASR_DEVICE_LABEL is not None:
        return _ASR_PIPELINE, _ASR_DEVICE_LABEL

    device, torch_dtype, device_label = _resolve_device()
    _ASR_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model=DEFAULT_ASR_MODEL,
        device=device,
        dtype=torch_dtype,
    )
    _ASR_DEVICE_LABEL = device_label
    return _ASR_PIPELINE, _ASR_DEVICE_LABEL


def _format_timestamp(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message

    return exc.__class__.__name__


def _format_transcript(result: dict[str, Any]) -> str:
    chunks = result.get("chunks")
    if not chunks:
        text = str(result.get("text", "")).strip()
        if not text:
            raise TranscriptionError("The model did not return any transcription text.")
        return f"[0.0s] {text}"

    return _format_segmented_transcript(chunks)


def _format_segmented_transcript(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        timestamp = chunk.get("timestamp")
        if timestamp is None:
            timestamp = (0.0, None)
        if not isinstance(timestamp, tuple) or not timestamp:
            raise TranscriptionError(
                "The ASR pipeline returned an unexpected timestamp format."
            )

        start = float(timestamp[0])
        end = (
            float(timestamp[1])
            if len(timestamp) > 1 and timestamp[1] is not None
            else None
        )
        for segment_start, segment_text in _split_segment_text(start, end, text):
            lines.append(f"[{_format_timestamp(segment_start)}] {segment_text}")

    if not lines:
        raise TranscriptionError(
            "The model returned chunks, but all chunk text was empty."
        )

    return "\n".join(lines)


def _split_segment_text(
    start: float,
    end: float | None,
    text: str,
) -> list[tuple[float, str]]:
    clean_text = " ".join(text.split())
    if len(clean_text) <= DEFAULT_SEGMENT_MAX_CHARS:
        return [(start, clean_text)]

    parts: list[str] = []
    current = ""
    for char in clean_text:
        current += char
        should_split = (
            char in SEGMENT_BREAK_PUNCTUATION
            or len(current) >= DEFAULT_SEGMENT_MAX_CHARS
        )
        if should_split:
            part = current.strip()
            if part:
                parts.append(part)
            current = ""

    if current.strip():
        parts.append(current.strip())

    if len(parts) <= 1 or end is None or end <= start:
        return [(start, clean_text)]

    total_chars = sum(len(part) for part in parts)
    duration = end - start
    split_segments: list[tuple[float, str]] = []
    elapsed = 0.0
    for part in parts:
        split_start = start + elapsed
        split_segments.append((split_start, part))
        elapsed += duration * (len(part) / total_chars)

    return split_segments


def _detect_language(pipe: Any, input_path: Path) -> str | None:
    """Detect spoken language from the first 30 seconds of audio."""
    try:
        with wave.open(str(input_path), "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = min(int(framerate * 30), wf.getnframes())
            raw = wf.readframes(n_frames)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        audio /= np.iinfo(dtype).max
        if nchannels > 1:
            audio = audio.reshape(-1, nchannels).mean(axis=1)

        inputs = pipe.feature_extractor(
            audio, sampling_rate=framerate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(pipe.device)
        with torch.no_grad():
            language_token_ids = pipe.model.detect_language(input_features)

        decoded = pipe.tokenizer.batch_decode(
            language_token_ids, skip_special_tokens=False
        )
        return decoded[0].strip("<>|")
    except Exception:
        return None


def _transcribe_with_pipeline(pipe: Any, input_path: Path) -> dict[str, Any]:
    language = _detect_language(pipe, input_path)
    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if language is not None:
        generate_kwargs["language"] = language

    result = pipe(
        str(input_path),
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=[5, 0],
        batch_size=8,
        generate_kwargs=generate_kwargs,
    )
    if not isinstance(result, dict):
        raise TranscriptionError("The ASR pipeline returned an unexpected result.")

    return result


def transcribe_media(
    input_path: Path,
    output_path: Path | None = None,
) -> TranscriptionResult:
    """Transcribe a local media file into timestamped plain text."""
    resolved_input_path = input_path.expanduser().resolve()
    if not resolved_input_path.is_file():
        raise TranscriptionError(
            f"Input media file was not found: {resolved_input_path}"
        )

    resolved_output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else resolved_input_path.with_suffix(".txt")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pipe, device_label = _get_asr_pipeline()
        with _suppress_known_transformers_warnings():
            result = _transcribe_with_pipeline(pipe, resolved_input_path)
    except Exception as exc:
        _clear_asr_pipeline()
        raise TranscriptionError(_format_exception_message(exc)) from exc

    transcript = _format_transcript(result)
    resolved_output_path.write_text(f"{transcript}\n", encoding="utf-8")

    return TranscriptionResult(
        output_path=resolved_output_path,
        source="whisper",
        device_label=device_label,
    )
