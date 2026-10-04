import gc
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_SEGMENT_MAX_CHARS = 36
SEGMENT_BREAK_PUNCTUATION = frozenset(
    "\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a,.!?;"
)


class TranscriptionError(RuntimeError):
    """Raised when transcription cannot be completed."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Represents a completed transcription job."""

    output_path: Path
    source: str
    device_label: str | None = None
    language: str | None = None


def _resolve_device() -> tuple[str, Any, str]:
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda:0", dtype, "CUDA"
    return "cpu", torch.float32, "CPU"


def _clear_asr_model() -> None:
    import torch

    _get_asr_model.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@cache
def _get_asr_model() -> tuple[Any, str]:
    from qwen_asr import Qwen3ASRModel

    device_map, torch_dtype, device_label = _resolve_device()
    model = Qwen3ASRModel.from_pretrained(
        DEFAULT_ASR_MODEL,
        dtype=torch_dtype,
        device_map=device_map,
        forced_aligner=DEFAULT_FORCED_ALIGNER_MODEL,
        forced_aligner_kwargs={"dtype": torch_dtype, "device_map": device_map},
        max_inference_batch_size=1,
        max_new_tokens=2048,
    )
    return model, device_label


def _format_timestamp(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message

    return exc.__class__.__name__


def _format_transcript(result: Any) -> str:
    text = " ".join(str(getattr(result, "text", "")).split())
    if not text:
        raise TranscriptionError("The model did not return any transcription text.")

    aligned_positions = _locate_aligned_tokens(
        text,
        getattr(result, "time_stamps", None) or (),
    )
    if not aligned_positions:
        return f"[0.0s] {text}"

    parts = _split_text_with_offsets(text)
    lines: list[str] = []
    previous_start = aligned_positions[0][1]
    token_index = 0
    for part_start, part_end, part_text in parts:
        while (
            token_index < len(aligned_positions)
            and aligned_positions[token_index][0] < part_start
        ):
            previous_start = aligned_positions[token_index][1]
            token_index += 1

        if (
            token_index < len(aligned_positions)
            and aligned_positions[token_index][0] < part_end
        ):
            previous_start = aligned_positions[token_index][1]

        lines.append(f"[{_format_timestamp(previous_start)}] {part_text}")

    return "\n".join(lines)


def _split_text_with_offsets(text: str) -> list[tuple[int, int, str]]:
    parts: list[tuple[int, int, str]] = []
    part_start = 0

    def flush(raw_start: int, raw_part: str) -> None:
        clean_part = raw_part.strip()
        if not clean_part:
            return
        clean_start = raw_start + len(raw_part) - len(raw_part.lstrip())
        parts.append((clean_start, clean_start + len(clean_part), clean_part))

    for index, char in enumerate(text):
        if (
            char not in SEGMENT_BREAK_PUNCTUATION
            and index - part_start + 1 < DEFAULT_SEGMENT_MAX_CHARS
        ):
            continue

        flush(part_start, text[part_start : index + 1])
        part_start = index + 1

    flush(part_start, text[part_start:])
    return parts


def _locate_aligned_tokens(
    text: str,
    time_stamps: Any,
) -> list[tuple[int, float]]:
    positions: list[tuple[int, float]] = []
    search_text = text.casefold()
    search_start = 0
    for item in time_stamps:
        token = str(getattr(item, "text", "")).strip()
        if not token:
            continue

        position = search_text.find(token.casefold(), search_start)
        if position < 0:
            continue

        try:
            start_time = float(item.start_time)
        except (TypeError, ValueError):
            continue

        positions.append((position, start_time))
        search_start = position + len(token)

    return positions


def _transcribe_with_model(model: Any, input_path: Path) -> Any:
    results = model.transcribe(
        audio=str(input_path),
        language=None,
        return_time_stamps=True,
    )
    if not isinstance(results, list) or len(results) != 1:
        raise TranscriptionError("The Qwen ASR model returned an unexpected result.")

    return results[0]


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
        model, device_label = _get_asr_model()
    except Exception as exc:
        _clear_asr_model()
        raise TranscriptionError(_format_exception_message(exc)) from exc

    try:
        result = _transcribe_with_model(model, resolved_input_path)
    except Exception as exc:
        raise TranscriptionError(_format_exception_message(exc)) from exc

    transcript = _format_transcript(result)
    resolved_output_path.write_text(f"{transcript}\n", encoding="utf-8")

    return TranscriptionResult(
        output_path=resolved_output_path,
        source="qwen",
        device_label=device_label,
        language=str(getattr(result, "language", "")).strip() or None,
    )
