from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from yt_helper.transcription import (
    DEFAULT_ASR_MODEL,
    DEFAULT_FORCED_ALIGNER_MODEL,
    TranscriptionError,
    _clear_asr_model,
    _format_timestamp,
    _format_transcript,
    _get_asr_model,
    _transcribe_with_model,
    transcribe_media,
)

# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------


def test_format_timestamp_zero():
    assert _format_timestamp(0.0) == "0.0s"


def test_format_timestamp_fractional():
    assert _format_timestamp(12.345) == "12.3s"


def test_format_timestamp_integer():
    assert _format_timestamp(60.0) == "60.0s"


# ---------------------------------------------------------------------------
# _format_transcript
# ---------------------------------------------------------------------------


def test_format_qwen_transcript_preserves_mixed_language_and_punctuation():
    result = SimpleNamespace(
        text="今天 deploy the backend\uff0c然後測試 Qwen。",
        time_stamps=[
            SimpleNamespace(text="今", start_time=0.0),
            SimpleNamespace(text="天", start_time=0.2),
            SimpleNamespace(text="deploy", start_time=0.8),
            SimpleNamespace(text="the", start_time=1.2),
            SimpleNamespace(text="backend", start_time=1.5),
            SimpleNamespace(text="然", start_time=3.0),
            SimpleNamespace(text="後", start_time=3.2),
            SimpleNamespace(text="測", start_time=3.5),
            SimpleNamespace(text="試", start_time=3.7),
            SimpleNamespace(text="Qwen", start_time=4.0),
        ],
    )

    assert _format_transcript(result).splitlines() == [
        "[0.0s] 今天 deploy the backend\uff0c",
        "[3.0s] 然後測試 Qwen。",
    ]


def test_format_qwen_transcript_without_alignment_uses_zero_timestamp():
    result = SimpleNamespace(text="中 English 混合", time_stamps=None)

    assert _format_transcript(result) == "[0.0s] 中 English 混合"


def test_format_qwen_transcript_empty_text_raises():
    result = SimpleNamespace(text="", time_stamps=[])

    with pytest.raises(TranscriptionError):
        _format_transcript(result)


# ---------------------------------------------------------------------------
# Qwen ASR backend
# ---------------------------------------------------------------------------


def test_get_asr_model_loads_qwen_with_forced_aligner():
    fake_model = object()
    with (
        patch(
            "qwen_asr.Qwen3ASRModel.from_pretrained",
            return_value=fake_model,
        ) as from_pretrained,
        patch(
            "yt_helper.transcription._resolve_device",
            return_value=("cpu", torch.float32, "CPU"),
        ),
    ):
        _clear_asr_model()
        model, device_label = _get_asr_model()

    assert model is fake_model
    assert device_label == "CPU"
    from_pretrained.assert_called_once_with(
        DEFAULT_ASR_MODEL,
        dtype=torch.float32,
        device_map="cpu",
        forced_aligner=DEFAULT_FORCED_ALIGNER_MODEL,
        forced_aligner_kwargs={"dtype": torch.float32, "device_map": "cpu"},
        max_inference_batch_size=1,
        max_new_tokens=2048,
    )
    _clear_asr_model()


def test_transcribe_with_model_keeps_language_detection_automatic():
    qwen_result = SimpleNamespace(
        text="你好 hello",
        language="Chinese,English",
        time_stamps=[],
    )
    model = Mock()
    model.transcribe.return_value = [qwen_result]
    input_path = Path("audio.wav")

    assert _transcribe_with_model(model, input_path) is qwen_result
    model.transcribe.assert_called_once_with(
        audio="audio.wav",
        language=None,
        return_time_stamps=True,
    )


def test_transcribe_media_writes_qwen_result(tmp_path):
    input_path = tmp_path / "audio.wav"
    input_path.touch()
    qwen_result = SimpleNamespace(
        text="你好 hello。",
        language="Chinese,English",
        time_stamps=[],
    )
    model = Mock()
    model.transcribe.return_value = [qwen_result]

    with patch(
        "yt_helper.transcription._get_asr_model",
        return_value=(model, "CPU"),
    ):
        result = transcribe_media(input_path)

    assert result.source == "qwen"
    assert result.device_label == "CPU"
    assert result.language == "Chinese,English"
    assert result.output_path.read_text(encoding="utf-8") == "[0.0s] 你好 hello。\n"
