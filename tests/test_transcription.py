import pytest

from yt_helper.transcription import (
    TranscriptionError,
    _format_segmented_transcript,
    _format_timestamp,
    _format_transcript,
    _split_segment_text,
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
# _split_segment_text
# ---------------------------------------------------------------------------


def test_split_segment_short_text_is_unchanged():
    # 11 chars — well under DEFAULT_SEGMENT_MAX_CHARS (36)
    result = _split_segment_text(0.0, 5.0, "Short text.")
    assert result == [(0.0, "Short text.")]


def test_split_segment_normalises_whitespace():
    result = _split_segment_text(0.0, 5.0, "  hello   world  ")
    assert result == [(0.0, "hello world")]


def test_split_segment_long_text_no_end_returns_whole():
    long_text = "A" * 80
    result = _split_segment_text(0.0, None, long_text)
    assert result == [(0.0, long_text)]


def test_split_segment_long_text_end_lte_start_returns_whole():
    long_text = "A" * 80
    result = _split_segment_text(5.0, 5.0, long_text)
    assert result == [(5.0, long_text)]


def test_split_segment_long_text_splits_at_punctuation():
    # Two clearly separated sentences, each < 36 chars
    text = "Hello world, nice to see you today. And goodbye now, farewell."
    result = _split_segment_text(0.0, 10.0, text)
    # Should produce more than one segment
    assert len(result) > 1
    # First segment starts at 0.0
    assert result[0][0] == 0.0
    # Subsequent starts must be increasing
    starts = [seg[0] for seg in result]
    assert starts == sorted(starts)


def test_split_segment_timestamps_are_proportional():
    # Two equal-length parts separated by punctuation
    text = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA. BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB."
    #       34 A's + ". " = 36 chars first part, then 34 B's + "."
    result = _split_segment_text(0.0, 20.0, text)
    assert len(result) == 2
    # First segment at t=0.0; second somewhere in the middle
    assert result[0][0] == pytest.approx(0.0)
    assert 0.0 < result[1][0] < 20.0


def test_split_segment_all_text_is_preserved():
    text = "The quick brown fox jumps over it. And even more text follows here."
    result = _split_segment_text(0.0, 10.0, text)
    reconstructed = " ".join(seg[1] for seg in result)
    # Every word from original must appear in the reconstructed output
    for word in text.replace(".", "").replace(",", "").split():
        assert word in reconstructed


# ---------------------------------------------------------------------------
# _format_transcript
# ---------------------------------------------------------------------------


def test_format_transcript_no_chunks_uses_text():
    result = _format_transcript({"text": "Hello world."})
    assert result == "[0.0s] Hello world."


def test_format_transcript_no_chunks_strips_whitespace():
    result = _format_transcript({"text": "  Hello  "})
    assert result == "[0.0s] Hello"


def test_format_transcript_empty_text_raises():
    with pytest.raises(TranscriptionError):
        _format_transcript({"text": ""})


def test_format_transcript_missing_text_raises():
    with pytest.raises(TranscriptionError):
        _format_transcript({})


def test_format_transcript_empty_chunks_list_uses_text():
    # Empty list is falsy — falls through to text path
    result = _format_transcript({"chunks": [], "text": "Fallback text."})
    assert result == "[0.0s] Fallback text."


def test_format_transcript_with_chunks_delegates():
    chunks = [{"text": "Hello.", "timestamp": (0.0, 2.0)}]
    result = _format_transcript({"chunks": chunks})
    assert "[0.0s]" in result
    assert "Hello." in result


# ---------------------------------------------------------------------------
# _format_segmented_transcript
# ---------------------------------------------------------------------------


def test_segmented_transcript_single_chunk():
    chunks = [{"text": "Hello.", "timestamp": (0.0, 2.0)}]
    result = _format_segmented_transcript(chunks)
    assert result == "[0.0s] Hello."


def test_segmented_transcript_multiple_chunks():
    chunks = [
        {"text": "First.", "timestamp": (0.0, 2.0)},
        {"text": "Second.", "timestamp": (2.0, 4.0)},
    ]
    lines = _format_segmented_transcript(chunks).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[0.0s]")
    assert lines[1].startswith("[2.0s]")


def test_segmented_transcript_skips_empty_text_chunks():
    chunks = [
        {"text": "", "timestamp": (0.0, 1.0)},
        {"text": "Only this.", "timestamp": (1.0, 3.0)},
    ]
    result = _format_segmented_transcript(chunks)
    assert result == "[1.0s] Only this."


def test_segmented_transcript_all_empty_raises():
    chunks = [{"text": "", "timestamp": (0.0, 1.0)}]
    with pytest.raises(TranscriptionError):
        _format_segmented_transcript(chunks)


def test_segmented_transcript_missing_timestamp_defaults_to_zero():
    chunks = [{"text": "No timestamp here."}]
    result = _format_segmented_transcript(chunks)
    assert result.startswith("[0.0s]")


def test_segmented_transcript_invalid_timestamp_type_raises():
    chunks = [{"text": "Hello.", "timestamp": "bad-value"}]
    with pytest.raises(TranscriptionError):
        _format_segmented_transcript(chunks)


def test_segmented_transcript_empty_tuple_timestamp_raises():
    chunks = [{"text": "Hello.", "timestamp": ()}]
    with pytest.raises(TranscriptionError):
        _format_segmented_transcript(chunks)
