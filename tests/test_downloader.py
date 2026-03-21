from yt_dlp.utils import DownloadError

from yt_helper.downloader import (
    DownloadProgressReporter,
    _build_download_error_message,
    _format_bytes,
    _format_seconds,
)

# ---------------------------------------------------------------------------
# _format_bytes
# ---------------------------------------------------------------------------


def test_format_bytes_none():
    assert _format_bytes(None) == "unknown size"


def test_format_bytes_zero():
    assert _format_bytes(0) == "0.0 B"


def test_format_bytes_below_kib():
    assert _format_bytes(1023) == "1023.0 B"


def test_format_bytes_exact_kib():
    assert _format_bytes(1024) == "1.0 KiB"


def test_format_bytes_mib():
    assert _format_bytes(1024 * 1024) == "1.0 MiB"


def test_format_bytes_gib():
    assert _format_bytes(1024**3) == "1.0 GiB"


def test_format_bytes_fractional():
    assert _format_bytes(1536) == "1.5 KiB"


# ---------------------------------------------------------------------------
# _format_seconds
# ---------------------------------------------------------------------------


def test_format_seconds_none():
    assert _format_seconds(None) == "unknown ETA"


def test_format_seconds_zero():
    assert _format_seconds(0) == "00:00"


def test_format_seconds_below_minute():
    assert _format_seconds(59) == "00:59"


def test_format_seconds_exact_minute():
    assert _format_seconds(60) == "01:00"


def test_format_seconds_mixed_minutes():
    assert _format_seconds(125) == "02:05"


def test_format_seconds_exact_hour():
    assert _format_seconds(3600) == "1:00:00"


def test_format_seconds_hours_minutes_seconds():
    assert _format_seconds(3661) == "1:01:01"


def test_format_seconds_negative_clamped_to_zero():
    assert _format_seconds(-5) == "00:00"


# ---------------------------------------------------------------------------
# _build_download_error_message
# ---------------------------------------------------------------------------


def _err(msg: str) -> DownloadError:
    return DownloadError(msg)


def test_error_message_permission_denied():
    result = _build_download_error_message(_err("Permission denied: /some/path"))
    assert "permissions" in result.lower()


def test_error_message_operation_not_permitted():
    result = _build_download_error_message(_err("Operation not permitted"))
    assert "permissions" in result.lower()


def test_error_message_ssl():
    result = _build_download_error_message(_err("certificate verify failed: ..."))
    assert "SSL" in result or "ssl" in result.lower()


def test_error_message_ffmpeg_not_found():
    result = _build_download_error_message(_err("ffmpeg not found"))
    assert "ffmpeg" in result.lower()


def test_error_message_timeout():
    result = _build_download_error_message(_err("timed out after 30 seconds"))
    assert "network" in result.lower() or "timed out" in result.lower()


def test_error_message_sign_in():
    result = _build_download_error_message(_err("Sign in to confirm your age"))
    assert "verification" in result.lower() or "rejected" in result.lower()


def test_error_message_requested_format_not_available():
    result = _build_download_error_message(
        _err("The requested format is not available")
    )
    assert "verification" in result.lower() or "rejected" in result.lower()


def test_error_message_generic_passthrough():
    result = _build_download_error_message(_err("Something completely unexpected"))
    assert result == "Something completely unexpected"


# ---------------------------------------------------------------------------
# DownloadProgressReporter._extract_download_key
# ---------------------------------------------------------------------------


def test_extract_key_prefers_filename():
    data = {"filename": "/tmp/video.mp4", "tmpfilename": "/tmp/video.part"}
    assert DownloadProgressReporter._extract_download_key(data) == "/tmp/video.mp4"


def test_extract_key_falls_back_to_tmpfilename():
    data = {"tmpfilename": "/tmp/video.part"}
    assert DownloadProgressReporter._extract_download_key(data) == "/tmp/video.part"


def test_extract_key_falls_back_to_info_dict():
    data = {"info_dict": {"id": "abc123", "format_id": "137"}}
    assert DownloadProgressReporter._extract_download_key(data) == "abc123:137"


def test_extract_key_falls_back_to_literal_download():
    assert DownloadProgressReporter._extract_download_key({}) == "download"


def test_extract_key_ignores_empty_filename():
    data = {"filename": "", "tmpfilename": "/tmp/video.part"}
    assert DownloadProgressReporter._extract_download_key(data) == "/tmp/video.part"
