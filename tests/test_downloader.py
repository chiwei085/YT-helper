from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from yt_dlp.utils import DownloadError

from yt_helper import downloader
from yt_helper.downloader import (
    AudioFormat,
    DownloadProgressReporter,
    PlaylistInfo,
    VideoDownloadError,
    _build_download_error_message,
    _format_bytes,
    _format_seconds,
    download_audio,
    fetch_playlist_info,
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


FORBIDDEN_ERROR = "ERROR: unable to download video data: HTTP Error 403: Forbidden"


class _FakeManagedYoutubeDL:
    """Record each attempt's options, failing the first `failures` attempts."""

    attempts: ClassVar[list[dict]] = []
    failures: ClassVar[int] = 0
    error: ClassVar[str] = FORBIDDEN_ERROR

    def __init__(self, options, **_kwargs):
        type(self).attempts.append(options)

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def extract_info(self, _url, **_kwargs):
        if len(type(self).attempts) <= type(self).failures:
            raise DownloadError(type(self).error)
        return {"requested_downloads": [{"filepath": "/tmp/downloaded.wav"}]}


def _fake_youtube_dl(
    failures: int = 0,
    error: str = FORBIDDEN_ERROR,
) -> type[_FakeManagedYoutubeDL]:
    return type(
        "_Fake",
        (_FakeManagedYoutubeDL,),
        {"attempts": [], "failures": failures, "error": error},
    )


def test_download_media_retries_youtube_403_with_compatibility_client(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(downloader, "_use_compatibility_client", False)
    fake = _fake_youtube_dl(failures=1)

    with patch("yt_helper.downloader.ManagedYoutubeDL", fake):
        result = download_audio(
            "https://youtu.be/example",
            tmp_path,
            audio_format=AudioFormat.WAV,
        )

    assert result == Path("/tmp/downloaded.wav")
    assert len(fake.attempts) == 2
    assert "extractor_args" not in fake.attempts[0]
    assert fake.attempts[0]["format"] == "bestaudio/best"
    assert fake.attempts[1]["format"] == "bestaudio/best"
    assert fake.attempts[1]["extractor_args"] == {
        "youtube": {"player_client": ["tv_simply"]}
    }
    assert fake.attempts[1]["js_runtimes"] == {"node": {}}
    assert fake.attempts[1]["remote_components"] == {"ejs:github"}


def test_download_media_reuses_compatibility_client_after_first_fallback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(downloader, "_use_compatibility_client", True)
    fake = _fake_youtube_dl()

    with patch("yt_helper.downloader.ManagedYoutubeDL", fake):
        download_audio("https://youtu.be/example", tmp_path)

    assert len(fake.attempts) == 1
    assert fake.attempts[0]["extractor_args"] == {
        "youtube": {"player_client": ["tv_simply"]}
    }


def test_download_media_reports_non_403_failures_without_retrying(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(downloader, "_use_compatibility_client", False)
    fake = _fake_youtube_dl(
        failures=1,
        error="ERROR: requested format is not available",
    )

    with (
        patch("yt_helper.downloader.ManagedYoutubeDL", fake),
        pytest.raises(VideoDownloadError, match="rejected the requested media"),
    ):
        download_audio("https://youtu.be/example", tmp_path)

    assert len(fake.attempts) == 1
    assert downloader._use_compatibility_client is False


# ---------------------------------------------------------------------------
# Playlist metadata
# ---------------------------------------------------------------------------


def test_fetch_playlist_info_returns_playable_entries():
    playlist_data = {
        "_type": "playlist",
        "title": "Mixed language lessons",
        "entries": [
            {
                "title": "First lesson",
                "url": "https://www.youtube.com/watch?v=video-one",
            },
            None,
            {
                "title": "Second lesson",
                "url": "https://www.youtube.com/watch?v=video-two",
            },
        ],
    }
    with patch("yt_helper.downloader.YoutubeDL") as youtube_dl:
        youtube_dl.return_value.__enter__.return_value.extract_info.return_value = (
            playlist_data
        )
        result = fetch_playlist_info("https://youtube.test/playlist")

    assert isinstance(result, PlaylistInfo)
    assert result.title == "Mixed language lessons"
    assert [(entry.title, entry.url) for entry in result.entries] == [
        ("First lesson", "https://www.youtube.com/watch?v=video-one"),
        ("Second lesson", "https://www.youtube.com/watch?v=video-two"),
    ]


def test_fetch_playlist_info_rejects_empty_playlist():
    with patch("yt_helper.downloader.YoutubeDL") as youtube_dl:
        youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "_type": "playlist",
            "title": "Empty",
            "entries": [],
        }

        try:
            fetch_playlist_info("https://youtube.test/playlist")
        except VideoDownloadError as exc:
            assert "does not contain any playable videos" in str(exc)
        else:
            raise AssertionError("Expected an empty playlist to fail")
