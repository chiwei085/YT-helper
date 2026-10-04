from pathlib import Path
from unittest.mock import call, patch

import pytest
from typer.testing import CliRunner

from yt_helper.cli import (
    _analyze_url_target,
    _clean_url,
    _normalize_batch_url,
    _queue_batch_url,
    _resolve_output_dir,
    app,
)
from yt_helper.downloader import (
    MediaFormatInfo,
    MediaInfo,
    PlaylistEntry,
    PlaylistInfo,
    VideoDownloadError,
    VideoFormat,
)
from yt_helper.transcription import TranscriptionError, TranscriptionResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# _resolve_output_dir
# ---------------------------------------------------------------------------


def test_resolve_output_dir_defaults_to_user_downloads():
    result = _resolve_output_dir(None)
    assert result == (Path.home() / "Downloads" / "yt-helper").resolve()
    assert result.is_absolute()


def test_resolve_output_dir_uses_provided_path(tmp_path):
    result = _resolve_output_dir(tmp_path)
    assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# _analyze_url_target
# ---------------------------------------------------------------------------


def test_analyze_youtube_playlist_url():
    target = _analyze_url_target("https://www.youtube.com/playlist?list=PL1234567890")
    assert target.canonical_url == "https://www.youtube.com/playlist?list=PL1234567890"
    assert target.dedupe_key == "youtube-playlist:PL1234567890"
    assert target.is_playlist is True


def test_analyze_watch_url_with_playlist_treated_as_playlist():
    target = _analyze_url_target(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890"
    )
    assert target.canonical_url == "https://www.youtube.com/playlist?list=PL1234567890"
    assert target.dedupe_key == "youtube-playlist:PL1234567890"
    assert target.is_playlist is True


# ---------------------------------------------------------------------------
# _normalize_batch_url
# ---------------------------------------------------------------------------


def test_normalize_youtu_be():
    url, key = _normalize_batch_url("https://youtu.be/dQw4w9WgXcQ")
    assert url == "https://youtu.be/dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_youtube_watch():
    url, key = _normalize_batch_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_youtube_watch_with_playlist_prefers_playlist():
    url, key = _normalize_batch_url(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30&list=PL"
    )
    assert url == "https://www.youtube.com/playlist?list=PL"
    assert key == "youtube-playlist:PL"


def test_normalize_youtube_watch_strips_non_playlist_params():
    url, key = _normalize_batch_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_youtube_shorts():
    url, key = _normalize_batch_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_youtube_live():
    url, key = _normalize_batch_url("https://www.youtube.com/live/dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_youtube_embed():
    url, key = _normalize_batch_url("https://www.youtube.com/embed/dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_m_youtube():
    url, key = _normalize_batch_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert key == "youtube:dQw4w9WgXcQ"


def test_normalize_playlist_url():
    url, key = _normalize_batch_url("https://www.youtube.com/playlist?list=PL123")
    assert url == "https://www.youtube.com/playlist?list=PL123"
    assert key == "youtube-playlist:PL123"


def test_normalize_generic_url_passthrough():
    url, key = _normalize_batch_url("https://vimeo.com/123456")
    assert url == "https://vimeo.com/123456"
    assert key == "https://vimeo.com/123456"


def test_normalize_strips_fragment():
    url, _ = _normalize_batch_url("https://vimeo.com/123456#comments")
    assert "#" not in url


# ---------------------------------------------------------------------------
# _queue_batch_url — deduplication
# ---------------------------------------------------------------------------


def test_queue_deduplication_same_youtu_be():
    urls: list[str] = []
    seen: set[str] = set()

    _queue_batch_url("https://youtu.be/dQw4w9WgXcQ", urls, seen)
    _queue_batch_url("https://youtu.be/dQw4w9WgXcQ", urls, seen)

    assert len(urls) == 1


def test_queue_deduplication_youtu_be_and_watch_same_id():
    urls: list[str] = []
    seen: set[str] = set()

    _queue_batch_url("https://youtu.be/dQw4w9WgXcQ", urls, seen)
    _queue_batch_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ", urls, seen)

    assert len(urls) == 1


def test_queue_different_urls_both_added():
    urls: list[str] = []
    seen: set[str] = set()

    _queue_batch_url("https://youtu.be/aaaaaaaaaaa", urls, seen)
    _queue_batch_url("https://youtu.be/bbbbbbbbbbb", urls, seen)

    assert len(urls) == 2


def test_queue_deduplication_playlist_url_and_watch_url():
    urls: list[str] = []
    seen: set[str] = set()

    _queue_batch_url("https://www.youtube.com/playlist?list=PL123", urls, seen)
    _queue_batch_url(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
        urls,
        seen,
    )

    assert len(urls) == 1


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


def test_info_success():
    fake_info = MediaInfo(
        title="Test video",
        duration_seconds=123,
        uploader="Test channel",
        subtitles_available=True,
        estimated_size_bytes=12 * 1024 * 1024,
        formats=(
            MediaFormatInfo(
                format_id="137",
                extension="mp4",
                description="1080p | video",
                filesize_bytes=10 * 1024 * 1024,
            ),
        ),
    )

    with patch("yt_helper.cli.fetch_media_info", return_value=fake_info):
        result = runner.invoke(app, ["info", "https://youtu.be/test"])

    assert result.exit_code == 0
    assert "Fetched media information." in result.output
    assert "Test video" in result.output
    assert "Available formats" in result.output


def test_info_failure():
    with patch(
        "yt_helper.cli.fetch_media_info",
        side_effect=VideoDownloadError("Info error"),
    ):
        result = runner.invoke(app, ["info", "https://youtu.be/test"])

    assert result.exit_code == 1
    assert "Download failed." in result.output
    assert "Info error" in result.output


# ---------------------------------------------------------------------------
# download command
# ---------------------------------------------------------------------------


def test_download_full_video_success(tmp_path):
    fake_file = tmp_path / "video.mp4"
    with patch("yt_helper.cli.download_video", return_value=fake_file) as mock_dl:
        result = runner.invoke(
            app, ["download", "--output-dir", str(tmp_path), "https://youtu.be/test"]
        )

    assert result.exit_code == 0
    assert "Video download complete." in result.output
    mock_dl.assert_called_once()


def test_download_audio_mode_calls_download_audio(tmp_path):
    fake_file = tmp_path / "audio.wav"
    with patch("yt_helper.cli.download_audio", return_value=fake_file) as mock_dl:
        result = runner.invoke(
            app,
            [
                "download",
                "--mode",
                "audio",
                "--output-dir",
                str(tmp_path),
                "https://youtu.be/test",
            ],
        )

    assert result.exit_code == 0
    mock_dl.assert_called_once()


def test_download_visual_only_mode_calls_download_video_only(tmp_path):
    with patch("yt_helper.cli.download_video_only", return_value=None) as mock_dl:
        result = runner.invoke(
            app,
            [
                "download",
                "--mode",
                "visual-only",
                "--output-dir",
                str(tmp_path),
                "https://youtu.be/test",
            ],
        )

    assert result.exit_code == 0
    mock_dl.assert_called_once()


def test_download_raises_video_download_error(tmp_path):
    with patch(
        "yt_helper.cli.download_video",
        side_effect=VideoDownloadError("Network timeout"),
    ):
        result = runner.invoke(
            app, ["download", "--output-dir", str(tmp_path), "https://youtu.be/test"]
        )

    assert result.exit_code == 1
    assert "Download failed." in result.output
    assert "Network timeout" in result.output


def test_download_playlist_url_uses_playlist_mode(tmp_path):
    fake_file = tmp_path / "video.mp4"
    with patch("yt_helper.cli.download_video", return_value=fake_file) as mock_dl:
        result = runner.invoke(
            app,
            [
                "download",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/playlist?list=PL123",
            ],
        )

    assert result.exit_code == 0
    assert "Detected target type: playlist" in result.output
    mock_dl.assert_called_once_with(
        "https://www.youtube.com/playlist?list=PL123",
        tmp_path.resolve(),
        is_playlist=True,
        video_format=VideoFormat.MP4,
    )


# ---------------------------------------------------------------------------
# transcribe command
# ---------------------------------------------------------------------------


def test_transcribe_success(tmp_path):
    audio_path = tmp_path / "audio.wav"
    transcript_path = tmp_path / "audio.txt"
    fake_result = TranscriptionResult(
        output_path=transcript_path,
        source="qwen",
        device_label="CPU",
        language="Chinese,English",
    )

    with (
        patch("yt_helper.cli.download_native_subtitles", return_value=None),
        patch("yt_helper.cli.download_audio", return_value=audio_path),
        patch("yt_helper.cli.transcribe_media", return_value=fake_result),
    ):
        result = runner.invoke(
            app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "https://youtu.be/test",
            ],
        )

    assert result.exit_code == 0
    assert "Transcription complete." in result.output
    assert "Source: qwen" in result.output
    assert "Detected language(s): Chinese,English" in result.output
    assert "CPU" in result.output


def test_transcribe_prefers_native_subtitles(tmp_path):
    subtitle_path = tmp_path / "video.en.vtt"
    subtitle_path.write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nHello\n",
        encoding="utf-8",
    )

    with (
        patch(
            "yt_helper.cli.download_native_subtitles",
            return_value=type(
                "SubtitleResult",
                (),
                {
                    "output_path": subtitle_path,
                    "language": "en",
                    "extension": "vtt",
                },
            )(),
        ),
        patch("yt_helper.cli.download_audio") as mock_download_audio,
        patch("yt_helper.cli.transcribe_media") as mock_transcribe_media,
    ):
        result = runner.invoke(
            app,
            ["transcribe", "--output-dir", str(tmp_path), "https://youtu.be/test"],
        )

    assert result.exit_code == 0
    assert "Source: native subtitles" in result.output
    assert str(subtitle_path) in result.output
    mock_download_audio.assert_not_called()
    mock_transcribe_media.assert_not_called()


def test_transcribe_download_fails(tmp_path):
    with (
        patch(
            "yt_helper.cli.download_native_subtitles",
            return_value=None,
        ),
        patch(
            "yt_helper.cli.download_audio",
            side_effect=VideoDownloadError("Download error"),
        ),
    ):
        result = runner.invoke(
            app,
            ["transcribe", "--output-dir", str(tmp_path), "https://youtu.be/test"],
        )

    assert result.exit_code == 1
    assert "Download failed." in result.output


def test_transcribe_playlist_processes_each_video_and_continues_on_error(tmp_path):
    playlist = PlaylistInfo(
        title="L2 (Crypto)",
        entries=(
            PlaylistEntry(
                title="Encryption definition",
                url="https://www.youtube.com/watch?v=first",
            ),
            PlaylistEntry(
                title="Notion part1",
                url="https://www.youtube.com/watch?v=second",
            ),
        ),
    )
    transcript_path = tmp_path / "first.txt"
    first_result = TranscriptionResult(
        output_path=transcript_path,
        source="qwen",
        device_label="CPU",
        language="English",
    )

    with (
        patch("yt_helper.cli.fetch_playlist_info", return_value=playlist),
        patch(
            "yt_helper.cli._transcribe_url",
            side_effect=[first_result, TranscriptionError("ASR failed")],
        ) as transcribe_url,
    ):
        result = runner.invoke(
            app,
            [
                "transcribe",
                "--output-dir",
                str(tmp_path),
                "https://www.youtube.com/playlist?list=PL123",
            ],
        )

    assert result.exit_code == 0
    assert "Detected target type: playlist" in result.output
    assert "Playlist: L2 (Crypto)" in result.output
    assert "[1/2] Encryption definition" in result.output
    assert "[2/2] Notion part1" in result.output
    assert "Successful: 1" in result.output
    assert "Failed: 1" in result.output
    assert transcribe_url.call_args_list == [
        call("https://www.youtube.com/watch?v=first", tmp_path.resolve()),
        call("https://www.youtube.com/watch?v=second", tmp_path.resolve()),
    ]


def test_transcribe_playlist_rejects_single_output_path(tmp_path):
    with patch("yt_helper.cli.fetch_playlist_info") as fetch_playlist:
        result = runner.invoke(
            app,
            [
                "transcribe",
                "--output",
                str(tmp_path / "all.txt"),
                "https://www.youtube.com/playlist?list=PL123",
            ],
        )

    assert result.exit_code == 2
    assert "--output cannot be used with a playlist" in result.output
    fetch_playlist.assert_not_called()


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


def test_batch_playlist_url_uses_playlist_mode(tmp_path):
    with (
        patch(
            "yt_helper.cli._collect_batch_urls",
            return_value=["https://www.youtube.com/playlist?list=PL123"],
        ),
        patch("yt_helper.cli.download_video", return_value=None) as mock_dl,
    ):
        result = runner.invoke(app, ["batch", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Detected target type: playlist" in result.output
    mock_dl.assert_called_once_with(
        "https://www.youtube.com/playlist?list=PL123",
        tmp_path.resolve(),
        is_playlist=True,
        video_format=VideoFormat.MP4,
    )


@pytest.mark.parametrize(
    "pasted",
    [
        "https://www.youtube.com/playlist\\?list\\=PL1234567890",
        "  https://www.youtube.com/playlist?list=PL1234567890\n",
        "<https://www.youtube.com/playlist?list=PL1234567890>",
        "'https://www.youtube.com/playlist?list=PL1234567890'",
        "www.youtube.com/playlist?list=PL1234567890",
        "https://www.youtube.com/playlist?list=PL1234567890\u200b",
        "https://www.youtube.com/watch?v=abc&list=PL1234567890&si=tracking",
    ],
)
def test_analyze_url_target_normalizes_pasted_urls(pasted):
    target = _analyze_url_target(pasted)
    assert target.is_playlist
    assert target.canonical_url == (
        "https://www.youtube.com/playlist?list=PL1234567890"
    )


def test_clean_url_leaves_a_clean_url_alone():
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert _clean_url(url) == url
