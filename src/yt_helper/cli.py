from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, NoReturn
from urllib.parse import parse_qs, urlparse, urlunparse

import typer

from yt_helper.downloader import (
    AudioFormat,
    MediaInfo,
    PlaylistInfo,
    VideoDownloadError,
    VideoFormat,
    download_audio,
    download_native_subtitles,
    download_video,
    download_video_only,
    fetch_media_info,
    fetch_playlist_info,
)
from yt_helper.transcription import (
    TranscriptionError,
    TranscriptionResult,
    transcribe_media,
)


class DownloadMode(StrEnum):
    """Supported download output modes."""

    FULL_VIDEO = "full-video"
    VISUAL_ONLY = "visual-only"
    AUDIO = "audio"


@dataclass(frozen=True)
class UrlTarget:
    """Represents the normalized target and how yt-helper should treat it."""

    canonical_url: str
    dedupe_key: str
    is_playlist: bool


app = typer.Typer(
    add_completion=False,
    help="Download media with a guided CLI.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

DEFAULT_OUTPUT_DIR = Path.home() / "Downloads" / "yt-helper"


def _resolve_output_dir(output_dir: Path | None) -> Path:
    return (output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()


def _prompt_for_video_url(initial_value: str | None = None) -> str:
    if initial_value and initial_value.strip():
        return initial_value.strip()

    typer.echo()
    typer.secho("No video URL was provided.", fg=typer.colors.YELLOW, bold=True)
    typer.echo("Paste a video URL below to start the download.")
    typer.echo("Examples: https://www.youtube.com/watch?v=... or https://youtu.be/...")
    typer.echo()

    while True:
        url = typer.prompt("Video URL", prompt_suffix=": ").strip()
        if url.startswith(("http://", "https://")):
            return url

        typer.secho(
            "Please enter a valid URL that starts with http:// or https://.",
            fg=typer.colors.RED,
        )


def _analyze_url_target(url: str) -> UrlTarget:
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)

    if hostname in {
        "youtu.be",
        "www.youtu.be",
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        playlist_id = query.get("list", [None])[0]
        if playlist_id:
            canonical = f"https://www.youtube.com/playlist?list={playlist_id}"
            return UrlTarget(
                canonical_url=canonical,
                dedupe_key=f"youtube-playlist:{playlist_id}",
                is_playlist=True,
            )

    if hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
        if video_id:
            canonical = f"https://youtu.be/{video_id}"
            return UrlTarget(
                canonical_url=canonical,
                dedupe_key=f"youtube:{video_id}",
                is_playlist=False,
            )

    if hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
    }:
        if parsed.path == "/watch":
            video_id = query.get("v", [None])[0]
            if video_id:
                canonical = f"https://www.youtube.com/watch?v={video_id}"
                return UrlTarget(
                    canonical_url=canonical,
                    dedupe_key=f"youtube:{video_id}",
                    is_playlist=False,
                )

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "live", "embed"}:
            video_id = path_parts[1]
            canonical = f"https://www.youtube.com/watch?v={video_id}"
            return UrlTarget(
                canonical_url=canonical,
                dedupe_key=f"youtube:{video_id}",
                is_playlist=False,
            )

    canonical = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )
    return UrlTarget(
        canonical_url=canonical,
        dedupe_key=canonical,
        is_playlist=False,
    )


def _normalize_batch_url(url: str) -> tuple[str, str]:
    target = _analyze_url_target(url)
    return target.canonical_url, target.dedupe_key


def _queue_batch_url(url: str, urls: list[str], seen_urls: set[str]) -> None:
    canonical_url, dedupe_key = _normalize_batch_url(url)
    if dedupe_key in seen_urls:
        typer.secho(
            f"Skipping duplicate URL: {canonical_url}",
            fg=typer.colors.YELLOW,
        )
        return

    seen_urls.add(dedupe_key)
    urls.append(canonical_url)
    typer.secho(
        f"Queued URL #{len(urls)}: {canonical_url}",
        fg=typer.colors.GREEN,
    )


def _collect_batch_urls(initial_url: str | None = None) -> list[str]:
    urls: list[str] = []
    seen_urls: set[str] = set()

    typer.echo()
    typer.secho("Batch input mode", fg=typer.colors.CYAN, bold=True)
    typer.echo("Add video URLs one by one, then stop to begin batch downloading.")
    typer.echo()

    first_url = _prompt_for_video_url(initial_url)
    _queue_batch_url(first_url, urls, seen_urls)

    while typer.confirm("Continue adding another URL?", default=True):
        next_url = _prompt_for_video_url()
        _queue_batch_url(next_url, urls, seen_urls)

    return urls


def _handle_download_error(exc: VideoDownloadError) -> NoReturn:
    typer.echo()
    typer.secho("Download failed.", fg=typer.colors.RED, bold=True)
    typer.echo(str(exc))
    raise typer.Exit(code=1) from exc


def _handle_transcription_error(exc: TranscriptionError) -> NoReturn:
    typer.echo()
    typer.secho("Transcription failed.", fg=typer.colors.RED, bold=True)
    typer.echo(str(exc))
    raise typer.Exit(code=1) from exc


def _execute_downloader(
    *,
    downloader: Callable[..., Path | None],
    url: str,
    output_dir: Path,
    is_playlist: bool = False,
    audio_format: AudioFormat | None = None,
    video_format: VideoFormat | None = None,
) -> Path | None:
    kwargs: dict[str, AudioFormat | VideoFormat | bool] = {
        "is_playlist": is_playlist,
    }
    if audio_format is not None:
        kwargs["audio_format"] = audio_format
    if video_format is not None:
        kwargs["video_format"] = video_format
    return downloader(url, output_dir, **kwargs)


def _resolve_download_plan(
    mode: DownloadMode,
) -> tuple[Callable[..., Path | None], str, str]:
    if mode is DownloadMode.FULL_VIDEO:
        return (
            download_video,
            "Video download complete.",
            "Downloading the best available video with audio.",
        )

    if mode is DownloadMode.VISUAL_ONLY:
        return (
            download_video_only,
            "Visual-only download complete.",
            "Downloading the best available video stream without audio.",
        )

    return (
        download_audio,
        "Audio download complete.",
        "Downloading audio only from the provided video.",
    )


def _run_media_download(
    *,
    url: str | None,
    output_dir: Path | None,
    heading: str,
    summary: str,
    downloader: Callable[..., Path | None],
    completion_message: str,
    audio_format: AudioFormat | None = None,
    video_format: VideoFormat | None = None,
) -> None:
    resolved_url = _prompt_for_video_url(url)
    target = _analyze_url_target(resolved_url)
    resolved_output_dir = _resolve_output_dir(output_dir)

    typer.echo()
    typer.secho("yt-helper", fg=typer.colors.CYAN, bold=True)
    typer.echo(summary)
    typer.echo(f"Source URL: {target.canonical_url}")
    if target.is_playlist:
        typer.echo("Detected target type: playlist")
    typer.echo(f"Output directory: {resolved_output_dir}")
    if audio_format is not None:
        typer.echo(f"Audio format: {audio_format.value}")
    if video_format is not None:
        typer.echo(f"Video format: {video_format.value}")
    typer.echo()

    try:
        downloaded_file = _execute_downloader(
            downloader=downloader,
            url=target.canonical_url,
            output_dir=resolved_output_dir,
            is_playlist=target.is_playlist,
            audio_format=audio_format,
            video_format=video_format,
        )
    except VideoDownloadError as exc:
        _handle_download_error(exc)

    typer.echo()
    typer.secho(heading, fg=typer.colors.GREEN, bold=True)
    if downloaded_file is not None:
        typer.echo(f"Saved to: {downloaded_file}")
    else:
        typer.echo(completion_message.format(output_dir=resolved_output_dir))


def _format_filesize_mib(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "Unknown"

    return f"{size_bytes / (1024 * 1024):.1f} MiB"


def _format_duration_for_info(duration_seconds: int | None) -> str:
    if duration_seconds is None:
        return "Unknown"

    hours, remainder = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def _render_media_info(info: MediaInfo, source_url: str) -> None:
    typer.echo()
    typer.secho("yt-helper", fg=typer.colors.CYAN, bold=True)
    typer.echo("Fetched media information.")
    typer.echo(f"Source URL: {source_url}")
    typer.echo(f"Title: {info.title}")
    typer.echo(f"Duration: {_format_duration_for_info(info.duration_seconds)}")
    typer.echo(f"Uploader / channel: {info.uploader or 'Unknown'}")
    typer.echo(f"Subtitles available: {'Yes' if info.subtitles_available else 'No'}")
    typer.echo(f"Estimated size: {_format_filesize_mib(info.estimated_size_bytes)}")
    typer.echo()
    typer.secho("Available formats", fg=typer.colors.CYAN, bold=True)
    if not info.formats:
        typer.echo("No format details were returned by yt-dlp.")
        return

    for format_info in info.formats:
        typer.echo(
            f"- {format_info.format_id} | .{format_info.extension} | "
            f"{format_info.description} | "
            f"{_format_filesize_mib(format_info.filesize_bytes)}"
        )


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Run the default download flow when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(download)


@app.command()
def info(
    url: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Video URL to inspect. If omitted, the CLI will ask for it "
                "interactively."
            ),
        ),
    ] = None,
) -> None:
    """Show media metadata and available formats before downloading."""
    resolved_url = _prompt_for_video_url(url)
    try:
        media_info = fetch_media_info(resolved_url)
    except VideoDownloadError as exc:
        _handle_download_error(exc)

    _render_media_info(media_info, resolved_url)


@app.command()
def download(
    url: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Video URL to download. If omitted, the CLI will ask for it "
                "interactively."
            ),
        ),
    ] = None,
    mode: Annotated[
        DownloadMode,
        typer.Option(
            "--mode",
            "-m",
            case_sensitive=False,
            help="Choose whether to download full video, visual-only, or audio.",
        ),
    ] = DownloadMode.FULL_VIDEO,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=False,
            help=(
                "Directory where downloaded files will be saved. "
                "Defaults to ~/Downloads/yt-helper"
            ),
        ),
    ] = None,
    audio_format: Annotated[
        AudioFormat,
        typer.Option(
            "--audio-format",
            "-f",
            case_sensitive=False,
            help="Audio format used when --mode audio is selected.",
        ),
    ] = AudioFormat.WAV,
    video_format: Annotated[
        VideoFormat,
        typer.Option(
            "--video-format",
            "-v",
            case_sensitive=False,
            help=(
                "Video container used when --mode full-video or visual-only "
                "is selected."
            ),
        ),
    ] = VideoFormat.MP4,
) -> None:
    """Download media using a single command with explicit output mode."""
    downloader, heading, summary = _resolve_download_plan(mode)
    _run_media_download(
        url=url,
        output_dir=output_dir,
        heading=heading,
        summary=summary,
        downloader=downloader,
        completion_message="Files were saved in: {output_dir}",
        audio_format=audio_format if mode is DownloadMode.AUDIO else None,
        video_format=video_format if mode is not DownloadMode.AUDIO else None,
    )


@app.command()
def batch(
    first_url: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Optional first video URL. If omitted, batch mode will ask "
                "for the first URL interactively."
            ),
        ),
    ] = None,
    mode: Annotated[
        DownloadMode,
        typer.Option(
            "--mode",
            "-m",
            case_sensitive=False,
            help="Choose whether to batch download full video, visual-only, or audio.",
        ),
    ] = DownloadMode.FULL_VIDEO,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=False,
            help=(
                "Directory where downloaded files will be saved. "
                "Defaults to ~/Downloads/yt-helper"
            ),
        ),
    ] = None,
    audio_format: Annotated[
        AudioFormat,
        typer.Option(
            "--audio-format",
            "-f",
            case_sensitive=False,
            help="Audio format used when --mode audio is selected.",
        ),
    ] = AudioFormat.WAV,
    video_format: Annotated[
        VideoFormat,
        typer.Option(
            "--video-format",
            "-v",
            case_sensitive=False,
            help=(
                "Video container used when --mode full-video or visual-only "
                "is selected."
            ),
        ),
    ] = VideoFormat.MP4,
) -> None:
    """Collect multiple URLs interactively, then process them as one batch."""
    urls = _collect_batch_urls(first_url)
    resolved_output_dir = _resolve_output_dir(output_dir)
    downloader, _, summary = _resolve_download_plan(mode)
    batch_audio_format = audio_format if mode is DownloadMode.AUDIO else None
    batch_video_format = video_format if mode is not DownloadMode.AUDIO else None

    typer.echo()
    typer.secho("yt-helper", fg=typer.colors.CYAN, bold=True)
    typer.echo("Starting batch download.")
    typer.echo(summary)
    typer.echo(f"Queued URLs: {len(urls)}")
    typer.echo(f"Output directory: {resolved_output_dir}")
    if batch_audio_format is not None:
        typer.echo(f"Audio format: {batch_audio_format.value}")
    if batch_video_format is not None:
        typer.echo(f"Video format: {batch_video_format.value}")
    typer.echo()

    success_count = 0
    failure_count = 0
    for index, queued_url in enumerate(urls, start=1):
        target = _analyze_url_target(queued_url)
        typer.secho(
            f"[{index}/{len(urls)}] Processing: {target.canonical_url}",
            fg=typer.colors.CYAN,
            bold=True,
        )
        if target.is_playlist:
            typer.echo("Detected target type: playlist")
        try:
            downloaded_file = _execute_downloader(
                downloader=downloader,
                url=target.canonical_url,
                output_dir=resolved_output_dir,
                is_playlist=target.is_playlist,
                audio_format=batch_audio_format,
                video_format=batch_video_format,
            )
        except VideoDownloadError as exc:
            failure_count += 1
            typer.secho("Failed.", fg=typer.colors.RED, bold=True)
            typer.echo(str(exc))
            typer.echo()
            continue

        success_count += 1
        typer.secho("Completed.", fg=typer.colors.GREEN, bold=True)
        if downloaded_file is not None:
            typer.echo(f"Saved to: {downloaded_file}")
        else:
            typer.echo(f"Files were saved in: {resolved_output_dir}")
        typer.echo()

    typer.secho("Batch download finished.", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Successful: {success_count}")
    typer.echo(f"Failed: {failure_count}")


def _transcribe_url(
    url: str,
    output_dir: Path,
    output_path: Path | None = None,
) -> TranscriptionResult:
    """Transcribe one URL; `output_path`, when given, must already be resolved."""
    native_subtitles = download_native_subtitles(url, output_dir)
    if native_subtitles is not None:
        subtitle_path = native_subtitles.output_path
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                subtitle_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            subtitle_path = output_path
        return TranscriptionResult(
            output_path=subtitle_path,
            source="native subtitles",
            language=native_subtitles.language,
        )

    with TemporaryDirectory(prefix="yt-helper-transcribe-") as temp_dir:
        downloaded_audio = download_audio(
            url,
            Path(temp_dir),
            audio_format=AudioFormat.WAV,
        )
        if downloaded_audio is None:
            raise TranscriptionError(
                "Audio download completed, but no audio file path was returned."
            )

        return transcribe_media(
            downloaded_audio,
            output_path=output_path or output_dir / f"{downloaded_audio.stem}.txt",
        )


def _render_transcription_details(result: TranscriptionResult) -> None:
    typer.echo(f"Source: {result.source}")
    if result.language is not None:
        typer.echo(f"Detected language(s): {result.language}")
    if result.device_label is not None:
        typer.echo(f"Accelerator: {result.device_label}")
    typer.echo(f"Saved to: {result.output_path}")


def _transcribe_playlist(playlist: PlaylistInfo, output_dir: Path) -> bool:
    typer.echo(f"Playlist: {playlist.title}")
    typer.echo(f"Videos: {len(playlist.entries)}")
    typer.echo()

    success_count = 0
    for position, entry in enumerate(playlist.entries, start=1):
        typer.secho(
            f"[{position}/{len(playlist.entries)}] {entry.title}",
            fg=typer.colors.CYAN,
            bold=True,
        )
        try:
            result = _transcribe_url(entry.url, output_dir)
        except (VideoDownloadError, TranscriptionError) as exc:
            typer.secho("Failed.", fg=typer.colors.RED, bold=True)
            typer.echo(str(exc))
            typer.echo()
            continue

        success_count += 1
        typer.secho("Completed.", fg=typer.colors.GREEN, bold=True)
        _render_transcription_details(result)
        typer.echo()

    typer.secho("Playlist transcription finished.", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Successful: {success_count}")
    typer.echo(f"Failed: {len(playlist.entries) - success_count}")
    return success_count > 0


@app.command()
def transcribe(
    url: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Video or playlist URL to transcribe. If omitted, the CLI will "
                "ask for it interactively."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-d",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=False,
            help=(
                "Directory where transcript text files will be saved. "
                "Defaults to ~/Downloads/yt-helper"
            ),
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=False,
            help=(
                "Text file path for the transcript. Defaults to a .txt file "
                "named after the downloaded audio. Not available for playlists."
            ),
        ),
    ] = None,
) -> None:
    """Transcribe one video or every video in a playlist."""
    resolved_url = _prompt_for_video_url(url)
    target = _analyze_url_target(resolved_url)
    resolved_output_dir = _resolve_output_dir(output_dir)

    if target.is_playlist and output_path is not None:
        raise typer.BadParameter(
            "--output cannot be used with a playlist because each video "
            "creates a separate transcript. Use --output-dir instead."
        )

    resolved_output_path = (
        output_path.expanduser().resolve() if output_path is not None else None
    )

    typer.echo()
    typer.secho("yt-helper", fg=typer.colors.CYAN, bold=True)
    typer.echo("Trying native subtitles first, then falling back to Qwen ASR.")
    typer.echo(f"Source URL: {target.canonical_url}")
    if target.is_playlist:
        typer.echo("Detected target type: playlist")
    typer.echo(f"Transcript directory: {resolved_output_dir}")
    if resolved_output_path is not None:
        typer.echo(f"Output file: {resolved_output_path}")
    typer.echo()

    if target.is_playlist:
        try:
            playlist = fetch_playlist_info(target.canonical_url)
        except VideoDownloadError as exc:
            _handle_download_error(exc)

        if not _transcribe_playlist(playlist, resolved_output_dir):
            raise typer.Exit(code=1)
        return

    try:
        result = _transcribe_url(
            target.canonical_url,
            resolved_output_dir,
            resolved_output_path,
        )
    except VideoDownloadError as exc:
        _handle_download_error(exc)
    except TranscriptionError as exc:
        _handle_transcription_error(exc)

    typer.echo()
    typer.secho("Transcription complete.", fg=typer.colors.GREEN, bold=True)
    _render_transcription_details(result)


def main() -> None:
    app()
