from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Any

import typer
from tqdm import tqdm
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


class VideoDownloadError(RuntimeError):
    """Raised when yt-dlp cannot complete the download."""


class AudioFormat(StrEnum):
    """Supported audio export formats for downstream speech workflows."""

    WAV = "wav"
    FLAC = "flac"
    M4A = "m4a"
    MP3 = "mp3"


class VideoFormat(StrEnum):
    """Supported video container formats for common download workflows."""

    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"


@dataclass(frozen=True)
class MediaFormatInfo:
    """Represents one available media format exposed by yt-dlp."""

    format_id: str
    extension: str
    description: str
    filesize_bytes: int | None


@dataclass(frozen=True)
class MediaInfo:
    """Summarized media metadata for pre-download inspection."""

    title: str
    duration_seconds: int | None
    uploader: str | None
    subtitles_available: bool
    estimated_size_bytes: int | None
    formats: tuple[MediaFormatInfo, ...]


@dataclass(frozen=True)
class SubtitleDownloadResult:
    """Represents a downloaded native subtitle file."""

    output_path: Path
    language: str
    extension: str


@dataclass(frozen=True)
class PlaylistEntry:
    """Represents one playable item in a playlist."""

    title: str
    url: str


@dataclass(frozen=True)
class PlaylistInfo:
    """Represents a playlist and its playable entries."""

    title: str
    entries: tuple[PlaylistEntry, ...]


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown size"

    size = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.1f} {units[unit_index]}"


def _format_seconds(value: int | float | None) -> str:
    if value is None:
        return "unknown ETA"

    total_seconds = max(int(value), 0)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def _pick_best_video_size(formats: list[dict[str, Any]]) -> int | None:
    video_candidates = [
        fmt
        for fmt in formats
        if fmt.get("vcodec") not in {None, "none"}
        and fmt.get("acodec") in {None, "none"}
        and (fmt.get("filesize") or fmt.get("filesize_approx"))
    ]
    if not video_candidates:
        return None

    best_video = max(
        video_candidates,
        key=lambda fmt: (
            int(fmt.get("height") or 0),
            float(fmt.get("tbr") or 0.0),
            int(fmt.get("filesize") or fmt.get("filesize_approx") or 0),
        ),
    )
    return int(best_video.get("filesize") or best_video.get("filesize_approx") or 0)


def _pick_best_audio_size(formats: list[dict[str, Any]]) -> int | None:
    audio_candidates = [
        fmt
        for fmt in formats
        if fmt.get("acodec") not in {None, "none"}
        and fmt.get("vcodec") in {None, "none"}
        and (fmt.get("filesize") or fmt.get("filesize_approx"))
    ]
    if not audio_candidates:
        return None

    best_audio = max(
        audio_candidates,
        key=lambda fmt: (
            float(fmt.get("abr") or 0.0),
            float(fmt.get("tbr") or 0.0),
            int(fmt.get("filesize") or fmt.get("filesize_approx") or 0),
        ),
    )
    return int(best_audio.get("filesize") or best_audio.get("filesize_approx") or 0)


def _estimate_media_size(info: Mapping[str, Any]) -> int | None:
    direct_size = info.get("filesize") or info.get("filesize_approx")
    if direct_size:
        return int(direct_size)

    formats = [fmt for fmt in info.get("formats", []) if isinstance(fmt, dict)]
    if not formats:
        return None

    muxed_candidates = [
        fmt
        for fmt in formats
        if fmt.get("vcodec") not in {None, "none"}
        and fmt.get("acodec") not in {None, "none"}
        and (fmt.get("filesize") or fmt.get("filesize_approx"))
    ]
    if muxed_candidates:
        best_muxed = max(
            muxed_candidates,
            key=lambda fmt: (
                int(fmt.get("height") or 0),
                float(fmt.get("tbr") or 0.0),
                int(fmt.get("filesize") or fmt.get("filesize_approx") or 0),
            ),
        )
        return int(best_muxed.get("filesize") or best_muxed.get("filesize_approx") or 0)

    best_video_size = _pick_best_video_size(formats)
    best_audio_size = _pick_best_audio_size(formats)
    if best_video_size is not None and best_audio_size is not None:
        return best_video_size + best_audio_size

    return best_video_size or best_audio_size


def _build_format_description(format_info: Mapping[str, Any]) -> str:
    parts: list[str] = []
    resolution = format_info.get("resolution")
    if isinstance(resolution, str) and resolution and resolution != "audio only":
        parts.append(resolution)
    elif format_info.get("height"):
        parts.append(f"{int(format_info['height'])}p")

    if format_info.get("fps"):
        parts.append(f"{int(format_info['fps'])}fps")

    if format_info.get("vcodec") not in {None, "none"}:
        parts.append("video")
    if format_info.get("acodec") not in {None, "none"}:
        parts.append("audio")

    note = format_info.get("format_note")
    if isinstance(note, str) and note:
        parts.append(note)

    tbr = format_info.get("tbr")
    if tbr:
        parts.append(f"{float(tbr):.0f}kbps")

    return " | ".join(parts) if parts else "No additional format details"


_COMPATIBILITY_OPTIONS: dict[str, Any] = {
    "extractor_args": {"youtube": {"player_client": ["tv_simply"]}},
    "js_runtimes": {"node": {}},
    "remote_components": {"ejs:github"},
}

_use_compatibility_client = False


def _probe_info(url: str, **overrides: Any) -> Mapping[str, Any]:
    """Extract metadata for a URL without downloading it."""
    options: dict[str, Any] = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        **overrides,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise VideoDownloadError(_build_download_error_message(exc)) from exc

    if not isinstance(info, Mapping):
        raise VideoDownloadError(
            "yt-dlp did not return usable media information for this URL."
        )

    return info


def fetch_media_info(url: str) -> MediaInfo:
    """Fetch media metadata without downloading the file."""
    info = _probe_info(url)

    formats: list[MediaFormatInfo] = []
    for fmt in info.get("formats", []):
        if not isinstance(fmt, dict):
            continue

        filesize_bytes = fmt.get("filesize") or fmt.get("filesize_approx")
        formats.append(
            MediaFormatInfo(
                format_id=str(fmt.get("format_id") or "unknown"),
                extension=str(fmt.get("ext") or "unknown"),
                description=_build_format_description(fmt),
                filesize_bytes=int(filesize_bytes) if filesize_bytes else None,
            )
        )

    return MediaInfo(
        title=str(info.get("title") or "Unknown title"),
        duration_seconds=(
            int(info["duration"]) if info.get("duration") is not None else None
        ),
        uploader=(
            str(info.get("uploader") or info.get("channel"))
            if info.get("uploader") or info.get("channel")
            else None
        ),
        subtitles_available=bool(
            info.get("subtitles") or info.get("automatic_captions")
        ),
        estimated_size_bytes=_estimate_media_size(info),
        formats=tuple(formats),
    )


def fetch_playlist_info(url: str) -> PlaylistInfo:
    """Fetch a playlist's playable entries without downloading media."""
    info = _probe_info(
        url,
        noplaylist=False,
        extract_flat=True,
        ignoreerrors=True,
    )
    if info.get("_type") != "playlist":
        raise VideoDownloadError(
            "yt-dlp did not return usable playlist information for this URL."
        )

    entries: list[PlaylistEntry] = []
    for entry in info.get("entries") or ():
        if not isinstance(entry, Mapping):
            continue

        entry_url = entry.get("webpage_url") or entry.get("url")
        if not isinstance(entry_url, str) or not entry_url.startswith(
            ("http://", "https://")
        ):
            continue

        entries.append(
            PlaylistEntry(
                title=str(entry.get("title") or "Video"),
                url=entry_url,
            )
        )

    if not entries:
        raise VideoDownloadError("The playlist does not contain any playable videos.")

    return PlaylistInfo(
        title=str(info.get("title") or "Untitled playlist"),
        entries=tuple(entries),
    )


class DownloadProgressReporter:
    """Render yt-dlp progress updates through tqdm."""

    _RENDER_INTERVAL = 0.1

    def __init__(self) -> None:
        self._bars: dict[str, tqdm[float]] = {}
        self._last_renders: dict[str, float] = {}

    def __call__(self, data: Mapping[str, Any]) -> None:
        status = data.get("status")
        if status == "downloading":
            self._render_download_progress(data)

    def _render_download_progress(self, data: Mapping[str, Any]) -> None:
        key = self._extract_download_key(data)
        total = data.get("total_bytes") or data.get("total_bytes_estimate")
        downloaded = data.get("downloaded_bytes")

        if key not in self._bars:
            self._bars[key] = tqdm(
                total=float(total) if total is not None else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
                leave=False,
                dynamic_ncols=True,
            )
            self._last_renders[key] = 0.0
        elif total is not None:
            self._bars[key].total = float(total)

        now = monotonic()
        if now - self._last_renders[key] < self._RENDER_INTERVAL:
            return
        self._last_renders[key] = now

        bar = self._bars[key]
        bar.n = float(downloaded) if downloaded is not None else 0.0
        bar.set_postfix_str(self._build_postfix(data), refresh=False)
        bar.refresh()

    @staticmethod
    def _build_postfix(data: Mapping[str, Any]) -> str:
        speed = _format_bytes(data.get("speed"))
        eta = _format_seconds(data.get("eta"))
        downloaded = _format_bytes(data.get("downloaded_bytes"))
        total = _format_bytes(
            data.get("total_bytes") or data.get("total_bytes_estimate")
        )
        return f"{downloaded}/{total}, {speed}/s, ETA {eta}"

    @staticmethod
    def _extract_download_key(data: Mapping[str, Any]) -> str:
        filename = data.get("filename")
        tmpfilename = data.get("tmpfilename")
        info_dict = data.get("info_dict")
        if isinstance(filename, str) and filename:
            return filename
        if isinstance(tmpfilename, str) and tmpfilename:
            return tmpfilename
        if isinstance(info_dict, Mapping):
            info_id = info_dict.get("id")
            format_id = info_dict.get("format_id")
            if info_id or format_id:
                return f"{info_id}:{format_id}"
        return "download"

    def finalize(self) -> None:
        for bar in self._bars.values():
            bar.leave = True
            bar.close()
        self._bars.clear()
        self._last_renders.clear()


class PostprocessingProgressReporter:
    """Show an indeterminate progress line for ffmpeg post-processing."""

    def __init__(self) -> None:
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._progress_bar: tqdm[Any] | None = None
        self._started_at: float | None = None
        self._last_warning_elapsed = 0.0

    def start(self, description: str) -> None:
        self.stop()
        self._stop_event = Event()
        self._started_at = monotonic()
        self._last_warning_elapsed = 0.0
        self._progress_bar = tqdm(
            total=None,
            desc=description,
            bar_format="{desc} | elapsed: {elapsed}",
            leave=False,
            dynamic_ncols=True,
        )
        self._thread = Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self, *, completed: bool = False, failed: bool = False) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None

        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        self._started_at = None

        if completed:
            tqdm.write("Post-processing finished.")
        elif failed:
            tqdm.write("Post-processing failed.")

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(1):
            bar = self._progress_bar
            if bar is not None:
                bar.refresh()
            self._maybe_warn_about_long_processing()

    def _maybe_warn_about_long_processing(self) -> None:
        started_at = self._started_at
        if started_at is None:
            return

        elapsed = monotonic() - started_at
        if elapsed < 120 or elapsed - self._last_warning_elapsed < 120:
            return

        self._last_warning_elapsed = elapsed
        tqdm.write(
            "Post-processing is still running. Large videos can take several "
            "minutes to merge."
        )


def _is_merge_postprocessor(pp: Any, info_dict: Mapping[str, Any]) -> bool:
    pp_key = getattr(pp, "pp_key", None)
    if not callable(pp_key):
        return False

    return pp_key() == "Merger" and "__files_to_merge" in info_dict


class ManagedYoutubeDL(YoutubeDL):
    """YoutubeDL wrapper that exposes ffmpeg post-processing progress."""

    def __init__(
        self,
        params: Mapping[str, Any],
        *,
        postprocessing_reporter: PostprocessingProgressReporter,
    ) -> None:
        super().__init__(dict(params))
        self._postprocessing_reporter = postprocessing_reporter

    def run_pp(self, pp: Any, infodict: dict[str, Any]) -> dict[str, Any]:
        is_merge_step = _is_merge_postprocessor(pp, infodict)
        if is_merge_step:
            self._postprocessing_reporter.start("Merging media streams")

        try:
            result = super().run_pp(pp, infodict)
        except Exception:
            if is_merge_step:
                self._postprocessing_reporter.stop(failed=True)
            raise

        if is_merge_step:
            self._postprocessing_reporter.stop(completed=True)

        return result


def _build_download_error_message(error: DownloadError) -> str:
    message = str(error).strip()
    lowered = message.lower()

    if "permission denied" in lowered or "operation not permitted" in lowered:
        return (
            "The download could not write files to the selected directory. "
            "Check folder permissions and try a different output location."
        )

    if "certificate verify failed" in lowered or "ssl" in lowered:
        return (
            "The download failed because SSL certificate verification did not "
            "complete successfully. Check the local certificate store or try "
            "again on a network that is not intercepting HTTPS traffic."
        )

    if "ffmpeg" in lowered and ("not found" in lowered or "not installed" in lowered):
        return (
            "The video was downloaded, but the final merge step requires ffmpeg. "
            "Install ffmpeg and run the command again."
        )

    if "timed out" in lowered or "temporary failure" in lowered:
        return (
            "The download timed out after multiple retries. Check your network "
            "connection and try again."
        )

    if (
        "sign in to confirm" in lowered
        or "requested format is not available" in lowered
    ):
        return (
            "The source site rejected the requested media or requires extra "
            "verification. Try another video or a different account/network setup."
        )

    return message


def _download_media(
    url: str,
    output_dir: Path,
    options: Mapping[str, Any],
    *,
    is_playlist: bool = False,
) -> Path | None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoDownloadError(
            "Could not prepare the output directory. Check permissions, "
            "available disk space, and the selected path."
        ) from exc

    def attempt(extra_options: Mapping[str, Any]) -> Any:
        progress_reporter = DownloadProgressReporter()
        postprocessing_reporter = PostprocessingProgressReporter()
        merged_options = {
            "noplaylist": not is_playlist,
            "outtmpl": str(output_dir / "%(title)s [%(id)s].%(ext)s"),
            "paths": {"home": str(output_dir)},
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "continuedl": True,
            "retries": 10,
            "fragment_retries": 10,
            "extractor_retries": 5,
            "file_access_retries": 3,
            "socket_timeout": 30,
            "concurrent_fragment_downloads": 4,
            "progress_hooks": [progress_reporter],
            **options,
            **extra_options,
        }
        try:
            with ManagedYoutubeDL(
                merged_options,
                postprocessing_reporter=postprocessing_reporter,
            ) as ydl:
                return ydl.extract_info(url, download=True)
        finally:
            progress_reporter.finalize()
            postprocessing_reporter.stop()

    global _use_compatibility_client

    try:
        info = attempt(_COMPATIBILITY_OPTIONS if _use_compatibility_client else {})
    except DownloadError as exc:
        if _use_compatibility_client or "http error 403" not in str(exc).lower():
            raise VideoDownloadError(_build_download_error_message(exc)) from exc

        typer.secho(
            "The default YouTube stream was rejected; retrying with a "
            "compatibility client.",
            fg=typer.colors.YELLOW,
        )
        _use_compatibility_client = True
        try:
            info = attempt(_COMPATIBILITY_OPTIONS)
        except DownloadError as retry_exc:
            raise VideoDownloadError(
                _build_download_error_message(retry_exc)
            ) from retry_exc

    if info is None:
        typer.secho(
            "Warning: yt-dlp did not return media details for this URL.",
            fg=typer.colors.YELLOW,
        )
        return None

    if info.get("_type") == "playlist":
        return None

    requested_downloads = info.get("requested_downloads", [])
    if requested_downloads:
        filepath = requested_downloads[0].get("filepath")
        if filepath:
            return Path(filepath)

    filepath = info.get("_filename")
    if filepath:
        return Path(filepath)

    typer.secho(
        "Warning: the download finished, but the saved file path could not "
        "be determined.",
        fg=typer.colors.YELLOW,
    )
    return None


def _resolve_requested_subtitle_path(
    info: Mapping[str, Any],
    output_dir: Path,
) -> Path | None:
    requested_subtitles = info.get("requested_subtitles")
    if isinstance(requested_subtitles, Mapping):
        for subtitle_info in requested_subtitles.values():
            if not isinstance(subtitle_info, Mapping):
                continue

            filepath = subtitle_info.get("filepath")
            if isinstance(filepath, str) and filepath:
                return Path(filepath)

    title = info.get("title")
    video_id = info.get("id")
    if not isinstance(title, str) or not title or not isinstance(video_id, str):
        return None

    candidates = sorted(output_dir.glob(f"{title} [{video_id}].*.vtt"))
    if candidates:
        return candidates[0]

    candidates = sorted(output_dir.glob(f"{title} [{video_id}].*.srt"))
    if candidates:
        return candidates[0]

    return None


def download_native_subtitles(
    url: str,
    output_dir: Path,
) -> SubtitleDownloadResult | None:
    """Download native subtitles when the source provides them."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoDownloadError(
            "Could not prepare the output directory. Check permissions, "
            "available disk space, and the selected path."
        ) from exc

    info = _probe_info(url)

    subtitles = info.get("subtitles")
    if not isinstance(subtitles, Mapping) or not subtitles:
        return None

    language = next((lang for lang, entries in subtitles.items() if entries), None)
    if language is None:
        return None

    download_options = {
        "noplaylist": True,
        "outtmpl": str(output_dir / "%(title)s [%(id)s].%(ext)s"),
        "paths": {"home": str(output_dir)},
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": [language],
        "subtitlesformat": "vtt/best",
    }
    try:
        with YoutubeDL(download_options) as ydl:
            downloaded_info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        raise VideoDownloadError(_build_download_error_message(exc)) from exc

    if not isinstance(downloaded_info, dict):
        return None

    subtitle_path = _resolve_requested_subtitle_path(downloaded_info, output_dir)
    if subtitle_path is None:
        return None

    return SubtitleDownloadResult(
        output_path=subtitle_path,
        language=language,
        extension=subtitle_path.suffix.lstrip("."),
    )


def download_video(
    url: str,
    output_dir: Path,
    video_format: VideoFormat = VideoFormat.MP4,
    *,
    is_playlist: bool = False,
) -> Path | None:
    """Download a video using the best available combined format."""
    return _download_media(
        url,
        output_dir,
        {
            "format": "bestvideo*+bestaudio/best",
            "merge_output_format": video_format.value,
            "remux_video": video_format.value,
        },
        is_playlist=is_playlist,
    )


def download_video_only(
    url: str,
    output_dir: Path,
    video_format: VideoFormat = VideoFormat.MP4,
    *,
    is_playlist: bool = False,
) -> Path | None:
    """Download the best available video stream without audio."""
    return _download_media(
        url,
        output_dir,
        {
            "format": "bestvideo/best",
            "remux_video": video_format.value,
        },
        is_playlist=is_playlist,
    )


def download_audio(
    url: str,
    output_dir: Path,
    audio_format: AudioFormat = AudioFormat.WAV,
    *,
    is_playlist: bool = False,
) -> Path | None:
    """Download audio only and convert it to a standalone audio file."""
    postprocessor = {
        "key": "FFmpegExtractAudio",
        "preferredcodec": audio_format.value,
    }
    if audio_format in {AudioFormat.MP3, AudioFormat.M4A}:
        postprocessor["preferredquality"] = "192"

    return _download_media(
        url,
        output_dir,
        {
            "format": "bestaudio/best",
            "postprocessors": [postprocessor],
        },
        is_playlist=is_playlist,
    )
