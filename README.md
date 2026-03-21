# yt-helper

`yt-helper` is a command-line tool for downloading online media and turning video speech into timestamped plain-text transcripts.

## Supported Features

- Download the best available full video with audio, including playlist URLs.
- Download visual-only video streams, including playlist URLs.
- Download audio-only files for speech or audio workflows, including playlist URLs.
- Choose common output containers for video downloads: `mp4`, `mkv`, `webm`.
- Choose audio formats suited for downstream processing: `wav`, `flac`, `m4a`, `mp3`.
- Resume interrupted downloads and retry unstable transfers through `yt-dlp`.
- Show guided interactive prompts when a URL is not provided.
- Inspect a video before downloading, including title, duration, uploader, subtitle availability, estimated size, and available formats.
- Batch multiple URLs in one session, then start processing them together.
- Skip duplicate URLs during batch input and keep only the first occurrence.
- Prefer native subtitles when they exist, then fall back to Whisper ASR.
- Save transcripts as timestamped plain text, for example:

```text
[0.0s] Welcome, today we're going to talk about...
[12.3s] The first topic is X, which relates to...
[45.1s] So the key insight here is that...
```

- Prefer CUDA automatically when it is available for transcription.

## Installation

Install `yt-helper` as a uv-managed CLI tool:

```bash
uv tool install .
```

## Commands

Show all commands:

```bash
yt-helper --help
```

Download media:

```bash
yt-helper download --help
```

Inspect media information before downloading:

```bash
yt-helper info --help
```

Batch download:

```bash
yt-helper batch --help
```

Transcribe a video URL, preferring native subtitles first:

```bash
yt-helper transcribe --help
```

## Examples

Download the best available full video:

```bash
yt-helper download "https://youtu.be/dQw4w9WgXcQ"
```

Inspect a video before deciding what to download:

```bash
yt-helper info "https://youtu.be/dQw4w9WgXcQ"
```

Download visual-only video as `mkv`:

```bash
yt-helper download --mode visual-only --video-format mkv "https://youtu.be/dQw4w9WgXcQ"
```

Download audio only as `wav`:

```bash
yt-helper download --mode audio --audio-format wav "https://youtu.be/dQw4w9WgXcQ"
```

Start interactive batch input:

```bash
yt-helper batch
```

Transcribe a video URL:

```bash
yt-helper transcribe "https://youtu.be/dQw4w9WgXcQ"
```
