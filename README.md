# yt-helper

[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![yt-dlp](https://img.shields.io/badge/powered%20by-yt--dlp-111111)](https://github.com/yt-dlp/yt-dlp)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`yt-helper` is a command-line tool for downloading YouTube media and turning spoken content into timestamped plain-text transcripts.

## Overview

| Area | What it does |
| --- | --- |
| Download modes | Download full video, video-only streams, or audio-only files, including playlist URLs. |
| Output formats | Export video as `mp4`, `mkv`, or `webm`, and audio as `wav`, `flac`, `m4a`, or `mp3`. |
| Reliability | Resume interrupted downloads and retry unstable transfers through `yt-dlp`. |
| Guided CLI | Prompt interactively for a URL when one is not provided. |
| Media inspection | Show title, duration, uploader, subtitle availability, estimated size, and available formats before downloading. |
| Batch workflow | Queue multiple URLs in one session and skip duplicates automatically. |
| Transcription | Prefer native subtitles when available, then fall back to Whisper ASR. |
| Acceleration | Use CUDA automatically for transcription when it is available. |

## Transcript Output

Transcripts are saved as timestamped plain text:

```text
[0.0s] Welcome, today we're going to talk about...
[12.3s] The first topic is X, which relates to...
[45.1s] So the key insight here is that...
```

## Installation

Install `yt-helper` as a `uv`-managed CLI tool:

```bash
uv tool install .
```

By default, downloaded files and generated transcripts are saved under
`~/Downloads/yt-helper`.

## Commands

| Command | Purpose |
| --- | --- |
| `yt-helper --help` | Show the top-level CLI help. |
| `yt-helper download --help` | Download media in full-video, video-only, or audio-only mode. |
| `yt-helper info --help` | Inspect media information before downloading. |
| `yt-helper batch --help` | Collect multiple URLs and process them in one batch. |
| `yt-helper transcribe --help` | Generate a transcript, preferring native subtitles first. |

## Examples

| Task | Command |
| --- | --- |
| Download the best available full video | `yt-helper download "https://youtu.be/dQw4w9WgXcQ"` |
| Inspect a video before downloading | `yt-helper info "https://youtu.be/dQw4w9WgXcQ"` |
| Download a video-only stream as `mkv` | `yt-helper download --mode visual-only --video-format mkv "https://youtu.be/dQw4w9WgXcQ"` |
| Download audio only as `wav` | `yt-helper download --mode audio --audio-format wav "https://youtu.be/dQw4w9WgXcQ"` |
| Start an interactive batch session | `yt-helper batch` |
| Generate a transcript from a video URL | `yt-helper transcribe "https://youtu.be/dQw4w9WgXcQ"` |
