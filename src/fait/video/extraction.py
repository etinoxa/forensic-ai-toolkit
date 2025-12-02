# src/fait/video/extraction.py
"""
Video evidence extraction utilities.
Extracts audio tracks and video frames from video files using FFmpeg.
"""
from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import Optional, Literal

import numpy as np

try:
    from static_ffmpeg import run as static_ffmpeg_run

    _has_static_ffmpeg = True
except ImportError:
    _has_static_ffmpeg = False

from fait.core.utils import ensure_folder, file_md5

log = logging.getLogger("fait.video.extraction")


def _get_ffmpeg_exe() -> str:
    """Get FFmpeg executable path"""
    if not _has_static_ffmpeg:
        raise RuntimeError(
            "static-ffmpeg is not installed. Please run: pip install static-ffmpeg"
        )

    ffmpeg_path, _ = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
    return str(ffmpeg_path)


def _get_ffprobe_exe() -> str:
    """Get FFprobe executable path"""
    if not _has_static_ffmpeg:
        raise RuntimeError(
            "static-ffmpeg is not installed. Please run: pip install static-ffmpeg"
        )

    _, ffprobe_path = static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
    return str(ffprobe_path)


def get_video_metadata(video_path: str | Path) -> dict:
    """
    Extract metadata from video file using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing:
            - duration: Duration in seconds (float)
            - fps: Frames per second (float)
            - width: Video width in pixels (int)
            - height: Video height in pixels (int)
            - has_audio: Whether video contains audio stream (bool)
            - format: Container format (str)
            - video_codec: Video codec name (str)
            - audio_codec: Audio codec name or None (str | None)
            - file_size: File size in bytes (int)
            - md5_hash: MD5 hash for chain of custody (str)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffprobe = _get_ffprobe_exe()

    # Get JSON output from ffprobe
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json
    data = json.loads(result.stdout)

    # Parse streams
    video_stream = None
    audio_stream = None

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Extract metadata
    format_data = data.get("format", {})

    # Calculate FPS (handle various formats)
    fps_str = video_stream.get("r_frame_rate", "0/1")
    if "/" in fps_str:
        num, denom = fps_str.split("/")
        fps = float(num) / float(denom) if float(denom) > 0 else 0.0
    else:
        fps = float(fps_str)

    metadata = {
        "duration": float(format_data.get("duration", 0.0)),
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "has_audio": audio_stream is not None,
        "format": format_data.get("format_name", "unknown"),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "file_size": int(format_data.get("size", video_path.stat().st_size)),
        "md5_hash": file_md5(video_path),
    }

    log.info("video_metadata", extra={
        "file": str(video_path),
        "duration": f"{metadata['duration']:.2f}s",
        "resolution": f"{metadata['width']}x{metadata['height']}",
        "fps": f"{metadata['fps']:.2f}",
        "has_audio": metadata["has_audio"],
    })

    return metadata


def extract_audio(
        video_path: str | Path,
        output_dir: str | Path,
        format: Literal["wav", "mp3", "flac", "m4a"] = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
) -> Path:
    """
    Extract audio track from video file.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted audio
        format: Output audio format (wav, mp3, flac, m4a)
        sample_rate: Target sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
        offset: Start extraction at this time (seconds)
        duration: Extract only this duration (seconds)

    Returns:
        Path to extracted audio file

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video has no audio stream
        subprocess.CalledProcessError: If FFmpeg extraction fails
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_folder(output_dir)

    # Check if video has audio
    metadata = get_video_metadata(video_path)
    if not metadata["has_audio"]:
        raise RuntimeError(f"Video has no audio stream: {video_path}")

    # Build output filename
    output_file = output_dir / f"{video_path.stem}.{format}"

    ffmpeg = _get_ffmpeg_exe()

    # Build FFmpeg command
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-loglevel", "error",
    ]

    # Input options
    if offset is not None and offset > 0:
        cmd += ["-ss", str(offset)]

    cmd += ["-i", str(video_path)]

    # Duration limit
    if duration is not None and duration > 0:
        cmd += ["-t", str(duration)]

    # Audio codec and format options
    if format == "wav":
        cmd += [
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
        ]
    elif format == "mp3":
        cmd += [
            "-acodec", "libmp3lame",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-q:a", "2",  # VBR quality
        ]
    elif format == "flac":
        cmd += [
            "-acodec", "flac",
            "-ar", str(sample_rate),
            "-ac", str(channels),
        ]
    elif format == "m4a":
        cmd += [
            "-acodec", "aac",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-b:a", "128k",
        ]

    cmd += [str(output_file)]

    log.info("extract_audio:start", extra={
        "video": str(video_path),
        "output": str(output_file),
        "format": format,
        "sample_rate": sample_rate,
        "channels": channels,
    })

    # Execute FFmpeg
    subprocess.run(cmd, check=True, capture_output=True)

    if not output_file.exists():
        raise RuntimeError(f"Audio extraction failed: {output_file}")

    log.info("extract_audio:complete", extra={
        "output_file": str(output_file),
        "size_mb": output_file.stat().st_size / (1024 * 1024),
    })

    return output_file


def extract_frames(
        video_path: str | Path,
        output_dir: str | Path,
        fps: float = 1.0,
        format: Literal["jpg", "png", "bmp"] = "jpg",
        quality: int = 95,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_dimension: Optional[int] = None,
) -> Path:
    """
    Extract frames from video file at specified rate.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        fps: Extract this many frames per second (e.g., 1.0 = 1 frame/sec, 0.1 = 1 frame/10sec)
        format: Output image format (jpg, png, bmp)
        quality: JPEG quality (2-31, lower is better) or PNG compression (0-9)
        start_time: Start extraction at this time (seconds)
        end_time: End extraction at this time (seconds)
        max_dimension: Resize frames so largest dimension is this size (preserves aspect ratio)

    Returns:
        Path to directory containing extracted frames (same as output_dir)

    Raises:
        FileNotFoundError: If video file doesn't exist
        subprocess.CalledProcessError: If FFmpeg extraction fails
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_folder(output_dir)

    # Get video metadata
    metadata = get_video_metadata(video_path)

    ffmpeg = _get_ffmpeg_exe()

    # Build output pattern (frame_0001.jpg, frame_0002.jpg, etc.)
    output_pattern = output_dir / f"frame_%04d.{format}"

    # Build FFmpeg command
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
    ]

    # Input options
    if start_time is not None and start_time > 0:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", str(video_path)]

    # Duration/end time
    if end_time is not None:
        if start_time is not None:
            duration = end_time - start_time
        else:
            duration = end_time
        cmd += ["-t", str(duration)]

    # Frame rate
    cmd += ["-vf", f"fps={fps}"]

    # Resize if requested
    if max_dimension is not None:
        scale_filter = f"scale='if(gt(iw,ih),{max_dimension},-2)':'if(gt(iw,ih),-2,{max_dimension})'"
        cmd[-1] += f",{scale_filter}"

    # Quality settings
    if format == "jpg":
        cmd += ["-q:v", str(quality)]  # 2-31, lower is better
    elif format == "png":
        cmd += ["-compression_level", str(quality)]  # 0-9

    cmd += [str(output_pattern)]

    log.info("extract_frames:start", extra={
        "video": str(video_path),
        "output_dir": str(output_dir),
        "fps": fps,
        "format": format,
        "video_fps": f"{metadata['fps']:.2f}",
        "video_duration": f"{metadata['duration']:.2f}s",
    })

    # Execute FFmpeg
    subprocess.run(cmd, check=True, capture_output=True)

    # Count extracted frames
    frame_files = list(output_dir.glob(f"frame_*.{format}"))

    if not frame_files:
        raise RuntimeError(f"Frame extraction failed: no frames in {output_dir}")

    log.info("extract_frames:complete", extra={
        "output_dir": str(output_dir),
        "frames_extracted": len(frame_files),
        "total_size_mb": sum(f.stat().st_size for f in frame_files) / (1024 * 1024),
    })

    return output_dir