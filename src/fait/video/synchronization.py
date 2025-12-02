# src/fait/video/synchronization.py
"""
Synchronization utilities for aligning multi-modal analysis results.
Creates unified timelines combining speaker recognition, speech-to-text, and facial recognition.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

log = logging.getLogger("fait.video.synchronization")


@dataclass
class TimelineEvent:
    """Single event in the forensic timeline"""
    timestamp: float  # Seconds from video start
    frame_number: Optional[int] = None
    speaker_id: Optional[str] = None
    speaker_confidence: Optional[float] = None
    transcript: Optional[str] = None
    transcript_confidence: Optional[float] = None
    detected_faces: Optional[List[Dict[str, Any]]] = None  # [{"identity": str, "confidence": float}, ...]

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values"""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


def align_results(
        video_metadata: Dict[str, Any],
        speaker_results: Optional[Dict] = None,
        stt_results: Optional[Dict] = None,
        face_results: Optional[Dict] = None,
        tolerance_ms: float = 500.0,
) -> List[dict]:
    """
    Create unified timeline by synchronizing multi-modal analysis results.

    Args:
        video_metadata: Video metadata from get_video_metadata()
        speaker_results: Speaker recognition results (from speaker pipeline)
        stt_results: Speech-to-text results with timestamps
        face_results: Facial recognition results with frame numbers
        tolerance_ms: Tolerance window for temporal alignment (milliseconds)

    Returns:
        List of timeline events, sorted by timestamp. Each event contains:
            - timestamp: Time in seconds from video start
            - frame_number: Video frame number (if applicable)
            - speaker_id: Identified speaker (if any)
            - speaker_confidence: Speaker match confidence
            - transcript: Transcribed text (if any)
            - transcript_confidence: Transcription confidence
            - detected_faces: List of face detections with identities
    """
    fps = video_metadata.get("fps", 30.0)
    duration = video_metadata.get("duration", 0.0)

    tolerance_sec = tolerance_ms / 1000.0

    # Build timeline from all sources
    timeline: Dict[float, TimelineEvent] = {}

    # Process speech-to-text results
    if stt_results and "segments" in stt_results:
        for segment in stt_results["segments"]:
            timestamp = segment.get("start", 0.0)

            # Find or create event at this timestamp
            event = _get_or_create_event(timeline, timestamp, tolerance_sec)
            event.transcript = segment.get("text", "").strip()
            event.transcript_confidence = segment.get("confidence")

    # Process speaker recognition results
    if speaker_results and "matches" in speaker_results:
        for match in speaker_results["matches"]:
            # Speaker results may have time ranges
            start_time = match.get("start_time", 0.0)
            end_time = match.get("end_time", start_time)

            # Create events at start and end (or just start if same)
            event = _get_or_create_event(timeline, start_time, tolerance_sec)
            event.speaker_id = match.get("speaker_id")
            event.speaker_confidence = match.get("confidence")

            if end_time > start_time:
                # Mark the range
                end_event = _get_or_create_event(timeline, end_time, tolerance_sec)
                if not end_event.speaker_id:
                    end_event.speaker_id = match.get("speaker_id")
                    end_event.speaker_confidence = match.get("confidence")

    # Process facial recognition results
    if face_results and "matches" in face_results:
        for match in face_results["matches"]:
            # Face results are tied to specific frames
            frame_num = match.get("frame_number")
            if frame_num is None:
                continue

            # Convert frame number to timestamp
            timestamp = frame_num / fps if fps > 0 else 0.0

            event = _get_or_create_event(timeline, timestamp, tolerance_sec)
            event.frame_number = frame_num

            # Add face detection
            face_info = {
                "identity": match.get("identity", "unknown"),
                "confidence": match.get("confidence"),
                "bbox": match.get("bbox"),  # Bounding box if available
            }

            if event.detected_faces is None:
                event.detected_faces = []
            event.detected_faces.append(face_info)

    # Convert to sorted list
    events = sorted(timeline.values(), key=lambda e: e.timestamp)

    log.info("timeline:synchronized", extra={
        "total_events": len(events),
        "duration": f"{duration:.2f}s",
        "has_speaker": speaker_results is not None,
        "has_stt": stt_results is not None,
        "has_faces": face_results is not None,
    })

    return [e.to_dict() for e in events]


def _get_or_create_event(
        timeline: Dict[float, TimelineEvent],
        timestamp: float,
        tolerance: float
) -> TimelineEvent:
    """
    Get existing event near timestamp or create new one.

    Uses tolerance window to merge events that are temporally close.
    """
    # Check if there's an event within tolerance
    for existing_ts in list(timeline.keys()):
        if abs(existing_ts - timestamp) <= tolerance:
            return timeline[existing_ts]

    # Create new event
    event = TimelineEvent(timestamp=timestamp)
    timeline[timestamp] = event
    return event


def export_timeline_json(
        timeline: List[dict],
        output_path: str | Path,
        indent: int = 2
) -> Path:
    """
    Export timeline to JSON file.

    Args:
        timeline: Timeline events from align_results()
        output_path: Output JSON file path
        indent: JSON indentation (None for compact)

    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=indent, ensure_ascii=False)

    log.info("timeline:exported", extra={
        "output": str(output_path),
        "events": len(timeline),
    })

    return output_path


def export_timeline_srt(
        timeline: List[dict],
        output_path: str | Path,
        include_speakers: bool = True
) -> Path:
    """
    Export timeline as SRT subtitle file (useful for video review).

    Args:
        timeline: Timeline events from align_results()
        output_path: Output SRT file path
        include_speakers: Include speaker IDs in subtitles

    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    srt_entries = []

    for i, event in enumerate(timeline, start=1):
        transcript = event.get("transcript")
        if not transcript:
            continue

        start_time = event.get("timestamp", 0.0)

        # Estimate end time (next event or +3 seconds)
        if i < len(timeline):
            end_time = timeline[i].get("timestamp", start_time + 3.0)
        else:
            end_time = start_time + 3.0

        # Format timestamps as SRT requires (HH:MM:SS,mmm)
        start_srt = _seconds_to_srt_time(start_time)
        end_srt = _seconds_to_srt_time(end_time)

        # Build subtitle text
        text = transcript
        if include_speakers and event.get("speaker_id"):
            text = f"[{event['speaker_id']}] {text}"

        # SRT format: index, timecode, text, blank line
        srt_entries.append(f"{i}\n{start_srt} --> {end_srt}\n{text}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    log.info("timeline:exported_srt", extra={
        "output": str(output_path),
        "subtitles": len(srt_entries),
    })

    return output_path


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_forensic_report(
        timeline: List[dict],
        video_metadata: Dict[str, Any],
        output_path: str | Path,
        case_info: Optional[Dict[str, str]] = None
) -> Path:
    """
    Generate human-readable forensic report from timeline.

    Args:
        timeline: Timeline events from align_results()
        video_metadata: Video metadata
        output_path: Output text file path
        case_info: Optional case metadata (case_id, investigator, etc.)

    Returns:
        Path to written report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("FORENSIC VIDEO ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Case information
    if case_info:
        lines.append("CASE INFORMATION:")
        for key, value in case_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Video metadata
    lines.append("VIDEO METADATA:")
    lines.append(f"  File: {video_metadata.get('file', 'N/A')}")
    lines.append(f"  Duration: {video_metadata.get('duration', 0):.2f} seconds")
    lines.append(f"  Resolution: {video_metadata.get('width')}x{video_metadata.get('height')}")
    lines.append(f"  FPS: {video_metadata.get('fps', 0):.2f}")
    lines.append(f"  Format: {video_metadata.get('format', 'unknown')}")
    lines.append(f"  MD5 Hash: {video_metadata.get('md5_hash', 'N/A')}")
    lines.append("")

    # Analysis summary
    speaker_count = len(set(e.get("speaker_id") for e in timeline if e.get("speaker_id")))
    face_count = len(set(
        f["identity"]
        for e in timeline
        for f in e.get("detected_faces", [])
    ))
    transcript_segments = sum(1 for e in timeline if e.get("transcript"))

    lines.append("ANALYSIS SUMMARY:")
    lines.append(f"  Timeline Events: {len(timeline)}")
    lines.append(f"  Unique Speakers: {speaker_count}")
    lines.append(f"  Unique Faces: {face_count}")
    lines.append(f"  Transcript Segments: {transcript_segments}")
    lines.append("")

    # Timeline
    lines.append("DETAILED TIMELINE:")
    lines.append("-" * 80)

    for event in timeline:
        timestamp = event.get("timestamp", 0.0)
        mins = int(timestamp // 60)
        secs = timestamp % 60

        lines.append(f"\n[{mins:02d}:{secs:05.2f}]")

        if event.get("frame_number"):
            lines.append(f"  Frame: {event['frame_number']}")

        if event.get("speaker_id"):
            conf = event.get("speaker_confidence", 0.0)
            lines.append(f"  Speaker: {event['speaker_id']} (confidence: {conf:.2%})")

        if event.get("transcript"):
            lines.append(f"  Transcript: {event['transcript']}")

        if event.get("detected_faces"):
            lines.append(f"  Faces detected:")
            for face in event["detected_faces"]:
                identity = face.get("identity", "unknown")
                conf = face.get("confidence", 0.0)
                lines.append(f"    - {identity} (confidence: {conf:.2%})")

    # Footer
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("forensic_report:generated", extra={
        "output": str(output_path),
        "events": len(timeline),
    })

    return output_path