# src/fait/video/__init__.py
"""
Video evidence analysis module.

This module provides utilities for processing video evidence in forensic investigations.
It extracts audio and video frames, then orchestrates existing audio and vision pipelines
for comprehensive multi-modal analysis.
"""

from fait.video.extraction import (
    extract_audio,
    extract_frames,
    get_video_metadata,
)

from fait.video.synchronization import (
    align_results,
    export_timeline_json,
    export_timeline_srt,
    generate_forensic_report,
    TimelineEvent,
)

from fait.video.pipelines.video_pipeline import (
    run_video_analysis,
    run_video_analysis_auto,
)

__all__ = [
    # Extraction utilities
    "extract_audio",
    "extract_frames",
    "get_video_metadata",

    # Synchronization utilities
    "align_results",
    "export_timeline_json",
    "export_timeline_srt",
    "generate_forensic_report",
    "TimelineEvent",

    # Main pipeline
    "run_video_analysis",
    "run_video_analysis_auto",
]