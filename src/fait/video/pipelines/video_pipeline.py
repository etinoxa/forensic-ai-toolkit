# src/fait/video/video_pipeline.py
"""
Video evidence analysis pipeline.
Orchestrates audio and vision pipelines for comprehensive video forensic analysis.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Literal

from fait.video.extraction import extract_audio, extract_frames, get_video_metadata
from fait.video.synchronization import (
    align_results,
    export_timeline_json,
    export_timeline_srt,
    generate_forensic_report
)
from fait.core.utils import ensure_folder, is_video_file, ProgressMeter

log = logging.getLogger("fait.video.video_pipeline")


def run_video_analysis(
        gallery_dir: str | Path,
        output_dir: str | Path,
        speaker_references: Optional[str | Path] = None,
        face_references: Optional[str | Path] = None,
        enable_speaker_recognition: Optional[bool] = None,
        enable_speech_to_text: Optional[bool] = None,
        enable_facial_recognition: Optional[bool] = None,
        frame_extraction_fps: Optional[float] = None,
        audio_format: Optional[Literal["wav", "mp3", "flac", "m4a"]] = None,
        audio_sample_rate: Optional[int] = None,
        case_info: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Run comprehensive video forensic analysis pipeline.

    This function orchestrates existing pipelines - it doesn't implement new AI models.
    It extracts audio/frames from videos, then feeds them to existing audio and vision pipelines.

    Configuration precedence:
    1. Explicit parameters passed to this function (highest priority)
    2. config.yaml settings via get_app_config()
    3. Hard-coded defaults (lowest priority)

    Args:
        gallery_dir: Directory containing video files to analyze
        output_dir: Base directory for all outputs
        speaker_references: Directory containing reference audio samples for speaker matching
        face_references: Directory containing reference face images
        enable_speaker_recognition: Run speaker recognition on audio track (None = use config)
        enable_speech_to_text: Run STT on audio track (None = use config)
        enable_facial_recognition: Run face recognition on video frames (None = use config)
        frame_extraction_fps: Extract this many frames per second (None = use config)
        audio_format: Audio extraction format (None = use config)
        audio_sample_rate: Audio sample rate for extraction (None = use config)
        case_info: Optional case metadata for forensic report

    Returns:
        Dictionary containing:
            - total_videos: Number of videos processed
            - successful: Number successfully processed
            - failed: Number that failed
            - results: List of per-video results (each containing video_metadata, timeline, etc.)
            - run_dir: Path to complete output directory
    """
    t0 = time.time()

    gallery_dir = Path(gallery_dir)
    output_dir = Path(output_dir)

    ensure_folder(output_dir)

    # Load config defaults from app_config
    from fait.core.app_config import get_app_config
    app_config = get_app_config()
    video_cfg = app_config.video

    # Apply config defaults if parameters not specified
    if frame_extraction_fps is None:
        frame_extraction_fps = video_cfg.extraction.frame_fps
    if audio_format is None:
        audio_format = video_cfg.extraction.audio_format
    if audio_sample_rate is None:
        audio_sample_rate = video_cfg.extraction.audio_sample_rate
    if enable_speaker_recognition is None:
        enable_speaker_recognition = video_cfg.analysis.enable_speaker_recognition
    if enable_speech_to_text is None:
        enable_speech_to_text = video_cfg.analysis.enable_speech_to_text
    if enable_facial_recognition is None:
        enable_facial_recognition = video_cfg.analysis.enable_facial_recognition

    # Collect video files to process
    video_files = [f for f in sorted(gallery_dir.iterdir()) if f.is_file() and is_video_file(f)]

    total = len(video_files)
    log.info("video_analysis:start", extra={
        "gallery_dir": str(gallery_dir),
        "total_videos": total,
        "output_dir": str(output_dir),
        "speaker_enabled": enable_speaker_recognition,
        "stt_enabled": enable_speech_to_text,
        "face_enabled": enable_facial_recognition,
        "frame_fps": frame_extraction_fps,
        "audio_format": audio_format,
    })

    # Progress tracking
    pm = ProgressMeter(
        total=total,
        label="video_analysis:progress",
        logger=log,
        emit_every_n=1,
        emit_every_sec=5.0,
    )

    processed = 0
    successful = 0
    failed = 0
    all_results = []

    # Process each video
    for video_file in video_files:
        processed += 1
        video_name = video_file.stem

        log.info(f"video_analysis:processing", extra={
            "video": str(video_file),
            "progress": f"{processed}/{total}",
        })

        # Create per-video output directory
        video_output_dir = output_dir / video_name

        try:
            result = _process_single_video(
                video_file=video_file,
                output_dir=video_output_dir,
                speaker_references=speaker_references,
                face_references=face_references,
                enable_speaker_recognition=enable_speaker_recognition,
                enable_speech_to_text=enable_speech_to_text,
                enable_facial_recognition=enable_facial_recognition,
                frame_extraction_fps=frame_extraction_fps,
                audio_format=audio_format,
                audio_sample_rate=audio_sample_rate,
                video_cfg=video_cfg,
                case_info=case_info,
            )
            result["status"] = "success"
            all_results.append(result)
            successful += 1

        except Exception as e:
            log.error("video_analysis:failed", extra={
                "video": str(video_file),
                "error": str(e),
            })
            all_results.append({
                "video_file": str(video_file),
                "status": "failed",
                "error": str(e),
            })
            failed += 1

        pm.set_counts(processed, found=successful, review=failed)

    pm.close()

    duration = time.time() - t0
    log.info("video_analysis:complete", extra={
        "total_videos": total,
        "successful": successful,
        "failed": failed,
        "duration_sec": round(duration, 2),
    })

    return {
        "total_videos": total,
        "successful": successful,
        "failed": failed,
        "results": all_results,
        "run_dir": str(output_dir),
    }


def _process_single_video(
        video_file: Path,
        output_dir: Path,
        speaker_references: Optional[Path],
        face_references: Optional[Path],
        enable_speaker_recognition: bool,
        enable_speech_to_text: bool,
        enable_facial_recognition: bool,
        frame_extraction_fps: float,
        audio_format: str,
        audio_sample_rate: int,
        video_cfg,
        case_info: Optional[Dict[str, str]],
) -> Dict:
    """Process a single video file. Internal helper for run_video_analysis."""
    ensure_folder(output_dir)

    results = {}

    # Step 1: Extract video metadata and verify chain of custody
    video_metadata = get_video_metadata(video_file)
    video_metadata["file"] = str(video_file)
    results["video_file"] = str(video_file)
    results["video_metadata"] = video_metadata

    # Step 2: Extract audio track (if audio analysis is enabled)
    audio_file = None
    if (enable_speaker_recognition or enable_speech_to_text) and video_metadata["has_audio"]:
        audio_dir = output_dir / "audio"
        audio_file = extract_audio(
            video_path=video_file,
            output_dir=audio_dir,
            format=audio_format,
            sample_rate=audio_sample_rate,
            channels=1,  # Mono for forensic analysis
        )
        results["audio_file"] = str(audio_file)

    # Step 3: Extract video frames (if facial recognition is enabled)
    frames_dir = None
    if enable_facial_recognition:
        frames_dir = output_dir / "frames"
        extract_frames(
            video_path=video_file,
            output_dir=frames_dir,
            fps=frame_extraction_fps,
            format="jpg",
            quality=95,
        )
        results["frames_dir"] = str(frames_dir)

    # Step 4: Run speaker recognition (if enabled and audio exists)
    speaker_results = None
    if enable_speaker_recognition and audio_file and speaker_references:
        try:
            from fait.audio.pipelines.speaker_recognition_pipeline import run_speaker_match

            speaker_results = run_speaker_match(
                reference_dir=str(speaker_references),
                gallery_dir=str(audio_file.parent),
                output_dir=str(output_dir / "speaker_analysis"),
            )
            results["speaker_results"] = speaker_results
        except ImportError as e:
            log.warning("speaker_recognition:unavailable", extra={"error": str(e)})
        except Exception as e:
            log.error("speaker_recognition:failed", extra={"error": str(e)})

    # Step 5: Run speech-to-text (if enabled and audio exists)
    stt_results = None
    if enable_speech_to_text and audio_file:
        try:
            from fait.audio.pipelines.speech_to_text_pipeline import run_speech_to_text

            stt_results = run_speech_to_text(
                audio_file=str(audio_file),
                output_dir=str(output_dir / "transcripts"),
            )
            results["stt_results"] = stt_results
        except ImportError as e:
            log.warning("speech_to_text:unavailable", extra={"error": str(e)})
        except Exception as e:
            log.error("speech_to_text:failed", extra={"error": str(e)})

    # Step 6: Run facial recognition (if enabled and frames exist)
    face_results = None
    if enable_facial_recognition and frames_dir and face_references:
        try:
            # Import models to trigger registration decorators
            import fait.vision.facial_recognition.models.arcface as _  # noqa: F401
            import fait.vision.facial_recognition.models.clip as __  # noqa: F401

            from fait.vision.pipelines.facial_recognition_pipeline import run_facial_recognition
            from fait.core.registry import get_embedder
            from fait.core.app_config import get_app_config

            # Get face recognition config
            app_config = get_app_config()
            face_cfg = app_config.vision.face_recognition

            # Get the embedder (arcface or clip)
            embedder = get_embedder(face_cfg.recognizer)

            # Determine metric
            metric = face_cfg.metric
            if metric == "auto":
                metric = "euclidean" if face_cfg.recognizer == "arcface" else "cosine"

            face_results = run_facial_recognition(
                embedder=embedder,
                reference_dir=str(face_references),
                gallery_dir=str(frames_dir),
                thresholds=face_cfg.thresholds,
                metric=metric,
                plot_results=face_cfg.plot_results,
            )
            results["face_results"] = face_results
        except ImportError as e:
            log.warning("facial_recognition:unavailable", extra={"error": str(e)})
        except Exception as e:
            log.error("facial_recognition:failed", extra={"error": str(e)})

    # Step 7: Synchronize results into unified timeline
    tolerance_ms = video_cfg.analysis.tolerance_ms
    timeline = align_results(
        video_metadata=video_metadata,
        speaker_results=speaker_results,
        stt_results=stt_results,
        face_results=face_results,
        tolerance_ms=tolerance_ms,
    )
    results["timeline"] = timeline

    # Step 8: Export timeline in multiple formats (if enabled in config)
    timeline_dir = output_dir / "timeline"
    ensure_folder(timeline_dir)

    if video_cfg.analysis.export_timeline_json:
        timeline_json = export_timeline_json(
            timeline=timeline,
            output_path=timeline_dir / "timeline.json",
        )
        results["timeline_json"] = str(timeline_json)

    if video_cfg.analysis.export_timeline_srt:
        timeline_srt = export_timeline_srt(
            timeline=timeline,
            output_path=timeline_dir / "timeline.srt",
            include_speakers=True,
        )
        results["timeline_srt"] = str(timeline_srt)

    # Step 9: Generate forensic report (if enabled in config)
    if video_cfg.analysis.export_forensic_report:
        forensic_report = generate_forensic_report(
            timeline=timeline,
            video_metadata=video_metadata,
            output_path=output_dir / "forensic_report.txt",
            case_info=case_info,
        )
        results["forensic_report"] = str(forensic_report)

    # Final results
    results["run_dir"] = str(output_dir)

    return results


def run_video_analysis_auto(
        gallery_dir: str | Path,
        output_dir: Optional[str | Path] = None,
        references_dir: Optional[str | Path] = None,
) -> Dict:
    """
    Simplified interface for quick video analysis with auto-detection of reference materials.

    All settings come from config.yaml. This is a convenience wrapper that:
    - Auto-generates output directory
    - Auto-detects reference materials
    - Uses all config.yaml defaults

    Args:
        gallery_dir: Directory containing video files
        output_dir: Output directory (auto-generated if None)
        references_dir: Base directory containing 'audio/' and 'images/' subdirs with references

    Returns:
        Same as run_video_analysis()

    Example:
        >>> results = run_video_analysis_auto(
        ...     gallery_dir="datasets/video/gallery/",
        ...     references_dir="datasets/"
        ... )
    """
    gallery_dir = Path(gallery_dir)

    # Auto-generate output directory
    if output_dir is None:
        from fait.core.paths import get_paths
        paths = get_paths()
        timestamp = int(time.time())
        output_dir = paths.outputs / "video" / f"analysis_{timestamp}"

    # Auto-detect reference directories
    speaker_refs = None
    face_refs = None

    if references_dir:
        references_dir = Path(references_dir)

        # Look for audio references
        audio_ref = references_dir / "audio" / "reference"
        if audio_ref.exists():
            speaker_refs = audio_ref

        # Look for face references
        face_ref = references_dir / "images" / "face" / "reference_images"
        if face_ref.exists():
            face_refs = face_ref

    return run_video_analysis(
        gallery_dir=gallery_dir,
        output_dir=output_dir,
        speaker_references=speaker_refs,
        face_references=face_refs,
        # All other parameters use config.yaml defaults
    )