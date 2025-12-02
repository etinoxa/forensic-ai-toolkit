# examples/video_analysis_quickstart.py
"""
Video forensic analysis quickstart.
Simple example with hardcoded paths - no CLI arguments.
For CLI usage, see video_analysis_cli.py
"""
from __future__ import annotations

import sys
import pathlib

# Add repo /src to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from fait.core.logging_config import setup_logging

setup_logging()

from fait.video import run_video_analysis
from fait.core.paths import get_paths


def main():
    """Run video forensic analysis with default paths"""

    # Resolve paths
    paths = get_paths()
    gallery_dir = ROOT / "datasets" / "videos" / "gallery"
    speaker_refs = ROOT / "datasets" / "audio" / "reference"
    face_refs = ROOT / "datasets" / "images" / "face" / "reference_images"

    # Auto-generate output directory
    import time
    timestamp = int(time.time())
    output_dir = paths.outputs / "video" / f"analysis_{timestamp}"

    print("=" * 80)
    print("VIDEO FORENSIC ANALYSIS")
    print("=" * 80)
    print(f"Gallery: {gallery_dir}")
    print(f"Output: {output_dir}")
    print(f"Speaker refs: {speaker_refs}")
    print(f"Face refs: {face_refs}")
    print("=" * 80)
    print()

    # Run analysis (all settings from config.yaml)
    results = run_video_analysis(
        gallery_dir=gallery_dir,
        output_dir=output_dir,
        speaker_references=speaker_refs,
        face_references=face_refs,
    )

    # Display results
    print("\n" + "=" * 80)
    print("VIDEO ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total videos: {results.get('total_videos', 0)}")
    print(f"Successful: {results.get('successful', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Output directory: {results.get('run_dir')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()