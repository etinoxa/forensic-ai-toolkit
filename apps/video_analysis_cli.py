# examples/video_analysis_cli.py
"""
Video forensic analysis CLI.
Provides command-line interface with argument parsing.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from fait.core.logging_config import setup_logging
from fait.video import run_video_analysis


def main():
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Video forensic analysis - process video evidence with speaker and face recognition"
    )

    # Required arguments
    parser.add_argument(
        "--gallery-dir",
        required=True,
        help="Directory containing video files to analyze"
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--speaker-refs",
        default=None,
        help="Directory containing speaker reference audio samples"
    )
    parser.add_argument(
        "--face-refs",
        default=None,
        help="Directory containing face reference images"
    )

    # Analysis options
    parser.add_argument(
        "--enable-speaker",
        action="store_true",
        default=None,
        help="Enable speaker recognition (overrides config)"
    )
    parser.add_argument(
        "--disable-speaker",
        action="store_true",
        help="Disable speaker recognition"
    )
    parser.add_argument(
        "--enable-stt",
        action="store_true",
        default=None,
        help="Enable speech-to-text (overrides config)"
    )
    parser.add_argument(
        "--disable-stt",
        action="store_true",
        help="Disable speech-to-text"
    )
    parser.add_argument(
        "--enable-faces",
        action="store_true",
        default=None,
        help="Enable facial recognition (overrides config)"
    )
    parser.add_argument(
        "--disable-faces",
        action="store_true",
        help="Disable facial recognition"
    )

    # Extraction options
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame extraction rate (frames per second)"
    )
    parser.add_argument(
        "--audio-format",
        choices=["wav", "mp3", "flac", "m4a"],
        default=None,
        help="Audio extraction format"
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=None,
        help="Audio sample rate in Hz"
    )

    args = parser.parse_args()

    # Resolve enable/disable flags
    enable_speaker = None
    if args.enable_speaker:
        enable_speaker = True
    elif args.disable_speaker:
        enable_speaker = False

    enable_stt = None
    if args.enable_stt:
        enable_stt = True
    elif args.disable_stt:
        enable_stt = False

    enable_faces = None
    if args.enable_faces:
        enable_faces = True
    elif args.disable_faces:
        enable_faces = False

    # Run analysis
    results = run_video_analysis(
        gallery_dir=args.gallery_dir,
        output_dir=args.output_dir,
        speaker_references=args.speaker_refs,
        face_references=args.face_refs,
        enable_speaker_recognition=enable_speaker,
        enable_speech_to_text=enable_stt,
        enable_facial_recognition=enable_faces,
        frame_extraction_fps=args.fps,
        audio_format=args.audio_format,
        audio_sample_rate=args.audio_sample_rate,
    )

    # Display results
    print("\n" + "=" * 80)
    print("VIDEO ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total videos: {results.get('total_videos', 0)}")
    print(f"Successful: {results.get('successful', 0)}")
    print(f"Failed: {results.get('failed', 0)}")
    print(f"Output directory: {results.get('run_dir')}")
    print("=" * 80)


if __name__ == "__main__":
    main()