"""
Speech-to-Text Quickstart Example
---------------------------------
Runs the Speech-to-Text pipeline using the same config.yaml.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import pathlib
import time

# Add src/ to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from fait.core.paths import get_paths
from fait.core.utils import load_yaml
from fait.audio.services.speech_to_text_service import SpeechToTextService
from fait.audio.services.translation_service import TranslationService


# ==========================================================
# 🪵 LOGGING SETUP
# ==========================================================
def setup_logging():
    """Configure logging for both console and persistent file outputs."""
    paths = get_paths()
    log_dir = Path(paths.repo_root) / ".fait" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "fait.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # --- File handler ---
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    print(f"🪵 Logging initialized — file: {log_file}\n")
    return log_file


def write_separator(log_file: Path, title: str):
    """Write a visual separator to both console and log."""
    separator = (
        "\n" + "=" * 100 +
        f"\n🧭 {title} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" +
        "=" * 100 + "\n"
    )
    print(separator)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(separator)


# ==========================================================
# 🗂️ HELPERS
# ==========================================================
def collect_audio_files(audio_dir: Path):
    """Collect supported audio files under the given path."""
    files = []
    if audio_dir.is_dir():
        for ext in [".mp3", ".wav", ".m4a", ".flac"]:
            files.extend(audio_dir.rglob(f"*{ext}"))
    elif audio_dir.is_file():
        files.append(audio_dir)
    return files


def print_summary(results):
    """Pretty-print transcription summary."""
    lines = ["\n================= 🧾 TRANSCRIPTION SUMMARY =================\n"]
    for r in results:
        lines.append(f"🎵 File: {r['file_name']}")
        lines.append(f"🌍 Detected Language: {r['language']} ({r['probability']})")
        lines.append(f"⏱️ Duration: {r['duration']} seconds\n")
        lines.append("🗣️  Transcription:")
        lines.append(r["text"].strip())
        if r.get("translation"):
            lines.append("\n🌐 Translation:")
            lines.append(r["translation"].strip())
        lines.append("\n" + "=" * 80 + "\n")
    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text


def save_summary(summary_text, output_dir: Path):
    """Save summary text to output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"transcription_{timestamp}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"🗂️ Transcription saved to: {output_path}\n")


# ==========================================================
# 🚀 MAIN ENTRY
# ==========================================================
def main():
    log_file = setup_logging()
    global_start = time.time()

    parser = argparse.ArgumentParser(description="Speech-to-Text Quickstart Example")
    parser.add_argument(
        "--audio",
        help="Path to audio file or folder (default: datasets/audio/speech_to_text)",
        default=None,
    )
    parser.add_argument("--engine", help="Override engine (default from config.yaml)", default=None)
    parser.add_argument("--translate", action="store_true", help="Enable translation after transcription (optional)")
    parser.add_argument("--target-lang", default="en", help="Target language for translation (default: en)")
    args = parser.parse_args()

    paths = get_paths()
    config_path = Path(paths.repo_root) / "config.yaml"

    if not config_path.exists():
        print(f"⚠️ No config.yaml found at {config_path}. Using default settings.\n")
        cfg = {}
    else:
        cfg = load_yaml(config_path)
        print(f"✅ Loaded config from {config_path}\n")

    stt_cfg = cfg.get("audio", {}).get("speech_to_text", {})

    # Resolve dataset path
    default_dataset = Path(paths.repo_root) / "datasets" / "audio" / "speech_to_text"
    audio_path = Path(args.audio).expanduser().resolve() if args.audio else default_dataset

    if not audio_path.exists():
        raise FileNotFoundError(f"❌ Audio path not found: {audio_path}")

    audio_files = collect_audio_files(audio_path)
    if not audio_files:
        raise FileNotFoundError("❌ No valid audio files found in dataset folder.")

    total = len(audio_files)
    print(f"🎙️ Found {total} audio file(s) to process from: {audio_path}\n")

    # === Use config.yaml defaults or CLI overrides ===
    engine = args.engine or stt_cfg.get("engine", "whisper-faster")
    output_dir = Path(
        stt_cfg.get("output_dir", paths.repo_root / ".fait" / "outputs" / "audio" / "speech_to_text")
    ).expanduser().resolve()

    print("=== Speech-to-Text Configuration ===")
    print(f"Engine        : {engine}")
    print(f"Translate     : {'Yes' if args.translate else 'No'}")
    print(f"Target Lang   : {args.target_lang}")
    print(f"Dataset Path  : {audio_path}")
    print(f"Output Dir    : {output_dir}")
    print("=============================================================\n")

    # Initialize services
    service = SpeechToTextService(engine=engine)
    translator = TranslationService(target_lang=args.target_lang) if args.translate else None

    results = []

    for idx, audio_file in enumerate(audio_files, start=1):
        # Add separator for each file
        write_separator(log_file, f"RUN STARTED — Processing {idx}/{total}: {audio_file.name}")

        text, meta = service.transcribe(audio_file)
        translation_text = None

        if translator:
            print(f"🌍 Translating output from {meta.get('language', 'unknown')} → {args.target_lang}")
            try:
                translation_text = translator.translate_audio(audio_file, source_lang=meta.get("language"))
            except Exception as te:
                translation_text = f"[Translation failed: {te}]"

        results.append({
            "file_name": audio_file.name,
            "language": meta.get("language", "unknown"),
            "probability": meta.get("probability", 1.0),
            "duration": meta.get("duration", "?"),
            "text": text,
            "translation": translation_text,
        })

    # Global summary
    total_time = round(time.time() - global_start, 2)
    print("\n✅ All transcriptions complete!")
    summary_text = print_summary(results)
    save_summary(summary_text, output_dir)

    global_footer = (
        "\n" + "=" * 100 +
        f"\n🧾 RUN COMPLETE — Processed {total} file(s) in {total_time} seconds\n" +
        "=" * 100 + "\n"
    )
    print(global_footer)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(global_footer)


if __name__ == "__main__":
    main()