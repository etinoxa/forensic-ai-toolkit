import argparse
import logging
from pathlib import Path
from datetime import datetime
import time

from fait.audio.services.speech_to_text_service import SpeechToTextService
from fait.audio.services.translation_service import TranslationService
from fait.core.utils import load_yaml
from fait.core.paths import get_paths


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

    # File logger
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console logger
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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
def collect_audio_files(audio_input: Path):
    """Parse input path for multiple audio files or folders."""
    audio_files = []
    path = Path(audio_input)
    if path.is_file():
        audio_files.append(path)
    elif path.is_dir():
        for ext in [".mp3", ".wav", ".m4a", ".flac"]:
            audio_files.extend(path.rglob(f"*{ext}"))
    else:
        print(f"⚠️ Skipping invalid path: {path}")
    return audio_files


def resolve_output_dir(cli_output_dir: str = None):
    """Determine where to save transcription results."""
    if cli_output_dir:
        output_dir = Path(cli_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory resolved from CLI: {output_dir}\n")
        return output_dir

    try:
        paths = get_paths()
        config_path = Path(paths.repo_root) / "config.yaml"
        if config_path.exists():
            cfg = load_yaml(config_path)
            stt_cfg = cfg.get("audio", {}).get("speech_to_text", {})
            if isinstance(stt_cfg, dict) and stt_cfg.get("output_dir"):
                output_dir = Path(stt_cfg["output_dir"]).expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 Output directory resolved from config.yaml: {output_dir}\n")
                return output_dir
    except Exception:
        pass

    print("⚠️ No output directory specified (CLI or config). Results will not be saved.\n")
    return None


def print_summary(results):
    """Display and return a clean transcription summary."""
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


def save_summary_to_file(summary_text: str, output_dir: Path):
    """Save the transcription summary into the given output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"transcription_{timestamp}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"🗂️ Transcription saved to: {output_path}\n")


# ==========================================================
# 🚀 MAIN ENTRY POINT
# ==========================================================
def main():
    log_file = setup_logging()
    global_start = time.time()

    parser = argparse.ArgumentParser(description="FAIT Speech-to-Text CLI")
    parser.add_argument(
        "--audio",
        help="Path to audio file or folder (default: datasets/audio/speech_to_text)",
        default=None
    )
    parser.add_argument("--engine", default=None, help="Override Speech-to-Text engine (e.g. whisper-faster)")
    parser.add_argument("--translate", action="store_true", help="Enable translation after transcription (optional)")
    parser.add_argument("--target-lang", default="en", help="Target language for translation (default: en)")
    parser.add_argument("--output-dir", default=None, help="Custom directory to save transcription results (optional)")
    args = parser.parse_args()

    try:
        paths = get_paths()

        # ✅ Default datasets folder
        default_audio_path = Path(paths.repo_root) / "datasets" / "audio" / "speech_to_text"
        audio_input = Path(args.audio).expanduser().resolve() if args.audio else default_audio_path

        if not audio_input.exists():
            raise FileNotFoundError(f"❌ Audio path not found: {audio_input}")

        print("🎧 Starting FAIT Speech-to-Text...\n")
        print(f"📂 Using audio dataset: {audio_input}\n")

        audio_files = collect_audio_files(audio_input)
        if not audio_files:
            raise FileNotFoundError("❌ No valid audio files found in the given path.")

        total = len(audio_files)
        output_dir = resolve_output_dir(args.output_dir)

        # ✅ Initialize Speech-to-Text
        service = SpeechToTextService(engine=args.engine)
        log = logging.getLogger("fait.audio.speech_to_text_cli")
        log.info("🚀 Speech-to-Text service initialized successfully.")

        # === Display Configuration Summary ===
        print("=== Speech-to-Text Configuration ===")
        print(f"Engine        : {args.engine or 'Default (from config)'}")
        print(f"Model ID      : {getattr(service, 'model_id', 'Unknown')}")
        print(f"Translate     : {'Yes' if args.translate else 'No'}")
        print(f"Target Lang   : {args.target_lang}")
        print(f"Audio Path    : {audio_input}")
        print(f"Audio Count   : {total}")
        print(f"Output Dir    : {output_dir or '(none — results not saved)'}")
        print("=============================================================\n")

        results = []

        # ✅ Initialize translation (if enabled)
        translator = None
        if args.translate:
            translator = TranslationService(target_lang=args.target_lang)
            log.info(f"🌐 Translation enabled → Target: {args.target_lang}")
        else:
            log.info("🌐 Translation disabled (run with --translate to enable)")

        # Process each file
        for idx, audio_file in enumerate(audio_files, start=1):
            write_separator(log_file, f"RUN STARTED — Processing {idx}/{total}: {audio_file.name}")
            print(f"🔊 Processing {idx}/{total}: {audio_file.name}", flush=True)
            log.info(f"🎙️ Input audio: {audio_file}")

            text, meta = service.transcribe(audio_file)
            translation_text = None

            # ✅ Translation (safe whisper re-run)
            if translator:
                print(f"🌍 Translating output from {meta.get('language', 'unknown')} → {args.target_lang}...")
                try:
                    translation_text = translator.translate_audio(
                        audio_file, source_lang=meta.get("language")
                    )
                except Exception as te:
                    translation_text = f"[Translation failed: {te}]"
                    log.warning(f"Translation failed for {audio_file.name}: {te}")

            results.append({
                "file_name": meta.get("file_name", audio_file.name),
                "language": meta.get("language", "unknown"),
                "probability": meta.get("probability", 1.0),
                "duration": meta.get("duration", "?"),
                "text": text,
                "translation": translation_text,
            })

        log.info("📝 Summarization/reporting is enabled (future feature).")

        print("\n✅ All transcriptions complete!")
        summary_text = print_summary(results)

        if output_dir:
            save_summary_to_file(summary_text, output_dir)
        else:
            print("📝 Output directory not set — results not saved.\n")

        # Final footer (single)
        total_time = round(time.time() - global_start, 2)
        footer = (
            "\n" + "=" * 100 +
            f"\n🧾 RUN COMPLETE — Processed {total} file(s) in {total_time} seconds\n" +
            "=" * 100 + "\n"
        )
        print(footer)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(footer)

    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Unhandled error in Speech-to-Text CLI")
        exit(1)


if __name__ == "__main__":
    main()