import logging
from pathlib import Path
from fait.audio.speech_to_text.models.whisper_faster_model import WhisperFasterModel

log = logging.getLogger(__name__)


def run_speech_to_text_pipeline(audio_path: Path, config) -> dict:
    """
    Runs the Speech-to-Text pipeline and returns transcription with metadata.

    Args:
        audio_path (Path): Path to the input audio file.
        config (SimpleNamespace): Loaded configuration namespace.

    Returns:
        dict: {
            "file_name": str,
            "language": str,
            "probability": float,
            "duration": float,
            "text": str
        }
    """

    if not hasattr(config.audio, "speech_to_text"):
        raise RuntimeError("ASR pipeline config missing 'audio.speech_to_text' section")

    stt_cfg = config.audio.speech_to_text
    engine = getattr(stt_cfg, "engine", "whisper-faster")
    model_id = getattr(stt_cfg, "model_id", None)
    device = getattr(stt_cfg, "device", "cpu")
    cache_dir = getattr(stt_cfg, "cache_dir", None)

    # ✅ Enforce a default model size if missing or invalid
    if not model_id or str(model_id).lower() == "none":
        model_id = "small"
        log.warning("⚠️ No valid model_id found in config — defaulting to 'small'")

    log.info(f"🎙️ Starting Speech-to-Text pipeline with engine={engine}")

    # ------------------------------------------------------------------
    # Select backend
    # ------------------------------------------------------------------
    if engine == "whisper-faster":
        log.info("⚙️ Using Faster-Whisper Speech-to-Text backend (CTranslate2)")

        model = WhisperFasterModel(
            model_size=model_id,
            device=device,
            compute_type="int8",
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"❌ Unsupported Speech-to-Text engine: {engine}")

    # ------------------------------------------------------------------
    # Run transcription
    # ------------------------------------------------------------------
    log.info(f"🎧 Processing input audio: {audio_path}")

    try:
        text, info = model.transcribe(audio_path)

        return {
            "file_name": Path(audio_path).name,
            "language": info.get("language", "unknown"),
            "probability": info.get("probability", 1.0),
            "duration": round(info.get("duration", 0.0), 2),
            "text": text.strip(),
        }

    except Exception as e:
        log.error(f"❌ Error during transcription: {e}", exc_info=True)
        raise RuntimeError(f"Speech-to-Text pipeline failed: {e}")

    finally:
        log.info("✅ Speech-to-Text transcription complete")