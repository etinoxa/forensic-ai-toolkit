import logging
from pathlib import Path
from types import SimpleNamespace

from fait.core.utils import load_yaml
from fait.core.paths import get_paths


def _to_namespace(obj):
    """Recursively convert dicts into SimpleNamespace objects for dot access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_namespace(i) for i in obj]
    return obj


class SpeechToTextService:
    """
    Main Speech-to-Text service responsible for orchestrating transcription
    using the configured or overridden backend engine.
    """

    def __init__(self, engine: str = None, summarize: bool = True):
        self.log = logging.getLogger(__name__)

        # ✅ Load config from repo root
        paths = get_paths()
        config_path = Path(paths.repo_root) / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {config_path}")

        config_dict = load_yaml(config_path)
        self.config = _to_namespace(config_dict)

        # ✅ Validate config structure
        if not hasattr(self.config, "audio") or not hasattr(self.config.audio, "speech_to_text"):
            raise ValueError("❌ Config missing required 'audio.speech_to_text' section.")

        # ✅ CLI override for engine
        if engine:
            self.config.audio.speech_to_text.engine = engine

        # ✅ Model fallback
        if not getattr(self.config.audio.speech_to_text, "model_id", None):
            self.log.warning("⚠️ No model_id found in config — defaulting to 'small'")
            self.config.audio.speech_to_text.model_id = "small"

        # ✅ Handle cache_dir if defined
        cache_dir = getattr(self.config.audio.speech_to_text, "cache_dir", None)
        if cache_dir:
            cache_dir = Path(cache_dir).expanduser().resolve()
            self.config.audio.speech_to_text.cache_dir = str(cache_dir)
           # self.log.info(f"📦 Model cache directory set to: {cache_dir}")
        else:
            self.log.info("⚠️ No cache_dir specified — Faster-Whisper will use its default cache location")

        self.summarize = summarize

        # ✅ Log startup summary
        self.log.info("🚀 Starting Speech-to-Text service")
        self.log.info(f"🎙️ Engine selected: {self.config.audio.speech_to_text.engine}")
        self.log.info(f"🧠 Model ID: {self.config.audio.speech_to_text.model_id}")

    # ---------------------------------------------------------------------
    # 🔊 TRANSCRIBE METHOD
    # ---------------------------------------------------------------------
    def transcribe(self, audio_path: Path):
        """
        Perform speech-to-text transcription and return both text + metadata.

        Args:
            audio_path (Path): Path to the input audio file.

        Returns:
            tuple: (text: str, metadata: dict)
        """
        try:
            # ✅ Import here to prevent circular dependency
            from fait.audio.pipelines.speech_to_text_pipeline import run_speech_to_text_pipeline

            self.log.info(f"🎙️ Input audio: {audio_path}")

            # Run the actual pipeline
            result = run_speech_to_text_pipeline(audio_path=audio_path, config=self.config)

            # ✅ Normalize results
            text_result = ""
            metadata = {
                "file_name": audio_path.name,
                "language": "unknown",
                "probability": 1.0,
                "duration": "?",
            }

            if isinstance(result, tuple):
                # (text, metadata_dict)
                text_result, meta = result
                if isinstance(meta, dict):
                    metadata.update(meta)
            elif isinstance(result, dict):
                # Structured dict result
                text_result = result.get("text", "")
                metadata.update({
                    "language": result.get("language", "unknown"),
                    "probability": result.get("probability", 1.0),
                    "duration": result.get("duration", "?"),
                })
            else:
                # Plain string fallback
                text_result = str(result)

            # Estimate duration if segments exist
            if hasattr(result, "segments"):
                total_duration = sum(
                    (s.end - s.start)
                    for s in getattr(result, "segments", [])
                    if hasattr(s, "end") and hasattr(s, "start")
                )
                metadata["duration"] = round(total_duration, 2)

            if self.summarize:
                self.log.info("📝 Summarization/reporting is enabled (future feature).")

            return text_result, metadata

        except Exception as e:
            self.log.error(f"❌ Speech-to-Text service failed: {e}", exc_info=True)
            raise RuntimeError(f"Speech-to-Text failed: {e}")