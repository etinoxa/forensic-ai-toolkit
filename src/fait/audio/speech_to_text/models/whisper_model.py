"""
ASR model wrapper using OpenAI Whisper via Hugging Face Transformers.

This version supports:
- Dynamic model selection from config (e.g. openai/whisper-small, medium, large)
- Optional translation
- Compatibility with FAIT’s ASR pipeline (config + logging)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from transformers import pipeline
from fait.core.logging_config import get_logger
from fait.core.paths import get_paths, ensure_on_first_write

log = get_logger("fait.audio.speech_to_text.whisper")


class WhisperSpeechToText:
    """
    Wrapper for the Whisper model using Hugging Face Transformers.
    Provides multilingual speech recognition and optional translation.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        model_dir: Optional[str | Path] = None,
        device: str = "cpu",
        translate_to: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_id: Hugging Face model name (e.g., "openai/whisper-small")
            model_dir: Optional directory to cache model files.
            device: Device to run inference on ("cpu" or "cuda")
            translate_to: Optional language code (e.g. "en", "fr", etc.)
        """
        self.model_id = model_id
        self.device = device
        self.translate_to = translate_to

        paths = get_paths()
        default_cache = paths.models_audio / "speech_to_text" / "whisper" / model_id.replace("/", "_")
        self.model_dir = Path(model_dir or default_cache)
        ensure_on_first_write(self.model_dir)

        self._pipeline = None

    # ------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Lazy-loads the Whisper model via Hugging Face Transformers."""
        if self._pipeline is not None:
            return

        log.info(f"🔄 Loading Whisper model: {self.model_id}")
        try:
            self._pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1,
            )
            log.info(f"✅ Whisper model loaded successfully ({self.model_id})")
        except Exception as e:
            log.exception(f"❌ Failed to load Whisper model: {e}")
            raise

    # ------------------------------------------------------------------
    # TRANSCRIPTION
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribes audio using Whisper (with optional translation)."""
        if self._pipeline is None:
            self.load_model()

        log.info(f"🎙️ Transcribing audio: {audio_path}")
        try:
            result = self._pipeline(str(audio_path))
            text = result["text"]
            log.info("✅ Transcription completed successfully.")
        except Exception as e:
            log.exception(f"❌ Whisper transcription failed: {e}")
            raise

        # Optional translation (if translate_to specified)
        if self.translate_to and self.translate_to.lower() != "none":
            try:
                from fait.audio.speech_to_text.translation_utils import translate_text
                translated = translate_text(text, self.translate_to)
                log.info(f"🌍 Translated to {self.translate_to.upper()}")
                return translated
            except Exception as e:
                log.warning(f"⚠️ Translation failed, returning raw text: {e}")
                return text

        return text

    # ------------------------------------------------------------------
    # CONVENIENCE
    # ------------------------------------------------------------------
    def __call__(self, audio_path: str | Path) -> str:
        """Shortcut for transcribing."""
        return self.transcribe(audio_path)