# src/fait/audio/services/translation_service.py
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from fait.core.paths import get_paths
from fait.core.utils import load_yaml

# We’ll reuse your Faster-Whisper wrapper to avoid torch & conversion
from fait.audio.speech_to_text.models.whisper_faster_model import WhisperFasterModel


@dataclass
class TranslatorConfig:
    target_lang: str = "en"
    # optional future settings
    device: str = "cpu"
    model_size: Optional[str] = None  # if set, overrides STT model_size for translation runs


def _read_audio_stt_cfg():
    """
    Reads config.yaml and returns the audio.speech_to_text section as a dict,
    or {} if not available.
    """
    paths = get_paths()
    cfg_path = Path(paths.repo_root) / "config.yaml"
    if not cfg_path.exists():
        return {}

    cfg = load_yaml(cfg_path)
    return (cfg.get("audio") or {}).get("speech_to_text") or {}


class TranslationService:
    """
    Lightweight translation service.

    Strategy:
      • If target_lang == 'en' → use Faster-Whisper's built-in translation task.
        (No CTranslate2, no transformers, no torch; avoids macOS mutex issues.)
      • Otherwise → return a clear "not supported" message. You can later plug a
        secondary backend here (e.g., Argos or a pre-converted ct2 model).
    """

    def __init__(self, target_lang: str = "en", cache_dir: Optional[str] = None):
        self.log = logging.getLogger(__name__)
        self.cfg = TranslatorConfig(target_lang=target_lang)

        # Read STT config to inherit device/model_size and the model cache dir
        stt_cfg = _read_audio_stt_cfg()
        self.cfg.device = stt_cfg.get("device", "cpu")
        # Use same Whisper size as STT (default to 'small' if missing/None)
        self.cfg.model_size = stt_cfg.get("model_size") or "small"

        # We don't need cache_dir here specifically because Faster-Whisper will
        # reuse its existing cache. If you’ve added a custom download_root in your
        # WhisperFasterModel, it’ll be used from there.
        self.log.info("📦 Translation configured (whisper re-run strategy)")

    # ------------------------------------------------------------------
    # Public API: translate the AUDIO directly (recommended)
    # ------------------------------------------------------------------
    def translate_audio(self, audio_path: Path, source_lang: Optional[str] = None) -> str:
        """
        Translate the given audio to self.cfg.target_lang.

        For target_lang == 'en', we run Faster-Whisper with task='translate',
        which produces English text regardless of source language.

        Returns the translated text (string).
        """
        if self.cfg.target_lang.lower() != "en":
            return f"[Translation not supported: only target_lang='en' is implemented]"

        # Re-run whisper in translate mode; no torch, no conversion, thread-safe
        model = WhisperFasterModel(
            model_size=self.cfg.model_size,
            device=self.cfg.device,
            compute_type="int8",  # safe default
        )

        # Use the model's translate method if available; otherwise emulate via
        # a special flag in the wrapper.
        if hasattr(model, "transcribe"):
            # We’ll call a translate-aware pathway via a dedicated method if present,
            # else pass a flag to transcribe().
            if hasattr(model, "translate"):
                text, _meta = model.translate(audio_path, source_lang=source_lang)
                return text
            else:
                # Fallback: our wrapper’s transcribe knows how to accept task="translate"
                text, _meta = model.transcribe(audio_path, task="translate")
                return text

        return "[Translation failed: Whisper model wrapper missing 'transcribe' method]"

    # ------------------------------------------------------------------
    # Legacy API: translate raw TEXT (kept for compatibility)
    # ------------------------------------------------------------------
    def translate_text(self, text: str, source_lang: Optional[str] = None) -> str:
        """
        Kept only for compatibility with older CLI calls. For your flow we now recommend
        translate_audio(audio_path, source_lang) so we can leverage whisper's translate task.

        Here we just return a clear message rather than trying to spin up a heavy text
        translation backend that could reintroduce the macOS lock.
        """
        if not text.strip():
            return ""

        if self.cfg.target_lang.lower() == "en":
            # We can’t translate text to en without an external model.
            # Direct users to use translate_audio() so we re-run whisper cleanly.
            return "[Translation skipped: use translate_audio(...) so we can run whisper's translate task safely]"
        else:
            return f"[Translation not supported: only target_lang='en' is implemented]"