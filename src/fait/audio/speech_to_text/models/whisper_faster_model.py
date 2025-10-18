import logging
from pathlib import Path
import time
from faster_whisper import WhisperModel


class WhisperFasterModel:
    """
    Wrapper around the Faster-Whisper engine to provide a consistent
    interface for transcription and optional translation.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cache_dir: str = None,
    ):
        """
        Initialize a Faster-Whisper model with optional caching.

        Args:
            model_size (str): Model size to load (tiny, base, small, medium, large-v3, etc.)
            device (str): Device to use (e.g. "cpu", "cuda").
            compute_type (str): Compute precision (e.g. "int8", "float16").
            cache_dir (str, optional): Directory where models should be downloaded/cached.
        """
        self.log = logging.getLogger(__name__)
        self.model_size = model_size or "small"
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

        self.log.info(
            f"🔄 Loading faster-whisper model: {self.model_size} "
            f"(device={self.device}, type={self.compute_type})"
        )

        try:
            # ✅ Load model with cache_dir if provided
            if self.cache_dir:
                self.log.info(f"📦 Using cache directory: {self.cache_dir}")
                self.model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.cache_dir),
                )
            else:
                self.model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )

            self.log.info(f"✅ Faster-Whisper model '{self.model_size}' initialized successfully.")

        except Exception as e:
            self.log.error(f"❌ Failed to load Faster-Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Faster-Whisper model: {e}")

    # ---------------------------------------------------------------------
    # 🔊 TRANSCRIBE / TRANSLATE METHOD
    # ---------------------------------------------------------------------
    def transcribe(self, audio_path: Path, task: str = "transcribe"):
        """
        Transcribe or translate an audio file using Faster-Whisper.

        Args:
            audio_path (Path): Path to the audio file to process.
            task (str): Either "transcribe" (default) or "translate".

        Returns:
            tuple: (text: str, metadata: dict)
        """
        if task not in {"transcribe", "translate"}:
            raise ValueError(f"❌ Invalid task '{task}'. Must be 'transcribe' or 'translate'.")

        self.log.info(f"🎙️ Running Faster-Whisper task='{task}' on: {audio_path}")

        start_time = time.time()

        try:
            # ✅ Run inference with optional translation
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,
                task=task,
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if hasattr(segment, "text") and segment.text.strip()
            ).strip()

            duration = getattr(info, "duration", 0.0)
            language = getattr(info, "language", "unknown")
            probability = getattr(info, "language_probability", 1.0)

            metadata = {
                "language": language,
                "probability": round(probability, 2),
                "duration": round(duration, 2),
                "processing_time": round(time.time() - start_time, 2),
            }

            self.log.info(f"✅ Completed {task} ({language}) in {metadata['processing_time']}s")
            return text, metadata

        except Exception as e:
            self.log.error(f"❌ Error during {task}: {e}", exc_info=True)
            raise RuntimeError(f"{task.capitalize()} failed: {e}")

    # ---------------------------------------------------------------------
    # 🌐 TRANSLATE HELPER (accepts optional args safely)
    # ---------------------------------------------------------------------
    def translate(self, audio_path: Path, source_lang: str = None, **_):
        """
        Convenience method for English translation (task='translate').
        Accepts extra kwargs (e.g. source_lang) for compatibility.
        """
        self.log.info(
            f"🌐 Translating audio to English (source={source_lang or 'auto'}): {audio_path}"
        )
        return self.transcribe(audio_path, task="translate")