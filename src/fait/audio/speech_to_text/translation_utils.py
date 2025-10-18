import logging
from pathlib import Path
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    """
    Handles multilingual translation using Hugging Face models.

    - Uses the configured cache_dir (from config.yaml) for storing models.
    - Defaults to a lightweight model (Helsinki-NLP/opus-mt-mul-en).
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-mul-en",
        device: str = "cpu",
        cache_dir: str = None,
    ):
        """
        Args:
            model_name: Name or path of the translation model.
            device: "cpu" or "cuda".
            cache_dir: Directory where translation models are downloaded/cached.
        """
        self.log = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = 0 if device == "cuda" else -1
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None

        # ✅ Use the cache dir if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(f"📦 Translation model cache directory: {self.cache_dir}")

        try:
            # ✅ Load model & tokenizer manually (to control cache directory)
            self.log.info(f"🌍 Loading translation model: {self.model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )

            # ✅ Create pipeline from local model
            self.translator = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
            )
            self.log.info(f"✅ Translation model '{self.model_name}' initialized successfully.")

        except Exception as e:
            self.log.error(f"❌ Failed to initialize translation model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize translation model: {e}")

    # ------------------------------------------------------------------
    # 🌐 TRANSLATE
    # ------------------------------------------------------------------
    def translate(self, text: str, target_lang: str = "en") -> str:
        """Translate text to the target language."""
        if not text.strip():
            return ""

        try:
            self.log.info(f"🌐 Translating → {target_lang}")
            result = self.translator(text, max_length=400)
            translated_text = result[0]["translation_text"]
            return translated_text.strip()
        except Exception as e:
            self.log.error(f"❌ Translation failed: {e}", exc_info=True)
            return f"[Translation failed: {e}]"