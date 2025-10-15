# src/fait/vision/ocr/models/olmocr_engine.py
from __future__ import annotations
from typing import Optional
from PIL import Image
import torch
import os
import logging

from ..base import OcrResult

log = logging.getLogger("fait.vision.ocr.olmocr")

# Correct model ID with lowercase 'o' in olmOCR
DEFAULT_OLMOCR_ID = "allenai/olmOCR-7B-0725"


class OlmOCREngine:
    """
    OlmOCR engine using Allen AI's olmOCR-7B-0725 model.
    A 7-billion parameter vision-language model optimized for OCR tasks.
    Follows the project's lazy-loading pattern with proper cache management.
    """

    def __init__(self, model_id: Optional[str] = None, cache_dir: Optional[str] = None):
        self.name = "olmocr"
        self.lang = None  # olmOCR is multilingual

        # Check environment variables for model override
        env_id = os.getenv("FAIT_OLMOCR_MODEL") or os.getenv("FAIT_OCR_OLMOCR_MODEL")
        self.model_id = (model_id or env_id or DEFAULT_OLMOCR_ID).strip()

        if not self.model_id or self.model_id.lower() == "none":
            log.warning("olmocr:model_id_missing; falling back to default")
            self.model_id = DEFAULT_OLMOCR_ID

        # Set up cache directory
        if cache_dir is None:
            from fait.core.paths import get_paths
            from fait.core.utils import ensure_folder
            cache_dir = str(get_paths().models_ocr / "hf")
            ensure_folder(cache_dir)

        os.environ.setdefault("HF_HOME", cache_dir)
        self.cache_dir = cache_dir

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self._processor = None
        self._model = None

        log.info("olmocr:init_config", extra={
            "model": self.model_id,
            "cache": self.cache_dir,
            "device": self.device
        })

    def _ensure_loaded(self):
        """Lazy load the model and processor only when needed."""
        if self._model is not None:
            return

        import time
        import gc
        import warnings

        t0 = time.time()

        try:
            from transformers import AutoProcessor, AutoModel

            log.info("olmocr:loading", extra={
                "model": self.model_id,
                "cache": self.cache_dir,
                "device": self.device,
                "note": "Loading 7B vision-language model"
            })

            print(f"\n[OlmOCR] Loading {self.model_id}...")
            print(f"[OlmOCR] Device: {self.device}")
            print(f"[OlmOCR] Cache: {self.cache_dir}")

            # Check available memory
            if self.device == "cuda":
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[OlmOCR] GPU Memory: {gpu_mem:.1f} GB")
                if gpu_mem < 14:
                    print(f"[OlmOCR] ⚠️  Warning: Model requires ~14GB VRAM")
                torch.cuda.empty_cache()

            gc.collect()

            # Load processor
            print("[OlmOCR] [1/2] Loading processor...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
            print("[OlmOCR] ✓ Processor loaded")

            # Load model
            print(f"[OlmOCR] [2/2] Loading model (4 shards, ~1-2 minutes)...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device == "cuda" else None
                )

            self._model.eval()

            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - t0
            print(f"\n[OlmOCR] ✓ Model loaded in {elapsed:.1f}s")
            print(f"[OlmOCR] Model class: {type(self._model).__name__}\n")

            log.info("olmocr:loaded", extra={
                "secs": round(elapsed, 2),
                "model_class": type(self._model).__name__
            })

        except Exception as e:
            log.exception("olmocr:load_error")
            print(f"\n[OlmOCR] ✗ Failed: {e}\n")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise RuntimeError(f"Failed to load olmOCR: {e}") from e

    @torch.inference_mode()
    def recognize(self, img: Image.Image) -> Optional[OcrResult]:
        """Recognize text from an image using olmOCR."""
        try:
            self._ensure_loaded()

            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")

            # Qwen2.5-VL conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text",
                         "text": "Extract all text from this image. Provide only the text content, no explanations."}
                    ]
                }
            ]

            text_prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self._processor(
                text=[text_prompt],
                images=[img],
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            if not text:
                log.debug("olmocr:no_text", extra={"image_size": img.size})
                return None

            log.debug("olmocr:success", extra={"text_length": len(text)})

            return OcrResult(
                text=text,
                lang=self.lang,
                confidence=None,
                engine=self.name
            )

        except Exception as e:
            log.error(f"olmocr:error: {e}")
            return None

    def ocr(self, img: Image.Image, lang: str = "auto") -> Optional[OcrResult]:
        """OCR interface for pipeline compatibility."""
        return self.recognize(img)