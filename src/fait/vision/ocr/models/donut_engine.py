# src/fait/vision/ocr/models/donut_engine.py
from __future__ import annotations
from typing import Optional
from PIL import Image
import torch
from ..base import OcrResult
import os

class DonutEngine:
    def __init__(self, model_id: str = "naver-clova-ix/donut-base", cache_dir: Optional[str] = None):
        self.name = "donut";
        self.lang = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        env_id = os.getenv("FAIT_OCR_DONUT_MODEL")
        self.model_id = (model_id or env_id).strip()
        self.cache_dir = cache_dir
        self._proc = None
        self._model = None
        self._prompt_ids = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        self._proc = DonutProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(
            self.device).eval()
        # Create task prompt for general OCR
        self._prompt_ids = self._proc.tokenizer("<s_ocr>", add_special_tokens=False, return_tensors="pt").input_ids.to(
            self.device)

    @torch.inference_mode()
    def recognize(self, img: Image) -> OcrResult | None:
        try:
            self._ensure_loaded()
            pixel_values = self._proc(images=img, return_tensors="pt").pixel_values.to(self.device)
            gen = self._model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=self._prompt_ids,
                max_new_tokens=512
            )
            text = self._proc.batch_decode(gen, skip_special_tokens=True)[0]

            # Donut often returns JSON-like structured output
            # Try to extract just the text content
            if text.startswith("{") and text.endswith("}"):
                import json
                try:
                    data = json.loads(text)
                    # Extract text from various possible fields
                    text = str(data.get("text", data.get("content", data)))
                except:
                    pass

            return OcrResult(text=text.strip(), engine=self.name) if text else None
        except Exception as e:
            import logging
            logging.error(f"Donut failed: {e}")
            return None

    def ocr(self, img: Image, lang: str = "auto") -> OcrResult | None:
        return self.recognize(img)
