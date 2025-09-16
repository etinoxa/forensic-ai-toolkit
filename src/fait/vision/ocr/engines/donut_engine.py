from __future__ import annotations
from typing import Optional
from PIL import Image
import torch
from ..base import OcrResult

class DonutEngine:
    """
    NAVER Donut. General free-form OCR-ish extraction.
    Confidence left blank. Heavy model; optional.
    """
    def __init__(self, model_id: str = "naver-clova-ix/donut-base", cache_dir: Optional[str] = None):
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except Exception as e:
            raise RuntimeError("transformers (Donut) not installed") from e

        self.name = "donut"
        self.lang = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._proc = DonutProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id, cache_dir=cache_dir).to(self.device).eval()

        # A generic prompt often used for Donut demos
        self._prompt_ids = self._proc.tokenizer("<s>", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

    @torch.inference_mode()
    def recognize(self, img: Image) -> OcrResult | None:
        try:
            pixel_values = self._proc(images=img, return_tensors="pt").pixel_values.to(self.device)
            gen = self._model.generate(pixel_values=pixel_values, decoder_input_ids=self._prompt_ids, max_new_tokens=512)
            text = self._proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            if not text:
                return None
            return OcrResult(text=text, engine=self.name)
        except Exception:
            return None
