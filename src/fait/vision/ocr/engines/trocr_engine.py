from __future__ import annotations
from typing import Optional
from PIL import Image
import torch
from ..base import OcrResult

class TrOCREngine:
    """
    Microsoft TrOCR (printed or handwritten). Confidence left blank (not standard).
    """
    def __init__(self, model_id: str = "microsoft/trocr-base-printed", cache_dir: Optional[str] = None):
        try:
            from transformers import VisionEncoderDecoderModel, AutoProcessor
        except Exception as e:
            raise RuntimeError("transformers not installed") from e

        self.name = "trocr"
        self.lang = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._proc = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id, cache_dir=cache_dir).to(self.device).eval()

    @torch.inference_mode()
    def recognize(self, img: Image) -> OcrResult | None:
        try:
            enc = self._proc(images=img, return_tensors="pt").to(self.device)
            gen = self._model.generate(**enc, max_new_tokens=512)
            text = self._proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            if not text:
                return None
            return OcrResult(text=text, engine=self.name)
        except Exception:
            return None
