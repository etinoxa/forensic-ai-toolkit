from __future__ import annotations
from typing import Optional
from PIL import Image
import torch
from ..base import OcrResult

class TrOCREngine:
    def __init__(self, model_id: str = "microsoft/trocr-base-printed", cache_dir: Optional[str] = None):
        self.name = "trocr"; self.lang = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id; self.cache_dir = cache_dir
        self._proc = None; self._model = None

    def _ensure_loaded(self):
        if self._model is not None: return
        from transformers import VisionEncoderDecoderModel, AutoProcessor
        self._proc  = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device).eval()

    @torch.inference_mode()
    def recognize(self, img: Image) -> OcrResult | None:
        try:
            self._ensure_loaded()
            enc = self._proc(images=img, return_tensors="pt").to(self.device)
            gen = self._model.generate(**enc, max_new_tokens=512)
            text = self._proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
            return OcrResult(text=text, engine=self.name) if text else None
        except Exception:
            return None

    def ocr(self, img, lang: str = "auto"):
        # TrOCR ignores `lang`; delegate to recognize
        return self.recognize(img)

