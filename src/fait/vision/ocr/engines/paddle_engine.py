from __future__ import annotations
from typing import Optional
from PIL import Image
import os
from ..base import OcrResult

class PaddleEngine:
    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None):
        # (keep your cache env setup exactly as-is)
        self.lang = lang; self.name = "paddle"; self._impl = None

    def _ensure_loaded(self):
        if self._impl is not None: return
        from paddleocr import PaddleOCR
        self._impl = PaddleOCR(use_angle_cls=True, lang=self.lang)

    def recognize(self, img: Image) -> OcrResult | None:
        import numpy as np
        im = np.array(img.convert("RGB"))
        try:
            self._ensure_loaded()
            result = self._impl.ocr(im, cls=False)
            if not result or not result[0]:
                return None
            texts, confs = [], []
            for _, (txt, conf) in result[0]:
                if str(txt).strip():
                    texts.append(str(txt)); confs.append(float(conf))
            text = " ".join(texts).strip()
            avg = (sum(confs)/len(confs)) if confs else None
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name) if text else None
        except Exception:
            return None

    def ocr(self, img: Image, lang: str = "auto") -> OcrResult | None:
        if lang != "auto" and lang != self.lang:
            pass
        return self.recognize(img)
