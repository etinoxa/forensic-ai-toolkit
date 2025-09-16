from __future__ import annotations
from typing import Optional
from PIL import Image
import os
from ..base import OcrResult

class PaddleEngine:
    """
    PaddleOCR wrapper. Confidence = mean of line confidences (0..1).
    """
    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None):
        # If a cache_dir is provided, set cache env vars BEFORE importing PaddleOCR
        if cache_dir:
            try:
                from fait.core.utils import ensure_folder
                ensure_folder(cache_dir)
            except Exception:
                os.makedirs(cache_dir, exist_ok=True)
            cache_dir_str = str(cache_dir)
            # Try all relevant envs that PaddleX/PaddleOCR may respect
            os.environ["PADDLEX_HOME"] = cache_dir_str
            os.environ["PPOCR_HOME"] = cache_dir_str
            os.environ.setdefault("PADDLEHUB_HOME", cache_dir_str)
            # Force expanduser and XDG cache to land under the project cache
            os.environ["HOME"] = cache_dir_str
            os.environ.setdefault("XDG_CACHE_HOME", cache_dir_str)

        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise RuntimeError("paddleocr not installed") from e

        self.lang = lang
        self.name = "paddle"
        # use angle_cls=True for auto-rotation at the engine level (we still do our own rotation sweep)
        # Note: some paddleocr versions don't accept `show_log`; pass only supported args.
        self._impl = PaddleOCR(use_angle_cls=True, lang=lang)

    def recognize(self, img: Image) -> OcrResult | None:
        import numpy as np
        im = np.array(img.convert("RGB"))
        try:
            result = self._impl.ocr(im, cls=False)
            # result is list per image (we pass single), each entry: [ [box, (text, conf)], ... ]
            if not result or not result[0]:
                return None
            lines = result[0]
            texts = []
            confs = []
            for _, (txt, conf) in lines:
                if str(txt).strip():
                    texts.append(str(txt))
                    confs.append(float(conf))
            text = " ".join(texts).strip()
            if not text:
                return None
            avg = (sum(confs)/len(confs)) if confs else None
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name)
        except Exception:
            return None
