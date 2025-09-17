from __future__ import annotations
from typing import Optional
from PIL import Image
from fait.vision.ocr.base import OcrResult, OcrEngine

class TesseractEngine:
    """
    pytesseract wrapper. Confidence = mean of word-level confidences (0..100) / 100.
    """
    def __init__(self, lang: str = "eng", tesseract_cmd: Optional[str] = None, psm: Optional[int] = None, oem: Optional[int] = None):
        try:
            import pytesseract as _pt
            self._pt = _pt
        except Exception as e:
            raise RuntimeError("pytesseract not installed") from e

        if tesseract_cmd:
            self._pt.pytesseract.tesseract_cmd = tesseract_cmd

        self.name = "tesseract"
        self.lang = lang
        self._psm = psm
        self._oem = oem

    def recognize(self, img: Image) -> OcrResult | None:
        from pytesseract import image_to_data, Output

        conf = ""
        try:
            cfg = []
            if self._psm is not None: cfg += [f"--psm {self._psm}"]
            if self._oem is not None: cfg += [f"--oem {self._oem}"]
            config = " ".join(cfg) if cfg else None

            data = image_to_data(img, lang=self.lang, output_type=Output.DATAFRAME, config=config)
            # text is joined words (keep original spacing minimal)
            words = [str(t) for t in data["text"].fillna("").tolist() if str(t).strip()]
            text = " ".join(words).strip()

            confidences = [float(c) for c in data["conf"].fillna("-1").tolist() if str(c).strip() != "-1"]
            avg_conf = (sum(confidences)/len(confidences)/100.0) if confidences else None

            if not text:
                return None

            return OcrResult(text=text, lang=self.lang, confidence=avg_conf, engine=self.name)
        except Exception:
            return None

    def ocr(self, img: Image, lang: str = "auto") -> OcrResult | None:
        """
        OCR interface wrapper for compatibility with the pipeline.
        Uses the engine's configured language unless overridden.
        """
        # Use provided language if different from configured
        if lang != "auto" and lang != self.lang:
            # Create a temporary instance with the requested language
            temp_engine = TesseractEngine(
                lang=lang, 
                tesseract_cmd=getattr(self._pt.pytesseract, 'tesseract_cmd', None),
                psm=self._psm,
                oem=self._oem
            )
            return temp_engine.recognize(img)
        return self.recognize(img)
