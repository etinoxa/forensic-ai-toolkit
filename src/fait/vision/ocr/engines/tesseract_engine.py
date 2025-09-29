from __future__ import annotations
from typing import Optional
from PIL import Image
from fait.vision.ocr.base import OcrResult, OcrEngine
import os


class TesseractEngine:
    def __init__(self, lang: str = "eng", tesseract_cmd: Optional[str] = None, psm: Optional[int] = None,
                 oem: Optional[int] = None):
        # Map "auto" to "eng"
        if lang == "auto":
            lang = "eng"

        try:
            import pytesseract as _pt
            self._pt = _pt
        except Exception as e:
            raise RuntimeError("pytesseract not installed") from e

        # Set Tesseract path
        if tesseract_cmd:
            self._pt.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            import shutil
            if not shutil.which("tesseract") and not shutil.which("tesseract.exe"):
                default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(default_path):
                    self._pt.pytesseract.tesseract_cmd = default_path

        self.name = "tesseract"
        self.lang = lang  # Now guaranteed to be "eng" not "auto"
        self._psm = psm
        self._oem = oem

    def ocr(self, img: Image.Image, lang: str = "auto") -> OcrResult | None:
        """OCR interface wrapper for compatibility with the pipeline."""
        # Map "auto" to "eng" for the runtime call too
        if lang == "auto":
            lang = "eng"

        # Use provided language if different from configured
        if lang != self.lang:
            # Create a temporary instance with the requested language
            temp_engine = TesseractEngine(
                lang=lang,
                tesseract_cmd=getattr(self._pt.pytesseract, 'tesseract_cmd', None),
                psm=self._psm,
                oem=self._oem
            )
            return temp_engine.recognize(img)
        return self.recognize(img)

    def recognize(self, img: Image.Image) -> OcrResult | None:
        try:
            import logging
            log = logging.getLogger("fait.vision.ocr.tesseract")

            # Build config
            config_parts = []
            if self._psm is not None:
                config_parts.append(f"--psm {self._psm}")
            if self._oem is not None:
                config_parts.append(f"--oem {self._oem}")

            # Use image_to_string for simplicity
            if config_parts:
                text = self._pt.image_to_string(img, lang=self.lang, config=" ".join(config_parts))
            else:
                text = self._pt.image_to_string(img, lang=self.lang)

            text = text.strip()

            if not text:
                log.warning(f"Tesseract found no text in image {img.size}")
                return None

            log.info(f"Tesseract found: {len(text)} chars")

            # For confidence, we can use image_to_data separately if needed
            # But for now, just return the text without confidence
            return OcrResult(text=text, lang=self.lang, confidence=None, engine=self.name)

        except Exception as e:
            import logging
            log = logging.getLogger("fait.vision.ocr.tesseract")
            log.error(f"Tesseract error: {e}")
            return None

    def preprocess_for_tesseract(img: Image.Image) -> Image.Image:
        """Preprocess image for better Tesseract OCR"""
        import cv2
        import numpy as np

        # Convert to grayscale
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Apply threshold to get black and white
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL
        return Image.fromarray(thresh)

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for better Tesseract OCR"""
        import cv2
        import numpy as np

        # Convert PIL to numpy
        img_array = np.array(img.convert('RGB'))

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Increase contrast
        alpha = 1.5  # Contrast control
        beta = 0  # Brightness control
        adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Apply threshold to get black text on white background
        _, binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(binary, 1)

        # Convert back to PIL
        return Image.fromarray(denoised)