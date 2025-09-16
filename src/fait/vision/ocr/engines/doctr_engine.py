from __future__ import annotations
from typing import Optional
from PIL import Image
from ..base import OcrResult

class DocTREngine:
    """
    DocTR end-to-end OCR. Confidence aggregated from words if available.
    """
    def __init__(self, reco_arch: str = "crnn_vgg16_bn", det_arch: str = "db_resnet50"):
        try:
            import doctr
            from doctr.models import ocr_predictor
        except Exception as e:
            raise RuntimeError("python-doctr not installed") from e

        self.name = "doctr"
        self.lang = None
        self._predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)

    def recognize(self, img: Image) -> OcrResult | None:
        try:
            from doctr.io import DocumentFile
            doc = DocumentFile.from_images([img])
            out = self._predictor(doc)
            exp = out.export()
            # aggregate text and confidence
            texts, confs = [], []
            for page in exp.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for word in line.get("words", []):
                            val = str(word.get("value", "")).strip()
                            if val:
                                texts.append(val)
                                c = word.get("confidence", None)
                                if c is not None:
                                    try:
                                        confs.append(float(c))
                                    except Exception:
                                        pass
            text = " ".join(texts).strip()
            if not text:
                return None
            avg = (sum(confs)/len(confs)) if confs else None
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name)
        except Exception:
            return None
