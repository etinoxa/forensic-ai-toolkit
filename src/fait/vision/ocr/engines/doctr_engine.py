from __future__ import annotations
from PIL import Image
from ..base import OcrResult

class DocTREngine:
    def __init__(self, reco_arch: str = "crnn_vgg16_bn", det_arch: str = "db_resnet50"):
        self.name = "doctr"; self.lang = None
        self.det_arch = det_arch; self.reco_arch = reco_arch
        self._predictor = None

    def _ensure_loaded(self):
        if self._predictor is not None: return
        import doctr
        from doctr.models import ocr_predictor
        self._predictor = ocr_predictor(det_arch=self.det_arch, reco_arch=self.reco_arch, pretrained=True)

    def recognize(self, img: Image) -> OcrResult | None:
        try:
            self._ensure_loaded()
            from doctr.io import DocumentFile
            doc = DocumentFile.from_images([img])
            out = self._predictor(doc)
            exp = out.export()
            texts, confs = [], []
            for page in exp.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for word in line.get("words", []):
                            val = str(word.get("value", "")).strip()
                            if val:
                                texts.append(val)
                                c = word.get("confidence")
                                if c is not None:
                                    try: confs.append(float(c))
                                    except: pass
            text = " ".join(texts).strip()
            avg = (sum(confs)/len(confs)) if confs else None
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name) if text else None
        except Exception:
            return None

    def ocr(self, img: Image, lang: str = "auto") -> OcrResult | None:
        return self.recognize(img)
