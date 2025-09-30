# src/fait/vision/ocr/models/doctr_engine.py
from __future__ import annotations
from PIL import Image
from ..base import OcrResult

class DocTREngine:
    def __init__(self, reco_arch: str = "crnn_vgg16_bn", det_arch: str = "db_resnet50"):
        self.name = "doctr"; self.lang = None
        # Defensive defaults (in case kwargs were passed as None)
        self.det_arch  = det_arch or "db_resnet50"
        self.reco_arch = reco_arch or "crnn_vgg16_bn"
        self._predictor = None

    def _ensure_loaded(self):
        if self._predictor is not None:
            return

        # 1) Pin caches under FAIT models/ocr (without overriding user-provided env)
        import os, logging
        from fait.core.paths import get_paths
        from fait.core.utils import ensure_folder

        cache_root = get_paths().models_ocr / "doctr"
        ensure_folder(cache_root)
        os.environ.setdefault("DOCTR_CACHE_DIR", str(cache_root))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
        os.environ.setdefault("TORCH_HOME", str(cache_root))

        # 2) Import doctr AFTER cache envs are set
        import doctr
        from doctr.models import ocr_predictor
        log = logging.getLogger("fait.vision.ocr.doctr")

        # (Optional but helpful) log where caches will land
        log.info("doctr:init", extra={
            "det_arch": self.det_arch,
            "reco_arch": self.reco_arch,
            "pretrained": True,
            "DOCTR_CACHE_DIR": os.environ.get("DOCTR_CACHE_DIR"),
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
            "doctr_ver": getattr(doctr, "__version__", "?"),
        })

        # 3) Build predictor
        self._predictor = ocr_predictor(
            det_arch=self.det_arch or "db_resnet50",
            reco_arch=self.reco_arch or "crnn_vgg16_bn",
            pretrained=True,
        )

    def recognize(self, img: Image) -> OcrResult | None:
        try:
            self._ensure_loaded()
            import io, numpy as np
            from doctr.io import DocumentFile

            # Normalize to RGB PIL once
            pil = img.convert("RGB")

            # 1) Preferred path for DocTR v1.0.0: PNG-encoded bytes
            try:
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                doc = DocumentFile.from_images([png_bytes])
            except Exception:
                # 2) Fallback: ndarray (uint8 HxWx3)
                try:
                    arr = np.asarray(pil, dtype=np.uint8)
                    doc = DocumentFile.from_images([arr])
                except Exception:
                    # 3) Last resort: pass PIL
                    doc = DocumentFile.from_images([pil])

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
                                    try:
                                        confs.append(float(c))
                                    except Exception:
                                        pass

            text = " ".join(texts).strip()
            avg = (sum(confs) / len(confs)) if confs else None
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name) if text else None
        except Exception as e:
            import logging
            logging.error(f"DocTR failed in recognize: {e}")
            return None

    def ocr(self, img: Image, lang: str = "auto") -> OcrResult | None:
        return self.recognize(img)
