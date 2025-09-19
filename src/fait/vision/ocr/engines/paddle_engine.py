# src/fait/vision/ocr/engines/paddle_engine.py
from __future__ import annotations
from typing import Optional, List, Tuple
import os, time, logging

import numpy as np
from PIL import Image

from ..base import OcrResult
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

log = logging.getLogger("fait.vision.ocr.paddle")


class PaddleEngine:
    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None):
        cache_root = cache_dir or str(get_paths().models_cache / "ocr")
        ensure_folder(cache_root)

        # Force Paddle caches into FAIT
        os.environ["PADDLEX_HOME"]       = cache_root
        os.environ["PPOCR_HOME"]         = cache_root
        os.environ["PADDLEHUB_HOME"]     = cache_root
        os.environ["PADDLE_HOME"]        = cache_root
        os.environ["PADDLE_MODELS_HOME"] = cache_root
        os.environ["XDG_CACHE_HOME"]     = cache_root
        os.environ["HOME"]               = cache_root
        if os.name == "nt":
            os.environ["USERPROFILE"] = cache_root

        self._impl = None
        self.lang = lang
        self.name = "paddle"

    def _ensure_loaded(self):
        if self._impl is not None:
            return
        t0 = time.time()
        try:
            from paddleocr import PaddleOCR
            self._impl = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang  # CPU-safe defaults
            )
            log.info("paddle:init_done", extra={"secs": round(time.time()-t0, 2)})
        except Exception:
            log.exception("paddle:init_error")
            raise

    def detect(self, img: Image.Image):
        self._ensure_loaded()
        im = np.array(img.convert("RGB"))

        # det only, no recognition
        # paddleocr returns: [[ [poly, score], ... ]] when rec=False
        det_out = self._impl.ocr(im, det=True, rec=False, cls=False)
        polys = det_out[0] if det_out else []

        boxes = []
        for poly, _score in polys:
            poly = np.asarray(poly, dtype=np.int32)
            x0, y0 = poly[:, 0].min(), poly[:, 1].min()
            x1, y1 = poly[:, 0].max(), poly[:, 1].max()
            crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
            boxes.append((poly.tolist(), crop))
        return boxes

    # ---------- public APIs ----------

    def ocr(self, img: Image.Image, lang: str = "auto"):
        if lang and lang != "auto" and lang != self.lang:
            self.lang = lang
            self._impl = None  # re-init next call
        t0 = time.time()
        res = self.recognize(img)
        log.info("paddle:ocr", extra={"secs": round(time.time()-t0, 3)})
        if res is None:
            return None
        return {"text": res.text, "confidence": res.confidence, "lang": res.lang, "engine": self.name}

    def recognize(self, img: Image.Image) -> OcrResult | None:
        self._ensure_loaded()
        im = np.array(img.convert("RGB"))
        try:
            result = self._impl.ocr(im, det=True, rec=True, cls=True)
            page = _first_page(result)
            if not page:
                log.info("paddle:empty", extra={"boxes": 0})
                return None
            texts, confs = [], []
            for item in page:
                txt, conf = _extract_text_conf(item)
                if txt:
                    texts.append(txt)
                    if conf is not None: confs.append(conf)
            if not texts:
                log.info("paddle:no_text", extra={"boxes": len(page)})
                return None
            text = " ".join(texts).strip()
            avg = (sum(confs)/len(confs)) if confs else None
            log.info("paddle:got_text", extra={"chars": len(text), "boxes": len(texts), "avg_conf": (avg or 0)})
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name)
        except Exception:
            log.exception("paddle:error")
            return None

    def detect(self, img: Image.Image) -> List[Tuple[np.ndarray, Image.Image]]:
        """
        Returns: list of (polygon[4x2], cropped_pil)
        """
        self._ensure_loaded()
        im = np.array(img.convert("RGB"))
        out = []
        try:
            result = self._impl.ocr(im)  # detection only
            page = _first_page(result)
            for item in page:
                if not isinstance(item, (list, tuple)) or not item:
                    continue
                poly = np.array(item[0], dtype=np.float32)  # 4 points
                # simple rectangular crop (bbox) – robust & fast
                x0, y0 = np.min(poly[:,0]), np.min(poly[:,1])
                x1, y1 = np.max(poly[:,0]), np.max(poly[:,1])
                crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
                out.append((poly, crop))
            log.info("paddle:detect", extra={"boxes": len(out)})
        except Exception:
            log.exception("paddle:detect_error")
        return out


# ---------- helpers ----------

def _first_page(result):
    if not result:
        return []
    if isinstance(result, list):
        return result[0] if result and isinstance(result[0], (list, tuple)) else result
    return result

def _extract_text_conf(item) -> tuple[str, float | None]:
    # item ≈ [poly, (text, conf)] or [poly, [text, conf]]
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return "", None
    rec = item[1]
    if isinstance(rec, (list, tuple)) and rec:
        txt = str(rec[0]).strip()
        conf = None
        if len(rec) > 1:
            try: conf = float(rec[1])
            except Exception: conf = None
        return txt, conf
    if isinstance(rec, dict):
        txt = str(rec.get("text", "")).strip()
        conf = rec.get("confidence")
        try: conf = float(conf) if conf is not None else None
        except Exception: conf = None
        return txt, conf
    return "", None
