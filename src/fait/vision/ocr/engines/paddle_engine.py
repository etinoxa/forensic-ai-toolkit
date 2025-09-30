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
    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None, **kwargs):
        log.info(f"PaddleEngine init with kwargs: {kwargs}")

        if lang == "auto":
            lang = "en"  # Default to English when "auto" is specified
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

        # Add debug flag
        self._debug = True

        # Store detection parameters with defaults
        self.det_db_thresh = kwargs.get('det_db_thresh', 0.20)
        self.det_db_box_thresh = kwargs.get('det_db_box_thresh', 0.40)
        self.det_db_unclip_ratio = kwargs.get('det_db_unclip_ratio', 2.0)

        log.info(f"Detection params: thresh={self.det_db_thresh}, "
                 f"box_thresh={self.det_db_box_thresh}, "
                 f"unclip={self.det_db_unclip_ratio}")

    def _ensure_loaded(self):
        if self._impl is not None:
            return
        t0 = time.time()
        try:
            from paddleocr import PaddleOCR

            # Add explicit parameters matching your successful test
            self._impl = PaddleOCR(
                lang=self.lang,
                use_angle_cls=False,
                det_db_thresh=self.det_db_thresh,
                det_db_box_thresh=self.det_db_box_thresh,
                det_db_unclip_ratio=self.det_db_unclip_ratio,
            )
            log.info("paddle:init_done", extra={"secs": round(time.time() - t0, 2)})
        except Exception as e:
            log.exception("paddle:init_error")
            raise

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

    def detect(self, img: Image.Image) -> List[Tuple[np.ndarray, Image.Image]]:
        """Returns: list of (polygon[4x2], cropped_pil)"""
        self._ensure_loaded()
        im = np.array(img.convert("RGB"))
        out = []

        try:
            result = self._impl.ocr(im)

            if result and isinstance(result, list) and result[0]:
                first = result[0]

                # New-style dict result: {'dt_polys': ...}
                if isinstance(first, dict) and 'dt_polys' in first:
                    dt_polys = first.get('dt_polys') or []
                    for poly in dt_polys:
                        poly_array = np.array(poly, dtype=np.float32)
                        x0 = int(max(0, np.min(poly_array[:, 0])));
                        y0 = int(max(0, np.min(poly_array[:, 1])))
                        x1 = int(min(img.width, np.max(poly_array[:, 0])));
                        y1 = int(min(img.height, np.max(poly_array[:, 1])))
                        if x1 > x0 and y1 > y0:
                            crop = img.crop((x0, y0, x1, y1))
                            out.append((poly_array, crop))

                # Legacy list-of-polygons: [[poly1, poly2, ...]]  (what the test returns)
                elif isinstance(first, list) and first and isinstance(first[0], (list, tuple)):
                    for poly in first:
                        poly_array = np.array(poly, dtype=np.float32)
                        x0 = int(max(0, np.min(poly_array[:, 0])));
                        y0 = int(max(0, np.min(poly_array[:, 1])))
                        x1 = int(min(img.width, np.max(poly_array[:, 0])));
                        y1 = int(min(img.height, np.max(poly_array[:, 1])))
                        if x1 > x0 and y1 > y0:
                            crop = img.crop((x0, y0, x1, y1))
                            out.append((poly_array, crop))

            log.info(f"paddle:detect boxes={len(out)}")

        except Exception as e:
            log.exception(f"paddle:detect_error: {str(e)}")

        return out

    def recognize(self, img: Image.Image) -> OcrResult | None:
        """Recognize text from an image"""
        self._ensure_loaded()
        im = np.array(img.convert("RGB"))

        try:
            result = self._impl.ocr(im)

            if result and isinstance(result, list) and result[0]:
                first = result[0]

                # OCRResult uses dictionary-like access
                if hasattr(first, '__getitem__'):
                    try:
                        rec_texts = first['rec_texts']
                        rec_scores = first['rec_scores']

                        if rec_texts:
                            text = ' '.join(rec_texts).strip()
                            avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else None
                            log.info(f"paddle:recognize text={text[:50]}... conf={avg_conf}")
                            return OcrResult(text=text, lang=self.lang, confidence=avg_conf, engine=self.name)
                    except KeyError as e:
                        log.error(f"Missing key in OCRResult: {e}")

        except Exception:
            log.exception("paddle:recognize_error")

        return None

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
