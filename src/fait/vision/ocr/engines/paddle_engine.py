from __future__ import annotations
from typing import Optional
import os, time, logging
import numpy as np
from PIL import Image

from ..base import OcrResult
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

log = logging.getLogger("fait.vision.ocr.paddle")

class PaddleEngine:
    def __init__(self, lang: str = "en", cache_dir: Optional[str] = None):
        # Route all Paddle/PaddleX caches to FAIT models cache
        cache_root = cache_dir or str(get_paths().models_cache / "ocr")
        ensure_folder(cache_root)

        # FORCE env so ~ expands under our cache and Paddle stacks use it
        os.environ["PADDLEX_HOME"]       = cache_root
        os.environ["PPOCR_HOME"]         = cache_root
        os.environ["PADDLEHUB_HOME"]     = cache_root
        os.environ["PADDLE_HOME"]        = cache_root
        os.environ["PADDLE_MODELS_HOME"] = cache_root
        os.environ["XDG_CACHE_HOME"]     = cache_root
        os.environ["HOME"]               = cache_root
        if os.name == "nt":
            os.environ["USERPROFILE"] = cache_root  # expanduser() on Windows

        self._impl = None
        self.lang = lang
        self.name = "paddle"

    def _ensure_loaded(self):
        if self._impl is not None:
            return
        t0 = time.time()
        try:
            from paddleocr import PaddleOCR
            # On Windows/CPU, be explicit:
            self._impl = PaddleOCR(use_angle_cls=True, lang=self.lang)
            log.info("paddle:init_done", extra={"secs": round(time.time() - t0, 2)})
        except Exception:
            log.exception("paddle:init_error")  # full traceback
            raise

    def ocr(self, img: Image.Image, lang: str = "auto"):
        if lang and lang != "auto" and lang != self.lang:
            self.lang = lang
            self._impl = None
        t0 = time.time()
        res = self.recognize(img)
        log.info("paddle:ocr", extra={"secs": round(time.time() - t0, 3)})
        if res is None:
            return None
        return {"text": res.text, "confidence": res.confidence, "lang": res.lang, "engine": self.name}

    def recognize(self, img: Image.Image) -> OcrResult | None:
        self._ensure_loaded()
        im = np.array(img.convert("RGB"))
        try:
            # explicit flags; some builds default det/rec incorrectly if not set
            result = self._impl.ocr(im)
            if not result:
                log.info("paddle:empty", extra={"boxes": 0})
                return None

            page = result[0] if isinstance(result, list) else result
            if not page:
                log.info("paddle:empty", extra={"boxes": 0})
                return None

            texts, confs = [], []
            for item in page:
                # item ≈ [poly, (text, conf)] or [poly, [text, conf]]
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                rec = item[1]
                txt, conf = "", None
                if isinstance(rec, (list, tuple)) and rec:
                    txt = str(rec[0]).strip()
                    if len(rec) > 1:
                        try:
                            conf = float(rec[1])
                        except Exception:
                            conf = None
                elif isinstance(rec, dict):
                    txt = str(rec.get("text", "")).strip()
                    conf = float(rec.get("confidence")) if rec.get("confidence") is not None else None
                if txt:
                    texts.append(txt)
                    if conf is not None: confs.append(conf)

            if not texts:
                log.info("paddle:no_text", extra={"boxes": len(page)})
                return None

            text = " ".join(texts).strip()
            avg = (sum(confs) / len(confs)) if confs else None
            log.info("paddle:got_text", extra={"chars": len(text), "boxes": len(texts), "avg_conf": (avg or 0)})
            return OcrResult(text=text, lang=self.lang, confidence=avg, engine=self.name)

        except Exception:
            # This prints full stack trace so we can see the real failure reason.
            log.exception("paddle:error")
            return None


