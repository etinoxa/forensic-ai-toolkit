# src/fait/vision/ocr/engines/trocr_engine.py

from __future__ import annotations
from typing import Optional
from PIL import Image
import torch, time, os, logging
from ..base import OcrResult
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

log = logging.getLogger("fait.vision.ocr.trocr")

DEFAULT_TROCR_ID = "microsoft/trocr-large-printed"

class TrOCREngine:
    def __init__(self, model_id: Optional[str] = None, cache_dir: Optional[str] = None):
        self.name = "trocr"
        env_id = os.getenv("FAIT_TROCR_MODEL") or os.getenv("FAIT_OCR_TROCR_MODEL")
        self.model_id = (model_id or env_id or DEFAULT_TROCR_ID).strip()
        if not self.model_id or self.model_id.lower() == "none":
            log.warning("trocr:model_id_missing; falling back to default")
            self.model_id = DEFAULT_TROCR_ID

        if cache_dir is None:
            cache_dir = str(get_paths().models_cache / "ocr" / "hf")
        ensure_folder(cache_dir)
        os.environ.setdefault("HF_HOME", cache_dir)
        self.cache_dir = cache_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._proc = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        t0 = time.time()
        # IMPORTANT: use TrOCRProcessor, not AutoProcessor
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        log.info("trocr:init", extra={"model": self.model_id, "cache": self.cache_dir, "device": self.device})
        self._proc  = TrOCRProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device).eval()
        log.info("trocr:init_done", extra={"secs": round(time.time() - t0, 2)})

    @torch.inference_mode()
    def recognize(self, img: Image.Image) -> OcrResult | None:
        try:
            self._ensure_loaded()
            # Ensure RGB PIL -> pixel_values tensor
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            inputs = self._proc(images=img, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            generated_ids = self._model.generate(pixel_values, max_new_tokens=256)
            text = self._proc.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return OcrResult(text=text, engine=self.name) if text else None
        except Exception:
            log.exception("trocr:error")
            return None

    # Pipeline calls .ocr(...); delegate to recognize (TrOCR ignores lang)
    def ocr(self, img: Image.Image, lang: str = "auto"):
        return self.recognize(img)
