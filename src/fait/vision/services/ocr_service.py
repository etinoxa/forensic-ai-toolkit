from __future__ import annotations
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from fait.vision.ocr.base import OcrEngine
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

@dataclass
class TesseractCfg:
    lang: str = "eng"
    tesseract_cmd: str | None = None
    psm: int | None = None
    oem: int | None = None

@dataclass
class PaddleCfg:
    lang: str = "en"

@dataclass
class TrOCRCfg:
    model_id: str = "microsoft/trocr-base-printed"

@dataclass
class DonutCfg:
    model_id: str = "naver-clova-ix/donut-base"

@dataclass
class DocTRCfg:
    det_arch: str = "db_resnet50"
    reco_arch: str = "crnn_vgg16_bn"

@dataclass
class OcrServiceCfg:
    tesseract: TesseractCfg = TesseractCfg()
    paddle: PaddleCfg = PaddleCfg()
    trocr: TrOCRCfg = TrOCRCfg()
    donut: DonutCfg = DonutCfg()
    doctr: DocTRCfg = DocTRCfg()

class OcrService:
    def __init__(self, cfg: OcrServiceCfg = OcrServiceCfg()):
        self.cfg = cfg
        self.paths = get_paths()
        self.cache_dir = (self.paths.models_cache / "ocr")
        ensure_folder(self.cache_dir)
        self._pool: Dict[Tuple[str, str], OcrEngine] = {}

    def get(self, name: str) -> OcrEngine:
        n = name.lower()
        key = (n, "default")
        if key in self._pool:
            return self._pool[key]

        if n == "tesseract":
            from fait.vision.ocr.engines.tesseract_engine import TesseractEngine
            c = self.cfg.tesseract
            eng = TesseractEngine(lang=c.lang, tesseract_cmd=c.tesseract_cmd, psm=c.psm, oem=c.oem)
        elif n == "paddle":
            from fait.vision.ocr.engines.paddle_engine import PaddleEngine
            c = self.cfg.paddle
            eng = PaddleEngine(lang=c.lang)
        elif n == "trocr":
            from fait.vision.ocr.engines.trocr_engine import TrOCREngine
            c = self.cfg.trocr
            eng = TrOCREngine(model_id=c.model_id, cache_dir=str(self.cache_dir))
        elif n == "donut":
            from fait.vision.ocr.engines.donut_engine import DonutEngine
            c = self.cfg.donut
            eng = DonutEngine(model_id=c.model_id, cache_dir=str(self.cache_dir))
        elif n == "doctr":
            from fait.vision.ocr.engines.doctr_engine import DocTREngine
            c = self.cfg.doctr
            eng = DocTREngine(det_arch=c.det_arch, reco_arch=c.reco_arch)
        else:
            raise ValueError(f"Unknown OCR engine '{name}'")

        self._pool[key] = eng
        return eng

# factory
_service_singleton: Optional[OcrService] = None
def get_ocr_service(cfg: Optional[OcrServiceCfg] = None) -> OcrService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = OcrService(cfg or OcrServiceCfg())
    return _service_singleton
