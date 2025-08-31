from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from fait.core.paths import get_paths, ensure_on_first_write
from fait.vision.detectors.grounding_dino import GroundingDINO, GDINOConfig
from fait.vision.detectors.deformable_detr import DeformableDETR, DefDETRConfig
from fait.vision.detectors.yolo import YOLODetector, YoloConfig

log = logging.getLogger("fait.vision.services.object")

@dataclass(frozen=True)
class ObjectServiceConfig:
    # default configs can be overridden by YAML before construction
    gdino: GDINOConfig = GDINOConfig()
    detr:  DefDETRConfig = DefDETRConfig()
    yolo:  YoloConfig = YoloConfig()

class ObjectModelsService:
    def __init__(self, cfg: ObjectServiceConfig = ObjectServiceConfig()):
        self.cfg = cfg
        self.paths = get_paths()
        ensure_on_first_write(self.paths.models_object_screen)
        self._gdino_pool: Dict[Tuple[str], GroundingDINO] = {}
        self._detr_pool:  Dict[Tuple[str], DeformableDETR] = {}
        self._yolo_pool:  Dict[Tuple[str], YOLODetector] = {}

    def get_gdino(self, cfg: Optional[GDINOConfig] = None) -> GroundingDINO:
        c = cfg or self.cfg.gdino
        key = (c.model_id, f"{c.box_threshold:.3f}", f"{c.text_threshold:.3f}")
        if key not in self._gdino_pool:
            self._gdino_pool[key] = GroundingDINO(c, cache_dir=str(self.paths.models_object_screen))
        return self._gdino_pool[key]

    def get_detr(self, cfg: Optional[DefDETRConfig] = None) -> DeformableDETR:
        c = cfg or self.cfg.detr
        key = (c.model_id, f"{c.score_threshold:.3f}", f"{c.nms_iou:.3f}")
        if key not in self._detr_pool:
            self._detr_pool[key] = DeformableDETR(c, cache_dir=str(self.paths.models_object_screen))
        return self._detr_pool[key]

    def get_yolo(self, cfg: Optional[YoloConfig] = None) -> YOLODetector:
        c = cfg or self.cfg.yolo
        key = (c.model_id, f"{c.score_threshold:.3f}", f"{c.nms_iou:.3f}", str(c.imgsz))
        if key not in self._yolo_pool:
            self._yolo_pool[key] = YOLODetector(c, cache_dir=str(self.paths.models_object_screen))
        return self._yolo_pool[key]

# singleton accessor
_OBJ_SVC: Optional[ObjectModelsService] = None
def get_object_service() -> ObjectModelsService:
    global _OBJ_SVC
    if _OBJ_SVC is None:
        _OBJ_SVC = ObjectModelsService()
    return _OBJ_SVC
