# src/fait/vision/object_detection/models/yolo.py
from __future__ import annotations
import logging
import numpy as np
import torch


from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from typing import Union
from pathlib import Path
from PIL import Image


from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from fait.vision.object_detection.base import Detection

from fait.core.paths import get_paths, ensure_on_first_write

log = logging.getLogger("fait.vision.object_detection.yolo")

@dataclass
class YoloConfig:
    # Use a small default; change to "yolo11n.pt" if you prefer
    model_id: str = "yolov8n.pt"
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    imgsz: int = 640
    class_whitelist: Optional[List[str]] = None  # e.g., ["knife","laptop","cell phone"]

def _resolve_yolo_weights(model_id: str, cache_dir: Path) -> Path:
    p = Path(model_id)
    if p.exists():
        return p.resolve()
    ensure_on_first_write(cache_dir)

    candidates = []
    if "/" in model_id and model_id.endswith(".pt"):
        repo_id, filename = model_id.rsplit("/", 1)
        candidates.append((repo_id, filename))
    else:
        fname = Path(model_id).name
        candidates.extend([
            ("ultralytics/assets", fname),
            ("ultralytics/yolov8", fname),
        ])

    for repo_id, filename in candidates:
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            return Path(local).resolve()
        except Exception:
            continue

    fallback = cache_dir / Path(model_id).name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Could not resolve YOLO weights '{model_id}'. Place the file at: {fallback}"
    )

def _to_source(img_or_path: Union[str, Path, Image.Image, np.ndarray]):
    """Return a source YOLO can consume (path or numpy RGB array)."""
    if isinstance(img_or_path, (str, Path)):
        return str(img_or_path)
    if isinstance(img_or_path, Image.Image):
        return np.asarray(img_or_path.convert("RGB"))
    if isinstance(img_or_path, np.ndarray):
        return img_or_path  # assume HWC RGB
    raise TypeError(f"Unsupported image type: {type(img_or_path)}")

class YOLODetector:
    def __init__(self, cfg: YoloConfig, cache_dir: Optional[str] = None):
        self.cfg = cfg
        paths = get_paths()
        cache = Path(cache_dir) if cache_dir else paths.models_object_detection
        ensure_on_first_write(cache)

        weights_path = _resolve_yolo_weights(cfg.model_id, cache)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(str(weights_path))
        try:
            self.model.to(self.device)
        except Exception:
            pass

        log.info("yolo:init", extra={
            "weights": str(weights_path),
            "cache_dir": str(cache),
            "imgsz": cfg.imgsz,
            "score_threshold": cfg.score_threshold,
            "nms_iou": cfg.nms_iou,
            "device": self.device,
        })

    @torch.inference_mode()
    def detect(self, image_or_path: Union[str, Path, Image.Image, np.ndarray]):
        source = _to_source(image_or_path)
        results = self.model.predict(
            source=source,
            conf=self.cfg.score_threshold,
            iou=self.cfg.nms_iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )

        dets: list[Detection] = []
        names = getattr(self.model, "names", {}) or {}
        for res in results:
            boxes = res.boxes
            # compat: support either a Boxes object or a list of per-box objects
            if isinstance(boxes, list):
                for b in boxes:
                    for box, c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
                        label = names.get(int(k), str(int(k)))
                        if self.cfg.class_whitelist and label not in self.cfg.class_whitelist:
                            continue
                        x1, y1, x2, y2 = map(float, box)
                        dets.append(Detection(bbox=[x1, y1, x2, y2], score=float(c), label=label))
            else:
                xyxy = boxes.xyxy.tolist()
                conf = boxes.conf.tolist()
                cls_ = boxes.cls.tolist()
                for box, c, k in zip(xyxy, conf, cls_):
                    label = names.get(int(k), str(int(k)))
                    if self.cfg.class_whitelist and label not in self.cfg.class_whitelist:
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    dets.append(Detection(bbox=[x1, y1, x2, y2], score=float(c), label=label))

        return dets