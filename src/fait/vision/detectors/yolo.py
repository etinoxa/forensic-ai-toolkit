# src/fait/vision/detectors/yolo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
import logging

import torch
from PIL import Image

log = logging.getLogger("fait.vision.detectors.yolo")

@dataclass
class YoloConfig:
    # Use a small default; change to "yolo11n.pt" if you prefer
    model_id: str = "yolov8n.pt"
    score_threshold: float = 0.25
    nms_iou: float = 0.50
    imgsz: int = 640
    class_whitelist: Optional[List[str]] = None  # e.g., ["knife","laptop","cell phone"]

class YOLODetector:
    """Closed-set detector wrapper for Ultralytics YOLO (v8/v11)."""

    def __init__(self, cfg: YoloConfig = YoloConfig(), cache_dir: str | None = None):
        from ultralytics import YOLO
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try to steer Ultralytics to our cache/outputs (best-effort)
        if cache_dir:
            try:
                from ultralytics.utils import settings as ysettings
                cache = Path(cache_dir)
                ysettings.update({
                    "weights_dir": str(cache / "yolo"),
                    "runs_dir": str(cache.parent / "yolo_runs"),
                })
            except Exception:
                pass

        self.model = YOLO(cfg.model_id)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # name map (COCO etc.)
        self.names = getattr(self.model, "names", None)
        if not self.names:
            self.names = {i: str(i) for i in range(1000)}

        log.info("yolo:init", extra={
            "model": cfg.model_id, "device": self.device,
            "score_thr": cfg.score_threshold, "iou": cfg.nms_iou, "imgsz": cfg.imgsz
        })

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> List[Dict]:
        # Ultralytics accepts PIL.Image directly
        results = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.score_threshold,
            iou=self.cfg.nms_iou,
            device=self.device,
            verbose=False
        )
        r = results[0]
        boxes = r.boxes

        out: List[Dict] = []
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        for b, c, k in zip(xyxy, conf, cls):
            label = self.names.get(int(k), str(int(k)))
            if self.cfg.class_whitelist and label not in self.cfg.class_whitelist:
                continue
            out.append({
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "score": float(c),
                "label": label,
            })
        return out
