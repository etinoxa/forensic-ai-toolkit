# src/fait/vision/detectors/yolo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from fait.core.paths import get_paths, ensure_on_first_write

log = logging.getLogger("fait.vision.detectors.yolo")

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

class YOLODetector:
    def __init__(self, cfg: YoloConfig, cache_dir: Optional[str] = None):
        self.cfg = cfg
        paths = get_paths()
        cache = Path(cache_dir) if cache_dir else paths.models_object_screen
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
    def detect(self, image) -> List[Dict[str, Any]]:
        """
        Accepts PIL.Image, numpy array, or file path. Returns a list of dicts:
        {"label": str, "score": float, "box": [x1,y1,x2,y2]} in absolute pixels.
        """
        results = self.model.predict(
            source=image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.score_threshold,
            iou=self.cfg.nms_iou,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        names = getattr(self.model, "names", None) or getattr(r, "names", {})

        out: List[Dict[str, Any]] = []
        for b in r.boxes:
            conf = float(b.conf.item())
            cls_id = int(b.cls.item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            if self.cfg.class_whitelist and label not in self.cfg.class_whitelist:
                continue

            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({"label": label, "score": conf, "box": [x1, y1, x2, y2]})
        return out