# src/fait/vision/object_detection/models/deformable_detr.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from typing import Union
import numpy as np

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from fait.vision.object_detection.base import Detection

log = logging.getLogger("fait.vision.object_detection.deformable_detr")

@dataclass
class DefDETRConfig:
    model_id: str = "SenseTime/deformable-detr"
    score_threshold: float = 0.25
    nms_iou: float = 0.5
    class_whitelist: Optional[List[str]] = None  # e.g., ["knife","laptop","cell phone","handbag","backpack","remote"]

def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> List[int]:
    if boxes.numel() == 0:
        return []
    idxs = torch.ops.torchvision.nms(boxes, scores, iou_thresh) if hasattr(torch.ops, "torchvision") else \
           _slow_nms(boxes, scores, iou_thresh)
    return idxs.cpu().tolist()

def _slow_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = _iou_pairwise(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return torch.tensor(keep, device=boxes.device)

def _iou_pairwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _as_pil(img_or_path: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Return a RGB PIL.Image from a path, PIL image, or numpy array."""
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert("RGB")
    if isinstance(img_or_path, (str, Path)):
        return Image.open(img_or_path).convert("RGB")
    if isinstance(img_or_path, np.ndarray):
        arr = img_or_path
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:  # drop alpha
            arr = arr[..., :3]
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img_or_path)}")

class DeformableDETR:
    """Closed-set detector run on ROI crops."""
    def __init__(self, cfg: DefDETRConfig = DefDETRConfig(), cache_dir: str | None = None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Auto-resolve the right model class from model_id
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id, cache_dir=cache_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(cfg.model_id, cache_dir=cache_dir).to(
            self.device).eval()
        # label map
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        log.info("deformabledetr:init", extra={"model": cfg.model_id, "device": self.device})

    @torch.inference_mode()
    def detect(self, image_or_path: Union[str, Path, Image.Image, np.ndarray]):
        img = _as_pil(image_or_path)
        w, h = img.size

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        post = self.processor.post_process_object_detection(
            outputs,
            threshold=self.cfg.score_threshold,
            target_sizes=[(h, w)],
        )[0]

        dets: list[Detection] = []
        id2label = getattr(self.model.config, "id2label", {}) or {}
        for box, score, cls_id in zip(post["boxes"], post["scores"], post["labels"]):
            x1, y1, x2, y2 = map(float, box.tolist())
            label = id2label.get(int(cls_id), str(int(cls_id)))
            if self.cfg.class_whitelist and label not in self.cfg.class_whitelist:
                continue
            dets.append(Detection(bbox=[x1, y1, x2, y2], score=float(score), label=label))
        return dets
