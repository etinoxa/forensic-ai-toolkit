from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

log = logging.getLogger("fait.vision.detectors.deformabledetr")

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
    def detect(self, image: Image.Image) -> List[Dict]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (h,w)
        results = self.processor.post_process_object_detection(out, target_sizes=target_sizes)[0]
        boxes = results["boxes"]  # xyxy
        scores = results["scores"]
        labels = results["labels"]

        # filter by score and whitelist
        keep = scores >= self.cfg.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        names = [self.id2label[int(i)] for i in labels.cpu().tolist()]
        if self.cfg.class_whitelist:
            mask = torch.tensor([n in self.cfg.class_whitelist for n in names], device=boxes.device, dtype=torch.bool)
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
            names = [n for n, m in zip(names, mask.cpu().tolist()) if m]

        # class-wise NMS
        final = []
        for cls in set(names):
            idxs = [i for i, n in enumerate(names) if n == cls]
            b = boxes[idxs]
            s = scores[idxs]
            keep_idx = _nms_xyxy(b, s, self.cfg.nms_iou)
            for j in keep_idx:
                final.append({"box": b[j].cpu().tolist(), "score": float(s[j].cpu()), "label": cls})
        return final
