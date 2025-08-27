from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

log = logging.getLogger("fait.vision.detectors.groundingdino")

@dataclass
class GDINOConfig:
    model_id: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    nms_iou: float = 0.5
    box_expand: float = 0.15  # 15% padding around proposals

def _nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> List[int]:
    # boxes: [N, 4] (x1,y1,x2,y2)
    if boxes_xyxy.numel() == 0:
        return []
    idxs = torch.ops.torchvision.nms(boxes_xyxy, scores, iou_thresh) if hasattr(torch.ops, "torchvision") else \
           _slow_nms(boxes_xyxy, scores, iou_thresh)
    return idxs.cpu().tolist()

def _slow_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    # Minimal PyTorch-only NMS (fallback)
    keep = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = _iou_pairwise(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
        mask = ious <= iou_thresh
        order = order[1:][mask]
    return torch.tensor(keep, device=boxes.device)

def _iou_pairwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [Na,4], b: [Nb,4] both xyxy
    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _expand_xyxy(box: torch.Tensor, w: int, h: int, pct: float) -> torch.Tensor:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * (1 + pct)
    bh = (y2 - y1) * (1 + pct)
    x1n = (cx - bw / 2).clamp(min=0, max=w - 1)
    y1n = (cy - bh / 2).clamp(min=0, max=h - 1)
    x2n = (cx + bw / 2).clamp(min=0, max=w - 1)
    y2n = (cy + bh / 2).clamp(min=0, max=h - 1)
    return torch.tensor([x1n, y1n, x2n, y2n], device=box.device)

class GroundingDINO:
    """Open-vocabulary proposals from prompts."""
    def __init__(self, cfg: GDINOConfig = GDINOConfig(), cache_dir: str | None = None):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id, cache_dir=cache_dir).to(self.device).eval()
        log.info("groundingdino:init", extra={"model": cfg.model_id, "device": self.device})

    @torch.inference_mode()
    def propose(self, image: Image.Image, prompts: List[str]) -> List[Dict]:
        # Returns list of dicts: {box:[x1,y1,x2,y2], score: float, prompt: str}
        inputs = self.processor(images=image, text=prompts, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # HF helper: convert to xyxy on CPU
        target_sizes = torch.tensor([image.size[::-1]])  # (h,w)
        kwargs = dict(
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
            target_sizes=torch.tensor([image.size[::-1]]),  # (h, w)
        )
        try:
            # Newer transformers: pass input_ids positionally
            results = self.processor.post_process_grounded_object_detection(
                out, inputs["input_ids"], **kwargs
            )[0]
        except TypeError:
            # Older signatures may require explicit kw
            results = self.processor.post_process_grounded_object_detection(
                outputs=out, input_ids=inputs["input_ids"], **kwargs
            )[0]
        boxes = results["boxes"]  # [N,4] xyxy
        scores = results["scores"]  # [N]
        phrases = results["labels"]  # ["handgun", ...]

        if boxes.numel() == 0:
            return []

        # NMS
        keep = _nms(boxes, scores, self.cfg.nms_iou)
        boxes = boxes[keep]
        scores = scores[keep]
        phrases = [phrases[i] for i in keep]

        # expand boxes
        W, H = image.size
        expanded = torch.stack([_expand_xyxy(b, W, H, self.cfg.box_expand) for b in boxes])

        return [
            {"box": expanded[i].cpu().tolist(), "score": float(scores[i].cpu()), "prompt": phrases[i]}
            for i in range(len(phrases))
        ]
