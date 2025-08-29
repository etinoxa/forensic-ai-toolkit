# src/fait/vision/detectors/grounding_dino.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Set
from pathlib import Path
import logging

import torch
from PIL import Image
import torchvision.ops as tvops


log = logging.getLogger("fait.vision.detectors.gdino")


@dataclass
class GDINOConfig:
    """
    Configuration for GroundingDINO detector.
    """
    # Stronger than tiny; matches typical notebook defaults that perform better on web images
    model_id: str = "IDEA-Research/grounding-dino-base"

    # Proposal thresholds used inside HF post-process
    box_threshold: float = 0.20
    text_threshold: float = 0.20

    # Additional processing
    nms_iou: float = 0.50
    box_expand: float = 0.20         # final padding ratio on proposals (0.0 disables)
    long_side: int = 1024            # upsample small images (0 disables upsample)

    # Prompt handling
    normalize_prompts: bool = True   # lowercase + trailing period
    add_relational_prompts: bool = True  # add "person with X", "person holding X", "a photo of X"


class GroundingDINO:
    """
    GroundingDINO wrapper that returns open-vocabulary proposals as:
        [{'box':[x1,y1,x2,y2], 'score': float, 'prompt': str}, ...]
    """

    def __init__(self, cfg: GDINOConfig = GDINOConfig(), cache_dir: Optional[str] = None):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(cfg.model_id, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg.model_id, cache_dir=cache_dir
        ).to(self.device).eval()

        log.info(
            "gdino:init",
            extra={"model": cfg.model_id, "device": self.device},
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _norm_prompt(self, p: str) -> str:
        p = p.strip().lower()
        if not p.endswith("."):
            p += "."
        return p

    def _enrich_prompts(self, prompts: Iterable[str]) -> List[str]:
        """
        Normalize and add relational variants:
          - base
          - "person with {base}"
          - "person holding {base}"
          - "a photo of {base}"
        """
        if not self.cfg.normalize_prompts and not self.cfg.add_relational_prompts:
            # keep original list as-is
            return [p for p in prompts]

        out: Set[str] = set()
        ordered: List[str] = []

        def _push(s: str):
            if s not in out:
                out.add(s)
                ordered.append(s)

        for base in prompts:
            b = self._norm_prompt(base) if self.cfg.normalize_prompts else str(base).strip()
            if b:
                _push(b)
            if self.cfg.add_relational_prompts:
                _push(self._norm_prompt(f"person with {base}"))
                _push(self._norm_prompt(f"person holding {base}"))
                _push(self._norm_prompt(f"a photo of {base}"))
        return ordered

    def _maybe_resize(self, img: Image.Image) -> Image.Image:
        if not self.cfg.long_side:
            return img
        w, h = img.size
        longer = max(w, h)
        if longer >= self.cfg.long_side:
            return img
        if w >= h:
            nw, nh = self.cfg.long_side, int(self.cfg.long_side * h / w)
        else:
            nw, nh = int(self.cfg.long_side * w / h), self.cfg.long_side
        return img.resize((nw, nh), Image.BICUBIC)

    def _expand_xyxy(self, box, W, H, ratio) -> List[float]:
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1)
        h = (y2 - y1)
        w *= (1.0 + ratio)
        h *= (1.0 + ratio)
        nx1 = max(0.0, cx - 0.5 * w)
        ny1 = max(0.0, cy - 0.5 * h)
        nx2 = min(float(W), cx + 0.5 * w)
        ny2 = min(float(H), cy + 0.5 * h)
        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def _to_device(batch):
        """Move a transformers BatchFeature or dict of tensors to current device (robustly)."""
        try:
            return batch.to  # type: ignore[attr-defined]
        except Exception:
            pass
        return None

    # --------------------------------------------------------------------- #
    # API
    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def propose(self, img: Image.Image, prompts: List[str]) -> List[Dict]:
        """
        Return proposals:
          [{'box':[x1,y1,x2,y2], 'score': float, 'prompt': str}, ...]
        """
        # 1) Normalize/enrich prompts and build a single caption "p1 . p2 . p3 ."
        phrases = self._enrich_prompts(prompts)
        phrases = [self._norm_prompt(p) for p in phrases if isinstance(p, str) and p.strip()]
        # dedupe while keeping order
        seen: Set[str] = set()
        phrases = [p for p in phrases if not (p in seen or seen.add(p))]
        if not phrases:
            return []

        caption = " ".join(phrases)  # each already ends with "."

        # 2) Optional upsample to help on small/web images
        original = img
        img = self._maybe_resize(img)

        # 3) Tokenize (pad/truncate) and forward
        inputs = self.processor(
            images=img,
            text=caption,           # single caption string (dot-separated phrases)
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # Move to device (BatchFeature may or may not support .to())
        try:
            inputs = inputs.to(self.device)  # type: ignore[attr-defined]
        except Exception:
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # 4) Post-process with correct target sizes (H, W)
        H, W = img.size[1], img.size[0]
        target_sizes = torch.tensor([[H, W]], device=self.device)
        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=target_sizes
            )[0]
        except TypeError:
            # older transformers signature
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs, input_ids=inputs["input_ids"],
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=target_sizes
            )[0]

        boxes = processed["boxes"].detach().cpu()    # Tensor [N, 4] xyxy in resized image space
        scores = processed["scores"].detach().cpu()  # Tensor [N]
        labels = processed["labels"]                 # List[str] aligned to phrases

        if boxes.numel() == 0:
            return []

        # 5) Single NMS pass
        keep = tvops.nms(boxes, scores, self.cfg.nms_iou)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep.tolist()]

        # 6) Scale boxes back to original size if we upsampled
        if original.size != img.size:
            sx = original.size[0] / img.size[0]
            sy = original.size[1] / img.size[1]
            boxes = boxes * torch.tensor([sx, sy, sx, sy])

        # 7) Optional box expansion (padding)
        W0, H0 = original.size
        proposals: List[Dict] = []
        for b, s, lab in zip(boxes.tolist(), scores.tolist(), labels):
            out_box = self._expand_xyxy(b, W0, H0, self.cfg.box_expand) if self.cfg.box_expand > 0 else b
            proposals.append({
                "box": [float(x) for x in out_box],
                "score": float(s),
                "prompt": str(lab),
            })

        return proposals
