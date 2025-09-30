# src/fait/vision/object_detection/grounding_dino.py
from __future__ import annotations

import logging
import inspect
import torchvision.ops as tvops
import torch

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Set
from PIL import Image

log = logging.getLogger("fait.vision.object_detection.gdino")

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
    def _first_key(d: dict, keys: tuple[str, ...]):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

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
            text=caption,  # single caption string (dot-separated phrases)
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

        # 4) Post-process with version-proof arg names
        H, W = img.size[1], img.size[0]  # PIL: (W, H)
        target_sizes = torch.tensor([[H, W]], device=self.device)

        pp = self.processor.post_process_grounded_object_detection
        sig = set(inspect.signature(pp).parameters)

        kw = {}
        # ids parameter name differs on some versions
        if "input_ids" in sig:
            kw["input_ids"] = inputs["input_ids"]
        elif "text_inputs" in sig:
            kw["text_inputs"] = inputs["input_ids"]

        # thresholds changed names across releases
        bt = float(self.cfg.box_threshold)
        tt = float(self.cfg.text_threshold)
        nms = float(getattr(self.cfg, "nms_iou", 0.5))

        if "box_threshold" in sig:
            kw["box_threshold"] = bt
        elif "boxes_threshold" in sig:
            kw["boxes_threshold"] = bt
        elif "threshold" in sig:
            kw["threshold"] = bt

        if "text_threshold" in sig:
            kw["text_threshold"] = tt
        elif "phrase_threshold" in sig:
            kw["phrase_threshold"] = tt

        if "nms_threshold" in sig:  # some versions expose this here
            kw["nms_threshold"] = nms

        try:
            processed = pp(outputs=outputs, target_sizes=target_sizes, **kw)
        except TypeError:
            # final fallback: call with minimal args
            processed = pp(outputs=outputs, target_sizes=target_sizes)

        # Some versions return a list of dicts
        if isinstance(processed, list):
            processed = processed[0]

        # 5) Robust extraction of boxes/scores/labels
        boxes = self._first_key(processed, ("boxes", "pred_boxes", "bboxes"))
        scores = self._first_key(processed, ("scores", "logits", "confidences"))
        labels = self._first_key(processed, ("text_labels", "labels", "phrases"))

        # Convert lists to tensors; squeeze singleton dims
        if isinstance(boxes, list):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        if isinstance(scores, list):
            scores = torch.tensor(scores, dtype=torch.float32)
        if isinstance(boxes, torch.Tensor) and boxes.ndim > 2:
            boxes = boxes.squeeze(0)
        if isinstance(scores, torch.Tensor) and scores.ndim > 1:
            scores = scores.squeeze()

        # Bail out cleanly if nothing there
        if boxes is None or scores is None or (isinstance(boxes, torch.Tensor) and boxes.numel() == 0):
            return []

        # If labels are integer ids, map back to the caption phrases
        if labels is not None and torch.is_tensor(labels):
            idxs = labels.tolist()
            labels = [phrases[i] if 0 <= i < len(phrases) else str(i) for i in idxs]

        # 6) Single NMS pass
        # Move to CPU floats for NMS
        boxes = boxes.detach().to("cpu").float()
        scores = scores.detach().to("cpu").float()

        # If labels are integer ids, map to phrase text
        if labels is not None and torch.is_tensor(labels):
            idxs = labels.tolist()
            labels = [phrases[i] if 0 <= i < len(phrases) else str(i) for i in idxs]
        elif labels is None:
            labels = [""] * boxes.shape[0]

        # 7) Scale boxes back to original size if we upsampled
        if original.size != img.size:
            sx = original.size[0] / img.size[0]
            sy = original.size[1] / img.size[1]
            boxes = boxes * torch.tensor([sx, sy, sx, sy])

        # 8) Optional box expansion (padding)
        W0, H0 = original.size
        proposals: List[Dict] = []
        for i, (b, s) in enumerate(zip(boxes.tolist(), scores.tolist())):
            out_box = self._expand_xyxy(b, W0, H0, self.cfg.box_expand) if self.cfg.box_expand > 0 else b
            lab = (labels[i] if labels is not None and i < len(labels) else "") or ""
            proposals.append({
                "box": [float(x) for x in out_box],
                "score": float(s),
                "prompt": str(lab),
            })

        return proposals