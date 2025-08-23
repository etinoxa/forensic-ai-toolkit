from __future__ import annotations
import os, cv2, torch, numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass
from insightface.app import FaceAnalysis

from fait.core.utils import ensure_folder, is_image_file, l2_normalize
from fait.core.paths import get_paths
from fait.vision.recognizers.base import Detection

SelectStrategy = Literal["first", "largest", "best"]

@dataclass(frozen=True)
class FaceServiceConfig:
    model_name: str = "buffalo_l"
    cache_dir: str = str(get_paths().models_cache)   # <— changed default
    det_width: int = 640
    det_height: int = 640

class FaceService:
    """
    Shared InsightFace FaceAnalysis instance for both detection & embedding.
    """
    def __init__(self, cfg: FaceServiceConfig = FaceServiceConfig()):
        self.cfg = cfg
        ensure_folder(cfg.cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        ctx_id = 0 if self.device == "cuda" else -1
        self.app = FaceAnalysis(name=cfg.model_name, root=cfg.cache_dir, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(cfg.det_width, cfg.det_height))

    # ---- detect ----
    def detect(self, image_path: str) -> List[Detection]:
        if not is_image_file(image_path): return []
        img = cv2.imread(image_path)
        if img is None: return []
        faces = self.app.get(img)
        out: List[Detection] = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            kps = [(float(px), float(py)) for (px, py) in getattr(f, "kps", [])] if getattr(f, "kps", None) is not None else None
            out.append(Detection(bbox=[x1, y1, x2, y2], score=float(f.det_score), label="face", kps=kps))
        return out

    # ---- embed (select one face) ----
    def embed(self, image_path: str, select: SelectStrategy = "best") -> Optional[np.ndarray]:
        if not is_image_file(image_path): return None
        img = cv2.imread(image_path)
        if img is None: return None
        faces = self.app.get(img)
        if not faces: return None

        idx = 0
        if select != "first":
            areas = [max(1.0, (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
            if select == "largest":
                idx = int(np.argmax(areas))
            else:  # best = score * area
                scores = [float(f.det_score) * a for f, a in zip(faces, areas)]
                idx = int(np.argmax(scores))

        emb = faces[idx].embedding.astype(np.float32)
        return l2_normalize(emb)

# lightweight singleton
_SERVICE: Optional[FaceService] = None
def get_face_service() -> FaceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = FaceService()
    return _SERVICE
