from __future__ import annotations
import os, numpy as np
from typing import Optional

from fait.core.registry import register_embedder
from fait.core.interfaces import Embedder
from fait.core.utils import ensure_folder, is_image_file, cache_path, save_embedding, load_embedding, l2_normalize
from fait.core.paths import get_paths
from fait.vision.services.face_service import get_face_service, FaceService

@register_embedder("arcface")
class ArcFaceEmbedder(Embedder):
    """
    ArcFace embeddings via shared FaceService (no duplicate model load).
    """
    def __init__(self, embed_cache_dir: str | None = None, face_service: Optional[FaceService] = None):
        paths = get_paths()
        self.embed_cache_dir = embed_cache_dir or str(paths.embeddings_cache)
        ensure_folder(self.embed_cache_dir)
        self.fs = face_service or get_face_service()

    def name(self) -> str:
        return f"ArcFace({self.fs.device})"

    def _cache_base(self, image_path: str) -> str:
        return cache_path(self.embed_cache_dir, image_path, "arcface")

    def embed_image(self, image_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        if not is_image_file(image_path): return None
        base = self._cache_base(image_path)
        for ext in (".pkl", ".npy"):
            p = base + ext
            if use_cache and os.path.exists(p):
                return load_embedding(p)
        emb = self.fs.embed(image_path, select="best")
        if emb is None: return None
        save_embedding(base, emb)  # writes .pkl default
        return emb

    def mean_embedding(self, folder: str, use_cache: bool = True) -> np.ndarray:
        import os
        embs = []
        for fn in os.listdir(folder):
            fp = os.path.join(folder, fn)
            if not is_image_file(fp): continue
            e = self.embed_image(fp, use_cache=use_cache)
            if e is not None: embs.append(e)
        if not embs:
            raise ValueError("No valid reference embeddings found.")
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        return l2_normalize(mean.astype(np.float32))
