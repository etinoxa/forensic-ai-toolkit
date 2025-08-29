from __future__ import annotations
import os, torch, numpy as np
from typing import Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import time, logging
from fait.core.registry import register_embedder
from fait.core.interfaces import Embedder
from fait.core.utils import ensure_folder, is_image_file, cache_path, save_embedding, load_embedding, l2_normalize
from fait.core.paths import get_paths, ensure_on_first_write

log = logging.getLogger("fait.vision.embeddings.clip")

@register_embedder("clip")
class CLIPEmbedder(Embedder):
    """
    openai/clip-vit-base-patch32 (image features). L2-normalized.
    """
    def __init__(self, model_id="openai/clip-vit-base-patch32",
                 cache_dir: str | None = None,
                 embed_cache_dir: str | None = None):
        paths = get_paths()
        self.model_id = model_id
        self.cache_dir = paths.models_face_match
        self.embed_cache_dir = embed_cache_dir or str(paths.embedding_cache)
        ensure_folder(self.cache_dir);
        ensure_folder(self.embed_cache_dir)
        ensure_on_first_write(self.cache_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
        self.model = CLIPModel.from_pretrained(model_id, cache_dir=self.cache_dir).to(self.device).eval()

    def _cache_base(self, image_path: str) -> str:
        # include model id so different CLIP variants don’t share cache files
        return cache_path(self.embed_cache_dir, image_path, f"clip:{self.model_id}")

    def name(self) -> str:
        return f"CLIP({self.model_id}, {self.device})"

    def _cache_base(self, image_path: str) -> str:
        return cache_path(self.embed_cache_dir, image_path, "clip")

    def embed_image(self, image_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        if not is_image_file(image_path):
            return None
        base = self._cache_base(image_path)
        for ext in (".pkl", ".npy"):
            p = base + ext
            if use_cache and os.path.exists(p):
                log.debug("embed:cache_hit", extra={"image": image_path, "cache": p})
                return load_embedding(p)

        t0 = time.time()
        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                emb = feats.detach().cpu().numpy().reshape(-1).astype(np.float32)
            save_embedding(base, emb)
            log.debug("embed:ok",
                      extra={"image": image_path, "ms": int((time.time() - t0) * 1000), "cache": base + ".pkl"})
            return emb
        except Exception as e:
            log.debug("embed:error", extra={"image": image_path, "ms": int((time.time() - t0) * 1000), "err": str(e)})
            return None

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
