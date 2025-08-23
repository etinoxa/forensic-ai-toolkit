from __future__ import annotations
import os, torch, numpy as np
from typing import Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from fait.core.registry import register_embedder
from fait.core.interfaces import Embedder
from fait.core.utils import ensure_folder, is_image_file, cache_path, save_embedding, load_embedding, l2_normalize
from fait.core.paths import get_paths
@register_embedder("clip")
class CLIPEmbedder(Embedder):
    """
    openai/clip-vit-base-patch32 (image features). L2-normalized.
    """
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32",
                 cache_dir: str | None = None,
                 embed_cache_dir: str | None = None):
        paths = get_paths()
        self.model_id = model_id
        self.cache_dir = cache_dir or str(paths.models_cache)
        self.embed_cache_dir = embed_cache_dir or str(paths.embeddings_cache)
        ensure_folder(self.cache_dir); ensure_folder(self.embed_cache_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).to(self.device).eval()

    def name(self) -> str:
        return f"CLIP({self.model_id}, {self.device})"

    def _cache_base(self, image_path: str) -> str:
        return cache_path(self.embed_cache_dir, image_path, "clip")

    def embed_image(self, image_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        if not is_image_file(image_path): return None
        base = self._cache_base(image_path)
        for ext in (".pkl", ".npy"):
            p = base + ext
            if use_cache and os.path.exists(p):
                return load_embedding(p)

        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                emb = feats.detach().cpu().numpy().reshape(-1).astype(np.float32)
            save_embedding(base, emb)
            return emb
        except Exception:
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
