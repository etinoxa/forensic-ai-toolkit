from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, List
import os, time, numpy as np
from pathlib import Path

from fait.core.utils import (
    ensure_folder, is_audio_file, cache_path, save_embedding, load_embedding
)
from fait.core.paths import get_paths

class AudioEmbedder(Protocol):
    """Interface for audio embedders."""
    def name(self) -> str: ...
    @property
    def model_tag(self) -> str: ...   # short tag for cache keys
    def embed_file(self, audio_path: str, use_cache: bool = True) -> Optional[np.ndarray]: ...
    def mean_embedding_from_folder(self, folder: str, use_cache: bool = True) -> np.ndarray: ...

@dataclass
class _BaseAudioEmbedder:
    """Utility base with shared caching behavior."""
    embed_cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        paths = get_paths()
        self._embed_cache = self.embed_cache_dir or str(paths.embeddings_cache)
        ensure_folder(self._embed_cache)

    @property
    def _cache_dir(self) -> str:
        return self._embed_cache

    def _cache_base(self, audio_path: str) -> str:
        # Use model_tag to avoid collisions between different audio models
        return cache_path(self._cache_dir, audio_path, self.model_tag)

    # default implementation for mean over a folder
    def mean_embedding_from_folder(self, folder: str, use_cache: bool = True) -> np.ndarray:
        embs: List[np.ndarray] = []
        for fn in sorted(os.listdir(folder)):
            fp = os.path.join(folder, fn)
            if not is_audio_file(fp):
                continue
            e = self.embed_file(fp, use_cache=use_cache)
            if e is not None:
                embs.append(e)
        if not embs:
            raise ValueError(f"No valid audio embeddings from: {folder}")
        mean = np.mean(embs, axis=0).astype(np.float32)
        # L2 normalize
        n = np.linalg.norm(mean) + 1e-9
        return (mean / n).astype(np.float32)
