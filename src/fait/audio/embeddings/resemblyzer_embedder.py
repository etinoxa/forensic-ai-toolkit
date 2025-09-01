# src/fait/audio/embeddings/resemblyzer_embedder.py

from __future__ import annotations
from typing import Optional
import numpy as np
import librosa
import torch

from resemblyzer import VoiceEncoder, preprocess_wav
from fait.core.registry import register_audio_embedder
from fait.core.utils import save_embedding, load_embedding, ensure_folder
from fait.core.paths import get_paths
from .base import _BaseAudioEmbedder

@register_audio_embedder("resemblyzer")
class ResemblyzerEmbedder(_BaseAudioEmbedder):
    """
    Speaker embeddings via Resemblyzer (256-dim).
    """
    def __init__(
        self,
        model_tag_override: str | None = None,
        embed_cache_dir: str | None = None,
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)
        paths = get_paths()
        # Resemblyzer downloads into ~/.cache by default;
        # no weights arg; we just ensure cache dir exists for our embeddings.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = VoiceEncoder(device=self.device)
        self._tag = model_tag_override or "resemblyzer"

    def name(self) -> str:
        return "Resemblyzer"

    @property
    def model_tag(self) -> str:
        return self._tag

    def embed_file(self, audio_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        base = self._cache_base(audio_path)
        for ext in (".pkl", ".npy"):
            p = base + ext
            if use_cache and os.path.exists(p):
                return load_embedding(p)

        # Resemblyzer has its own robust WAV loader + VAD
        wav = preprocess_wav(audio_path)
        if wav is None or len(wav) == 0:
            return None
        emb = self.encoder.embed_utterance(wav).astype(np.float32)
        # L2 normalize for consistent cosine compare with other models
        n = np.linalg.norm(emb) + 1e-9
        emb = (emb / n).astype(np.float32)
        save_embedding(base, emb)
        return emb
