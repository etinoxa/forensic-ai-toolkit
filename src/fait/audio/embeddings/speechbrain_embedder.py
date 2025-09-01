# src/fait/audio/embeddings/speechbrain_embedder.py
from __future__ import annotations
import os
import warnings
from typing import Optional

import numpy as np
import torch

from pathlib import Path
from huggingface_hub import snapshot_download
from speechbrain.inference import EncoderClassifier
from speechbrain.utils import fetching

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, cache_path, save_embedding, load_embedding
from fait.core.registry import register_audio_embedder
from .base import _BaseAudioEmbedder

# Windows-safe + quieter
os.environ.setdefault("SPEECHBRAIN_LOCAL_FILES_STRATEGY", "copy")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*", module="webrtcvad")

@register_audio_embedder("speechbrain")
class SpeechBrainEmbedder(_BaseAudioEmbedder):
    """
    ECAPA-TDNN speaker embeddings via SpeechBrain.
    Model: 'speechbrain/spkrec-ecapa-voxceleb'
    """

    def __init__(
            self,
            model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
            cache_dir: str | None = None,
            embed_cache_dir: str | None = None,
            sample_rate: int = 16000,
            savedir: str | None = None,
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)

        # --- define paths early so they exist even if downloads fail ---
        paths = get_paths()
        base_cache = Path(cache_dir or (paths.models_cache / "audio"))
        self.savedir = Path(savedir) if savedir else (base_cache / "speechbrain")
        ensure_folder(self.savedir)

        # the rest of your init
        self.model_id = model_id
        self.cache_dir = str(base_cache)
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # avoid symlinks on Windows
        try:
            fetching.set_default_strategy(fetching.LocalStrategy())
        except Exception:
            pass

        try:
            self.clf = EncoderClassifier.from_hparams(
                source=self.model_id,
                run_opts={"device": self.device},
                savedir=str(self.savedir),
            )
        except OSError:
            # full local fallback (no symlinks)
            local_repo = self.savedir / "repo"
            ensure_folder(local_repo)
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(local_repo),
                local_dir_use_symlinks=False,
            )
            self.clf = EncoderClassifier.from_hparams(
                source=str(local_repo),
                run_opts={"device": self.device},
                savedir=str(self.savedir),
            )
        self.clf.eval()


    def name(self) -> str:
        # Use savedir name if present; otherwise fall back to model_id tail
        try:
            tag = Path(self.savedir).name
        except Exception:
            tag = Path(getattr(self, "model_id", "speechbrain")).name
        return f"SpeechBrain({tag}, {self.device})"

    def _cache_base(self, audio_path: str) -> str:
        return cache_path(self._embed_cache, audio_path, self._model_tag)

    def embed(self, audio_path: str) -> Optional[np.ndarray]:
        try:
            with torch.no_grad():
                emb = self.clf.encode_file(audio_path)
            return emb.squeeze().detach().cpu().numpy().astype(np.float32)
        except Exception:
            return None

    def embed_file(self, audio_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        base = self._cache_base(audio_path)
        for ext in (".pkl", ".npy"):
            p = base + ext
            if use_cache and os.path.exists(p):
                return load_embedding(p)
        emb = self.embed(audio_path)
        if emb is not None:
            save_embedding(base, emb)
        return emb
