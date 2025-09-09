# src/fait/audio/embeddings/titanet_embedder.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download

from fait.core.paths import get_paths
from fait.core.registry import register_audio_embedder
from fait.core.utils import (
    ensure_folder,
    cache_path,
    load_numpy_safe,
    save_numpy,
    load_audio_any,
)
from .base import _BaseAudioEmbedder


@register_audio_embedder("titanet")
class TitanetEmbedder(_BaseAudioEmbedder):
    """
    NVIDIA Titanet-Large speaker embeddings.

    - Repo: nvidia/speakerverification_en_titanet_large
    - Loads from a local .nemo (downloaded via HF Hub without symlinks)
    - Produces a single L2-normalized float32 embedding per file
    """

    def __init__(
        self,
        model_id: str = "nvidia/speakerverification_en_titanet_large",
        cache_dir: str | None = None,
        embed_cache_dir: str | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)
        paths = get_paths()

        # Cache locations
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Where the .nemo and related files will live
        self.cache_root = Path(cache_dir or (paths.models_cache / "audio" / "titanet"))
        self.repo_dir = self.cache_root / "repo"
        ensure_folder(self.repo_dir)

        # Avoid symlink warnings / behavior
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        # Download snapshot into a normal folder (no symlinks)
        snapshot_download(
            repo_id=self.model_id,
            local_dir=str(self.repo_dir),
            local_dir_use_symlinks=False,
        )

        # Locate the .nemo file
        nemo_files = list(self.repo_dir.glob("*.nemo"))
        if not nemo_files:
            raise RuntimeError(
                f"No .nemo checkpoint found in {self.repo_dir}. "
                f"Repo '{self.model_id}' should contain a NeMo checkpoint."
            )
        nemo_path = nemo_files[0]

        # Load NeMo model
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except Exception as e:
            raise RuntimeError(
                "NeMo is required for Titanet. Install with:\n"
                "  pip install 'nemo_toolkit[asr]'  \n"
                "or in Docker image ensure nemo_toolkit is included."
            ) from e

        self.model = EncDecSpeakerLabelModel.restore_from(
            restore_path=str(nemo_path),
            map_location=self.device,
        ).eval()

        # A small tag for cache file naming (prevents cross-model collisions)
        self._model_tag = "titanet"

    # ---------- Public API ----------

    def name(self) -> str:
        return f"TitanetLarge({Path(self.model_id).name}, {self.device})"

    def embed_file(self, audio_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Returns a single L2-normalized float32 vector or None on failure.
        Caches to .npy using the standard cache_path/save_numpy helpers.
        """
        # Cache path (per-file, per-model)
        base = cache_path(self.embed_cache_dir, audio_path, self._model_tag)
        npy = base + ".npy"

        if use_cache:
            arr = load_numpy_safe(npy)
            if arr is not None:
                return arr

        # Load audio (handles m4a/mp3/wav/flac via ffmpeg backend in utils)
        wav, sr = load_audio_any(audio_path, sr=self.sample_rate, mono=True)
        if wav is None or wav.size == 0:
            return None

        # Compute embedding
        emb = self._embed_wav(wav, sr)
        if emb is None:
            return None

        # Save cache and return
        ensure_folder(Path(npy).parent)
        save_numpy(npy, emb)
        return emb

    # ---------- Internals ----------

    def _embed_wav(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Core embedding path. Always passes input lengths to NeMo.
        Normalizes the output to unit L2.
        """
        # NeMo expects float32 tensors
        wav_t = torch.from_numpy(wav.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        len_t = torch.tensor([wav_t.shape[1]], dtype=torch.int64, device=self.device)

        with torch.no_grad():
            # 1) Preferred API: get_embedding (exists on most NeMo speaker models)
            if hasattr(self.model, "get_embedding"):
                # Try with positional arguments only
                try:
                    emb_t = self.model.get_embedding(wav_t)
                except Exception as e:
                    # If that fails, try with both positional arguments
                    try:
                        emb_t = self.model.get_embedding(wav_t, len_t)
                    except Exception:
                        # If both fail, fall back to the original approach
                        raise e

                # Some versions return (emb, ) or time-major outputs → squeeze + pool if needed
                if isinstance(emb_t, (tuple, list)):
                    emb_t = emb_t[0]

                if emb_t.dim() == 3:  # (B, T, C) or (B, C, T) → mean-pool over time
                    if emb_t.shape[1] > emb_t.shape[2]:
                        emb_t = emb_t.mean(dim=1)
                    else:
                        emb_t = emb_t.mean(dim=2)

                emb_t = torch.nn.functional.normalize(emb_t, p=2, dim=-1)
                return emb_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # 2) Fallback: preprocessor -> encoder -> mean-pool
            proc_sig, proc_len = self.model.preprocessor(input_signal=wav_t, length=len_t)
            enc = self.model.encoder(audio_signal=proc_sig, length=proc_len)  # (B,T,C) or (B,C,T)

            if enc.dim() == 3:
                if enc.shape[1] > enc.shape[2]:
                    emb_t = enc.mean(dim=1)
                else:
                    emb_t = enc.mean(dim=2)
            else:
                emb_t = enc

            emb_t = torch.nn.functional.normalize(emb_t, p=2, dim=-1)
            return emb_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
