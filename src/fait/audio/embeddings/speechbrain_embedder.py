# src/fait/audio/embeddings/speechbrain_embedder.py
from __future__ import annotations
import os, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa

from speechbrain.utils import fetching as sb_fetching

from huggingface_hub import snapshot_download
from speechbrain.inference import EncoderClassifier

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, cache_path, load_audio_any
from fait.audio.embeddings.base import _BaseAudioEmbedder
from fait.core.registry import register_audio_embedder

if os.name == "nt":
    _real_symlink_to = Path.symlink_to  # keep original for non-error cases

    def _symlink_or_copy(self: Path, target, target_is_directory=False):
        """
        Try to create a real symlink; if not permitted, copy files/dirs instead.
        This makes SpeechBrain's fetch/link steps work on Windows without admin/dev mode.
        """
        try:
            # Try real symlink first (works if user has privileges)
            return _real_symlink_to(self, target, target_is_directory)
        except Exception:
            src = Path(target)
            dst = self

            # Ensure parent exists
            dst.parent.mkdir(parents=True, exist_ok=True)

            # If destination exists, remove appropriately
            if dst.exists():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    try:
                        dst.unlink()
                    except Exception:
                        # last resort: remove tree
                        shutil.rmtree(dst, ignore_errors=True)

            if src.is_dir():
                # Copy whole directory tree
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                # If dst is a directory path (sometimes SB passes a dir as target),
                # place the file under it:
                if dst.exists() and dst.is_dir():
                    shutil.copy2(src, dst / src.name)
                else:
                    # Ensure parent exists (again in case we changed dst)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            return None  # for API compatibility

    # Monkey-patch globally for this process
    Path.symlink_to = _symlink_or_copy  # type: ignore[assignment]

@dataclass
class _Cfg:
    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    sample_rate: int = 16000

@register_audio_embedder("speechbrain")
class SpeechBrainEmbedder(_BaseAudioEmbedder):
    """ECAPA-TDNN speaker embeddings via SpeechBrain, Windows-safe (no symlinks)."""

    def __init__(
            self,
            model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
            cache_dir: str | None = None,
            embed_cache_dir: str | None = None,
            sample_rate: int = 16000,  # <-- keep this arg
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)
        paths = get_paths()

        self.model_id = model_id
        self.sample_rate = int(sample_rate)  # <-- **restore this line**
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # keep your chosen cache location
        self.cache_dir = cache_dir or str(paths.models_cache / "audio" / "speechbrain")
        ensure_folder(self.cache_dir)

        # load model (your existing loader logic is fine)
        self.clf = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=self.cache_dir,
            run_opts={"device": self.device},
        )
        self.clf.eval()

    # ---- Embedding API ----
    def name(self) -> str:
        return f"SpeechBrain({Path(self.repo_dir).name}, {self.device})"

    def embed_wave(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg.sample_rate)
            sr = self.cfg.sample_rate
        wav = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            emb = self.clf.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
        return emb.astype(np.float32, copy=False)

    def embed_file(self, audio_path: str, use_cache: bool = True) -> np.ndarray | None:
        # load with fallback (handles .m4a)
        wav, sr = load_audio_any(audio_path, sr=self.sample_rate, mono=True)
        if wav.size == 0:
            return None

        wav_t = torch.from_numpy(wav).unsqueeze(0).to(self.device)  # [1, T]
        with torch.no_grad():
            emb = self.clf.encode_batch(wav_t).squeeze(0).mean(dim=0).cpu().numpy()
        return emb.astype(np.float32, copy=False)
