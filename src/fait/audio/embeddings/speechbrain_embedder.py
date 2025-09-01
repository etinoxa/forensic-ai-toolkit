# src/fait/audio/embeddings/speechbrain_embedder.py
from __future__ import annotations
import os, tempfile, subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample

try:
    import imageio_ffmpeg as ffmpegio  # fallback converter for m4a/mp3/etc.
except Exception:
    ffmpegio = None

from speechbrain.pretrained import EncoderClassifier

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, save_embedding, load_embedding
from fait.core.registry import register_audio_embedder
from .base import _BaseAudioEmbedder  # <- package import (no relative import from a script)


@register_audio_embedder("speechbrain")
class SpeechBrainEmbedder(_BaseAudioEmbedder):
    """
    ECAPA-TDNN speaker embeddings via SpeechBrain.
    Model hub id: 'speechbrain/spkrec-ecapa-voxceleb'
    This implementation avoids Windows symlink errors by NOT using `savedir`
    (we point HF cache to .fait and load directly from there).
    """

    def __init__(
        self,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        embed_cache_dir: Optional[str] = None,
        hf_home: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)

        paths = get_paths()
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Route Hugging Face cache under our project so it's offline-friendly
        # and doesn't try to symlink into a custom savedir.
        self.hf_home = hf_home or str(paths.models_cache / "audio" / "hf_home")
        ensure_folder(self.hf_home)
        os.environ.setdefault("HF_HOME", self.hf_home)

        # CRITICAL: savedir=None prevents SpeechBrain from creating symlinks.
        self.clf = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=None,
            run_opts={"device": self.device},
        )
        self.clf.eval()

    # ---------- public API ----------
    def name(self) -> str:
        return f"SpeechBrain({Path(self.model_id).name},{self.device})"

    @property
    def model_tag(self) -> str:
        # short tag for cache keys; include model id tail to avoid collisions
        return f"speechbrain_{Path(self.model_id).name}"

    def embed_file(self, path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        base = self._cache_base(path)
        if use_cache:
            for ext in (".pkl", ".npy"):
                p = base + ext
                if os.path.exists(p):
                    try:
                        return load_embedding(p)
                    except Exception:
                        # fall through to recompute if cache read fails
                        pass
        try:
            wav, sr = self._load_mono_16k(path)
        except Exception:
            return None
        emb = self.embed_tensor(wav, sr)
        try:
            save_embedding(base, emb)
        except Exception:
            pass
        return emb

    def embed_tensor(self, wav: torch.Tensor, sr: int) -> np.ndarray:
        """Accepts torch waveform [C,T] or [T]; returns L2-normalized emb (float32)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = Resample(sr, self.sample_rate)(wav)

        with torch.no_grad():
            emb = self.clf.encode_batch(wav.to(self.device)).squeeze().cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb.astype(np.float32)

    # ---------- helpers ----------
    def _load_mono_16k(self, path: str) -> Tuple[torch.Tensor, int]:
        """
        Try torchaudio first (fast for WAV/FLAC/etc). If it fails (e.g., M4A),
        use ffmpeg to convert to a temporary 16kHz mono WAV, then load.
        """
        try:
            wav, sr = torchaudio.load(path)  # [C,T]
        except Exception:
            if ffmpegio is None:
                raise
            tmpdir = Path(tempfile.mkdtemp(prefix="audio_conv_"))
            dst = tmpdir / (Path(path).stem + "_16k_mono.wav")
            cmd = [
                ffmpegio.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-i", path, "-ar", str(self.sample_rate), "-ac", "1", str(dst)
            ]
            subprocess.run(cmd, check=True)
            wav, sr = torchaudio.load(str(dst))

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = Resample(sr, self.sample_rate)(wav)

        peak = wav.abs().max()
        if float(peak) > 0:
            wav = wav / peak.clamp(min=1e-9)
        return wav, self.sample_rate
