# src/fait/audio/speaker_recognition/models/resnet_embedder.py
from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import snapshot_download
from speechbrain.inference.speaker import EncoderClassifier

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, cache_path, is_audio_file
from fait.audio.speaker_recognition.base import _BaseAudioEmbedder
from fait.core.registry import register_audio_embedder


@register_audio_embedder("resnet")
class ResNetEmbedder(_BaseAudioEmbedder):
    """
    Speaker models from SpeechBrain ResNet34:
      - HF repo: 'speechbrain/spkrec-resnet34-v1'
    """

    def __init__(
        self,
        model_id: str = "speechbrain/spkrec-resnet34-v1",
        cache_dir: str | None = None,
        embed_cache_dir: str | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)
        paths = get_paths()
        self.model_id = model_id
        self.cache_root = Path(cache_dir or (paths.models_speaker_recognition/ "resnet"))
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Where SpeechBrain will look for its files
        self.savedir = self.cache_root  # keep flat layout to avoid symlinks

        # Environment knobs to reduce symlink noise on Windows
        os.environ.setdefault("SPEECHBRAIN_LOCAL_FILES_STRATEGY", "copy")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        # Pre-download the repo into a subfolder, then copy the few files we know SpeechBrain expects
        repo_dir = self.cache_root / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.model_id,
            local_dir=str(repo_dir),
            local_dir_use_symlinks=False,  # force real files
            ignore_patterns=[],            # keep everything; small repo
        )

        # Copy required files up one level (if they exist), so SB won't try to symlink
        for fname in [
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.txt",
        ]:
            src = repo_dir / fname
            dst = self.savedir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load classifier straight from savedir (already populated above)
        # NOTE: we pass source=str(self.savedir) to avoid another fetch/symlink.
        self.clf = EncoderClassifier.from_hparams(
            source=str(self.savedir),
            run_opts={"device": self.device},
        )
        self.clf.eval()

    # --- API required by pipelines -------------------------------------------------

    def name(self) -> str:
        return f"ResNet34({Path(self.model_id).name}, {self.device})"

    def embed_file(self, audio_path: str | Path, use_cache: bool = True) -> Optional[np.ndarray]:
        audio_path = Path(audio_path)
        if not audio_path.exists() or not is_audio_file(audio_path):
            return None

        # cache
        base = cache_path(self.embed_cache_dir, audio_path, tag="resnet34")
        npy = Path(base).with_suffix(".npy")
        if use_cache and npy.exists():
            try:
                return np.load(npy)
            except Exception:
                pass

        wav, sr = self._load_audio(audio_path, target_sr=self.sample_rate)
        if wav is None:
            return None

        with torch.inference_mode():
            wav_t = torch.from_numpy(wav).float().to(self.device).unsqueeze(0)  # (1, T)
            emb = self.clf.encode_batch(wav_t).squeeze(0).squeeze(0).cpu().numpy()  # (D,)

        # write cache
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, emb.astype(np.float32, copy=False))
        return emb

    # --- helpers -------------------------------------------------------------------

    def _load_audio(self, path: Path, target_sr: int) -> tuple[Optional[np.ndarray], Optional[int]]:
        """
        Robust loader: try soundfile -> librosa (audioread/ffmpeg) for m4a/mp4.
        Returns mono float32 waveform in [-1,1] and the sample rate.
        """
        try:
            y, sr = sf.read(str(path), always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception:
            try:
                y, sr = librosa.load(str(path), sr=None, mono=True)
            except Exception:
                return None, None

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        y = y.astype(np.float32, copy=False)
        return y, sr
