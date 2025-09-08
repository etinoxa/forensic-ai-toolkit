from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, cache_path, is_audio_file, load_audio_any
from .base import _BaseAudioEmbedder
from fait.core.registry import register_audio_embedder


@register_audio_embedder("titanet")
class TitanetEmbedder(_BaseAudioEmbedder):
    """
    NVIDIA Titanet-Large speaker embeddings.

    Defaults to Hugging Face repo 'nvidia/speakerverification_en_titanet_large'.
    We avoid symlinks and copy files into <repo>/.fait/cache/models/audio/titanet.
    Requires: nemo-toolkit >= 1.21 (nemo.collections.asr.models.EncDecSpeakerLabelModel)
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
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Windows-safe: no symlinks
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        self.cache_root = Path(cache_dir or (paths.models_cache / "audio" / "titanet"))
        ensure_folder(self.cache_root)

        # 1) Pull the repo locally
        repo_dir = self.cache_root / "repo"
        ensure_folder(repo_dir)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(repo_dir),
            local_dir_use_symlinks=False,
        )

        # .nemo file path (common layout)
        nemo_candidates = list(repo_dir.glob("*.nemo"))
        if not nemo_candidates:
            raise RuntimeError(
                f"No .nemo file found in {repo_dir}. Make sure the repo {model_id} "
                "contains a NeMo checkpoint."
            )
        self.nemo_path = nemo_candidates[0]
        self.model_id = model_id

        # 2) Load NeMo model
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except Exception as e:
            raise RuntimeError(
                "NeMo is required for Titanet. Install with:\n"
                "  pip install nemo_toolkit[asr]\n"
            ) from e

        self.model = EncDecSpeakerLabelModel.restore_from(
            restore_path=str(self.nemo_path),
            map_location=self.device,
        ).eval()
        # Optional: fast inference + TF32
        torch.backends.cuda.matmul.allow_tf32 = True

    def name(self) -> str:
        return f"TitanetLarge({Path(self.model_id).name}, {self.device})"

    def _load_audio(self, path: Path, target_sr: int) -> tuple[Optional[np.ndarray], Optional[int]]:
        # Robust loader: try soundfile, then librosa
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

    # def embed_file(self, audio_path: str | Path, use_cache: bool = True) -> Optional[np.ndarray]:
    #     audio_path = Path(audio_path)
    #     if not audio_path.exists() or not is_audio_file(audio_path):
    #         return None
    #
    #     base = cache_path(self.embed_cache_dir, audio_path, tag="titanet")
    #     npy = Path(base).with_suffix(".npy")
    #     if use_cache and npy.exists():
    #         try:
    #             return np.load(npy)
    #         except Exception:
    #             pass
    #
    #     wav, sr = self._load_audio(audio_path, target_sr=self.sample_rate)
    #     if wav is None:
    #         return None
    #
    #     wav_t = torch.from_numpy(wav).float().to(self.device).unsqueeze(0)
    #     length_t = torch.tensor([wav_t.shape[1]], device=self.device).long()
    #
    #     with torch.inference_mode():
    #         # NeMo API: get_embedding(audio_signal, length) -> (B, D) or (B, 1, D)
    #         out = self.model.get_embedding(audio_signal=wav_t, length=length_t)
    #         if isinstance(out, (list, tuple)):
    #             emb = out[0]
    #         else:
    #             emb = out
    #         emb = emb.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
    #
    #     npy.parent.mkdir(parents=True, exist_ok=True)
    #     np.save(npy, emb)
    #     return emb

    def embed_file(self, audio_path: str, use_cache: bool = True) -> np.ndarray | None:
        wav, sr = load_audio_any(audio_path, sr=self.sample_rate, mono=True)
        if wav.size == 0:
            return None
        wav_t = torch.from_numpy(wav).to(self.device).unsqueeze(0)  # [1, T]
        with torch.no_grad():
            out = self.model(wav_t)  # or appropriate forward
            emb = out.squeeze(0).cpu().numpy()
        return emb.astype(np.float32, copy=False)
