# src/fait/audio/speaker_recognition/models/titanet_embedder.py
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
from fait.audio.speaker_recognition.base import _BaseAudioEmbedder


@register_audio_embedder("titanet")
class TitanetEmbedder(_BaseAudioEmbedder):
    """
    NVIDIA Titanet-Large speaker models.

    Strategy:
      - Download local snapshot (no symlinks).
      - Restore NeMo model.
      - Always use get_embedding(input_signal=..., input_signal_length=...).
      - Reduce to (C,) and L2-normalize. No fixed-dim enforcement.
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
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Local snapshot (no symlinks)
        self.cache_root = Path(cache_dir or (paths.models_speaker_recognition / "titanet"))
        self.repo_dir = self.cache_root / "repo"
        ensure_folder(self.repo_dir)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        snapshot_download(
            repo_id=self.model_id,
            local_dir=str(self.repo_dir),
            local_dir_use_symlinks=False,
        )

        nemo_files = list(self.repo_dir.glob("*.nemo"))
        if not nemo_files:
            raise RuntimeError(
                f"No .nemo checkpoint found in {self.repo_dir} for {self.model_id}"
            )
        nemo_path = nemo_files[0]

        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except Exception as e:
            raise RuntimeError(
                "NeMo is required for Titanet. Install e.g.\n"
                "  pip install 'nemo_toolkit[asr]'"
            ) from e

        self.model = EncDecSpeakerLabelModel.restore_from(
            restore_path=str(nemo_path),
            map_location=self.device,
        ).eval()

        # A stable cache tag that won't fight old shapes
        self._model_tag = "titanet.v6"  # constant; do not encode dim/variant

    def name(self) -> str:
        return f"TitanetLarge({Path(self.model_id).name}, {self.device})"

    # ---------- public API ----------

    def embed_file(self, audio_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        base = cache_path(self.embed_cache_dir, audio_path, self._model_tag)
        npy = base + ".npy"

        if use_cache:
            arr = load_numpy_safe(npy)
            if arr is not None:
                return arr

        wav, sr = load_audio_any(audio_path, sr=self.sample_rate, mono=True)
        if wav is None or wav.size == 0:
            return None

        emb = self._embed_wav(wav, sr)
        if emb is None:
            return None

        ensure_folder(Path(npy).parent)
        save_numpy(npy, emb)
        return emb

    # ---------- internals ----------

    def _reduce_to_vector(self, t) -> torch.Tensor:
        """
        Reduce outputs to shape (C,), then L2-normalize.
        Handles tensor/tuple/list/dict. Pools batch/time if necessary.
        """
        if isinstance(t, (list, tuple)):
            t = t[0]
        if isinstance(t, dict):
            # prefer common keys
            for k in ("models", "embedding", "x", "out"):
                if k in t:
                    t = t[k]
                    break

        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=self.device)

        # Common cases:
        #  (B, C)          -> (C)
        #  (B, T, C)       -> mean over T -> (B, C) -> (C)
        #  (B, C, T)       -> mean over T -> (B, C) -> (C)
        #  Weird shapes    -> flatten -> (N,)
        if t.dim() == 3:
            # choose time dim heuristically (the larger of dim1/2)
            time_dim = 1 if t.shape[1] >= t.shape[2] else 2
            t = t.mean(dim=time_dim)  # -> (B, C)

        if t.dim() == 2:
            # pool batch if needed
            if t.shape[0] > 1:
                t = t.mean(dim=0)
            else:
                t = t[0]

        if t.dim() != 1:
            t = t.view(-1)

        t = torch.nn.functional.normalize(t.float(), p=2, dim=0)
        return t

    @torch.inference_mode()
    def _embed_wav(self, wav: np.ndarray, sr: int) -> Optional[np.ndarray]:
        # The current NeMo get_embedding API expects file paths, not tensors
        # Let's try to use the model's forward method directly

        # Prepare input tensor
        x = torch.from_numpy(wav.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        xlen = torch.tensor([x.shape[1]], dtype=torch.int64, device=self.device)

        try:
            # Try using the model's forward method directly
            if hasattr(self.model, 'forward'):
                out = self.model.forward(input_signal=x, input_signal_length=xlen)
            else:
                # Fallback to using get_embedding with temporary file
                import tempfile
                import soundfile as sf
                import os

                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name

                # Save audio to temporary file
                sf.write(temp_path, wav, sr)

                try:
                    # Use NeMo's get_embedding method with file path
                    out = self.model.get_embedding(temp_path)
                finally:
                    # Clean up temporary file
                    os.unlink(temp_path)

        except Exception as e:
            raise RuntimeError(f"Failed to extract embedding: {e}") from e

        vec = self._reduce_to_vector(out)
        if not torch.isfinite(vec).all():
            return None

        return vec.cpu().numpy().astype(np.float32)


