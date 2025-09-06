# src/fait/audio/embeddings/wavlm_embedder.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector
from dataclasses import dataclass
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, save_numpy, load_numpy_safe
from .base import _BaseAudioEmbedder
from fait.core.utils import load_audio_any  # your helper that uses imageio-ffmpeg/soundfile/librosa as needed


@dataclass
class WavLMConfig:
    model_id: str = "microsoft/wavlm-base-plus-sv"
    sample_rate: int = 16000
    # NEW: chunking to avoid OOM on long files
    chunk_sec: float = 8.0       # length of each window
    hop_sec: float = 4.0         # step between windows
    max_chunks: int = 2000       # hard safety cap (optional)

class WavLMEmbedder(_BaseAudioEmbedder):
    """
    Microsoft WavLM x-vector speaker embeddings with safe chunking.
    Default model: microsoft/wavlm-base-plus-sv
    """

    def __init__(self, model_id: str = "microsoft/wavlm-base-plus-sv", cache_dir: Optional[str] = None,
                 embed_cache_dir: Optional[str] = None, sample_rate: int = 16000, chunk_sec: float = 8.0,
                 hop_sec: float = 4.0, max_chunks: int = 2000):
        super().__init__(embed_cache_dir)
        paths = get_paths()
        self.model_id = model_id
        self.cache_dir = cache_dir or str(paths.models_cache / "audio" / "wavlm")
        self.embed_cache_dir = embed_cache_dir or str(paths.embeddings_cache)
        self.sample_rate = sample_rate
        self.chunk_sec = float(chunk_sec)
        self.hop_sec = float(hop_sec)
        self.max_chunks = int(max_chunks)

        ensure_folder(self.cache_dir)
        ensure_folder(self.embed_cache_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use feature extractor (no tokenizer for SV models)
        self.feat = AutoFeatureExtractor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = WavLMForXVector.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device).eval()

    # ---------- public API ----------

    def name(self) -> str:
        return f"WavLM({Path(self.model_id).name}, {self.device})"

    @torch.inference_mode()
    def embed_file(self, audio_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        # Per-model cache file
        stem = Path(audio_path).with_suffix("")
        npy = Path(self.embed_cache_dir) / f"{stem.name}__{Path(self.model_id).name}.npy"

        if use_cache:
            cached = load_numpy_safe(npy)
            if cached is not None:
                return cached

        # load_audio_any handles m4a via ffmpeg bridge if installed (imageio-ffmpeg)
        wav, sr = load_audio_any(audio_path, sr=self.sample_rate, mono=True)
        if wav is None:
            return None

        emb = self._embed_wav(wav)
        save_numpy(npy, emb)
        return emb

    # ---------- internals ----------

    def _forward_chunk(self, wav_chunk: np.ndarray) -> np.ndarray:
        """Forward a single chunk through WavLM and return 1D embedding."""
        with torch.no_grad():
            inputs = self.feat(wav_chunk, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)  # returns .embeddings
            emb = out.embeddings.detach().cpu().numpy().squeeze()
        return emb.astype(np.float32, copy=False)

    def _model_tag(self) -> str:
        # keep short stable tag used in cache file names
        return f"{Path(self.model_id).name.replace('/', '_')}_sr{self.sample_rate}"

    @torch.inference_mode()
    def _embed_wav(self, wav: np.ndarray) -> np.ndarray:
        """Chunked embedding to avoid huge sequence lengths."""
        # Ensure mono 1-D
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = np.asarray(wav, dtype=np.float32, order="C")

        samples_per_chunk = int(round(self.chunk_sec * self.sample_rate))
        hop = int(round(self.hop_sec * self.sample_rate))

        # Short audio → single pass
        if len(wav) <= samples_per_chunk:
            emb = self._forward_chunk(wav)
            # L2-normalize
            n = np.linalg.norm(emb) + 1e-12
            return (emb / n).astype(np.float32, copy=False)

        # Long audio → sliding-window
        embeddings = []
        start = 0
        chunks = 0
        while start < len(wav):
            end = min(start + samples_per_chunk, len(wav))
            chunk = wav[start:end]
            emb = self._forward_chunk(chunk)
            # Normalize per-chunk to keep scale consistent
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            embeddings.append(emb)
            chunks += 1
            if end >= len(wav) or chunks >= self.max_chunks:
                break
            start += hop

        embs = np.stack(embeddings, axis=0)  # [num_chunks, d]
        mean = embs.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        return mean.astype(np.float32, copy=False)

    def _forward_single(self, wav: np.ndarray, max_len: int, *, return_tensor: bool = False) -> np.ndarray | torch.Tensor:
        """
        One WavLM forward on a single chunk, padded/truncated to max_len.
        """
        inputs = self.proc(
            wav,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,  # counts raw audio samples
        )
        # Move to device
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        out = self.model(**inputs)  # WavLMForXVector -> has .embeddings (B, D)
        emb = out.embeddings.squeeze(0)  # (D,)
        if return_tensor:
            return emb
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)
