# src/fait/audio/models/resemblyzer_embedder.py
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder
from fait.core.registry import register_audio_embedder
from fait.audio.speaker_recognition.base import _BaseAudioEmbedder

try:
    import imageio_ffmpeg as ffmpegio  # downloads/locates an ffmpeg binary cross-platform
except Exception:
    ffmpegio = None


@register_audio_embedder("resemblyzer")
class ResemblyzerEmbedder(_BaseAudioEmbedder):
    """
    Speaker models using Resemblyzer.
    This version adds an FFmpeg fallback so formats like .m4a/.mp3 work on Windows.
    """
    def __init__(
        self,
        embed_cache_dir: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        super().__init__(embed_cache_dir=embed_cache_dir)
        self.sample_rate = sample_rate
        self.encoder = VoiceEncoder()  # uses torch under the hood

        # just to ensure cache roots exist
        paths = get_paths()
        ensure_folder(paths.models_speaker_recognition / "resemblyzer")

    def name(self) -> str:
        return "Resemblyzer"

    # ---------- public API ----------
    def embed_file(self, audio_path: str, use_cache: bool = True) -> np.ndarray:
        """
        Returns a 256-D float32 embedding.
        Transparent FFmpeg conversion if librosa can't read (e.g., .m4a on Windows).
        """
        p = Path(audio_path)
        tmp_wav: Optional[Path] = None

        # Try the normal path first
        try:
            wav = preprocess_wav(str(p))
        except Exception:
            # If that failed, convert to 16k mono wav via FFmpeg, then retry
            tmp_wav = self._ffmpeg_to_wav(p, sr=self.sample_rate)
            wav = preprocess_wav(str(tmp_wav))
        finally:
            # cleanup temp
            if tmp_wav and tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except Exception:
                    pass

        emb = self.encoder.embed_utterance(wav).astype(np.float32)
        return emb

    # ---------- helpers ----------
    def _ffmpeg_to_wav(self, src: Path, sr: int = 16000) -> Path:
        """
        Convert arbitrary audio (m4a/mp3/…) -> temporary 16kHz mono WAV using imageio-ffmpeg.
        """
        if ffmpegio is None:
            raise RuntimeError(
                "imageio-ffmpeg is not installed. Please `pip install imageio-ffmpeg` "
                "to enable .m4a/.mp3 conversion on Windows."
            )
        tmpdir = Path(tempfile.mkdtemp(prefix="ffconv_"))
        dst = tmpdir / f"{src.stem}_16k_mono.wav"

        cmd = [
            ffmpegio.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "error",
            "-i", str(src),
            "-ar", str(sr),
            "-ac", "1",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        return dst
