# src/fait/audio/services/speaker_service.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from fait.core.paths import get_paths, ensure_on_first_write
from ..embeddings.speechbrain_embedder import SpeechBrainEmbedder
from ..embeddings.resemblyzer_embedder import ResemblyzerEmbedder

@dataclass
class SpeakerServiceConfig:
    # future knobs (e.g., default model)
    pass

class SpeakerModelsService:
    """Small pooled service to reuse embedders in a process."""
    def __init__(self, cfg: SpeakerServiceConfig = SpeakerServiceConfig()):
        self.cfg = cfg
        self.paths = get_paths()
        ensure_on_first_write(self.paths.models_cache / "audio")
        self._pool: Dict[Tuple[str], object] = {}

    def get_speechbrain(self) -> SpeechBrainEmbedder:
        key = ("speechbrain",)
        if key not in self._pool:
            self._pool[key] = SpeechBrainEmbedder()
        return self._pool[key]  # type: ignore[return-value]

    def get_resemblyzer(self) -> ResemblyzerEmbedder:
        key = ("resemblyzer",)
        if key not in self._pool:
            self._pool[key] = ResemblyzerEmbedder()
        return self._pool[key]  # type: ignore[return-value]

# convenience
_service_singleton: Optional[SpeakerModelsService] = None
def get_speaker_service() -> SpeakerModelsService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = SpeakerModelsService()
    return _service_singleton
