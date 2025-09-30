# src/fait/audio/services/speaker_recognition_service.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from fait.core.paths import get_paths, ensure_on_first_write
from fait.audio.speaker_recognition.models.speechbrain_embedder import SpeechBrainEmbedder
from fait.audio.speaker_recognition.models.wavlm_embedder import WavLMEmbedder
from fait.audio.speaker_recognition.models.titanet_embedder import TitanetEmbedder

@dataclass
class SpeakerServiceConfig:
    # future knobs (e.g., default model)
    pass

class SpeakerModelsService:
    def __init__(self, cfg: SpeakerServiceConfig = SpeakerServiceConfig()):
        self.cfg = cfg
        self.paths = get_paths()
        ensure_on_first_write(self.paths.models_cache / "audio")
        self._pool: Dict[Tuple[str,str], object] = {}

    def _key(self, kind: str, model_id: str) -> Tuple[str,str]:
        return (kind, model_id)

    def get_speechbrain(self, model_id: str) -> SpeechBrainEmbedder:
        k = self._key("speechbrain", model_id)
        if k not in self._pool:
            self._pool[k] = SpeechBrainEmbedder(model_id=model_id)
        return self._pool[k]

    def get_titanet(self, model_id: str = "nvidia/speakerverification_en_titanet_large") -> TitanetEmbedder:
        k = ("titanet", model_id)
        if k not in self._pool:
            self._pool[k] = TitanetEmbedder(model_id=model_id)
        return self._pool[k]

    def get_wavlm(self, model_id: str = "microsoft/wavlm-base-plus-sv") -> WavLMEmbedder:
        k = ("wavlm", model_id)
        if k not in self._pool:
            self._pool[k] = WavLMEmbedder(model_id=model_id)
        return self._pool[k]

_service: Optional[SpeakerModelsService] = None
def get_speaker_service() -> SpeakerModelsService:
    global _service
    if _service is None:
        _service = SpeakerModelsService()
    return _service