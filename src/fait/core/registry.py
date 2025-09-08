from __future__ import annotations
from typing import Callable, Dict, Type

# Embedding plugins
_EMBEDDERS: Dict[str, Callable[[], object]] = {}

def register_embedder(name: str):
    def deco(cls: Type):
        _EMBEDDERS[name] = cls
        return cls
    return deco

def get_embedder(name: str):
    if name not in _EMBEDDERS:
        raise KeyError(f"Unknown embedder '{name}'. Available: {list(_EMBEDDERS)}")
    return _EMBEDDERS[name]()


# ===== Audio embedders registry (parallel to vision) =====
_AUDIO_EMBEDDERS = {}

def register_audio_embedder(name: str):
    def deco(cls):
        _AUDIO_EMBEDDERS[name.lower()] = cls
        return cls
    return deco

def get_audio_embedder(name: str, *args, **kwargs):
    key = name.lower()
    if key not in _AUDIO_EMBEDDERS:
        raise KeyError(f"Unknown audio embedder '{name}'. Available: {list(_AUDIO_EMBEDDERS)}")
    return _AUDIO_EMBEDDERS[key](*args, **kwargs)