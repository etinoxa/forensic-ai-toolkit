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
