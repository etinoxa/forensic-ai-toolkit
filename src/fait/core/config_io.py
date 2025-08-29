# src/fait/core/config_io.py
from __future__ import annotations
from dataclasses import is_dataclass, fields
from typing import Any, Mapping
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merge_into_dataclass(dc_obj, cfg: Mapping[str, Any]):
    """Recursively merge dict into a dataclass instance (in-place)."""
    if not is_dataclass(dc_obj) or not isinstance(cfg, Mapping):
        return dc_obj
    name2field = {f.name: f for f in fields(dc_obj)}
    for k, v in cfg.items():
        if k not in name2field:
            continue
        cur = getattr(dc_obj, k)
        if is_dataclass(cur) and isinstance(v, Mapping):
            merge_into_dataclass(cur, v)
        else:
            setattr(dc_obj, k, v)
    return dc_obj
