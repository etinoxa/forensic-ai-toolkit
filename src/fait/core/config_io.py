# src/fait/core/config_io.py
from __future__ import annotations
from dataclasses import is_dataclass, fields
from typing import Any, Mapping, Dict
import os, yaml


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

def csv_list(s: str) -> list[str]:
    return [t.strip() for t in str(s).split(",") if str(t).strip()]

def optional_env(name: str, default=None):
    v = os.getenv(name)
    if v is None: return default
    v = v.strip()
    return v if v else default

def int_env(name: str, default: int | None = None):
    v = os.getenv(name, "").strip()
    try: return int(v) if v != "" else default
    except: return default

def float_env(name: str, default: float | None = None):
    v = os.getenv(name, "").strip()
    try: return float(v) if v != "" else default
    except: return default

def bool_env(name: str, default: bool | None = None):
    v = os.getenv(name)
    if v is None: return default
    return v.strip().lower() in {"1","true","yes","y","on"}

def rotations_env(name: str, default: tuple[int, ...]):
    v = os.getenv(name, "").strip()
    if not v: return default
    try: return tuple(int(x) for x in csv_list(v))
    except: return default

