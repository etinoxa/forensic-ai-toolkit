from __future__ import annotations
import os, logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# Optional 4-bit (GPU only). Safe to import-gate for Windows/CPU.
try:
    from transformers import BitsAndBytesConfig  # optional
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from langchain_huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

log = logging.getLogger("fait.llm.local_hf")

# Small, CPU/GPU-friendly defaults (you can change anytime)
_DEFAULT_LLM_ID = os.getenv("FAIT_LLM_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# If installed, we use these and avoid the deprecation warning.
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    _HAS_LC_HF = True
except Exception:
    # ---- FALLBACK: old community class (will emit deprecation warning) ----
    _HAS_LC_HF = False
    ChatHuggingFace = None  # type: ignore
    from langchain_community.llms.huggingface_pipeline import (
        HuggingFacePipeline as CommunityHFPipeline,
    )
    HuggingFacePipeline = CommunityHFPipeline  # type: ignore

from fait.core.paths import get_paths
from fait.core.utils import ensure_folder

log = logging.getLogger("fait.llm.local_hf")

_DEFAULT_LLM_ID = os.getenv("FAIT_LLM_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # bfloat16 is a good default on modern GPUs; fallback to float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def build_local_hf_chat(
    model_id: str = _DEFAULT_LLM_ID,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False,
    use_4bit: Optional[bool] = None,
) -> ChatHuggingFace:
    """
    Create a LangChain Chat model backed by a local HF pipeline.
    Weights are cached under .fait/models/llm (no Ollama required).
    """
    # Resolve local LLM cache dir: .fait/models/llm
    paths = get_paths()
    llm_dir = Path(paths.home) / "cache" / "models" / "llm"
    ensure_folder(llm_dir)

    # Ensure HF also uses this directory
    os.environ.setdefault("TRANSFORMERS_CACHE", str(llm_dir))

    torch_dtype = _pick_dtype()
    device = 0 if torch.cuda.is_available() else -1

    # 4-bit quantization (optional, if bitsandbytes present)
    load_in_4bit = False
    quant_cfg = None
    if use_4bit is None:
        # Enable 4-bit on GPU + bitsandbytes present
        load_in_4bit = _HAS_BNB and torch.cuda.is_available()
    else:
        load_in_4bit = bool(use_4bit) and _HAS_BNB and torch.cuda.is_available()

    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)
        log.info("LLM: using 4-bit quantization via bitsandbytes", extra={"model": model_id})
    else:
        log.info("LLM: using full precision/half precision", extra={"model": model_id, "dtype": str(torch_dtype)})

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(llm_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(llm_dir),
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_cfg,
        trust_remote_code=False,
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Wrap HF pipeline into a Chat model LangChain understands
    llm = HuggingFacePipeline(pipeline=gen)  # new class if installed; alias to community fallback otherwise
    if ChatHuggingFace is not None:
        # Use chat wrapper when available (nicer with agents)
        return ChatHuggingFace(llm=llm)
    return llm
