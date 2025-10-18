import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # prevents MPS lockups on macOS

import argparse
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    # Load input JSON
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = data.get("text", "").strip()

    if not text:
        print("")
        return

    cache_dir = Path(args.cache-dir).expanduser().resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = "Helsinki-NLP/opus-mt-mul-en"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)

    translator = pipeline("translation", model=model, tokenizer=tokenizer, device=-1)
    result = translator(text, max_length=400)
    print(result[0]["translation_text"].strip())


if __name__ == "__main__":
    main()