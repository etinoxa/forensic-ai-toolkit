## 📂 Project Structure

```
Forensic_ai_toolkit/
├─ apps/
│  └─ agent_face_match.py            # NL agent (LangChain + local HF) to run face matching
│  └─ face_match_cl.py              # CLI to run face matching
│
├─ datasets/                         # (your local data; ignored by git as needed)
│  ├─ images/
│  │  ├─ face/
│  │  │  ├─ gallery/
│  │  │  ├─ matched_images/          # generated results live under .fait/outputs by default
│  │  │  └─ reference_images/
│  │  └─ objects/
│  │     └─ text_ocr/
│  ├─ audio/
│  ├─ text/
│  │  ├─ csv/ pdf/ txt/
│  └─ videos/
│
├─ examples/
│  └─ face_match_quickstart.py       # code-only example (no LLM)
│
├─ src/
│  └─ fait/
│     ├─ core/                       # reusable infrastructure
│     │  ├─ interface.py             # serves as a contract definition
│     │  ├─ logging_config.py        # structured logging + env config
│     │  ├─ paths.py                 # central path resolver (anchors .fait at repo root)
│     │  ├─ registry.py              # component registry (embedders/tools…)
│     │  └─ utils.py                 # shared helpers (I/O, plotting, hashing, etc.)
│     │
│     ├─ llm/
│     │  └─ local_hf.py              # local HuggingFace chat runtime (TinyLlama/Qwen/Phi-3)
│     │
│     └─ vision/
│        ├─ embeddings/              # feature extractors (recognition backends)
│        │  ├─ arcface.py
│        │  └─ clip.py
│        │
│        ├─ services/
│        │  └─ face_service.py       # InsightFace init/wrappers (detection+embedding)
│        │
│        ├─ pipelines/
│        │  └─ face_match.py         # end-to-end face matching pipeline
│        │
│        └─ integrations/
│           └─ langchain_tools.py    # LangChain tool(s): vision_face_match (ReAct-friendly)
│
├─ .fait/                            # CREATED AT RUNTIME (not committed)
│  ├─ cache/
│  │  ├─ models/                     # e.g., InsightFace, CLIP, HF weights (LLM -> .fait/models/llm)
│  │  └─ embeddings/                 # npy/pkl feature caches (model-tagged keys)
│  ├─ logs/                          # structured logs (json + pretty console)
│  ├─ outputs/                       # pipeline artifacts (reports, plots, matched images)
│  └─ tmp/
│
├─ .env.example                      # safe defaults; copy to .env locally
├─ requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── DATASETS.md
├── DCO.md
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt

```