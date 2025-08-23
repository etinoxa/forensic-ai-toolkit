from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from fait.core.registry import get_embedder
from fait.vision.pipelines.face_match import run_face_match

# Face matching (ArcFace/CLIP)
class FaceMatchInput(BaseModel):
    model: Literal["arcface", "clip"]
    reference_dir: str
    gallery_dir: str
    output_dir: str
    thresholds: List[float]
    metric: Literal["euclidean", "cosine"] = "euclidean"
    use_cache: bool = True
    plot_results: bool = False
    topk_preview: int = 10

def _face_match(args: FaceMatchInput) -> Dict[str, Any]:
    emb = get_embedder(args.model)
    return run_face_match(emb, args.reference_dir, args.gallery_dir, args.output_dir,
                          args.thresholds, args.metric, args.use_cache, args.plot_results, args.topk_preview)

face_match_tool = StructuredTool.from_function(
    name="vision_face_match",
    description="Run face matching with ArcFace or CLIP. Copies matches into per-threshold folders and writes a report.",
    func=_face_match,
    args_schema=FaceMatchInput,
)

# Single image embed
class EmbedOneInput(BaseModel):
    model: Literal["arcface", "clip"]
    image_path: str
    use_cache: bool = True
    out_path: Optional[str] = None

def _embed_one(args: EmbedOneInput) -> str:
    import numpy as np, os
    from fait.core.utils import ensure_folder
    emb = get_embedder(args.model).embed_image(args.image_path, use_cache=args.use_cache)
    if emb is None: return "No embedding (invalid image or no face for ArcFace)."
    if args.out_path:
        ensure_folder(os.path.dirname(args.out_path)); np.save(args.out_path, emb); return f"Saved -> {args.out_path}"
    return f"Embedding OK (shape={emb.shape})"

embed_one_tool = StructuredTool.from_function(
    name="vision_embed_one",
    description="Embed a single image with ArcFace or CLIP.",
    func=_embed_one,
    args_schema=EmbedOneInput,
)

ALL_RECOGNITION_TOOLS = [face_match_tool, embed_one_tool]
