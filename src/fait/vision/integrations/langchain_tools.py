from __future__ import annotations
import os, json, re, ast
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import Tool
from langchain_core.tools import StructuredTool

from fait.core.registry import get_embedder
from fait.vision.pipelines.face_match import run_face_match

# ----- Tool schemas -----

class FaceMatchInput(BaseModel):
    """Run face matching between a reference dir and a gallery dir."""
    model: Literal["arcface", "clip"] = Field(..., description="Which embedder to use.")
    reference_dir: str = Field(..., description="Folder of reference images.")
    gallery_dir: str = Field(..., description="Folder of gallery images to search.")
    output_dir: Optional[str] = Field(None, description="Optional. If omitted, defaults to .fait/outputs.")
    thresholds: List[float] = Field(..., description="Distance thresholds for matches.")
    metric: Literal["auto", "euclidean", "cosine"] = Field(
        "auto",
        description="Distance metric. 'auto' => euclidean for arcface, cosine for clip."
    )
    use_cache: bool = True
    plot_results: bool = False
    topk_preview: int = 10

# ----- tolerant parser for ReAct string inputs -----
_CLEAN_CODE_FENCE = re.compile(r"^```(?:json|python)?\s*(.*?)\s*```$", re.DOTALL)

def _parse_react_input(s: str) -> Dict[str, Any]:
    """
    Accepts:
      - JSON object string (preferred)
      - fenced code blocks with JSON
      - simple 'key=value; key2=value2' (lists via [..], booleans true/false)
    Returns a dict ready for validation with FaceMatchInput.
    """
    s = s.strip()
    # strip code fences if present
    m = _CLEAN_CODE_FENCE.match(s)
    if m:
        s = m.group(1).strip()

    # pure JSON object?
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass  # fallthrough

    # key=value; key2=value2; ... (very tolerant)
    kv = {}
    if "=" in s and "{" not in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            # lists / numbers / booleans via literal_eval when it looks like a literal
            if v.startswith("[") or v.startswith("{") or v.lower() in {"true", "false"} or re.fullmatch(r"-?\d+(\.\d+)?", v):
                try:
                    kv[k] = ast.literal_eval(v.replace("true", "True").replace("false", "False"))
                    continue
                except Exception:
                    pass
            # strip surrounding quotes if present
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            kv[k] = v
        return kv

    # last resort: try to pull the biggest {...} block and parse JSON
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass

    raise ValueError(f"Could not parse Action Input. Got: {s}")

def _run_face_match_validated(args: FaceMatchInput) -> Dict[str, Any]:
    # path validation for friendlier errors
    if not os.path.isdir(args.reference_dir):
        raise ValueError(f"reference_dir not found or not a directory: {args.reference_dir}")
    if not os.path.isdir(args.gallery_dir):
        raise ValueError(f"gallery_dir not found or not a directory: {args.gallery_dir}")

    metric = args.metric
    if metric == "auto":
        metric = "cosine" if args.model == "clip" else "euclidean"

    emb = get_embedder(args.model)
    result = run_face_match(
        embedder=emb,
        reference_dir=args.reference_dir,
        gallery_dir=args.gallery_dir,
        output_dir=args.output_dir,         # None -> defaults to .fait/outputs
        thresholds=args.thresholds,
        metric=metric,
        use_cache=args.use_cache,
        plot_results=args.plot_results,
        topk_preview=args.topk_preview,
    )
    return {
        "model": result["model"],
        "metric": result["metric"],
        "processed": result["processed"],
        "matches_per_threshold": result["matches_per_threshold"],
        "closest": result["closest"],
        "report_path": result["report_path"],
        "plot_path": result["plot_path"],
        "output_dir": result["output_dir"],
    }

def _face_match_tool_fn(tool_input: str) -> str:
    """
    ReAct tool entrypoint. Accepts a *string* Action Input, parses/validates,
    then runs the pipeline and returns a concise JSON string.
    """
    try:
        data = _parse_react_input(tool_input)
        args = FaceMatchInput(**data)
        res = _run_face_match_validated(args)
        return json.dumps({
            "output_dir": res.get("output_dir", args.output_dir or ""),
            "report_path": res.get("report_path", ""),
            "plot_path": res.get("plot_path", ""),
            "processed": res.get("processed", 0),
            "matches": res.get("matches_per_threshold", {}),
        })
    except ValidationError as ve:
        return f"ValidationError: {ve}"
    except Exception as e:
        return f"Error: {e}"

def _face_match(args: FaceMatchInput) -> Dict[str, Any]:
    # pick metric automatically if requested
    metric = args.metric
    if metric == "auto":
        metric = "cosine" if args.model == "clip" else "euclidean"

    emb = get_embedder(args.model)
    result = run_face_match(
        embedder=emb,
        reference_dir=args.reference_dir,
        gallery_dir=args.gallery_dir,
        output_dir=args.output_dir,         # may be None -> pipeline resolves .fait/outputs
        thresholds=args.thresholds,
        metric=metric,
        use_cache=args.use_cache,
        plot_results=args.plot_results,
        topk_preview=args.topk_preview,
    )
    return {
        "model": result["model"],
        "metric": result["metric"],
        "processed": result["processed"],
        "matches_per_threshold": result["matches_per_threshold"],
        "closest": result["closest"],
        "report_path": result["report_path"],
        "plot_path": result["plot_path"],
        "output_dir": result["output_dir"],
    }

face_match_tool = Tool(
    name="vision_face_match",
    description=(
        "Run face matching between a reference folder and a gallery folder.\n"
        "Action Input must be a JSON object string with keys:\n"
        "  model: 'arcface' | 'clip'\n"
        "  reference_dir: path\n"
        "  gallery_dir: path\n"
        "  thresholds: list of floats\n"
        "Optional: output_dir, metric ('auto'|'euclidean'|'cosine'), use_cache, plot_results, topk_preview.\n"
        "Example Action Input: "
        "{\"model\":\"arcface\",\"reference_dir\":\"/ref\",\"gallery_dir\":\"/gal\",\"thresholds\":[0.8,0.9],\"metric\":\"auto\",\"plot_results\":true}"
    ),
    func=_face_match_tool_fn,
)

ALL_RECOGNITION_TOOLS = [face_match_tool]

