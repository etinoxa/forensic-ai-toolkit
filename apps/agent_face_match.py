# --- load .env from repo root & silence TF/Protobuf BEFORE any heavy imports ---
import os, warnings, pathlib, sys, re, ast, json, argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]   # repo root (parent of apps/)
# Make sure src/ is importable
sys.path.insert(0, str(ROOT / "src"))

# Load .env from the repo root explicitly (not CWD)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# Ensure the keys exist even if .env is missing; do NOT wait for later imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 0=all; 1=INFO off; 2=+WARN off; 3=+ERROR off (keep only FATAL)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # hide oneDNN banner

# Silence protobuf version warnings
warnings.filterwarnings(
    "ignore",
    message=r"Protobuf gencode version .* is exactly one major version older",
    category=UserWarning,
    module="google\.protobuf\.runtime_version",
)


from fait.core.logging_config import setup_logging
setup_logging()

# Ensure embedders register
import fait.vision.embeddings.arcface  # noqa: F401
import fait.vision.embeddings.clip     # noqa: F401

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent  # ReAct agent works with any LLM
from fait.vision.integrations.langchain_tools import ALL_RECOGNITION_TOOLS
from fait.llm.local_hf import build_local_hf_chat
from fait.vision.integrations.langchain_tools import face_match_tool

# Hide protobuf runtime_version warnings
warnings.filterwarnings(
    "ignore",
    message=r"Protobuf gencode version .* is exactly one major version older",
    category=UserWarning,
    module="google.protobuf.runtime_version",
)


def build_agent(model_id: str | None = None, use_4bit: bool | None = None) -> AgentExecutor:
    llm = build_local_hf_chat(model_id=model_id, use_4bit=use_4bit)

    SYSTEM = """You are a helpful assistant for a forensic vision toolkit.
    - When asked to 'match faces', call the vision_face_match tool with the correct parameters.
    - If metric is not specified: use euclidean for arcface, cosine for clip.
    - Threshold hints:
        - arcface (euclidean): smaller is stricter (e.g., 0.75, 0.80, 0.90)
        - clip (cosine distance = 1 - similarity): smaller is stricter (e.g., 0.40, 0.35, 0.30)
    - Only run the tool when both reference_dir and gallery_dir are provided.
    - IMPORTANT: Use the ReAct format EXACTLY (no code blocks).
    """

    FEWSHOT = r"""Example:
    User request:
    match faces with arcface; reference_dir=/ref; gallery_dir=/gal; thresholds=[0.8,0.9]; plot_results=true

    Thought: I should run the face match tool with those parameters.
    Action: vision_face_match
    Action Input: {{"model":"arcface","reference_dir":"/ref","gallery_dir":"/gal","thresholds":[0.8,0.9],"metric":"auto","plot_results":true}}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nAvailable tools:\n{tools}\nUse tool names exactly from: {tool_names}\n\n" + FEWSHOT),
        MessagesPlaceholder("chat_history"),
        # Last message must be Human; include string scratchpad
        ("human", "User request:\n{input}\n\nScratchpad:\n{agent_scratchpad}")
    ])

    agent = create_react_agent(llm, tools=ALL_RECOGNITION_TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_RECOGNITION_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True,
    )

# accept: with/use/using/model= ; and various key synonyms
_QUICK = re.compile(
    r"(?ix)"                                     # i: case-insens, x: verbose
    r"(?:match\s+faces|face\s+match).*?"         # intent
    r"(?:(?:with|use|using|model\s*[:=]\s*)(arcface|clip))?.*?"  # model (optional)
    r"(?:reference_dir|reference|refs|ref)\s*=\s*([^\s;]+).*?"   # reference path
    r"(?:gallery_dir|gallery|gal|g)\s*=\s*([^\s;]+).*?"          # gallery path
    r"thresholds\s*=\s*(\[[^\]]+\])"                             # thresholds list
)

def _quick_parse(text: str):
    m = _QUICK.search(text)
    if not m:
        return None
    # model: if not captured, infer 'clip' if the word appears, else arcface
    if m.group(1):
        model = m.group(1).lower()
    else:
        model = "clip" if re.search(r"\bclip\b", text, re.I) else "arcface"

    ref = m.group(2).strip().strip("'\"")
    gal = m.group(3).strip().strip("'\"")
    try:
        th = ast.literal_eval(m.group(4))
        if not isinstance(th, list):
            return None
    except Exception:
        return None

    return {
        "model": model,
        "reference_dir": ref,
        "gallery_dir": gal,
        "thresholds": th,
        "metric": "auto",
        "plot_results": True,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.getenv("FAIT_LLM_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--four-bit", action="store_true")
    args = parser.parse_args()  # <-- assign BEFORE using

    executor = build_agent(model_id=args.model_id, use_4bit=args.four_bit)  # <-- now safe

    print("Local FAIT agent ready (HF). Examples:\n"
          "  match faces with arcface; reference_dir=...; gallery_dir=...; thresholds=[0.8,0.9]\n"
          "  use clip; refs=...; gallery=...; thresholds=[0.40,0.35]; plot_results=true\n")

    print("Agent ready. Type 'quit' to exit.")
    while True:
        try:
            q = input("You> ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break

            parsed = _quick_parse(q)
            if parsed:
                s = json.dumps(parsed)
                print("\n[direct] vision_face_match", s)
                print(face_match_tool.func(s))
                continue

            result = executor.invoke({"input": q, "chat_history": []})
            print("\nAssistant>", result.get("output", result), "\n")

            # (optional) see the tool steps
            # for i, step in enumerate(result.get("intermediate_steps", []), 1):
            #     action, observation = step
            #     print(f"Step {i}: {action.tool} -> {action.tool_input}\nObs: {observation}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        except Exception as e:
            print(f"[error] {e}")

if __name__ == "__main__":
    main()
