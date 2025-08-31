from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Literal, Any

Task = Literal["face", "object"]  # support both

@dataclass
class Detection:
    bbox: list[float]              # [x1, y1, x2, y2]
    score: float
    label: str                     # e.g., "face", "knife", "person"
    kps: Optional[list] = None     # optional (faces)
    mask: Optional[Any] = None     # reserved (segmentation)

class Recognizer(Protocol):
    task: Task
    def name(self) -> str: ...
    def detect(self, image_path: str) -> List[Detection]: ...