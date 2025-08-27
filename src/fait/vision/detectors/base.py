from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Literal, Any

Task = Literal["face"]

@dataclass
class Detection:
    bbox: list          # [x1, y1, x2, y2]
    score: float
    label: str = "face" # fixed for face recognition
    kps: Optional[list] = None  # landmarks
    mask: Optional[Any] = None  # reserved for future (segmentation)

class Recognizer(Protocol):
    task: Task
    def name(self) -> str: ...
    def detect(self, image_path: str) -> List[Detection]: ...
