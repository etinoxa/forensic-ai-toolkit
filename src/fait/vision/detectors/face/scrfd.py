from __future__ import annotations
from typing import List
from fait.vision.detectors.base import Recognizer, Detection
from fait.vision.services.face_service import get_face_service, FaceService
from fait.core.utils import is_image_file

class SCRFDRecognizer(Recognizer):
    """
    Face localization via the same shared FaceService used by ArcFace.
    """
    task = "face"

    def __init__(self, face_service: FaceService | None = None):
        self.fs = face_service or get_face_service()

    def name(self) -> str:
        return f"SCRFD({self.fs.device})"

    def detect(self, image_path: str) -> List[Detection]:
        if not is_image_file(image_path): return []
        return self.fs.detect(image_path)
