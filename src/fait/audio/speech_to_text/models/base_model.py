"""
Base interface for Automatic Speech Recognition (ASR) models.
"""

from abc import ABC, abstractmethod


class SpeechToTextModel(ABC):
    """Abstract base class for all ASR models."""

    @abstractmethod
    def transcribe(self, file_path: str) -> str:
        """Convert speech from an audio file to text."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the backend/model name."""
        raise NotImplementedError