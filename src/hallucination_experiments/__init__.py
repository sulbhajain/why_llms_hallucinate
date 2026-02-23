"""
Hallucination Experiments: Testing LLM Hallucination Theory with Real Models via Ollama
"""

__version__ = "0.1.0"
__author__ = "Sulbha Jain"

from .data import BirthdayDataset, DatasetConfig
from .ollama_client import OllamaClient
from .evaluation import ScoringScheme, evaluate_response

__all__ = [
    "BirthdayDataset",
    "DatasetConfig",
    "OllamaClient",
    "ScoringScheme",
    "evaluate_response",
]
