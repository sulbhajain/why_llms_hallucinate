"""
Evaluation metrics for hallucination detection and scoring.

Implements the scoring schemes from the paper:
- Binary grading: 1 if correct, 0 otherwise
- Penalized grading: 1 if correct, -penalty if wrong, 0 if IDK
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class ScoringScheme(Enum):
    """Scoring schemes for evaluating responses."""
    BINARY = "binary"  # Standard benchmark: 1 if correct, 0 otherwise
    PENALIZED = "penalized"  # Penalized: 1 if correct, -penalty if wrong, 0 if IDK


@dataclass
class ResponseEvaluation:
    """Result of evaluating a single response."""
    score: float
    label: str
    is_hallucination: bool
    is_correct: bool
    is_idk: bool


IDK = -1  # Sentinel value for "I don't know"


def evaluate_response(
    response: int,
    correct_answer: int,
    scheme: ScoringScheme = ScoringScheme.BINARY,
    penalty: float = 1.0,
) -> ResponseEvaluation:
    """
    Evaluate a model's response to a factual question.
    
    Args:
        response: Model's response (date value or IDK sentinel)
        correct_answer: Ground truth answer (date value)
        scheme: Scoring scheme to use
        penalty: Penalty multiplier for wrong answers (in penalized scheme)
        
    Returns:
        ResponseEvaluation with score and labels
    """
    is_idk = response == IDK
    is_correct = response == correct_answer and not is_idk
    is_hallucination = response != correct_answer and not is_idk
    
    if is_idk:
        score = 0.0
        label = "idk"
    elif is_correct:
        score = 1.0
        label = "correct"
    else:
        if scheme == ScoringScheme.BINARY:
            score = 0.0
            label = "hallucination"
        else:  # PENALIZED
            score = -penalty
            label = "hallucination"
    
    return ResponseEvaluation(
        score=score,
        label=label,
        is_hallucination=is_hallucination,
        is_correct=is_correct,
        is_idk=is_idk,
    )


class MetricsCalculator:
    """Calculate aggregate metrics over multiple evaluations."""
    
    def __init__(self):
        self.evaluations: list[ResponseEvaluation] = []
        self.total_score: float = 0.0
        self.correct_count: int = 0
        self.hallucination_count: int = 0
        self.idk_count: int = 0
    
    def add_evaluation(self, evaluation: ResponseEvaluation) -> None:
        """Add an evaluation result."""
        self.evaluations.append(evaluation)
        self.total_score += evaluation.score
        
        if evaluation.is_correct:
            self.correct_count += 1
        if evaluation.is_hallucination:
            self.hallucination_count += 1
        if evaluation.is_idk:
            self.idk_count += 1
    
    def get_metrics(self) -> dict:
        """Get aggregate metrics."""
        n = len(self.evaluations)
        
        if n == 0:
            return {
                "avg_score": 0.0,
                "accuracy": 0.0,
                "hallucination_rate": 0.0,
                "idk_rate": 0.0,
                "total_evaluations": 0,
            }
        
        return {
            "avg_score": self.total_score / n,
            "accuracy": self.correct_count / n,
            "hallucination_rate": self.hallucination_count / n,
            "idk_rate": self.idk_count / n,
            "total_evaluations": n,
            "total_correct": self.correct_count,
            "total_hallucinations": self.hallucination_count,
            "total_idk": self.idk_count,
        }
    
    def print_metrics(self) -> None:
        """Print formatted metrics."""
        metrics = self.get_metrics()
        if metrics["total_evaluations"] == 0:
            print("No evaluations yet.")
            return
        
        print("Evaluation Metrics:")
        print(f"  Total evaluations: {metrics['total_evaluations']}")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Hallucination rate: {metrics['hallucination_rate']:.1%}")
        print(f"  IDK rate: {metrics['idk_rate']:.1%}")
        print(f"  Average score: {metrics['avg_score']:.3f}")
