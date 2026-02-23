#!/usr/bin/env python3
"""
Main entry point for hallucination experiments.

Usage:
    python -m scripts.run_experiments [--model MODEL] [--samples N]
    
Example:
    python -m scripts.run_experiments --model llama2 --samples 20
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hallucination_experiments import experiments, data
from hallucination_experiments.evaluation import ScoringScheme


def main():
    parser = argparse.ArgumentParser(
        description="Run hallucination validation experiments with Ollama"
    )
    
    parser.add_argument(
        "--model",
        default="llama2",
        help="Ollama model to use (default: llama2)"
    )
    
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server address (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of test samples per experiment (default: 20)"
    )
    
    parser.add_argument(
        "--n-people",
        type=int,
        default=200,
        help="Number of people in synthetic dataset (default: 200)"
    )
    
    parser.add_argument(
        "--n-docs",
        type=int,
        default=400,
        help="Number of documents in training corpus (default: 400)"
    )
    
    parser.add_argument(
        "--scoring",
        choices=["binary", "penalized"],
        default="binary",
        help="Scoring scheme (default: binary)"
    )
    
    parser.add_argument(
        "--penalty",
        type=float,
        default=1.0,
        help="Penalty for wrong answers in penalized scheme (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Create configs
    dataset_config = data.DatasetConfig(
        n_people=args.n_people,
        n_docs=args.n_docs,
    )
    
    scoring_scheme = (
        ScoringScheme.PENALIZED 
        if args.scoring == "penalized" 
        else ScoringScheme.BINARY
    )
    
    exp_config = experiments.ExperimentConfig(
        model=args.model,
        ollama_host=args.host,
        dataset_config=dataset_config,
        n_test_samples=args.samples,
        scoring_scheme=scoring_scheme,
        penalty=args.penalty,
    )
    
    # Run experiments
    runner = experiments.ExperimentRunner(exp_config)
    runner.run_all()


if __name__ == "__main__":
    main()
