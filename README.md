# Why Language Models Hallucinate

A toy implementation exploring the theoretical foundations of hallucination in large language models, based on the paper by Kalai, Nachum, Vempala, and Zhang (arXiv:2509.04664, 2025).

## ⚡ Quick Start

**New:** Test with real LLMs using [Ollama](https://ollama.ai/)!

```bash
# Install uv and Python dependencies
uv venv && source .venv/bin/activate && uv pip install -e .

# Start Ollama server (in another terminal)
ollama serve

# Pull a model (in another terminal)
ollama pull llama2

# Run experiments
python scripts/run_experiments.py

# Or explore interactively
jupyter notebook ollama_experiments.ipynb
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Two Implementations

1. **Toy Notebook** (`why_llms_hallucinate.ipynb`)
   - Pure Python simulation of the paper's findings
   - No external models needed
   - Demonstrates core theory with synthetic data

2. **Ollama Implementation** (new!)
   - Tests theory against real language models
   - Uses `uv` + `pyproject.toml` for dependency management
   - Modular Python package in `src/hallucination_experiments/`
   - Interactive notebook: `ollama_experiments.ipynb`
   - CLI experiments: `scripts/run_experiments.py`

See [OLLAMA_README.md](OLLAMA_README.md) for full Ollama documentation.

## Overview

This notebook investigates two fundamental claims about why language models hallucinate:

1. **Pretraining causes hallucination** through a reduction to binary classification.
   - Generative error rate ≥ 2 × IIV misclassification rate
   - "Singleton facts" (seen only once in training) are almost certainly hallucinated

2. **Post-training preserves hallucination** due to standard evaluation benchmarks that penalize uncertainty.
   - Benchmarks reward guessing over saying "I don't know" (IDK)
   - The rational strategy becomes always to make a prediction

## Key Concepts

### Error Partitioning
All plausible text is partitioned into:
- **V (Valid outputs)**: Correct facts
- **E (Errors)**: Plausible falsehoods (e.g., "Einstein was born March 5")

### The Is-It-Valid (IIV) Problem
A binary classification problem trained on:
- 50/50 mix of valid examples (+) and random errors (−)
- Used to determine if a language model's output is likely correct

### The Birthday Model
A synthetic dataset used to demonstrate hallucination:
- N people, each with a random birthday in {0..364}
- No learnable pattern exists → hard IIV problem
- Serves as a proxy for arbitrary facts without systematic patterns

## Contents

- `why_llms_hallucinate.ipynb` - Main analysis notebook with:
  - Synthetic dataset generation
  - IIV binary classification simulation
  - Generative error rate analysis
  - Post-training behavior under different evaluation schemes
  - Visualizations and empirical validation

- `outputs/` - Generated results and figures from experiments

## Requirements

- Python 3.x
- Standard library only: `random`, `math`, `collections`
- `matplotlib` for visualization

## Running the Notebook

```bash
jupyter notebook why_llms_hallucinate.ipynb
```

The notebook is fully self-contained and runs on CPU using only standard Python libraries and matplotlib.

## Key Findings

The simulations demonstrate:
- Hard binary classification problems (high IIV error) necessarily lead to high generative error
- Singleton facts are particularly vulnerable to hallucination
- Standard evaluation metrics that penalize "IDK" responses incentivize hallucination

## References

Kalai, A., Nachum, O., Vempala, S., & Zhang, Y. (2025). Why Do Language Models Hallucinate? *arXiv preprint arXiv:2509.04664*.
