# Ollama Hallucination Experiments

Real-world validation of hallucination theory using local LLMs via Ollama.

This package implements the theoretical framework from **"Why Do Language Models Hallucinate?"** (Kalai, Nachum, Vempala, Zhang, 2025) and tests it against actual language models running locally.

## Quick Start

### Prerequisites

1. **Install Ollama**: https://ollama.ai/
2. **Pull a model**: `ollama pull llama2` (or another model)
3. **Start Ollama**: `ollama serve`

### Setup with uv

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Activate it
source .venv/bin/activate  # on macOS/Linux
# or
.venv\Scripts\activate  # on Windows

# Install the package and dependencies
uv pip install -e .
```

### Run Experiments

```bash
# Basic experiment with default settings
python scripts/run_experiments.py

# Custom model and sample size
python scripts/run_experiments.py --model mistral --samples 30

# With different scoring scheme
python scripts/run_experiments.py --scoring penalized --penalty 2.0

# See all options
python scripts/run_experiments.py --help
```

## Project Structure

```
hallucination_experiments/
├── pyproject.toml              # Project configuration with uv
├── src/hallucination_experiments/
│   ├── __init__.py
│   ├── data.py                 # Synthetic birthday dataset
│   ├── ollama_client.py        # Ollama integration
│   ├── evaluation.py           # Scoring and metrics
│   └── experiments.py          # Core experiment framework
├── scripts/
│   └── run_experiments.py      # Main entry point
└── README.md
```

## Core Concepts

### The Theory

The paper makes two key claims:

1. **Pretraining causes hallucination**: Generative error rate ≥ 2 × IIV classification error
2. **Post-training preserves it**: Standard metrics incentivize guessing over uncertainty

### The Birthday Model

A synthetic dataset where:
- Each of N people has a random birthday (no learnable pattern)
- Training corpus mentions people with exponential frequency (Zipfian)
- **Singletons** (mentioned once) cannot be distinguished from wrong dates → hallucinated
- Tests whether models hallucinate on facts seen only once

### IIV Classification

Binary classification task: **Is-It-Valid**
- 50% valid (correct facts)
- 50% errors (plausible falsehoods)
- Measures how hard the "is this true?" problem is
- Theory: Higher IIV error → higher generative error

## Usage Examples

### Basic Python API

```python
from hallucination_experiments import BirthdayDataset, OllamaClient
from hallucination_experiments.data import DatasetConfig

# Create synthetic dataset
config = DatasetConfig(n_people=200, n_docs=400)
dataset = BirthdayDataset(config)
dataset.print_stats()

# Initialize Ollama client
client = OllamaClient(model="llama2")

# Test connection
if client.test_connection():
    print("✓ Ollama is running")

# Generate a birthday
response = client.generate_birthday("John Doe", "mentioned in training")
print(f"Model generated: {response}")

# Classify if a date is valid
classification, confidence = client.classify_birthday("John Doe", "March 15")
print(f"Classification: {classification}")
```

### Run Experiment

```python
from hallucination_experiments.experiments import HallucinationExperiment, ExperimentConfig
from hallucination_experiments.evaluation import ScoringScheme

config = ExperimentConfig(
    model="llama2",
    n_test_samples=50,
    scoring_scheme=ScoringScheme.BINARY
)

exp = HallucinationExperiment(config)
exp.run_generation_experiment()
exp.run_iiv_classification_experiment()
```

## Command-Line Options

```
--model MODEL           Ollama model to use (default: llama2)
--host HOST            Ollama server address (default: http://localhost:11434)
--samples N            Number of test samples per experiment (default: 20)
--n-people N           People in dataset (default: 200)
--n-docs N             Docs in training corpus (default: 400)
--scoring SCHEME       binary or penalized (default: binary)
--penalty FLOAT        Wrong answer penalty (default: 1.0)
```

## Supported Models

Any model available in Ollama. Popular choices:

- `llama2` - 7B, good for local testing
- `mistral` - 7B, faster inference
- `neural-chat` - 7B, optimized for conversation
- `llama2:13b` - 13B variant (requires more VRAM)
- `openhermes` - 7B, instruction-tuned

Pull models with: `ollama pull <model-name>`

## Evaluation Metrics

### Binary Grading (Standard Benchmark)
- Correct: +1 point
- Wrong: 0 points
- IDK: 0 points
- **Problem**: Incentivizes guessing (better than IDK)

### Penalized Grading
- Correct: +1 point
- Wrong: -penalty points
- IDK: 0 points
- **Better**: Balances correct vs wrong trade-off

## Requirements

- Python 3.10+
- Ollama (running locally)
- Dependencies (auto-installed):
  - `ollama>=0.0.10` - Ollama client
  - `matplotlib>=3.7.0` - Visualization
  - `numpy>=1.24.0` - Numerical computing
  - `pydantic>=2.0.0` - Data validation

## Troubleshooting

### "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check the host is correct (default: `http://localhost:11434`)
- Verify model is pulled: `ollama list`

### Model not found
- Pull the model: `ollama pull llama2`
- Check spelling matches Ollama's model name

### Out of memory errors
- Try a smaller model: `ollama pull mistral`
- Reduce `--samples` and `--n-docs`

### Slow responses
- Larger models are slower. Try 7B models for faster feedback.
- Check your system has enough VRAM

## Contributing

To add new experiments:

1. Add experiment method to `HallucinationExperiment`
2. Update `ExperimentRunner.run_all()`
3. Add command-line args to `run_experiments.py`

## References

Kalai, A., Nachum, O., Vempala, S., & Zhang, Y. (2025). 
"Why Do Language Models Hallucinate?" 
*arXiv preprint arXiv:2509.04664*

## License

MIT
