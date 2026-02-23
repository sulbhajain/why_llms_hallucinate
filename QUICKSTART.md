# Quick Start Guide

Get up and running with the Ollama hallucination experiments in 5 minutes.

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **macOS** (this guide is for macOS; Linux/Windows similar)

## Step 1: Install Ollama

```bash
# Download and install from https://ollama.ai/
# Or use brew (macOS)
brew install ollama
```

Verify installation:
```bash
ollama --version
```

## Step 2: Start Ollama Server

Open a terminal and start the Ollama server:

```bash
ollama serve
```

This starts Ollama on `http://localhost:11434`. Leave this terminal open.

## Step 3: Pull a Model

In a **new terminal**, pull a model:

```bash
ollama pull llama2
```

This takes a few minutes on first run (~4GB download for llama2).

## Step 4: Clone/Setup Project

```bash
cd ~/Documents/Personal/github_repo/why_llms_hallucinate
```

## Step 5: Install Python Dependencies with uv

### Option A: Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install the package
uv pip install -e .
```

### Option B: Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Step 6: Run Experiments

```bash
# Run with default settings (llama2, 20 samples)
python scripts/run_experiments.py

# Or use the interactive notebook
jupyter notebook ollama_experiments.ipynb
```

## Troubleshooting

### "Cannot connect to Ollama"
- Check that `ollama serve` is running in another terminal
- Verify with: `curl http://localhost:11434/api/tags`

### "model not found"
- Pull the model: `ollama pull llama2`
- Check available: `ollama list`

### Slow responses
- Try a smaller model: `ollama pull mistral`
- Or check your system resources (VRAM)

## Next Steps

- Try different models:
  ```bash
  ollama pull mistral       # Smaller, faster
  ollama pull neural-chat   # Good for conversation
  ```

- Run with custom settings:
  ```bash
  python scripts/run_experiments.py --model mistral --samples 50
  python scripts/run_experiments.py --scoring penalized --n-people 500
  ```

- See all options:
  ```bash
  python scripts/run_experiments.py --help
  ```

## Project Structure

```
why_llms_hallucinate/
├── pyproject.toml                    # Project config (uv)
├── src/hallucination_experiments/    # Main package
│   ├── data.py                       # Dataset generation
│   ├── ollama_client.py              # Ollama integration
│   ├── evaluation.py                 # Scoring metrics
│   └── experiments.py                # Experiment framework
├── scripts/
│   └── run_experiments.py            # CLI entry point
├── why_llms_hallucinate.ipynb        # Original toy notebook
├── ollama_experiments.ipynb          # Ollama integration demo
├── OLLAMA_README.md                  # Full Ollama docs
└── README.md                         # This project's README
```

## Key Files to Know

- **`pyproject.toml`**: Project configuration and dependencies
- **`scripts/run_experiments.py`**: Main experiment runner
- **`src/hallucination_experiments/`**: Core library code
- **`ollama_experiments.ipynb`**: Interactive demo notebook

## Understanding the Experiments

The code tests two key claims from the paper:

1. **Generative Error ≥ 2 × IIV Error**
   - IIV = "Is-It-Valid" binary classification
   - Tests if something is fact or fabrication
   - Hard classification → high hallucination

2. **Singletons Are Hallucinated**
   - Facts seen only once in training
   - Model can't learn pattern from one example
   - Almost certainly hallucinated

3. **Evaluation Metrics Matter**
   - Binary grading: 1 if right, 0 if wrong or "IDK"
   - Incentivizes guessing over honesty
   - Penalized grading would fix this

## Questions?

- Check `OLLAMA_README.md` for detailed docs
- See `why_llms_hallucinate.ipynb` for toy implementation
- Read paper: arXiv:2509.04664
