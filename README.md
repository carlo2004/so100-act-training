# so100-act-training

Train an [ACT](https://tonyzhaozh.github.io/aloha/) (Action Chunking with Transformers) policy for the SO-100 robot arm using [LeRobot](https://github.com/huggingface/lerobot) on Google Colab, then run inference locally.

## Overview

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Collect data   │ ──>  │  Train on    │ ──>  │  Run inference  │
│  (SO-100 + cams)│      │  Google Colab│      │  (local machine)│
└─────────────────┘      └──────────────┘      └─────────────────┘
```

| Component | Details |
|-----------|---------|
| Robot | SO-100 (6-DoF) |
| Policy | ACT – Action Chunking with Transformers |
| Framework | [LeRobot](https://github.com/huggingface/lerobot) |
| Training | Google Colab (GPU) |
| Inference | Any machine with USB connection to robot |

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script for Google Colab |
| `inference.py` | Inference script for local robot control |

## Prerequisites

- A [HuggingFace](https://huggingface.co/) account
- A LeRobot-format dataset uploaded to HuggingFace Hub
- Google account (for Colab)
- SO-100 robot arm (for inference)

## 1. Training (Google Colab)

### Setup

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `train.py` or copy cells manually (each `# %%` = one cell)
3. Select a GPU runtime: `Runtime > Change runtime type > GPU`

### Configure

Edit these variables in `train.py` (Cell 3):

```python
DATASET_REPO_ID  = "your-hf-user/your-dataset"   # your HF dataset
CLEAN_EPISODES   = list(range(0, 60))              # episodes to train on
CORRUPT_EPISODES = [60, 61, 62]                     # episodes to skip
```

Edit these in Cell 7 (upload):

```python
HF_USER    = "your-hf-user"
MODEL_NAME = "act_policy"
```

### Run

Execute all cells in order. Training takes ~2–4 hours on a T4 GPU (100k steps).

The script automatically:
- Detects GPU and sets an appropriate batch size
- Installs LeRobot
- Loads and validates your dataset
- Trains the ACT policy
- Uploads the model to HuggingFace Hub

### Hyperparameters

Key defaults (tuned for ~60 episodes of pick & place):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `chunk_size` | 100 | ~3.3s lookahead at 30 Hz |
| `training_steps` | 100,000 | |
| `lr` | 1e-5 | Increase to 5e-5 if loss stagnates |
| `dropout` | 0.1 | Increase to 0.2 if overfitting |
| `temporal_ensemble_coeff` | 0.01 | Lower = smoother actions |

If the CLI training (Cell 6) fails, uncomment the fallback training loop in Cell 6b.

## 2. Inference (Local Machine)

### Install

```bash
pip install lerobot
```

### Run

```bash
python inference.py --model your-hf-user/act_policy
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model repo |
| `--fps` | 30 | Control loop frequency |
| `--device` | cpu | `cpu` or `cuda` |

### What it does

1. Downloads the trained policy from HuggingFace
2. Connects to the SO-100 robot
3. Runs a real-time control loop: observe → predict → act

Stop with `Ctrl+C`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Loss stagnates | Increase `lr` to `5e-5` |
| Overfitting | Increase `dropout` to `0.2`, reduce `training_steps` |
| Colab timeout | Resume from last checkpoint in `./outputs/act_policy/` |
| Jittery robot | Lower `temporal_ensemble_coeff` (e.g. `0.005`) |
| FPS drops (inference) | Use `--device cuda` if GPU available, or reduce camera resolution |

## License

MIT
