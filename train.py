#!/usr/bin/env python3
"""
ACT Policy Training for SO-100 Arm (Google Colab)

Train an ACT (Action Chunking with Transformers) policy on a LeRobot dataset.
Designed to run in Google Colab with GPU. Each "# %%" marks a Colab cell.

Usage:
  1. Upload to Google Colab or copy cells individually
  2. Configure DATASET_REPO_ID, CLEAN_EPISODES, CORRUPT_EPISODES below
  3. Run all cells
"""

# %% ============================================================
#  Cell 1 – GPU check & environment setup
# ===============================================================
import subprocess, sys, os

# --- GPU Check ---
gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print(f"GPU: {gpu_info.stdout.strip()}")

# --- Dynamic batch size based on GPU ---
gpu_name = gpu_info.stdout.strip().lower()
if "a100" in gpu_name or "v100" in gpu_name:
    GPU_TIER, BATCH_SIZE = "high", 16
elif "l4" in gpu_name:
    GPU_TIER, BATCH_SIZE = "mid", 12
else:  # T4, P100, etc.
    GPU_TIER, BATCH_SIZE = "low", 8

print(f"GPU tier: {GPU_TIER} -> batch_size={BATCH_SIZE}")


# %% ============================================================
#  Cell 2 – Install LeRobot
# ===============================================================
print("Installing LeRobot + dependencies...")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "lerobot", "huggingface_hub", "wandb", "tensorboard"],
    check=True,
)

import lerobot
print(f"LeRobot version: {lerobot.__version__}")


# %% ============================================================
#  Cell 3 – Load dataset & filter episodes
# ===============================================================
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
import torch

# =======================================================================
#  >>> CONFIGURE THESE FOR YOUR DATASET <<<
# =======================================================================
DATASET_REPO_ID = "your-hf-user/your-dataset"  # HuggingFace dataset repo
CLEAN_EPISODES  = list(range(0, 60))             # episodes to use
CORRUPT_EPISODES = [60, 61, 62]                   # episodes to skip (if any)
# =======================================================================

print(f"Dataset:         {DATASET_REPO_ID}")
print(f"Clean episodes:  {len(CLEAN_EPISODES)}")
print(f"Skipped:         {CORRUPT_EPISODES}")

dataset = LeRobotDataset(
    repo_id=DATASET_REPO_ID,
    episodes=CLEAN_EPISODES,
)

print(f"\nEpisodes loaded: {dataset.num_episodes}")
print(f"Total frames:    {dataset.num_frames}")
print(f"FPS:             {dataset.fps}")
print(f"Features:        {list(dataset.features.keys())}")

camera_keys = [k for k in dataset.features if "image" in k or "camera" in k]
state_keys  = [k for k in dataset.features if "state" in k or "position" in k]
action_keys = [k for k in dataset.features if "action" in k]

print(f"\nCameras: {camera_keys}")
print(f"States:  {state_keys}")
print(f"Actions: {action_keys}")


# %% ============================================================
#  Cell 4 – Validate data quality
# ===============================================================
import numpy as np

print("Validating data quality...\n")

sample_indices = [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]
all_ok = True

for idx in sample_indices:
    try:
        _ = dataset[idx]
        print(f"  Frame {idx}: OK")
    except Exception as e:
        print(f"  Frame {idx}: ERROR – {e}")
        all_ok = False

if all_ok:
    print("\nAll sampled frames are readable.")
else:
    print("\nSome frames have issues – consider filtering more episodes.")

sample = dataset[0]
print("\nSample shapes:")
for key, val in sample.items():
    if isinstance(val, torch.Tensor):
        print(f"  {key}: {val.shape} ({val.dtype})")


# %% ============================================================
#  Cell 5 – Training configuration (ACT hyperparameters)
# ===============================================================

TRAINING_CONFIG = {
    # --- Identification ---
    "experiment_name": "act_policy",

    # --- Dataset ---
    "dataset_repo_id": DATASET_REPO_ID,
    "episodes": CLEAN_EPISODES,

    # --- ACT Architecture ---
    "chunk_size": 100,            # ~3.3s lookahead at 30 Hz
    "n_encoder_layers": 4,
    "n_decoder_layers": 7,
    "d_model": 256,
    "dim_feedforward": 1024,
    "n_heads": 8,
    "dropout": 0.1,

    # --- VAE ---
    "kl_weight": 10.0,
    "use_vae": True,
    "latent_dim": 32,

    # --- Vision backbone ---
    "vision_backbone": "resnet18",
    "pretrained_backbone": True,

    # --- Training ---
    "training_steps": 100_000,
    "batch_size": BATCH_SIZE,
    "lr": 1e-5,
    "lr_warmup_steps": 500,
    "adam_betas": [0.95, 0.999],
    "adam_eps": 1e-8,
    "adam_weight_decay": 1e-6,
    "grad_clip_norm": 10.0,

    # --- Logging & Checkpoints ---
    "log_freq": 250,
    "save_checkpoint_freq": 25_000,
    "eval_freq": 25_000,
    "save_dir": "./outputs/act_policy",

    # --- Temporal Ensemble (inference smoothing) ---
    "temporal_ensemble_coeff": 0.01,
}

print("Training config:")
print("=" * 50)
for k, v in TRAINING_CONFIG.items():
    if k != "episodes":
        print(f"  {k}: {v}")
print(f"  episodes: {len(TRAINING_CONFIG['episodes'])} total")


# %% ============================================================
#  Cell 6 – Start training (LeRobot CLI)
# ===============================================================

os.makedirs(TRAINING_CONFIG["save_dir"], exist_ok=True)

episodes_str = str(CLEAN_EPISODES)

train_cmd = f"""python -m lerobot.scripts.train \
    --dataset.repo_id={DATASET_REPO_ID} \
    --dataset.episodes='{episodes_str}' \
    --policy.type=act \
    --policy.chunk_size={TRAINING_CONFIG['chunk_size']} \
    --policy.n_action_steps={TRAINING_CONFIG['chunk_size']} \
    --policy.dim_model={TRAINING_CONFIG['d_model']} \
    --policy.n_heads={TRAINING_CONFIG['n_heads']} \
    --policy.n_encoder_layers={TRAINING_CONFIG['n_encoder_layers']} \
    --policy.n_decoder_layers={TRAINING_CONFIG['n_decoder_layers']} \
    --policy.dim_feedforward={TRAINING_CONFIG['dim_feedforward']} \
    --policy.dropout={TRAINING_CONFIG['dropout']} \
    --policy.kl_weight={TRAINING_CONFIG['kl_weight']} \
    --policy.use_vae={TRAINING_CONFIG['use_vae']} \
    --policy.latent_dim={TRAINING_CONFIG['latent_dim']} \
    --policy.vision_backbone={TRAINING_CONFIG['vision_backbone']} \
    --policy.pretrained_backbone={TRAINING_CONFIG['pretrained_backbone']} \
    --policy.temporal_ensemble_coeff={TRAINING_CONFIG['temporal_ensemble_coeff']} \
    --batch_size={TRAINING_CONFIG['batch_size']} \
    --lr={TRAINING_CONFIG['lr']} \
    --lr_warmup_steps={TRAINING_CONFIG['lr_warmup_steps']} \
    --adam_betas {TRAINING_CONFIG['adam_betas'][0]} {TRAINING_CONFIG['adam_betas'][1]} \
    --adam_eps={TRAINING_CONFIG['adam_eps']} \
    --adam_weight_decay={TRAINING_CONFIG['adam_weight_decay']} \
    --grad_clip_norm={TRAINING_CONFIG['grad_clip_norm']} \
    --training_steps={TRAINING_CONFIG['training_steps']} \
    --log_freq={TRAINING_CONFIG['log_freq']} \
    --save_freq={TRAINING_CONFIG['save_checkpoint_freq']} \
    --output_dir={TRAINING_CONFIG['save_dir']} \
    --wandb.disable=true"""

print("Starting training...\n")
print(train_cmd)
print("\n" + "=" * 60)

os.system(train_cmd)


# %% ============================================================
#  Cell 6b – FALLBACK: Programmatic training loop
#  Uncomment if Cell 6 (CLI) fails due to API changes.
# ===============================================================

"""
from pathlib import Path
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from torch.optim.lr_scheduler import LambdaLR

dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, episodes=CLEAN_EPISODES)

cfg = ACTConfig(
    chunk_size=TRAINING_CONFIG["chunk_size"],
    n_action_steps=TRAINING_CONFIG["chunk_size"],
    dim_model=TRAINING_CONFIG["d_model"],
    n_heads=TRAINING_CONFIG["n_heads"],
    n_encoder_layers=TRAINING_CONFIG["n_encoder_layers"],
    n_decoder_layers=TRAINING_CONFIG["n_decoder_layers"],
    dim_feedforward=TRAINING_CONFIG["dim_feedforward"],
    dropout=TRAINING_CONFIG["dropout"],
    kl_weight=TRAINING_CONFIG["kl_weight"],
    use_vae=TRAINING_CONFIG["use_vae"],
    latent_dim=TRAINING_CONFIG["latent_dim"],
    vision_backbone=TRAINING_CONFIG["vision_backbone"],
    pretrained_backbone=TRAINING_CONFIG["pretrained_backbone"],
    temporal_ensemble_coeff=TRAINING_CONFIG["temporal_ensemble_coeff"],
    input_features=dataset.features,
)

policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
policy.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = policy.to(device)

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=TRAINING_CONFIG["lr"],
    betas=tuple(TRAINING_CONFIG["adam_betas"]),
    eps=TRAINING_CONFIG["adam_eps"],
    weight_decay=TRAINING_CONFIG["adam_weight_decay"],
)

def lr_lambda(step):
    if step < TRAINING_CONFIG["lr_warmup_steps"]:
        return step / TRAINING_CONFIG["lr_warmup_steps"]
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

print(f"Starting manual training on {device}...")
global_step = 0
best_loss = float("inf")
save_dir = Path(TRAINING_CONFIG["save_dir"])
save_dir.mkdir(parents=True, exist_ok=True)

while global_step < TRAINING_CONFIG["training_steps"]:
    for batch in dataloader:
        if global_step >= TRAINING_CONFIG["training_steps"]:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        loss_dict = policy.forward(batch)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), TRAINING_CONFIG["grad_clip_norm"])
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % TRAINING_CONFIG["log_freq"] == 0:
            lr_current = scheduler.get_last_lr()[0]
            kl_loss = loss_dict.get("kl_loss", 0)
            print(f"  Step {global_step:6d}/{TRAINING_CONFIG['training_steps']} | "
                  f"Loss: {loss.item():.4f} | KL: {kl_loss:.4f} | LR: {lr_current:.2e}")

        if global_step % TRAINING_CONFIG["save_checkpoint_freq"] == 0:
            ckpt_path = save_dir / f"checkpoint_{global_step:06d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = save_dir / "best_checkpoint"
                best_path.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(str(best_path))
                print(f"  New best checkpoint! Loss: {best_loss:.4f}")

print(f"\\nTraining complete after {global_step} steps. Best loss: {best_loss:.4f}")
"""


# %% ============================================================
#  Cell 7 – Upload trained model to Hugging Face
# ===============================================================
from huggingface_hub import HfApi, login

login()  # Interactive prompt in Colab

# =======================================================================
#  >>> CONFIGURE THESE <<<
# =======================================================================
HF_USER    = "your-hf-user"       # your HuggingFace username
MODEL_NAME = "act_policy"         # model repo name
# =======================================================================

REPO_ID = f"{HF_USER}/{MODEL_NAME}"

output_dir = TRAINING_CONFIG["save_dir"]
checkpoint_dirs = sorted(
    [d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
    key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
)

if os.path.exists(os.path.join(output_dir, "best_checkpoint")):
    upload_path = os.path.join(output_dir, "best_checkpoint")
elif checkpoint_dirs:
    upload_path = os.path.join(output_dir, checkpoint_dirs[-1])
else:
    upload_path = output_dir

print(f"Uploading: {upload_path}")

api = HfApi()
api.create_repo(repo_id=REPO_ID, exist_ok=True)
api.upload_folder(
    folder_path=upload_path,
    repo_id=REPO_ID,
    commit_message=f"ACT policy trained on {len(CLEAN_EPISODES)} episodes",
)

print(f"Model uploaded: https://huggingface.co/{REPO_ID}")


# %% ============================================================
#  Cell 8 – Summary
# ===============================================================
print(f"""
Training complete.

  Policy:     ACT (Action Chunking with Transformers)
  Dataset:    {len(CLEAN_EPISODES)} episodes from {DATASET_REPO_ID}
  Steps:      {TRAINING_CONFIG['training_steps']}
  Chunk size: {TRAINING_CONFIG['chunk_size']} (~{TRAINING_CONFIG['chunk_size']/30:.1f}s at 30 Hz)
  Model:      https://huggingface.co/{REPO_ID}

Next steps:
  1. Download the model on your robot PC
  2. Run: python inference.py --model {REPO_ID}
  3. If jittery, lower temporal_ensemble_coeff (e.g. 0.005)

Troubleshooting:
  - Loss stagnates  -> increase lr to 5e-5
  - Overfitting      -> increase dropout to 0.2, reduce steps
  - Colab timeout    -> resume from last checkpoint
""")
