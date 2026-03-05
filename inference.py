#!/usr/bin/env python3
"""
ACT Policy Inference for SO-100 Arm.

Downloads a trained ACT policy from HuggingFace and runs it
on the SO-100 robot arm in a real-time control loop.

Usage:
  python inference.py --model your-hf-user/your-model
"""

import argparse
import time
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


def parse_args():
    parser = argparse.ArgumentParser(description="ACT Policy Inference – SO-100")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model repo (e.g. your-user/act_policy)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Control loop frequency (default: 30)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Inference device (default: cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    dt = 1.0 / args.fps

    # --- Load policy ---
    print(f"Loading policy from {args.model}...")
    policy = ACTPolicy.from_pretrained(args.model)
    policy = policy.to(args.device)
    policy.eval()
    print(f"Policy loaded on {args.device}")

    # --- Connect robot ---
    # Adjust robot_type and config for your hardware setup.
    robot = ManipulatorRobot(robot_type="so100")
    robot.connect()
    print("Robot connected")

    # --- Control loop ---
    print(f"Running inference at {args.fps} Hz (Ctrl+C to stop)...")

    try:
        while True:
            t_start = time.time()

            # 1. Get observation (cameras + joint states)
            observation = robot.get_observation()

            # 2. Prepare tensors
            obs_tensor = {
                k: v.unsqueeze(0).to(args.device) if isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
            }

            # 3. Predict action
            with torch.no_grad():
                action = policy.select_action(obs_tensor)

            # 4. Execute action
            robot.send_action(action.squeeze(0).cpu())

            # 5. Maintain timing
            elapsed = time.time() - t_start
            time.sleep(max(0, dt - elapsed))

            actual_fps = 1.0 / (time.time() - t_start)
            if actual_fps < args.fps * 0.8:
                print(f"Warning: FPS drop ({actual_fps:.1f} Hz)")

    except KeyboardInterrupt:
        print("\nStopping inference.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
