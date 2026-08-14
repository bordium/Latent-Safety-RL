# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Roll out an env and save (state vector, RGB frame, per-term reward vector) triples.

"state" is read straight off the policy's own observation group
(env.observation_manager), so it's exactly what the policy's MLP branch sees:
last action, joint pos/vel, eef pose, hand/head joint state, and the 5 object
positions, all concatenated (see mdp/observations.py:state_obs). "vision" is
the *raw* camera sensor output, not the mean-centered/scaled version the
policy's CNN branch actually consumes (see PolicyCfg.vision in
nut_pour_env_cfg.py) -- raw pixels are what you want for a data/correlation
study, since the policy's own normalization shifts frame to frame.

--format csv (default) writes state + reward only, one row per (step, env)
pair, for a quick human-readable sanity check. A single RGB frame is 122,880
values (160x256x3) -- that does not fit sensibly as extra CSV columns, so
vision is skipped entirely in this mode (and never captured, to avoid the
memory/compute cost).

--format npz writes state + vision + reward together into one .npz (a flat
.npy can't hold three differently-shaped arrays -- state is (N, 198), vision
is (N, H, W, 3), reward is (N, 9) -- so this bundles them as named arrays
instead: "state", "vision", "reward", "reward_term_names").

Without --checkpoint, actions are sampled randomly (like scripts/random_agent.py).
With --checkpoint, an skrl PPO checkpoint drives the rollout instead.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (state, vision, reward-vector) data from a rollout.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Nut-Pour-Ppo-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="skrl checkpoint to act with (default: random actions).")
parser.add_argument("--num_steps", type=int, default=100, help="Number of env.step() calls to record.")
parser.add_argument("--format", type=str, default="csv", choices=["csv", "npz"], help="Output format.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help=(
        "Output path (default: baselines/ppo/results/<name>.<format>, where <name> is the training"
        " run's timestamp -- e.g. checkpoints/.../2026-08-12_17-48-07_ppo_torch/checkpoints/agent.pt"
        " -> results/2026-08-12_17-48-07.<format> -- or now() if run without --checkpoint)."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()
if args_cli.output is None:
    if args_cli.checkpoint:
        # checkpoint path looks like ".../<run_dir>/checkpoints/<file>.pt"; run_dir is named
        # "<timestamp>_<algorithm>_<ml_framework>" by train.py (see log_dir in scripts/skrl/train.py)
        run_dir_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args_cli.checkpoint))))
        match = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir_name)
        output_name = match.group(0) if match else run_dir_name
    else:
        output_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    args_cli.output = os.path.normpath(os.path.join(results_dir, f"{output_name}.{args_cli.format}"))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import ppo.tasks  # noqa: F401


def main():
    include_vision = args_cli.format == "npz"

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    camera = env.unwrapped.scene["robot_pov_cam"]
    reward_manager = env.unwrapped.reward_manager
    reward_term_names = list(reward_manager.active_terms)

    runner = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

        stepping_env = SkrlVecEnvWrapper(env, ml_framework="torch")
        runner = Runner(stepping_env, agent_cfg)
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        runner.agent.load(args_cli.checkpoint)
        runner.agent.enable_training_mode(False, apply_to_models=True)
        obs, _ = stepping_env.reset()
    else:
        print("[INFO] No --checkpoint given, stepping with random actions.")
        stepping_env = env
        stepping_env.reset()

    state_rows = []
    vision_rows = []
    reward_rows = []

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if runner is not None:
                outputs = runner.agent.act(obs, None, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = stepping_env.step(actions)
            else:
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                stepping_env.step(actions)

            # exactly what the policy's MLP branch conditions on
            state = env.unwrapped.observation_manager.compute_group("policy")["state"]
            state_rows.append(state.clone().cpu().numpy())
            if include_vision:
                # raw, un-normalized RGB -- shape (num_envs, H, W, 3), uint8
                vision_rows.append(camera.data.output["rgb"].clone().cpu().numpy())
            # per-term weighted reward, un-summed -- shape (num_envs, num_terms)
            reward_rows.append(reward_manager._step_reward.clone().cpu().numpy())

            if (step + 1) % 20 == 0:
                print(f"[INFO] Recorded step {step + 1}/{args_cli.num_steps}")

    states = np.stack(state_rows, axis=0)  # (steps, num_envs, state_dim)
    rewards = np.stack(reward_rows, axis=0)  # (steps, num_envs, num_terms)
    num_steps, num_envs, state_dim = states.shape

    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    if args_cli.format == "csv":
        state_columns = [f"state_{i}" for i in range(state_dim)]
        with open(args_cli.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "env_id"] + state_columns + reward_term_names)
            for step in range(num_steps):
                for env_id in range(num_envs):
                    writer.writerow(
                        [step, env_id] + states[step, env_id].tolist() + rewards[step, env_id].tolist()
                    )
        print(f"[INFO] Wrote {num_steps * num_envs} rows to {args_cli.output}")
    else:
        vision = np.stack(vision_rows, axis=0)  # (steps, num_envs, H, W, 3)
        np.savez_compressed(
            args_cli.output,
            state=states.reshape(num_steps * num_envs, state_dim),
            vision=vision.reshape(num_steps * num_envs, *vision.shape[2:]),
            reward=rewards.reshape(num_steps * num_envs, len(reward_term_names)),
            reward_term_names=np.array(reward_term_names),
        )
        print(
            f"[INFO] Saved state {states.shape}, vision {vision.shape}, reward {rewards.shape}"
            f" ({num_steps * num_envs} entries total) to {args_cli.output}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
