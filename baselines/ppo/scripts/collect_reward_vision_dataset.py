# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Roll out an env and save (per-term reward vector, raw camera frame) pairs to disk.

The policy's "vision" observation term is mean-centered/scaled for training
stability (see PolicyCfg.vision in nut_pour_env_cfg.py), which changes frame
to frame and isn't what you want for a visual-correlation study. This script
instead reads the camera sensor's raw RGB output directly, and reads the
un-summed per-term reward breakdown off `RewardManager` (the same numbers
Isaac Lab already logs to TensorBoard per-episode, here captured per-step).

Without --checkpoint, actions are sampled randomly (like scripts/random_agent.py).
With --checkpoint, an skrl PPO checkpoint drives the rollout instead.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect (reward-vector, vision-frame) pairs from a rollout.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Nut-Pour-Ppo-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="skrl checkpoint to act with (default: random actions).")
parser.add_argument("--num_steps", type=int, default=50, help="Number of env.step() calls to record.")
parser.add_argument("--output", type=str, default="reward_vision_dataset.npz", help="Output .npz path.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# always need the camera pipeline
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import ppo.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    camera = env.unwrapped.scene["robot_pov_cam"]
    reward_manager = env.unwrapped.reward_manager
    reward_term_names = list(reward_manager.active_terms)
    print(f"[INFO] Reward terms (columns of reward_vector): {reward_term_names}")

    runner = None
    if args_cli.checkpoint:
        import skrl
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
        runner.agent.set_running_mode("eval")
        obs, _ = stepping_env.reset()
        del skrl  # imported only for the version-checked Runner import above
    else:
        print("[INFO] No --checkpoint given, stepping with random actions.")
        stepping_env = env
        stepping_env.reset()

    vision_frames = []
    reward_vectors = []

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            if runner is not None:
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = stepping_env.step(actions)
            else:
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                stepping_env.step(actions)

            # raw, un-normalized RGB -- shape (num_envs, H, W, 3), uint8
            vision_frames.append(camera.data.output["rgb"].clone().cpu().numpy())
            # per-term weighted reward, un-summed -- shape (num_envs, num_terms)
            reward_vectors.append(reward_manager._step_reward.clone().cpu().numpy())

            if (step + 1) % 10 == 0:
                print(f"[INFO] Recorded step {step + 1}/{args_cli.num_steps}")

    vision_frames = np.stack(vision_frames, axis=0)  # (steps, num_envs, H, W, 3)
    reward_vectors = np.stack(reward_vectors, axis=0)  # (steps, num_envs, num_terms)

    np.savez_compressed(
        args_cli.output,
        vision=vision_frames,
        reward_vector=reward_vectors,
        reward_term_names=np.array(reward_term_names),
    )
    print(f"[INFO] Saved vision {vision_frames.shape} and reward_vector {reward_vectors.shape} to {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
