# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visually place the table / cube / target without editing the env config.

Spawns the scene in the GUI with the robot held at its default pose. Reachy2
uses ROS REP-103 (+X forward, +Y left, +Z up), which is easy to get wrong
headless. Prints a paste-ready snippet for pick_place_env_cfg.py.

    ./tune_scene.py
    ./tune_scene.py --table 0.6 0 --cube 0.45 -0.15
    ./tune_scene.py --show-frames     # EEF/head axis markers
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactively tune the Reachy2 pick-place scene layout.")
parser.add_argument("--task", type=str, default="Template-Reachy2-Pick-Place-v0", help="Name of the task.")
parser.add_argument(
    "--table", type=float, nargs=2, metavar=("X", "Y"), default=None, help="Table centre (metres, robot frame)."
)
parser.add_argument("--cube", type=float, nargs=2, metavar=("X", "Y"), default=None, help="Cube start position.")
parser.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"), default=None, help="Target marker position.")
parser.add_argument("--table-size", type=float, nargs=2, metavar=("DEPTH", "WIDTH"), default=None)
parser.add_argument("--show-frames", action="store_true", help="Draw axis markers on the EEF and head links.")
parser.add_argument("--steps", type=int, default=100000, help="How long to hold the scene open.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Only useful with a window, so force the GUI and cameras on.
args_cli.headless = False
args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym
import torch

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_tasks.utils import parse_env_cfg

import reachy2_ppo.tasks  # noqa: F401
from reachy2_ppo.assets.reachy2 import REACHY2_BOTH_EEF_LINKS, REACHY2_HEAD_LINK


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    # Overrides applied before the scene is built. Z is left alone.
    if args_cli.table is not None:
        x, y = args_cli.table
        z = env_cfg.scene.table.init_state.pos[2]
        env_cfg.scene.table.init_state.pos = (x, y, z)
    if args_cli.table_size is not None:
        depth, width = args_cli.table_size
        h = env_cfg.scene.table.spawn.size[2]
        env_cfg.scene.table.spawn.size = (depth, width, h)
    if args_cli.cube is not None:
        x, y = args_cli.cube
        z = env_cfg.scene.cube.init_state.pos[2]
        env_cfg.scene.cube.init_state.pos = (x, y, z)
    if args_cli.target is not None:
        x, y = args_cli.target
        z = env_cfg.scene.target.init_state.pos[2]
        env_cfg.scene.target.init_state.pos = (x, y, z)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    frames = None
    if args_cli.show_frames:
        # Streams from Nucleus on first use -- needs network once.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
        frames = VisualizationMarkers(
            marker_cfg.replace(prim_path="/Visuals/tune_frames")
        )
        body_names = REACHY2_BOTH_EEF_LINKS + [REACHY2_HEAD_LINK]
        body_ids = [env.scene["robot"].data.body_names.index(n) for n in body_names]

    env.reset()
    _report(env, env_cfg)

    # use_default_offset=True means zero action holds the default pose.
    zero = torch.zeros(env.action_space.shape, device=env.device)

    step = 0
    while simulation_app.is_running() and step < args_cli.steps:
        with torch.inference_mode():
            env.step(zero)
            if frames is not None:
                pos = env.scene["robot"].data.body_pos_w[:, body_ids].reshape(-1, 3)
                quat = env.scene["robot"].data.body_quat_w[:, body_ids].reshape(-1, 4)
                frames.visualize(translations=pos, orientations=quat)
        step += 1

    env.close()


def _report(env, env_cfg) -> None:
    """Print the robot-relative layout and a paste-ready config snippet."""
    robot = env.scene["robot"]
    origin = env.scene.env_origins[0]

    def rel(p):
        return [round(float(v), 3) for v in (torch.tensor(p, device=origin.device) if not torch.is_tensor(p) else p)]

    cube = env.scene["cube"].data.root_pos_w[0] - origin
    target = env.scene["target"].data.root_pos_w[0] - origin
    head_id = robot.data.body_names.index(REACHY2_HEAD_LINK)
    head = robot.data.body_pos_w[0, head_id] - origin

    print("\n" + "=" * 68)
    print("  Reachy2 frame: +X = FORWARD, +Y = LEFT, +Z = up")
    print("=" * 68)
    print(f"  table  centre  {rel(env_cfg.scene.table.init_state.pos)}   size {list(env_cfg.scene.table.spawn.size)}")
    print(f"  cube            {rel(cube)}")
    print(f"  target          {rel(target)}")
    print(f"  head ({REACHY2_HEAD_LINK})  {rel(head)}")
    for name in REACHY2_BOTH_EEF_LINKS:
        i = robot.data.body_names.index(name)
        p = robot.data.body_pos_w[0, i] - origin
        d = float(torch.linalg.norm(p - cube))
        print(f"  {name:16s}{rel(p)}   dist-to-cube {d:.3f} m")

    tx, ty, tz = env_cfg.scene.table.init_state.pos
    cx, cy, cz = rel(cube)
    gx, gy, gz = rel(target)
    print("-" * 68)
    print("  Paste into pick_place_env_cfg.py:")
    print(f"    table  init_state pos=({tx}, {ty}, TABLE_HEIGHT / 2)")
    print(f"    table  spawn size={tuple(env_cfg.scene.table.spawn.size)}")
    print(f"    cube   init_state pos=({cx}, {cy}, TABLE_HEIGHT + CUBE_SIZE)")
    print(f"    target init_state pos=({gx}, {gy}, TABLE_HEIGHT + CUBE_SIZE)")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
    # os._exit skips stdout flushing, losing the report when redirected.
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit's shutdown can hang for tens of minutes; see isaac-sim/IsaacSim#3730.
    os._exit(0)
