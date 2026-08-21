# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Reachy2 pick-and-place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action
from isaaclab.managers import SceneEntityCfg

from reachy2_ppo.assets.reachy2 import (
    REACHY2_LEFT_EEF_LINK,
    REACHY2_RIGHT_EEF_LINK,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def eef_pos(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """EEF position relative to the env origin, removing the per-env tiling offset."""
    robot: Articulation = env.scene["robot"]
    index = robot.data.body_names.index(link_name)
    return robot.data.body_pos_w[:, index] - env.scene.env_origins


def eef_quat(env: ManagerBasedRLEnv, link_name: str) -> torch.Tensor:
    """EEF orientation in the world frame; rotation needs no origin offset."""
    robot: Articulation = env.scene["robot"]
    index = robot.data.body_names.index(link_name)
    return robot.data.body_quat_w[:, index]


def object_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rigid-object position relative to the environment origin."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def object_to_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object -> target vector; the task drives this to zero."""
    obj = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    return target.data.root_pos_w - obj.data.root_pos_w


def eef_to_object(
    env: ManagerBasedRLEnv,
    link_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EEF -> object vector; drives the reach stage."""
    robot: Articulation = env.scene["robot"]
    index = robot.data.body_names.index(link_name)
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w - robot.data.body_pos_w[:, index]


def state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Every non-visual policy input as one flat vector, so the group is just state + vision."""
    cube = SceneEntityCfg("cube")
    target = SceneEntityCfg("target")
    return torch.cat(
        (
            last_action(env),
            joint_pos_rel(env),
            joint_vel_rel(env),
            eef_pos(env, REACHY2_LEFT_EEF_LINK),
            eef_quat(env, REACHY2_LEFT_EEF_LINK),
            eef_pos(env, REACHY2_RIGHT_EEF_LINK),
            eef_quat(env, REACHY2_RIGHT_EEF_LINK),
            object_pos_rel(env, cube),
            object_pos_rel(env, target),
            eef_to_object(env, REACHY2_LEFT_EEF_LINK, cube),
            eef_to_object(env, REACHY2_RIGHT_EEF_LINK, cube),
            object_to_target(env, cube, target),
        ),
        dim=-1,
    )
