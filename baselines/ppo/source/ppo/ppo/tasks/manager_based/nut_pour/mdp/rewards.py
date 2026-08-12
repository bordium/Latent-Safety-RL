# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.mdp.terminations import (
    task_done_nut_pour as _task_done_nut_pour,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for raising an object above a height threshold (e.g. picked up off the table)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_1_cfg: SceneEntityCfg,
    object_2_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Tanh-kernel reward for bringing one object close to another."""
    object_1: RigidObject = env.scene[object_1_cfg.name]
    object_2: RigidObject = env.scene[object_2_cfg.name]
    distance = torch.norm(object_1.data.root_pos_w - object_2.data.root_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def robot_body_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    body_names: list[str],
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Tanh-kernel reward for bringing the nearest of several robot bodies close to an object.

    Pass both hands (e.g. `["left_hand_roll_link", "right_hand_roll_link"]`) to
    reward whichever one is doing the reaching, since which hand grasps a
    given object isn't fixed ahead of time.
    """
    robot = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]
    body_indices = [robot.data.body_names.index(name) for name in body_names]
    body_pos = robot.data.body_pos_w[:, body_indices]  # (N, B, 3)
    distance = torch.norm(body_pos - obj.data.root_pos_w.unsqueeze(1), dim=-1)  # (N, B)
    closest_distance = torch.min(distance, dim=1).values
    return 1 - torch.tanh(closest_distance / std)


def gated_object_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    gate_object_cfg: SceneEntityCfg,
    object_1_cfg: SceneEntityCfg,
    object_2_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Like `object_object_distance`, but only active once `gate_object_cfg` is lifted off the table.

    Withholds "move X toward Y" shaping until an object has actually been
    picked up, so the policy isn't rewarded for e.g. dragging the nut across
    the table toward the bowl without ever grasping the beaker.
    """
    gate = object_is_lifted(env, minimal_height, gate_object_cfg)
    return gate * object_object_distance(env, std, object_1_cfg, object_2_cfg)


def task_success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward: 1.0 on steps where the nut-pour success condition is met."""
    return _task_done_nut_pour(env).float()
