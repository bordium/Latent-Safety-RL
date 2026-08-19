# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Reachy2 pick-and-place task.

Staged reach -> grasp -> lift -> transport -> place, so each stage stays
separable in the per-term reward vector that
`scripts/collect_state_reward_dataset.py` records. Later stages are gated on
earlier ones so they cannot be farmed out of order.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_positions(env: ManagerBasedRLEnv, body_names: list[str]) -> torch.Tensor:
    """World positions of the named bodies, (num_envs, len(body_names), 3)."""
    robot: Articulation = env.scene["robot"]
    indices = [robot.data.body_names.index(name) for name in body_names]
    return robot.data.body_pos_w[:, indices]


def eef_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    body_names: list[str],
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Stage 1 (reach): tanh kernel on the closest hand's distance.

    Minimum over `body_names` -- either hand may do the reaching.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos = _body_positions(env, body_names)
    distance = torch.norm(body_pos - obj.data.root_pos_w.unsqueeze(1), dim=-1)
    return 1.0 - torch.tanh(torch.min(distance, dim=1).values / std)


def object_is_grasped(
    env: ManagerBasedRLEnv,
    grasp_distance: float,
    body_names: list[str],
    object_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    closed_threshold: float,
) -> torch.Tensor:
    """Stage 2 (grasp): hand at the object AND gripper closing.

    The gripper term is what stops hovering from earning full credit.
    """
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene[object_cfg.name]

    body_pos = _body_positions(env, body_names)
    distance = torch.norm(body_pos - obj.data.root_pos_w.unsqueeze(1), dim=-1)
    is_near = torch.min(distance, dim=1).values < grasp_distance

    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    is_closing = torch.max(gripper_pos, dim=1).values > closed_threshold

    return (is_near & is_closing).float()


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Stage 3 (lift): 1.0 once the object clears a height threshold."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def gated_object_target_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Stage 4 (transport): closes object->target distance, gated on lift.

    The gate stops the cube being slid across the table for credit.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    lifted = object_is_lifted(env, minimal_height, object_cfg)
    distance = torch.norm(obj.data.root_pos_w - target.data.root_pos_w, dim=1)
    return lifted * (1.0 - torch.tanh(distance / std))


def object_at_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    max_velocity: float,
) -> torch.Tensor:
    """Stage 5 (place): sparse success -- object at rest on the target.

    Low velocity is required so flinging the cube through does not count.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    distance = torch.norm(obj.data.root_pos_w - target.data.root_pos_w, dim=1)
    speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
    return ((distance < threshold) & (speed < max_velocity)).float()
