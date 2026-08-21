# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Reachy2 pick-and-place task: reach, grasp, lift, transport, place."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

from reachy2_ppo.assets.reachy2 import REACHY2_FINGERTIP_OFFSET

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pinch_centres(env: ManagerBasedRLEnv, finger_links: list[list[str]]) -> torch.Tensor:
    """Fingertip midpoint per hand -- the true grasp point, unlike the knuckle EEF frames."""
    robot: Articulation = env.scene["robot"]
    offset = torch.tensor(REACHY2_FINGERTIP_OFFSET, device=env.device).expand(env.num_envs, 3)
    centres = []
    for pair in finger_links:
        ids = [robot.data.body_names.index(n) for n in pair]
        tips = [robot.data.body_pos_w[:, i] + quat_apply(robot.data.body_quat_w[:, i], offset) for i in ids]
        centres.append(0.5 * (tips[0] + tips[1]))
    return torch.stack(centres, dim=1)


def eef_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    finger_links: list[list[str]],
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Stage 1 (reach): tanh kernel on the closest hand's fingertip-midpoint distance."""
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos = _pinch_centres(env, finger_links)
    distance = torch.norm(body_pos - obj.data.root_pos_w.unsqueeze(1), dim=-1)
    return 1.0 - torch.tanh(torch.min(distance, dim=1).values / std)


def grasp_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    finger_pairs: list[list[str]],
    force_threshold: float,
) -> torch.Tensor:
    """Binary: both opposing fingertips pressing on the object. Squeezing air scores zero."""
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)

    magnitude = torch.norm(forces, dim=-1).sum(dim=-1)
    names = list(sensor.body_names)
    grasped = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for pair in finger_pairs:
        a, b = (names.index(n) for n in pair)
        grasped |= (magnitude[:, a] > force_threshold) & (magnitude[:, b] > force_threshold)
    return grasped.float()


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Stage 3 (lift): 1.0 once the object clears a height threshold."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_lowest_vertex_height(env: ManagerBasedRLEnv, half_extent: float, object_cfg: SceneEntityCfg):
    """World height of the cube's lowest corner; scores tipping at zero, unlike the centre."""
    obj: RigidObject = env.scene[object_cfg.name]
    # Third row of R is world-z in body coords.
    rot = matrix_from_quat(obj.data.root_quat_w)
    drop = half_extent * rot[:, 2, :].abs().sum(dim=-1)
    return obj.data.root_pos_w[:, 2] - drop


def object_lift_progress(
    env: ManagerBasedRLEnv,
    rest_height: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    half_extent: float,
) -> torch.Tensor:
    """Stage 3 shaping: lift ramp on the lowest vertex, so tipping cannot farm it."""
    lowest = object_lowest_vertex_height(env, half_extent, object_cfg)
    table = rest_height - half_extent
    return ((lowest - table) / (minimal_height - rest_height)).clamp(0.0, 1.0)


def gated_object_target_distance(
    env: ManagerBasedRLEnv,
    std: float,
    rest_height: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    half_extent: float,
) -> torch.Tensor:
    """Stage 4 (transport): object->target distance, gated on lift so sliding earns nothing."""
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    lifted = object_lift_progress(env, rest_height, minimal_height, object_cfg, half_extent)
    distance = torch.norm(obj.data.root_pos_w - target.data.root_pos_w, dim=1)
    return lifted * (1.0 - torch.tanh(distance / std))


class ObjectPlacedAfterLift(ManagerTermBase):
    """Stage 5 (place), payable only after a real lift; latches a per-env "was lifted" flag."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._was_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._was_lifted[:] = False
        else:
            self._was_lifted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        minimal_height: float,
        object_cfg: SceneEntityCfg,
        target_cfg: SceneEntityCfg,
        max_velocity: float,
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        target: RigidObject = env.scene[target_cfg.name]

        self._was_lifted |= obj.data.root_pos_w[:, 2] > minimal_height

        distance = torch.norm(obj.data.root_pos_w - target.data.root_pos_w, dim=1)
        speed = torch.norm(obj.data.root_lin_vel_w, dim=1)
        return ((distance < threshold) & (speed < max_velocity) & self._was_lifted).float()
