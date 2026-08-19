# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the Reachy2 pick-and-place environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hold_joints_at_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Point the drive targets of undriven joints at their default positions.

    `joint_pos_target` initializes to 0.0, not to `default_joint_pos`, so any
    joint outside the action space is actively pulled to zero regardless of
    `init_state`. The head would otherwise never keep its downward pitch.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    joint_ids = asset_cfg.joint_ids
    target = robot.data.default_joint_pos[env_ids][:, joint_ids]
    robot.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
