# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.mdp import (
    get_hand_state,
    get_head_state,
    get_left_eef_pos,
    get_left_eef_quat,
    get_right_eef_pos,
    get_right_eef_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position of a rigid object relative to its environment origin.

    World-frame positions differ per env just from the parallel-env tiling
    offset, so this subtracts `env.scene.env_origins` the same way the
    existing `get_left_eef_pos`/`get_right_eef_pos` helpers do for the robot.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Every non-visual policy input, concatenated into one flat vector.

    An `ObsGroup` either concatenates all of its terms or none of them, and
    we need the camera image left un-flattened as its own (H, W, C) term for
    the CNN branch. So all the vector-valued quantities are pre-concatenated
    here into a single "state" term, leaving the policy group with exactly
    two keys: "state" and "vision".
    """
    return torch.cat(
        (
            last_action(env),
            joint_pos_rel(env),
            joint_vel_rel(env),
            get_left_eef_pos(env),
            get_left_eef_quat(env),
            get_right_eef_pos(env),
            get_right_eef_quat(env),
            get_hand_state(env),
            get_head_state(env),
            object_pos_rel(env, SceneEntityCfg("sorting_scale")),
            object_pos_rel(env, SceneEntityCfg("sorting_bowl")),
            object_pos_rel(env, SceneEntityCfg("sorting_beaker")),
            object_pos_rel(env, SceneEntityCfg("factory_nut")),
            object_pos_rel(env, SceneEntityCfg("black_sorting_bin")),
        ),
        dim=-1,
    )
