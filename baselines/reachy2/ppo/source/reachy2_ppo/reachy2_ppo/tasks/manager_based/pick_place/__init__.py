# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# The "Template-" prefix is required: scripts/list_envs.py filters the registry on it.
gym.register(
    id="Template-Reachy2-Pick-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:Reachy2PickPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Same task, state-only: ~7x faster (no camera render, no conv stack), for iteration.
gym.register(
    id="Template-Reachy2-Pick-Place-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_env_cfg:Reachy2PickPlaceStateEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_state_cfg.yaml",
    },
)
