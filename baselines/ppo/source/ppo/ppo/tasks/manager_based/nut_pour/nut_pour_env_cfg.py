# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_base_env_cfg import NutPourGR1T2BaseEnvCfg

from . import mdp

# GR1T2's 14 arm joints + 22 hand joints, in the order the PPO action vector
# drives them. The base task leaves the action term MISSING (it's normally
# filled in by a teleop/mimic pipeline); this is the same 36-joint set NVIDIA's
# GR00T closed-loop eval config drives, copied here as data (not imported) so
# this extension doesn't need the separate IsaacLabEvalTasks package installed.
GR1T2_ARM_HAND_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]

# Both hands, used by reward terms that don't know ahead of time which hand
# is doing the grasping.
BOTH_HANDS = ["left_hand_roll_link", "right_hand_roll_link"]


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # `use_default_offset=True` + a small scale means action=0 holds the
    # robot's default pose and the policy only has to output small deltas
    # from there. This is the standard from-scratch-RL setup, and different
    # on purpose from the GR00T closed-loop eval config: that config drives
    # *absolute* joint targets (`use_default_offset=False`) because it's
    # replaying an imitation policy already calibrated in that space. A
    # freshly-initialized PPO policy outputs ~0 for every action at the
    # start of training, and absolute targets of 0 radians would snap the
    # arms away from their bent resting pose on the very first step.
    gr1_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=GR1T2_ARM_HAND_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class PolicyCfg(ObsGroup):
    """Policy observations: a compact state vector plus the raw camera frame.

    Two terms, both fed to the network (unlike a logging-only setup): the CNN
    branch in the skrl config reads "vision", the MLP branch reads "state".
    """

    state = ObsTerm(func=mdp.state_obs)
    vision = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": True},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the nut-pour task.

    The base `NutPourGR1T2BaseEnvCfg` ships with `rewards = None`: it's built
    for GR00T imitation-learning eval, not reward-driven RL. These terms are
    dense shaping toward the three sub-goals `task_done_nut_pour` checks
    (nut -> bowl, bowl -> scale, beaker -> bin), plus a sparse bonus on full
    success. The pour/bin terms are gated on the beaker having left the
    table -- see `gated_object_object_distance` -- so the policy isn't
    rewarded for sliding objects around without ever picking anything up.
    """

    # smoothness / effort
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)

    # penalize triggering an early termination (a dropped object, etc.)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    # stage 1: reach for the beaker with whichever hand is closer
    reach_beaker = RewTerm(
        func=mdp.robot_body_object_distance,
        weight=1.0,
        params={"std": 0.15, "body_names": BOTH_HANDS, "object_cfg": SceneEntityCfg("sorting_beaker")},
    )

    # stage 2: lift it off the table
    beaker_lifted = RewTerm(
        func=mdp.object_is_lifted,
        weight=3.0,
        params={"minimal_height": 1.04, "object_cfg": SceneEntityCfg("sorting_beaker")},
    )

    # stage 3: pour -- bring the nut toward the bowl (only once the beaker is picked up)
    nut_to_bowl = RewTerm(
        func=mdp.gated_object_object_distance,
        weight=2.0,
        params={
            "std": 0.2,
            "minimal_height": 1.04,
            "gate_object_cfg": SceneEntityCfg("sorting_beaker"),
            "object_1_cfg": SceneEntityCfg("factory_nut"),
            "object_2_cfg": SceneEntityCfg("sorting_bowl"),
        },
    )

    # stage 4a: move the bowl onto the scale
    bowl_to_scale = RewTerm(
        func=mdp.object_object_distance,
        weight=1.0,
        params={"std": 0.3, "object_1_cfg": SceneEntityCfg("sorting_bowl"), "object_2_cfg": SceneEntityCfg("sorting_scale")},
    )

    # stage 4b: stow the emptied beaker in the bin (only once picked up)
    beaker_to_bin = RewTerm(
        func=mdp.gated_object_object_distance,
        weight=1.0,
        params={
            "std": 0.3,
            "minimal_height": 1.04,
            "gate_object_cfg": SceneEntityCfg("sorting_beaker"),
            "object_1_cfg": SceneEntityCfg("sorting_beaker"),
            "object_2_cfg": SceneEntityCfg("black_sorting_bin"),
        },
    )

    # full task success (sparse)
    task_success = RewTerm(func=mdp.task_success_bonus, weight=50.0)


##
# Environment configuration
##


@configclass
class NutPourPpoEnvCfg(NutPourGR1T2BaseEnvCfg):
    """Nut-pour task configured for from-scratch PPO training with a vision-based policy."""

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()

    # NutPourGR1T2BaseEnvCfg sets this to a raw torch.Tensor. Nothing in
    # isaaclab/isaaclab_tasks/isaaclab_rl actually reads it -- it looks like
    # metadata for an external teleop/eval tool -- but Hydra's config
    # serialization (used by train.py/play.py, not list_envs.py/random_agent.py)
    # walks every attribute on the env cfg and can't serialize a raw Tensor,
    # so it has to be neutralized here.
    idle_action = None

    # NutPourGR1T2BaseEnvCfg turns XR (VR teleop) on; `None` is core Isaac
    # Lab's own default for "no XR device" (ManagerBasedEnvCfg.xr). We don't
    # do XR teleop here, and XrCfg's anchor_rotation_custom_func defaults to
    # an inline lambda that the same Hydra round-trip can't turn back into a
    # callable after serializing it, so this has to go too.
    xr = None

    def __post_init__(self) -> None:
        super().__post_init__()

        # Number of parallel envs. Every env renders its own camera frame and
        # runs a CNN forward pass each step, so this is far more
        # compute-bound than a state-only task -- tune down/up per GPU with
        # `--num_envs` before reaching for anything else.
        self.scene.num_envs = 64

        # TiledCameraCfg batches every env's render into one pass, unlike the
        # base task's plain CameraCfg (fine for the single-env GR00T eval,
        # too slow per-env at training scale). Same placement/optics as the
        # GR00T closed-loop eval config.
        self.scene.robot_pov_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/RobotPOVCam",
            height=160,
            width=256,
            data_types=["rgb"],
            update_period=0,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.12, 1.67675), rot=(-0.19848, 0.9801, 0.0, 0.0), convention="ros"
            ),
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2)),
        )

        # Render exactly one frame per control step (env.step() call): the
        # policy needs a fresh image each decision, not just for logging, so
        # this is now a functional requirement and not only a cost knob.
        self.sim.render_interval = self.decimation

        self.viewer.eye = (0.0, 1.8, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        # WAR to skip while loop bug after calling env.reset() followed by env.sim.reset()
        # https://github.com/isaac-sim/IsaacLab-Internal/blob/devel/source/isaaclab/isaaclab/envs/manager_based_env.py#L311C13-L311C53
        self.wait_for_textures = False
