# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reachy2 bimanual pick-and-place environment.

The robot must pick a cube off a table and place it at a target location. The
reward is deliberately staged -- reach, grasp, lift, transport, place -- so the
task decomposes into a task graph and each stage stays separable in the
per-term reward vector.

Unlike the nutpouring task (which subclasses NVIDIA's GR1T2 scene), this env is
built from scratch on `ManagerBasedRLEnvCfg`, since there is no upstream
Reachy2 task to inherit from.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from reachy2_ppo.assets.reachy2 import (
    REACHY2_ARM_GRIPPER_JOINTS,
    REACHY2_BOTH_EEF_LINKS,
    REACHY2_CFG,
    REACHY2_HEAD_LINK,
)

from . import mdp

##
# Scene
##

TABLE_HEIGHT = 0.75
CUBE_SIZE = 0.05
#: Height above the table the cube must clear to count as lifted.
LIFT_HEIGHT = TABLE_HEIGHT + CUBE_SIZE / 2 + 0.06


@configclass
class Reachy2PickPlaceSceneCfg(InteractiveSceneCfg):
    """Reachy2 in front of a table with a cube to move to a target."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=3000.0),
    )

    # Reachy2's base is pinned (its wheel joints are fixed in the URDF), so the
    # robot is simply placed in front of the table.
    robot: ArticulationCfg = REACHY2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, TABLE_HEIGHT / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.6, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.45, TABLE_HEIGHT + CUBE_SIZE)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2)),
        ),
    )

    # Kinematic marker: the goal pose. Not physically interactive -- it exists
    # so reward/observation terms can reference a scene entity rather than a
    # hardcoded constant, which keeps goal randomization easy to add later.
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.45, TABLE_HEIGHT + CUBE_SIZE)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE * 1.4, CUBE_SIZE * 1.4, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
    )

    # Head-mounted POV camera. `head` survives fixed-joint merging only because
    # prepare_for_isaac.py marks its parent joint <dont_collapse/>.
    robot_pov_cam = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{REACHY2_HEAD_LINK}/RobotPOVCam",
        height=160,
        width=256,
        data_types=["rgb"],
        update_period=0,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.05, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 5.0)),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """16-DOF bimanual action space: 7 joints per arm + 1 gripper per side.

    `preserve_order=True` so the action vector layout matches
    REACHY2_ARM_GRIPPER_JOINTS rather than the USD's internal DOF ordering --
    without it the mapping silently depends on import details.

    `use_default_offset=True` means action 0 holds the robot's default pose and
    the policy only outputs deltas, which is the right starting point for
    from-scratch RL (a freshly initialized policy outputs ~0).
    """

    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=REACHY2_ARM_GRIPPER_JOINTS,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class PolicyCfg(ObsGroup):
    """Two terms: a flat state vector and the raw camera frame for the CNN branch."""

    state = ObsTerm(func=mdp.state_obs)
    vision = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": True},
    )

    def __post_init__(self):
        self.enable_corruption = False
        # Must stay False: the image has to remain (H, W, C) for the conv stack.
        self.concatenate_terms = False


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    # Modest randomization so the policy can't memorize a single cube pose.
    reset_cube_pose = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Staged rewards forming the pick-and-place task graph.

    reach -> grasp -> lift -> transport -> place. Transport is gated on the
    cube actually being lifted, so the policy cannot farm it by sliding the
    cube along the table.
    """

    # --- stage 1: reach ---
    reach_cube = RewTerm(
        func=mdp.eef_object_distance,
        weight=1.0,
        params={"std": 0.15, "body_names": REACHY2_BOTH_EEF_LINKS, "object_cfg": SceneEntityCfg("cube")},
    )

    # --- stage 2: grasp (proximity AND a closing gripper) ---
    grasp_cube = RewTerm(
        func=mdp.object_is_grasped,
        weight=2.0,
        params={
            "grasp_distance": 0.08,
            "body_names": REACHY2_BOTH_EEF_LINKS,
            "object_cfg": SceneEntityCfg("cube"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["l_hand_finger", "r_hand_finger"]),
            "closed_threshold": 0.3,
        },
    )

    # --- stage 3: lift ---
    lift_cube = RewTerm(
        func=mdp.object_is_lifted,
        weight=5.0,
        params={"minimal_height": LIFT_HEIGHT, "object_cfg": SceneEntityCfg("cube")},
    )

    # --- stage 4: transport (gated on lift) ---
    transport_to_target = RewTerm(
        func=mdp.gated_object_target_distance,
        weight=8.0,
        params={
            "std": 0.2,
            "minimal_height": LIFT_HEIGHT,
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # --- stage 5: place (sparse success) ---
    place_success = RewTerm(
        func=mdp.object_at_target,
        weight=50.0,
        params={
            "threshold": 0.05,
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target"),
            "max_velocity": 0.1,
        },
    )

    # --- shaping penalties ---
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1.0e-4)
    terminating = RewTerm(func=base_mdp.is_terminated, weight=-5.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    cube_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_HEIGHT - 0.2, "asset_cfg": SceneEntityCfg("cube")},
    )


##
# Environment
##


@configclass
class Reachy2PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Reachy2 bimanual pick-and-place, configured for from-scratch PPO with vision."""

    scene: Reachy2PickPlaceSceneCfg = Reachy2PickPlaceSceneCfg(num_envs=64, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 10.0

        self.sim.dt = 1.0 / 100.0
        # One camera frame per control step: the policy consumes the image
        # every decision, so this is functional, not just a cost knob.
        self.sim.render_interval = self.decimation

        self.viewer.eye = (1.5, 1.5, 1.8)
        self.viewer.lookat = (0.0, 0.5, TABLE_HEIGHT)
