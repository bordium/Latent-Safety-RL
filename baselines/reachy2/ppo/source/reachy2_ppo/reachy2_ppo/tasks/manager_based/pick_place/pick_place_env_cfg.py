# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reachy2 bimanual pick-and-place environment.

Pick a cube off a table and place it at a target. Rewards are staged (reach,
grasp, lift, transport, place) so each stage stays separable in the per-term
reward vector. Built directly on `ManagerBasedRLEnvCfg` -- there is no upstream
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

# Reachy2 uses ROS REP-103: +X forward, +Y left, +Z up (verified in the URDF --
# torso at x=+0.08, shoulders at y=+-0.2). Props belong at +X, not +Y.
TABLE_HEIGHT = 0.75
CUBE_SIZE = 0.05
#: Base to table centre, along +X.
TABLE_DIST = 0.65
#: Base to cube/target, along +X. Pushed out for camera framing; much beyond
#: this and the cube leaves the arm's reach.
REACH_DIST = 0.52
#: Height the cube must clear to count as lifted.
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

    # Base is pinned (wheel joints are fixed in the URDF).
    robot: ArticulationCfg = REACHY2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_DIST, 0.0, TABLE_HEIGHT / 2)),
        spawn=sim_utils.CuboidCfg(
            # (depth_x, width_y, height). Front edge at 0.35 m clears the
            # mobile base's 0.163 m collision sphere.
            size=(0.5, 1.0, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        # Exact rest height, not dropped -- a 25 mm gap made it bounce and roll
        # ~100 mm before the episode started.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(REACH_DIST, -0.15, TABLE_HEIGHT + CUBE_SIZE / 2)),
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

    # Kinematic goal marker. A scene entity rather than a constant so reward
    # terms can reference it and goal randomization stays easy to add.
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        # Flat, sits just proud of the table surface.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(REACH_DIST, 0.15, TABLE_HEIGHT + 0.002)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE * 1.4, CUBE_SIZE * 1.4, 0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
    )

    # Head POV camera, mounted on `neck_link` (the `head` link is merged)
    robot_pov_cam = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{REACHY2_HEAD_LINK}/RobotPOVCam",
        height=80,
        width=128,
        data_types=["rgb"],
        update_period=0,
        # Camera cfg in robot frame (won't appear visually in simulation)
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.05), rot=(0.3428, -0.6185, 0.6185, -0.3428), convention="ros"
        ),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 5.0)),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """16-DOF bimanual action space: 7 joints per arm + 1 gripper per side.

    `preserve_order=True` pins the layout to REACHY2_ARM_GRIPPER_JOINTS instead
    of the USD's DOF order. `use_default_offset=True` makes action 0 hold the
    default pose, so a freshly initialized policy starts at rest.
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

    # Undriven joints default to a zero drive target, which would flatten the head.
    hold_head = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_.*", "antenna_.*"])},
    )

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
    """reach -> grasp -> lift -> transport -> place.

    Transport is gated on the cube being lifted, so it cannot be farmed by
    sliding the cube along the table.
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
        # One camera frame per control step -- the policy consumes an image
        # every decision, so this is functional, not just a cost knob.
        self.sim.render_interval = self.decimation

        self.viewer.eye = (1.8, -1.4, 1.7)
        self.viewer.lookat = (REACH_DIST, 0.0, TABLE_HEIGHT)
