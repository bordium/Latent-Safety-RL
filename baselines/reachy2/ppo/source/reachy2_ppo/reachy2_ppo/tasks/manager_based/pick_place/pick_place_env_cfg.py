# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reachy2 bimanual pick-and-place: lift a cube off a table and place it at a target."""

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from reachy2_ppo.assets.reachy2 import (
    REACHY2_CFG,
    REACHY2_GRIPPER_GRIP,
    REACHY2_GRIPPER_OPEN,
    REACHY2_HEAD_LINK,
    REACHY2_LEFT_ARM_JOINTS,
    REACHY2_LEFT_FINGER_LINKS,
    REACHY2_RIGHT_ARM_JOINTS,
    REACHY2_RIGHT_FINGER_LINKS,
)

#: [left, right] fingertip link pairs -- the grasp frames the rewards aim at.
REACHY2_FINGER_LINK_PAIRS = [REACHY2_LEFT_FINGER_LINKS, REACHY2_RIGHT_FINGER_LINKS]

from . import mdp

##
# Scene
##

# Reachy2 uses ROS REP-103: +X forward, +Y left, +Z up. Props belong at +X.
TABLE_HEIGHT = 0.75
CUBE_SIZE = 0.05
#: Base to table centre, along +X.
TABLE_DIST = 0.65
#: Base to cube/target, along +X. Much beyond this and the cube leaves the arm's reach.
REACH_DIST = 0.52
#: Cube centre height at rest -- the zero point for the lift ramp.
CUBE_REST_HEIGHT = TABLE_HEIGHT + CUBE_SIZE / 2
#: Height the cube must clear to count as lifted.
LIFT_HEIGHT = CUBE_REST_HEIGHT + 0.06


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
            # Front edge at 0.35 m clears the mobile base's 0.163 m collision sphere.
            size=(0.5, 1.0, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        # Exact rest height; a 25 mm gap made it bounce and roll before the episode began.
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

    # Kinematic goal marker, as a scene entity so reward terms can reference it.
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

    # Fingertip contacts filtered against the cube, so the reward reads true grasp force.
    finger_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_hand_distal.*_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=0,
        track_air_time=False,
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
    """14 arm joints + one binary open/close per gripper."""

    arm_action = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=REACHY2_LEFT_ARM_JOINTS + REACHY2_RIGHT_ARM_JOINTS,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )
    left_gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["l_hand_finger"],
        open_command_expr={"l_hand_finger": REACHY2_GRIPPER_OPEN},
        close_command_expr={"l_hand_finger": REACHY2_GRIPPER_GRIP},
    )
    right_gripper_action = base_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["r_hand_finger"],
        open_command_expr={"r_hand_finger": REACHY2_GRIPPER_OPEN},
        close_command_expr={"r_hand_finger": REACHY2_GRIPPER_GRIP},
    )


@configclass
class PolicyCfg(ObsGroup):
    """A flat state vector plus the raw camera frame for the CNN branch."""

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
class StatePolicyCfg(ObsGroup):
    """State-only variant: the CNN duplicates privileged state at ~7x the cost."""

    state = ObsTerm(func=mdp.state_obs)

    def __post_init__(self):
        self.enable_corruption = False
        # Safe here unlike the vision group: one term, already flat.
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()


@configclass
class StateObservationsCfg:
    policy: StatePolicyCfg = StatePolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    # Undriven joints default to a zero drive target, flattening the head.
    hold_head = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_.*", "antenna_.*"])},
    )

    # Randomization so the policy can't memorize a single cube pose.
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
    """reach -> grasp -> lift -> transport -> place; later stages gated on a real lift."""

    # --- stage 1: reach ---
    reach_cube = RewTerm(
        func=mdp.eef_object_distance,
        weight=1.0,
        # std 0.1 as in Isaac Lab's lift task; 0.3 saturates before grasp precision matters.
        params={"std": 0.1, "finger_links": REACHY2_FINGER_LINK_PAIRS, "object_cfg": SceneEntityCfg("cube")},
    )

    # --- stage 2: grasp (real contact force, not aperture) ---
    grasp_cube = RewTerm(
        func=mdp.grasp_contact_force,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("finger_contacts"),
            "finger_pairs": REACHY2_FINGER_LINK_PAIRS,
            # Well under the ~25 N a real grip produces, so it latches reliably.
            "force_threshold": 0.0,
        },
    )
    # --- stage 3: lift (shaped ramp + the threshold bonus) ---
    lift_progress = RewTerm(
        func=mdp.object_lift_progress,
        weight=3.0,
        params={
            "rest_height": CUBE_REST_HEIGHT,
            "minimal_height": LIFT_HEIGHT,
            "object_cfg": SceneEntityCfg("cube"),
            "half_extent": CUBE_SIZE / 2,
        },
    )
    lift_cube = RewTerm(
        func=mdp.object_is_lifted,
        # 15.0 as in Isaac Lab's lift task: lifting should dominate the shaping before it.
        weight=15.0,
        params={"minimal_height": LIFT_HEIGHT, "object_cfg": SceneEntityCfg("cube")},
    )

    # --- stage 4: transport (gated on lift) ---
    transport_to_target = RewTerm(
        func=mdp.gated_object_target_distance,
        weight=8.0,
        params={
            "std": 0.2,
            "rest_height": CUBE_REST_HEIGHT,
            "minimal_height": LIFT_HEIGHT,
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target"),
            "half_extent": CUBE_SIZE / 2,
        },
    )

    # --- stage 5: place (sparse success) ---
    # Lift-gated: an ungated bonus is collectable by sliding the cube across the table.
    place_success = RewTerm(
        func=mdp.ObjectPlacedAfterLift,
        weight=50.0,
        params={
            "threshold": 0.05,
            "minimal_height": LIFT_HEIGHT,
            "object_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target"),
            "max_velocity": 0.1,
        },
    )

    # --- shaping penalties ---
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.0005)
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
    """Reachy2 pick-and-place, for from-scratch PPO with vision."""

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
        # One camera frame per control step; the policy consumes an image per decision.
        self.sim.render_interval = self.decimation

        self.viewer.eye = (1.8, -1.4, 1.7)
        self.viewer.lookat = (REACH_DIST, 0.0, TABLE_HEIGHT)


@configclass
class Reachy2PickPlaceStateEnvCfg(Reachy2PickPlaceEnvCfg):
    """Same task, state-only; subclassed so asset and reward fixes apply to both."""

    observations: StateObservationsCfg = StateObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Nothing consumes the camera, and rendering it would cost the speed advantage.
        self.scene.robot_pov_cam = None
