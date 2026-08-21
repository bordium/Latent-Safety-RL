# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation config for the Reachy2. Regenerate the USD with reachy2_usd_scripts/."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# parents[4] == baselines/reachy2/ppo -> ../assets/reachy2.usd
REACHY2_USD_PATH = Path(__file__).resolve().parents[4].parent / "assets" / "reachy2.usd"

##
# Joint groups
##

REACHY2_LEFT_ARM_JOINTS = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_elbow_yaw",
    "l_elbow_pitch",
    "l_wrist_roll",
    "l_wrist_pitch",
    "l_wrist_yaw",
]
REACHY2_RIGHT_ARM_JOINTS = [
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_elbow_yaw",
    "r_elbow_pitch",
    "r_wrist_roll",
    "r_wrist_pitch",
    "r_wrist_yaw",
]

#: Gripper travel is the reverse of the joint's name: 0.0 is fully CLOSED, 2.27 OPEN.
REACHY2_GRIPPER_OPEN = 2.27
#: Reset aperture -- open enough to admit the cube.
REACHY2_GRIPPER_DEFAULT = 1.9
#: Aperture that retains the cube through a lift (25 N); 0.8 and above drops it.
REACHY2_GRIPPER_GRIP = 0.4

#: Fingertip links per hand; their midpoint is the true grasp point.
REACHY2_LEFT_FINGER_LINKS = ["l_hand_distal_link", "l_hand_distal_mimic_link"]
REACHY2_RIGHT_FINGER_LINKS = ["r_hand_distal_link", "r_hand_distal_mimic_link"]
#: Outermost collision sphere centre in each distal link's frame.
REACHY2_FINGERTIP_OFFSET = (0.0, 0.0, 0.01)

#: Follower-to-driver mimic multiplier; offsets are baked into the joint origins.
REACHY2_FINGER_MIMIC = 0.4689
#: Reset poses must satisfy the mimic relation or the constraint snaps on reset.
REACHY2_FINGER_PROXIMAL_DEFAULT = -REACHY2_FINGER_MIMIC * REACHY2_GRIPPER_DEFAULT
REACHY2_FINGER_DISTAL_DEFAULT = REACHY2_FINGER_MIMIC * REACHY2_GRIPPER_DEFAULT

#: Hand frames. These sit at the KNUCKLE, not between the fingers -- use the fingertip
#: links above as the grasp frame.
REACHY2_LEFT_EEF_LINK = "l_hand_virtual_gripper_link"
REACHY2_RIGHT_EEF_LINK = "r_hand_virtual_gripper_link"

#: Head/camera frame. `head` is merged away; `neck_link` absorbs it.
REACHY2_HEAD_LINK = "neck_link"

##
# Orbita hardware spec.
##

#: Arm joint speed [rad/s]. Spec 5.24-7.33.
REACHY2_ARM_VELOCITY_LIMIT = 5.24
#: Shoulder/elbow torque [N-m]. Spec 12-16.5.
REACHY2_SHOULDER_ELBOW_EFFORT_LIMIT = 16.5
#: Wrist torque [N-m]. Spec 3-15.
REACHY2_WRIST_EFFORT_LIMIT = 15.0
#: Gripper torque [N-m], from a 50 N fingertip force over a 0.06 m lever arm.
REACHY2_GRIPPER_EFFORT_LIMIT = 3.0
REACHY2_GRIPPER_VELOCITY_LIMIT = 3.0

REACHY2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(REACHY2_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Off deliberately: the upper-arm links have no collision geometry at all.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Neutral bent-arm pose; both hands can reach a table in front.
            "l_shoulder_pitch": 0.0,
            "l_shoulder_roll": 0.2,
            "l_elbow_yaw": 0.0,
            "l_elbow_pitch": -1.0,
            "l_wrist_roll": 0.0,
            "l_wrist_pitch": 0.0,
            "l_wrist_yaw": 0.0,
            "r_shoulder_pitch": 0.0,
            "r_shoulder_roll": -0.2,
            "r_elbow_yaw": 0.0,
            "r_elbow_pitch": -1.0,
            "r_wrist_roll": 0.0,
            "r_wrist_pitch": 0.0,
            "r_wrist_yaw": 0.0,
            # Grippers open -- 0.0 would be fully CLOSED.
            ".*_hand_finger": REACHY2_GRIPPER_DEFAULT,
            ".*_hand_finger_proximal.*": REACHY2_FINGER_PROXIMAL_DEFAULT,
            ".*_hand_finger_distal.*": REACHY2_FINGER_DISTAL_DEFAULT,
            "neck_.*": 0.0,
            "antenna_.*": 0.0,
            "tripod_joint": 0.0,
            ".*_bar_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # Use the `_sim` limit fields only: implicit actuators discard `velocity_limit`.
    actuators={
        # 800 matches the USD; 50 sagged 8.1 deg (~7 cm), more than the cube is wide.
        "left_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_.*", "l_elbow_.*"],
            effort_limit_sim=REACHY2_SHOULDER_ELBOW_EFFORT_LIMIT,
            velocity_limit_sim=REACHY2_ARM_VELOCITY_LIMIT,
            stiffness=800.0,
            damping=40.0,
            armature=0.01,
        ),
        "left_wrist": ImplicitActuatorCfg(
            joint_names_expr=["l_wrist_.*"],
            effort_limit_sim=REACHY2_WRIST_EFFORT_LIMIT,
            velocity_limit_sim=REACHY2_ARM_VELOCITY_LIMIT,
            stiffness=300.0,
            damping=15.0,
            armature=0.01,
        ),
        "right_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["r_shoulder_.*", "r_elbow_.*"],
            effort_limit_sim=REACHY2_SHOULDER_ELBOW_EFFORT_LIMIT,
            velocity_limit_sim=REACHY2_ARM_VELOCITY_LIMIT,
            stiffness=800.0,
            damping=40.0,
            armature=0.01,
        ),
        "right_wrist": ImplicitActuatorCfg(
            joint_names_expr=["r_wrist_.*"],
            effort_limit_sim=REACHY2_WRIST_EFFORT_LIMIT,
            velocity_limit_sim=REACHY2_ARM_VELOCITY_LIMIT,
            stiffness=300.0,
            damping=15.0,
            armature=0.01,
        ),
        # Driven joint only -- driving the mimic followers would fight the constraint.
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_finger"],
            effort_limit_sim=REACHY2_GRIPPER_EFFORT_LIMIT,
            velocity_limit_sim=REACHY2_GRIPPER_VELOCITY_LIMIT,
            stiffness=200.0,
            damping=10.0,
            armature=0.001,
        ),
        # Mimic-driven, so slack; the group exists only to cover every DOF.
        "gripper_followers": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_finger_(proximal|distal).*"],
            stiffness=0.0,
            damping=0.0,
            armature=0.001,
        ),
        # Not in the action space -- stiff purely so they hold position.
        "neck": ImplicitActuatorCfg(
            joint_names_expr=["neck_.*"],
            effort_limit_sim=10.0,
            velocity_limit_sim=3.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "antennas": ImplicitActuatorCfg(
            joint_names_expr=["antenna_.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=5.0,
            stiffness=50.0,
            damping=5.0,
        ),
        "torso_lift": ImplicitActuatorCfg(
            joint_names_expr=["tripod_joint"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.5,
            stiffness=5000.0,
            damping=200.0,
        ),
        # Mimic-coupled four-bar; driving it fights the constraint.
        "torso_bars": ImplicitActuatorCfg(
            joint_names_expr=[".*_bar_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Reachy2 with a fixed base (the URDF wheel joints are fixed), for tabletop work."""
