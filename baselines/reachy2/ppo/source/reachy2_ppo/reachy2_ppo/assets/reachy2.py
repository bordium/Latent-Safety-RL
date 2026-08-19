# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the Pollen Robotics Reachy2 robot.

The USD is converted locally from URDF (not streamed from Nucleus) and is
committed alongside this package. Regenerate with
`reachy2_assets/reachy2_usd_scripts/convert_reachy2_usd.py`.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# parents[4] == baselines/reachy2/ppo -> ../assets/reachy2.usd
REACHY2_USD_PATH = Path(__file__).resolve().parents[4].parent / "assets" / "reachy2.usd"

##
# Joint groups
##

# The 16 driven joints: 7 per arm + 1 gripper finger per side. The gripper's
# follower joints became independent joints at conversion time (PhysX cannot
# represent Reachy2's mimic-of-a-mimic chain), so they are not driven here.
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
REACHY2_GRIPPER_JOINTS = ["l_hand_finger", "r_hand_finger"]

#: The 16-DOF action space, in fixed order. Paired with ``preserve_order=True``
#: so the action layout is this list, not the USD's DOF order.
REACHY2_ARM_GRIPPER_JOINTS = (
    REACHY2_LEFT_ARM_JOINTS + REACHY2_RIGHT_ARM_JOINTS + REACHY2_GRIPPER_JOINTS
)

#: End-effector frames. These are the WRIST links, not the palms: the importer
#: (ext 2.4.30) merges the palm/head fixed joints despite the <dont_collapse/>
#: tags in prepare_for_isaac.py. The palms are rigidly fixed to the wrists, so
#: this is only a constant offset. Switch back if an importer upgrade fixes it.
REACHY2_LEFT_EEF_LINK = "l_wrist_link"
REACHY2_RIGHT_EEF_LINK = "r_wrist_link"
REACHY2_BOTH_EEF_LINKS = [REACHY2_LEFT_EEF_LINK, REACHY2_RIGHT_EEF_LINK]

#: Head/camera frame. `head` is merged away; `neck_link` absorbs it.
REACHY2_HEAD_LINK = "neck_link"


##
# Actuator limits -- Orbita hardware spec, NOT the URDF.
#
# The URDF declares effort=1000 N-m / velocity=100 rad/s on every arm joint.
# Those placeholders import into the USD, so failing to override them lets
# PhysX drive the arms at ~13 rad/s. Ranges resolve to the lower bound for
# velocity and the upper bound for torque.
##

#: Arm joint speed [rad/s]. Spec 5.24-7.33.
REACHY2_ARM_VELOCITY_LIMIT = 5.24
#: Shoulder/elbow torque [N-m]. Spec 12-16.5.
REACHY2_SHOULDER_ELBOW_EFFORT_LIMIT = 16.5
#: Wrist torque [N-m]. Spec 3-15.
REACHY2_WRIST_EFFORT_LIMIT = 15.0

#: Gripper torque [N-m], from the 50 N peak fingertip force over an assumed
#: 0.06 m lever arm. Speed is an estimate -- the spec gives no figure.
REACHY2_GRIPPER_EFFORT_LIMIT = 3.0
REACHY2_GRIPPER_VELOCITY_LIMIT = 3.0

# Spec puts the gripper range at [-0.175, 2.653] rad; the URDF says [0, 2.27]
# and that is baked into the USD. Fixing it needs a re-run of the converter.

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
            # Off deliberately: the shoulder/upper-arm links have no collision
            # geometry, so self-collision would miss exactly the joints most
            # likely to intersect -- misleading rather than protective.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Neutral bent-arm pose; both EEFs can reach a table in front.
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
            # Grippers open.
            ".*_hand_finger.*": 0.0,
            # Head level, antennas neutral, torso lift at the bottom.
            "neck_.*": 0.0,
            "antenna_.*": 0.0,
            "tripod_joint": 0.0,
            ".*_bar_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # Use the `_sim` fields only: implicit actuators silently discard
    # `velocity_limit` (reset to None in ImplicitActuator.__init__), leaving
    # PhysX on the USD's imported 100 rad/s. `effort_limit` is merely deprecated.
    actuators={
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
            stiffness=800.0,
            damping=40.0,
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
            stiffness=800.0,
            damping=40.0,
            armature=0.01,
        ),
        # Compliant enough not to fling the object on contact.
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_finger.*"],
            effort_limit_sim=REACHY2_GRIPPER_EFFORT_LIMIT,
            velocity_limit_sim=REACHY2_GRIPPER_VELOCITY_LIMIT,
            stiffness=200.0,
            damping=10.0,
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
        # tripod_joint is prismatic and carries the torso; *_bar_* are its
        # four-bar linkage (formerly URDF mimics).
        "torso_lift": ImplicitActuatorCfg(
            joint_names_expr=["tripod_joint", ".*_bar_.*"],
            effort_limit_sim=500.0,
            velocity_limit_sim=0.5,
            stiffness=5000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Reachy2 with a fixed base, for bimanual tabletop manipulation.

The wheel joints are ``fixed`` in the URDF (no rolling DOF), so the USD uses
``fix_base=True`` and the robot is pinned to the world.
"""
