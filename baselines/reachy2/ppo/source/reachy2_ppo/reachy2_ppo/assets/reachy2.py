# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation configuration for the Pollen Robotics Reachy2 robot.

Modeled on `GR1T2_CFG` (isaaclab_assets/robots/fourier.py) for overall shape,
with `FRANKA_PANDA_CFG` as the reference for explicit gripper gains.

Unlike the stock Isaac Lab robots, whose USDs stream from the Nucleus cloud,
Reachy2's USD is converted locally from URDF -- see
`baselines/reachy2/convert_reachy2_usd.py`. Regenerate it if the URDF changes;
the asset directory is gitignored.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# baselines/reachy2/ppo/source/reachy2_ppo/reachy2_ppo/assets/reachy2.py
#   parents[4] == baselines/reachy2/ppo  ->  ../assets/reachy2.usd
REACHY2_USD_PATH = Path(__file__).resolve().parents[4].parent / "assets" / "reachy2.usd"

##
# Joint groups
##

# The 16 joints the PPO policy drives: 7 per arm + 1 gripper finger per side.
# The gripper's follower joints (*_hand_finger_proximal / _distal / their
# _mimic duplicates) were converted from URDF <mimic> joints into independent
# joints during USD conversion -- PhysX cannot represent Reachy2's
# mimic-of-a-mimic chain -- so they are NOT in the action space and are simply
# held by their drives.
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

#: The 16-DOF bimanual action space, in a fixed order. Used with
#: ``preserve_order=True`` so the action vector layout is this list, not
#: whatever DOF order the USD happens to have.
REACHY2_ARM_GRIPPER_JOINTS = (
    REACHY2_LEFT_ARM_JOINTS + REACHY2_RIGHT_ARM_JOINTS + REACHY2_GRIPPER_JOINTS
)

#: End-effector frames.
#:
#: NOTE: these are the *wrist* links, not the palm links, and that is
#: deliberate. `reachy2/prepare_for_isaac.py` marks the palm/head fixed joints
#: <dont_collapse/>, but the importer on this Isaac Sim build (extension 2.4.30;
#: 5.1 wants 2.4.31) merges them anyway -- verified empirically by
#: `inspect_reachy2_usd.py`, which found only the 35-body merged set.
#:
#: This is not a functional loss: `{l,r}_hand_palm_link` are attached to
#: `{l,r}_wrist_link` by a *fixed* joint, so the two frames are rigidly related
#: by a constant transform. Distances measured from the wrist differ from
#: palm-relative ones by a fixed few-centimetre offset, which reward shaping
#: absorbs. Re-run the inspector after any importer upgrade -- if the palm links
#: reappear, switching these two constants back is the only change needed.
REACHY2_LEFT_EEF_LINK = "l_wrist_link"
REACHY2_RIGHT_EEF_LINK = "r_wrist_link"
REACHY2_BOTH_EEF_LINKS = [REACHY2_LEFT_EEF_LINK, REACHY2_RIGHT_EEF_LINK]

#: Head frame -- camera mount point. `head` itself is merged away (see above);
#: `neck_link` is the surviving body that absorbs it.
REACHY2_HEAD_LINK = "neck_link"


##
# Configuration
##

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
            # Off deliberately: the shoulder and upper-arm links carry NO
            # collision geometry in the source URDF, so self-collision would
            # silently fail to protect exactly the joints most likely to
            # self-intersect -- misleading rather than useful.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Arms slightly forward and bent, a neutral pose from which both
            # end-effectors can reach a table in front of the robot.
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
    actuators={
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["l_shoulder_.*", "l_elbow_.*", "l_wrist_.*"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=800.0,
            damping=40.0,
            armature=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["r_shoulder_.*", "r_elbow_.*", "r_wrist_.*"],
            effort_limit=100.0,
            velocity_limit=10.0,
            stiffness=800.0,
            damping=40.0,
            armature=0.01,
        ),
        # Compliant enough not to fling the object on contact.
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_finger.*"],
            effort_limit=50.0,
            velocity_limit=5.0,
            stiffness=200.0,
            damping=10.0,
            armature=0.001,
        ),
        # Not in the action space -- stiff purely so they hold position.
        "neck": ImplicitActuatorCfg(
            joint_names_expr=["neck_.*"],
            effort_limit=50.0,
            velocity_limit=5.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "antennas": ImplicitActuatorCfg(
            joint_names_expr=["antenna_.*"],
            effort_limit=5.0,
            velocity_limit=10.0,
            stiffness=50.0,
            damping=5.0,
        ),
        # tripod_joint is prismatic and carries the whole torso; the *_bar_*
        # joints are its four-bar linkage, formerly URDF mimics.
        "torso_lift": ImplicitActuatorCfg(
            joint_names_expr=["tripod_joint", ".*_bar_.*"],
            effort_limit=500.0,
            velocity_limit=1.0,
            stiffness=5000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Reachy2 with a fixed base, configured for bimanual tabletop manipulation.

The mobile base's wheel joints are ``fixed`` in the source URDF (there is no
rolling DOF), so the USD is converted with ``fix_base=True`` -- the robot is
pinned to the world rather than free-floating.
"""
