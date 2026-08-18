#!/usr/bin/env python3
"""
Convert the Reachy2 URDF to USD for Isaac Lab.

Input:  reachy2/reachy2_isaac.urdf  (produced by reachy2/prepare_for_isaac.py --
        adds <dont_collapse/> to EE/head frames, fixes zero-effort antennas,
        strips ros2_control/gazebo)
Output: baselines/reachy2/assets/reachy2.usd  (gitignored -- regenerable)

The stock IsaacLab CLI (scripts/tools/convert_urdf.py) exposes only
--merge-joints/--fix-base/--joint-stiffness/--joint-damping/--joint-target-type.
Reachy2 needs per-joint-group gains and mimic-joint conversion, so this
instantiates UrdfConverterCfg directly instead.

Usage:
    python convert_reachy2_usd.py [--headless] [--force]
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the Reachy2 URDF to USD.")
parser.add_argument("--force", action="store_true", default=False, help="Re-convert even if the USD already exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pathlib import Path

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# This script lives in reachy2_assets/reachy2_usd_scripts/. The URDF sources sit
# one level up in reachy2_assets/, and the converted USD lands in the PPO
# baseline's asset directory, which is where REACHY2_CFG loads it from.
URDF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "baselines" / "reachy2" / "assets"
SOURCE_URDF = URDF_DIR / "reachy2_isaac.urdf"
OUTPUT_DIR = ASSETS_DIR
OUTPUT_NAME = "reachy2.usd"

# Joint-drive gains per joint group. Every drive_type/target_type/stiffness/
# damping field accepts a dict keyed by a joint-name REGEX (matched with
# re.search against URDF joint names), so one config covers all groups.
#
# Note: for non-prismatic joints the converter internally scales stiffness and
# damping by pi/180 (rad->deg), so these are pre-scaling values.
#
# The arms carry the payload and need to hold pose against gravity; the
# grippers are small and want to be compliant enough not to launch the cube;
# the neck/antennas/tripod are not in the 16-DOF action space and are stiff
# purely so they hold position rather than sag.
JOINT_STIFFNESS = {
    r"^[rl]_(shoulder|elbow|wrist)_.*": 800.0,
    r".*_hand_finger.*": 200.0,
    r"^neck_.*": 400.0,
    r"^antenna_.*": 50.0,
    r"^tripod_joint$": 5000.0,  # prismatic, carries the whole torso
    r".*_bar_.*": 1000.0,       # tripod four-bar linkage (was mimic)
}
JOINT_DAMPING = {
    r"^[rl]_(shoulder|elbow|wrist)_.*": 40.0,
    r".*_hand_finger.*": 10.0,
    r"^neck_.*": 20.0,
    r"^antenna_.*": 5.0,
    r"^tripod_joint$": 200.0,
    r".*_bar_.*": 50.0,
}


def main():
    output_path = OUTPUT_DIR / OUTPUT_NAME
    if output_path.exists() and not args_cli.force:
        print(f"[INFO] USD already exists: {output_path}")
        print("[INFO] Pass --force to re-convert.")
        return

    if not SOURCE_URDF.is_file():
        raise SystemExit(
            f"ERROR: {SOURCE_URDF} not found.\n"
            "Run 'python3 reachy2_assets/reachy2_usd_scripts/prepare_for_isaac.py' first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(SOURCE_URDF),
        usd_dir=str(OUTPUT_DIR),
        usd_file_name=OUTPUT_NAME,
        force_usd_conversion=True,
        make_instanceable=True,
        # The mobile-base wheel joints are FIXED in this URDF (no rolling DOF
        # exists), so a floating base has nothing to balance on and would just
        # topple. Pinning it is correct for tabletop manipulation.
        fix_base=True,
        root_link_name="base_link",
        # Mandatory: 55 links have mass < 0.01 kg (min 1e-05) and PhysX's
        # solver degrades badly on that mass ratio if they survive as bodies.
        # prepare_for_isaac.py has already exempted the frames we need to keep.
        merge_fixed_joints=True,
        # Reachy2 has 12 mimic joints including a mimic-of-a-mimic
        # (*_hand_finger_proximal_mimic -> *_hand_finger_proximal ->
        # *_hand_finger), a topology PhysX's Mimic Joint API does not support.
        # Converting them to independent joints avoids the unsupported coupling;
        # the gripper is then driven through the single *_hand_finger DOF and
        # the follower joints are held by their drives.
        convert_mimic_joints_to_normal_joints=True,
        # Collision geometry is 85 analytic sphere primitives, zero meshes --
        # so collider_type never actually applies. Self-collision is off because
        # the shoulder/upper-arm links carry NO collision geometry at all, which
        # would make self-collision misleading rather than protective.
        collider_type="convex_hull",
        self_collision=False,
        collision_from_visuals=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=JOINT_STIFFNESS,
                damping=JOINT_DAMPING,
            ),
        ),
    )

    print(f"[INFO] Converting {SOURCE_URDF}")
    print(f"[INFO]        -> {output_path}")
    converter = UrdfConverter(cfg)
    print(f"[INFO] Generated USD: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
