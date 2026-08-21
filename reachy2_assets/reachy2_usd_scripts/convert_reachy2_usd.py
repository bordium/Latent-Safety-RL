#!/usr/bin/env python3
"""
Convert the Reachy2 URDF to USD for Isaac Lab.

Input:  reachy2_assets/reachy2_isaac.urdf (from prepare_for_isaac.py)
Output: baselines/reachy2/assets/reachy2.usd (committed, not gitignored)

Uses UrdfConverterCfg directly.

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

# URDF sources sit one level up; the USD lands where REACHY2_CFG loads it.
URDF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "baselines" / "reachy2" / "assets"
SOURCE_URDF = URDF_DIR / "reachy2_isaac.urdf"
OUTPUT_DIR = ASSETS_DIR
OUTPUT_NAME = "reachy2.usd"

# Per-group drive gains, keyed by joint-name regex (matched with re.search).
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
        # Wheel joints are FIXED in this URDF (no rolling DOF), so a floating
        # base would just topple.
        fix_base=True,
        root_link_name="base_link",
        # Mandatory: 55 links have mass < 0.01 kg (min 1e-05) and PhysX degrades
        # badly on that mass ratio if they survive as bodies.
        merge_fixed_joints=True,
        # 12 mimic joints include a mimic-of-a-mimic chain
        # (*_hand_finger_proximal_mimic -> _proximal -> *_hand_finger) that
        # PhysX's Mimic Joint API cannot represent. Converting to independent
        # joints drives the gripper through the single *_hand_finger DOF.
        convert_mimic_joints_to_normal_joints=True,
        # collider_type never applies -- collision geometry is 85 analytic
        # spheres, no meshes. Self-collision is off because the shoulder and
        # upper-arm links have no collision geometry at all.
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
