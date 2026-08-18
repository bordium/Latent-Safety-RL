#!/usr/bin/env python3
"""
Load the converted Reachy2 USD as an Isaac Lab Articulation and report its
actual joint/body inventory.

This is the gate between "the converter emitted a file" and "the robot is
usable in a task": it confirms the articulation instantiates, reports the real
DOF count and ordering, and checks that the link names the task config depends
on survived fixed-joint merging.

Usage:
    python inspect_reachy2_usd.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the converted Reachy2 USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# This script lives in reachy2_assets/reachy2_usd_scripts/. The URDF sources sit
# one level up in reachy2_assets/, and the converted USD lands in the PPO
# baseline's asset directory, which is where REACHY2_CFG loads it from.
URDF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "baselines" / "reachy2" / "assets"
USD_PATH = ASSETS_DIR / "reachy2.usd"

# Link names the pick-and-place task config will depend on. prepare_for_isaac.py
# marked the fixed joints producing these with <dont_collapse/> so they survive
# merge_fixed_joints=True.
# The frames the task config actually uses. NOTE: these are the wrist/neck
# links, not the palm/head links -- <dont_collapse/> does not survive this
# importer build, so the task uses the merged survivors instead (they are
# rigidly related by a fixed joint). See assets/reachy2.py for the rationale.
REQUIRED_BODIES = ["r_wrist_link", "l_wrist_link", "neck_link"]

# The 16 joints intended for the PPO action space (14 arm + 2 gripper).
EXPECTED_ACTION_JOINTS = [
    "l_shoulder_pitch", "l_shoulder_roll", "l_elbow_yaw", "l_elbow_pitch",
    "l_wrist_roll", "l_wrist_pitch", "l_wrist_yaw", "l_hand_finger",
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw", "r_elbow_pitch",
    "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw", "r_hand_finger",
]


REPORT_PATH = ASSETS_DIR / "inventory.txt"
_lines: list[str] = []


def emit(text: str = "") -> None:
    """Print and also buffer, so the report survives Kit's noisy stdout."""
    print(text, flush=True)
    _lines.append(text)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=0.01))

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)
    light = sim_utils.DomeLightCfg(intensity=3000.0)
    light.func("/World/Light", light)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Reachy2",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={},
    )
    robot = Articulation(robot_cfg)

    sim.reset()

    emit("\n" + "=" * 70)
    emit("REACHY2 ARTICULATION INVENTORY")
    emit("=" * 70)

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    emit(f"\nDOF count : {robot.num_joints}")
    emit(f"Body count: {robot.num_bodies}")

    emit(f"\n--- all {len(joint_names)} joints (index: name) ---")
    for i, name in enumerate(joint_names):
        marker = "  <-- ACTION" if name in EXPECTED_ACTION_JOINTS else ""
        emit(f"  {i:3d}: {name}{marker}")

    emit(f"\n--- all {len(body_names)} bodies ---")
    for i, name in enumerate(body_names):
        marker = "  <-- REQUIRED" if name in REQUIRED_BODIES else ""
        emit(f"  {i:3d}: {name}{marker}")

    emit("\n" + "=" * 70)
    emit("CHECKS")
    emit("=" * 70)

    failures = []

    for body in REQUIRED_BODIES:
        ok = body in body_names
        emit(f"  [{'OK' if ok else 'FAIL'}] body survived merging: {body}")
        if not ok:
            failures.append(f"missing body {body}")

    for joint in EXPECTED_ACTION_JOINTS:
        ok = joint in joint_names
        emit(f"  [{'OK' if ok else 'FAIL'}] action joint present: {joint}")
        if not ok:
            failures.append(f"missing joint {joint}")

    # Joint limits sanity: a joint with zero range can't be controlled.
    limits = robot.data.joint_pos_limits[0]
    degenerate = [
        joint_names[i] for i in range(len(joint_names))
        if float(limits[i, 1] - limits[i, 0]) < 1e-6
    ]
    emit(f"  [{'OK' if not degenerate else 'WARN'}] {len(degenerate)} joints with ~zero range"
          f"{': ' + ', '.join(degenerate) if degenerate else ''}")

    emit("\n" + ("FAILED: " + "; ".join(failures) if failures else "All checks passed."))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    # Kit's shutdown reliably hangs (or segfaults) on this Isaac Sim build when
    # closing a headless app -- see isaac-sim/IsaacSim#3730. The report is
    # already written to disk by this point, so skip the graceful teardown and
    # exit immediately; otherwise this script never returns and cannot be
    # chained after other commands.
    simulation_app.close()
    os._exit(code)
