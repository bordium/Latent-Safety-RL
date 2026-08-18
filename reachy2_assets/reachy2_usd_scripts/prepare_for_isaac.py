#!/usr/bin/env python3
"""
Derive an Isaac-Sim-import-ready URDF from reachy2_meshed.urdf.

Reads  reachy2_meshed.urdf  (its mesh paths are relative to reachy2/)
Writes reachy2_isaac.urdf   (non-destructive; the input is never modified)

Three transformations, each addressing a verified Isaac Sim import problem:

1. `<dont_collapse/>` on selected fixed joints.
   The converter runs with merge_fixed_joints=True (mandatory here: 55 links
   have mass < 0.01 kg and PhysX chokes on that mass ratio if they survive).
   But merging collapses 100 bodies to 35 and destroys the link names the task
   config needs -- r_hand_palm_link/l_hand_palm_link would be absorbed into
   {r,l}_wrist_link, and `head` into neck_link. This tag (understood by the
   importer, see IsaacLab/docs/source/how-to/import_new_asset.rst) exempts
   specific fixed joints from merging so their child links survive as prims.

2. Non-zero antenna effort/velocity limits.
   The URDF declares effort="0.0" velocity="0.0" on antenna_left/antenna_right,
   which means PhysX can apply no torque at all and the antennas flop around.

3. Strip <ros2_control> and <gazebo> blocks.
   Confirmed silently ignored by the importer (its tinyxml2 walker whitelists
   known tags), so this is cosmetic rather than required -- but it matches Isaac
   Lab's own documented guidance and keeps the file readable.

Usage:
    python3 prepare_for_isaac.py            # write reachy2_isaac.urdf
    python3 prepare_for_isaac.py --verify   # report on the existing output
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# This script lives in reachy2_assets/reachy2_usd_scripts/. The URDF sources sit
# one level up in reachy2_assets/, and the converted USD lands in the PPO
# baseline's asset directory, which is where REACHY2_CFG loads it from.
URDF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "baselines" / "reachy2" / "assets"
SOURCE_URDF = URDF_DIR / "reachy2_meshed.urdf"
OUTPUT_URDF = URDF_DIR / "reachy2_isaac.urdf"

# Fixed joints whose child link must survive fixed-joint merging.
# Verified against the URDF: each is type="fixed" with the named child.
PRESERVE_JOINTS = {
    "r_hand_palm": "r_hand_palm_link",  # right end-effector frame
    "l_hand_palm": "l_hand_palm_link",  # left end-effector frame
    "neck_fixed": "head",               # head frame -- camera mount point
    "r_tip_joint": "r_arm_tip",         # SRDF-declared right gripper tip
    "l_tip_joint": "l_arm_tip",         # SRDF-declared left gripper tip
}

# Antennas: URDF ships effort=0 velocity=0, which makes them uncontrollable.
ANTENNA_JOINTS = ("antenna_left", "antenna_right")
ANTENNA_EFFORT = 5.0
ANTENNA_VELOCITY = 10.0

STRIP_TAGS = ("ros2_control", "gazebo")


def prepare() -> ET.ElementTree:
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()

    stripped = {tag: 0 for tag in STRIP_TAGS}
    for tag in STRIP_TAGS:
        for element in root.findall(tag):
            root.remove(element)
            stripped[tag] += 1

    preserved = []
    relimited = []
    for joint in root.findall("joint"):
        name = joint.get("name")

        if name in PRESERVE_JOINTS:
            if joint.find("dont_collapse") is None:
                joint.append(ET.Element("dont_collapse"))
            preserved.append(name)

        if name in ANTENNA_JOINTS:
            limit = joint.find("limit")
            if limit is not None:
                limit.set("effort", str(ANTENNA_EFFORT))
                limit.set("velocity", str(ANTENNA_VELOCITY))
                relimited.append(name)

    for tag, count in stripped.items():
        print(f"  stripped {count} <{tag}> block(s)")
    print(f"  marked <dont_collapse/> on {len(preserved)}: {', '.join(sorted(preserved))}")
    print(f"  fixed zero effort/velocity on {len(relimited)}: {', '.join(sorted(relimited))}")

    missing = set(PRESERVE_JOINTS) - set(preserved)
    if missing:
        raise SystemExit(f"ERROR: expected joints not found in URDF: {sorted(missing)}")
    if set(ANTENNA_JOINTS) - set(relimited):
        raise SystemExit(f"ERROR: antenna joints not found: {sorted(set(ANTENNA_JOINTS) - set(relimited))}")

    return tree


def verify(path: Path) -> int:
    """Re-parse the output and assert every transformation actually landed."""
    if not path.is_file():
        print(f"FAIL: {path} does not exist")
        return 1

    root = ET.parse(path).getroot()
    failures = []

    for tag in STRIP_TAGS:
        found = len(root.findall(tag))
        status = "OK" if found == 0 else "FAIL"
        print(f"  [{status}] no <{tag}> blocks remain (found {found})")
        if found:
            failures.append(tag)

    joints = {j.get("name"): j for j in root.findall("joint")}
    for name, child in PRESERVE_JOINTS.items():
        joint = joints.get(name)
        ok = joint is not None and joint.find("dont_collapse") is not None
        print(f"  [{'OK' if ok else 'FAIL'}] {name} -> {child} marked dont_collapse")
        if not ok:
            failures.append(name)

    for name in ANTENNA_JOINTS:
        joint = joints.get(name)
        limit = joint.find("limit") if joint is not None else None
        ok = limit is not None and float(limit.get("effort", 0)) > 0 and float(limit.get("velocity", 0)) > 0
        print(f"  [{'OK' if ok else 'FAIL'}] {name} has non-zero effort/velocity")
        if not ok:
            failures.append(name)

    # Mesh paths must still resolve -- this file feeds straight into the converter.
    # Mesh refs are relative to reachy2/, so resolve them against URDF_DIR
    # rather than the current working directory.
    meshes = {m.get("filename") for m in root.iter("mesh")}
    broken = sorted(
        m for m in meshes
        if m and not (Path(m) if Path(m).is_absolute() else URDF_DIR / m).is_file()
    )
    print(f"  [{'OK' if not broken else 'FAIL'}] {len(meshes)} unique mesh refs resolve ({len(broken)} broken)")
    failures.extend(broken)

    print(f"  links={len(root.findall('link'))} joints={len(root.findall('joint'))}")
    return 1 if failures else 0


def main():
    if "--verify" in sys.argv:
        raise SystemExit(verify(OUTPUT_URDF))

    print(f"=== preparing {SOURCE_URDF.name} -> {OUTPUT_URDF.name} ===")
    tree = prepare()
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUTPUT_URDF}")

    print("=== verifying ===")
    raise SystemExit(verify(OUTPUT_URDF))


if __name__ == "__main__":
    main()
