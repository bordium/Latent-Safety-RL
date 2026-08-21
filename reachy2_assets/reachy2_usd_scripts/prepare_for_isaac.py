#!/usr/bin/env python3
"""
Derive an Isaac-Sim-import-ready URDF from reachy2_meshed.urdf.

Reads  reachy2_meshed.urdf  (mesh paths relative to reachy2_assets/)
Writes reachy2_isaac.urdf   (non-destructive; the input is never modified)

Three transformations:

1. `<dont_collapse/>` on selected fixed joints.

2. Non-zero antenna effort/velocity limits.

3. Strip <ros2_control> and <gazebo> blocks.

Usage:
    python3 prepare_for_isaac.py            # write reachy2_isaac.urdf
    python3 prepare_for_isaac.py --verify   # report on the existing output
"""

import math
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

# URDF sources sit one level up; the USD lands where REACHY2_CFG loads it.
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

# Modifications to antenna and wrist
ANTENNA_JOINTS = ("antenna_left", "antenna_right")
ANTENNA_EFFORT = 5.0
ANTENNA_VELOCITY = 10.0
WRIST_JOINTS = ("l_wrist_roll", "l_wrist_pitch", "r_wrist_roll", "r_wrist_pitch")
WRIST_LIMIT = 0.785

STRIP_TAGS = ("ros2_control", "gazebo")

# Mimic joints whose reference is itself a mimic -- PhysX cannot represent that.
FLATTEN_MIMIC = {
    "l_hand_finger_proximal_mimic": "l_hand_finger",
    "r_hand_finger_proximal_mimic": "r_hand_finger",
}


MESH_SUFFIXES = (".dae", ".stl", ".obj", ".usd")


def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """URDF rpy convention: R = Rz(yaw) . Ry(pitch) . Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_to_rpy(m: list[list[float]]) -> tuple[float, float, float]:
    """Inverse of `_rpy_to_mat`, using the non-degenerate branch (|pitch| != pi/2)."""
    pitch = math.atan2(-m[2][0], math.hypot(m[0][0], m[1][0]))
    if math.isclose(abs(pitch), math.pi / 2, abs_tol=1e-9):
        # Gimbal lock: fold the whole rotation into roll.
        return math.atan2(-m[1][2], m[1][1]), pitch, 0.0
    return math.atan2(m[2][1], m[2][2]), pitch, math.atan2(m[1][0], m[0][0])


def _axis_angle_to_mat(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
    """Rodrigues' rotation formula for a (not necessarily normalised) axis."""
    n = math.sqrt(sum(a * a for a in axis)) or 1.0
    x, y, z = (a / n for a in axis)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _floats(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.replace(",", " ").split()]
    return parts[0], parts[1], parts[2]


@lru_cache(maxsize=None)
def mesh_index() -> dict:
    """Every mesh file under URDF_DIR, grouped by basename."""
    index: dict[str, list[Path]] = {}
    for path in URDF_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in MESH_SUFFIXES:
            index.setdefault(path.name, []).append(path)
    return index


def resolve_mesh(ref: str) -> str | None:
    """Find `ref` on disk: drop leading path components, then fall back to basename."""
    if Path(ref).is_absolute():
        return ref if Path(ref).is_file() else None
    parts = Path(ref).parts
    for start in range(len(parts)):
        candidate = Path(*parts[start:])
        if (URDF_DIR / candidate).is_file():
            return str(candidate)
    # Meshes may sit under a differently-shaped tree, so match on filename alone.
    matches = mesh_index().get(Path(ref).name, [])
    if len(matches) == 1:
        return str(matches[0].relative_to(URDF_DIR))
    return None


def flatten_mimic_chains(root: ET.Element) -> list[str]:
    """Repoint mimic-of-a-mimic joints at the driven joint; the composition is exact."""
    joints = {j.get("name"): j for j in root.findall("joint")}
    flattened = []
    for name, driver in FLATTEN_MIMIC.items():
        joint = joints.get(name)
        if joint is None:
            raise SystemExit(f"ERROR: joint to flatten not found: {name}")
        mimic = joint.find("mimic")
        if mimic is None:
            raise SystemExit(f"ERROR: {name} has no <mimic> to flatten")

        inner = joints.get(mimic.get("joint"))
        inner_mimic = inner.find("mimic") if inner is not None else None
        if inner_mimic is None:
            # Already points at a driven joint -- nothing to compose.
            continue

        m2 = float(mimic.get("multiplier", 1.0))
        o2 = float(mimic.get("offset", 0.0))
        m1 = float(inner_mimic.get("multiplier", 1.0))
        o1 = float(inner_mimic.get("offset", 0.0))
        if inner_mimic.get("joint") != driver:
            raise SystemExit(
                f"ERROR: {name} resolves to {inner_mimic.get('joint')}, expected {driver}"
            )

        mimic.set("joint", driver)
        mimic.set("multiplier", repr(m2 * m1))
        mimic.set("offset", repr(m2 * o1 + o2))
        flattened.append(name)
    return flattened


def bake_mimic_offsets(root: ET.Element) -> list[str]:
    """Fold mimic offsets into joint origins; the converter drops offsets, breaking the gripper."""
    baked = []
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        offset = float(mimic.get("offset", 0.0))
        if abs(offset) < 1e-12:
            continue

        jtype = joint.get("type")
        axis_el = joint.find("axis")
        axis = _floats(axis_el.get("xyz")) if axis_el is not None else (1.0, 0.0, 0.0)

        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        xyz = _floats(origin.get("xyz", "0 0 0"))
        rpy = _floats(origin.get("rpy", "0 0 0"))
        rot = _rpy_to_mat(*rpy)

        if jtype in ("revolute", "continuous"):
            # Rotations about a common axis commute, so the constant folds cleanly.
            new_rot = _matmul(rot, _axis_angle_to_mat(axis, offset))
            origin.set("rpy", " ".join(repr(v) for v in _mat_to_rpy(new_rot)))
        elif jtype == "prismatic":
            norm = math.sqrt(sum(a * a for a in axis)) or 1.0
            unit = [a / norm for a in axis]
            shift = [sum(rot[i][k] * unit[k] * offset for k in range(3)) for i in range(3)]
            origin.set("xyz", " ".join(repr(xyz[i] + shift[i]) for i in range(3)))
        else:
            raise SystemExit(f"ERROR: cannot bake a mimic offset into a {jtype} joint: {joint.get('name')}")

        limit = joint.find("limit")
        if limit is not None:
            for bound in ("lower", "upper"):
                if limit.get(bound) is not None:
                    limit.set(bound, repr(float(limit.get(bound)) - offset))

        mimic.set("offset", "0")
        baked.append(joint.get("name"))
    return baked


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
    widened = []
    for joint in root.findall("joint"):
        name = joint.get("name")

        if name in WRIST_JOINTS:
            limit = joint.find("limit")
            if limit is not None:
                limit.set("lower", str(-WRIST_LIMIT))
                limit.set("upper", str(WRIST_LIMIT))
                widened.append(name)

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

    # Order matters: flattening can introduce an offset that baking must then remove.
    flattened = flatten_mimic_chains(root)
    baked = bake_mimic_offsets(root)

    remapped, unresolved = [], []
    for mesh in root.iter("mesh"):
        ref = mesh.get("filename")
        if not ref:
            continue
        fixed = resolve_mesh(ref)
        if fixed is None:
            unresolved.append(ref)
        elif fixed != ref:
            mesh.set("filename", fixed)
            remapped.append(ref)

    for tag, count in stripped.items():
        print(f"  stripped {count} <{tag}> block(s)")
    print(f"  marked <dont_collapse/> on {len(preserved)}: {', '.join(sorted(preserved))}")
    print(f"  fixed zero effort/velocity on {len(relimited)}: {', '.join(sorted(relimited))}")
    print(f"  widened to +-{WRIST_LIMIT} on {len(widened)}: {', '.join(sorted(widened))}")
    print(f"  flattened {len(flattened)} mimic chain(s): {', '.join(sorted(flattened))}")
    print(f"  baked {len(baked)} mimic offset(s) into joint origins: {', '.join(sorted(baked))}")
    print(f"  remapped {len(set(remapped))} mesh path(s) onto the on-disk layout")
    if unresolved:
        print(f"  WARNING: {len(set(unresolved))} mesh(es) not found on disk -- these links will have no visual:")
        for ref in sorted(set(unresolved)):
            print(f"    {ref}")

    missing = set(PRESERVE_JOINTS) - set(preserved)
    if missing:
        raise SystemExit(f"ERROR: expected joints not found in URDF: {sorted(missing)}")
    if set(ANTENNA_JOINTS) - set(relimited):
        raise SystemExit(f"ERROR: antenna joints not found: {sorted(set(ANTENNA_JOINTS) - set(relimited))}")
    if set(WRIST_JOINTS) - set(widened):
        raise SystemExit(f"ERROR: wrist joints not found: {sorted(set(WRIST_JOINTS) - set(widened))}")

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

    for name in WRIST_JOINTS:
        joint = joints.get(name)
        limit = joint.find("limit") if joint is not None else None
        ok = limit is not None and abs(float(limit.get("upper", 0)) - WRIST_LIMIT) < 1e-9
        print(f"  [{'OK' if ok else 'FAIL'}] {name} widened to +-{WRIST_LIMIT}")
        if not ok:
            failures.append(name)

    # The two invariants the gripper fix depends on, checked over every mimic joint.
    mimics = {n: j.find("mimic") for n, j in joints.items() if j.find("mimic") is not None}
    chained = sorted(n for n, m in mimics.items() if m.get("joint") in mimics)
    print(f"  [{'OK' if not chained else 'FAIL'}] no mimic references another mimic"
          f"{' (' + ', '.join(chained) + ')' if chained else ''}")
    failures.extend(chained)

    offsets = sorted(n for n, m in mimics.items() if abs(float(m.get("offset", 0.0))) > 1e-12)
    print(f"  [{'OK' if not offsets else 'FAIL'}] all {len(mimics)} mimic offsets baked to zero"
          f"{' (' + ', '.join(offsets) + ')' if offsets else ''}")
    failures.extend(offsets)

    # Mesh paths must resolve relative to reachy2
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
