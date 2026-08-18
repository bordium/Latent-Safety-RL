# Reachy2 baselines

Isaac Lab baselines for the Pollen Robotics Reachy2 robot on a bimanual
**pick-and-place** task (reach → grasp → lift → transport → release), chosen so
the task decomposes cleanly into a task graph.

| Baseline | Status |
|---|---|
| [`ppo/`](ppo/) | **Working end-to-end.** Trains with skrl PPO. |
| [`gr00t/`](gr00t/) | **Scaffold only** — needs demonstration data. See its README. |

## Asset pipeline

The Reachy2 USD in `assets/` is **committed**, so there is nothing to build --
`REACHY2_CFG` loads it directly. It is fully self-contained (meshes baked in at
conversion time, zero external references), which is why the 61 MB
`reachy2_core` mesh checkout is not needed to *use* the robot, only to
regenerate it.

The pipeline that produced it lives in `reachy2_assets/reachy2_usd_scripts/`:

| script | role |
|---|---|
| `prepare_for_isaac.py` | `reachy2_meshed.urdf` -> `reachy2_isaac.urdf`: adds `<dont_collapse/>`, fixes zero-effort antennas, strips `<ros2_control>`/`<gazebo>` |
| `convert_reachy2_usd.py` | URDF -> `baselines/reachy2/assets/reachy2.usd` |
| `inspect_reachy2_usd.py` | loads the USD as an Articulation and verifies DOFs/bodies; writes `assets/inventory.txt` |

To regenerate, clone the mesh source into `reachy2_assets/reachy2_core/` first (see the
repo README) and run those three in order. Without it, `prepare_for_isaac.py`
will correctly report the 9 mesh references as missing.

### What the conversion does, and why

- **`fix_base=True`** -- Reachy2's mobile-base wheel joints are `fixed` in this
  URDF (there is no rolling DOF), so a floating base has nothing to balance on.
- **`merge_fixed_joints=True`** -- effectively mandatory: 55 links have mass
  < 0.01 kg (min 1e-05), and PhysX's solver degrades badly on that mass ratio.
- **`convert_mimic_joints_to_normal_joints=True`** -- Reachy2 has 12 `<mimic>`
  joints including a *mimic-of-a-mimic* (`*_hand_finger_proximal_mimic` ->
  `*_hand_finger_proximal` -> `*_hand_finger`), a topology PhysX's Mimic Joint
  API cannot represent. Converting them to independent joints sidesteps it; the
  gripper is driven through the single `*_hand_finger` DOF and the followers
  are held by their drives. Result: 34 DOF (22 originally-actuated + 12
  former mimics).

### Not needed: orbita2d_control / orbita3d_control

The URDF also references two sibling Pollen packages, but neither reaches the
converter, so they are not vendored and not required:

- `orbita3d_control` contributes **only** `<param name="config_file">` yamls,
  which live inside the `<ros2_control>` blocks that `prepare_for_isaac.py`
  strips.
- `orbita2d_control` contributes those same stripped configs plus one mesh,
  `2D_sphere_visual.dae` -- and that appears only in `reachy2_spherized.urdf`,
  a purely cosmetic variant (it draws the four orbita ball joints as real
  meshes instead of analytic spheres; physics is byte-identical). We import
  `reachy2_meshed.urdf`, which has zero orbita mesh references.

Net: `reachy2_isaac.urdf` has **0** orbita references of either kind.

### Kit hangs on shutdown

Isaac Sim's `simulation_app.close()` hangs (or segfaults) on this build when
running headless -- see [isaac-sim/IsaacSim#3730](https://github.com/isaac-sim/IsaacSim/issues/3730).
`inspect_reachy2_usd.py` therefore writes its report to `assets/inventory.txt`
*before* closing and then calls `os._exit()`, so it terminates instead of
hanging forever. If you write another script in this directory and it never
returns after finishing its work, this is why.

### Stale processes exhaust the 8 GB GPU

Because Kit hangs on shutdown (above), a killed or interrupted run can leave a
process holding GPU memory. The next run then fails during articulation init
with a misleading error:

```
Exception: Failed to get DOF velocities from backend
```

The real cause is higher up the log: `vkAllocateMemory failed` /
`Out of GPU memory`. Check with `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv` and kill anything left over before retrying.

### Known wart: `<dont_collapse/>` is ignored on this build

`prepare_for_isaac.py` marks the fixed joints producing `{l,r}_hand_palm_link`
and `head` with `<dont_collapse/>`, intending to keep them through merging.
**It does not work on this Isaac Sim build** (importer extension 2.4.30; 5.1
wants 2.4.31) — `inspect_reachy2_usd.py` confirms only the 35-body merged set
survives. So the task uses the surviving frames instead:

| wanted | actually used |
|---|---|
| `l_hand_palm_link` / `r_hand_palm_link` | `l_wrist_link` / `r_wrist_link` |
| `head` | `neck_link` |

This is not a functional loss — the palms attach to the wrists by a *fixed*
joint, so the frames differ by a constant few-centimetre transform that reward
shaping absorbs. If an importer upgrade makes the tags take effect, flipping
the two constants in `ppo/source/reachy2_ppo/reachy2_ppo/assets/reachy2.py`
is the only change needed.

## PPO quickstart

```bash
cd ppo
./reachy2_ppo_setup.sh install-extension      # pip install -e source/reachy2_ppo
python scripts/list_envs.py                   # expect Template-Reachy2-Pick-Place-v0

# sanity checks
python scripts/zero_agent.py   --task Template-Reachy2-Pick-Place-v0 --enable_cameras --num_envs 2 --headless
python scripts/random_agent.py --task Template-Reachy2-Pick-Place-v0 --enable_cameras --num_envs 2 --headless

# train  (--enable_cameras is REQUIRED: the policy has a vision branch)
python scripts/skrl/train.py --task Template-Reachy2-Pick-Place-v0 --enable_cameras --headless \
  --num_envs 32 --max_iterations 1000
```

Action space is **16 DOF** — 7 joints per arm + 1 gripper per side, in the
order given by `REACHY2_ARM_GRIPPER_JOINTS` (`preserve_order=True`, so the
action vector layout does not depend on USD DOF ordering). Head, antennas and
torso lift are held by their drives and are not in the action space.

Observations are split into `state` (113-dim) and `vision` (160×256×3) so the
skrl config can run an MLP branch and a CNN branch and concatenate them.
