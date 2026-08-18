# Reachy2 GR00T baseline — SCAFFOLD ONLY

This directory mirrors `baselines/nutpouring/gr00t/`, retargeted to a Reachy2
bimanual pick-and-place task. **It is not trainable yet.** The environment,
setup, and data-config plumbing are in place; the demonstration data is not.

## Why this is a scaffold and not a working baseline

GR00T is imitation learning, so unlike the PPO baseline (which needs only an
environment and a reward function) it needs *demonstrations*. There is no
Reachy2 pick-and-place demonstration set in the format this pipeline consumes,
so `gr00t_train_reachy2.sh train` will stop with a clear error until one exists.

Two honest caveats found while building this, both worth knowing before
investing effort here:

1. **The lab's existing Reachy2 GR00T checkpoints are probably not loadable
   here.** `erl-hub/reachy-groot*` on HuggingFace appear to be LeRobot-native
   GR00T checkpoints (no `experiment_cfg/` directory), whereas this pipeline is
   pinned to NVIDIA's Isaac-GR00T **N1** (commit `755876a`, the `n1-release`
   branch). Community Reachy2 checkpoints are generally N1.5, which will not
   load under an N1 pin. Verify before assuming a checkpoint drops in.

2. **The IsaacLabEvalTasks eval layer is hardcoded to GR1 and will silently
   produce all-zero actions on Reachy2.** `scripts/policies/joints_conversion.py`
   dispatches on `joint_name.split("_")[0]` with `case "left" | "right" | "L" |
   "R"`. Reachy2's joints are named `l_shoulder_pitch`, `r_elbow_yaw`, ... so
   every joint falls through to `continue` and the action vector stays zero --
   with no error raised. `scripts/policies/gr00t_n1_policy.py` also hardcodes
   `state.left_arm`(7)/`right_arm`(7)/`left_hand`(6)/`right_hand`(6) reshapes
   and the `video.ego_view` key, and `scripts/config/args.py::EvalTaskConfig`
   has no Reachy entry. Closed-loop sim eval needs real code changes, not config.

## What IS here

| File | Status |
|---|---|
| `deps/gr00t_common.sh` | Retargeted paths/vars; `EMBODIMENT_TAG=new_embodiment`, `DATA_CONFIG=reachy2_bimanual` |
| `gr00t_setup.sh` | Unchanged from nutpouring — clones/pins Isaac-GR00T, creates the `isaac-gr00t` conda env |
| `gr00t_train_reachy2.sh` | Retargeted; fails fast with instructions when the dataset is absent |
| `data_config/reachy2_data_config.py` | `Reachy2DataConfig` skeleton + registration helper |
| `data_config/modality.json.template` | Template describing the state/action layout |

## Where demonstrations plug in

Place a GR00T-LeRobot dataset at `$DATASET_PATH`
(default `isaac/datasets/reachy2/Reachy2-Pick-Place-task`) with this layout:

```
meta/{episodes.jsonl,modality.json,info.json,tasks.jsonl}
videos/chunk-000/observation.images.<name>/episode_000000.mp4
data/chunk-000/episode_000000.parquet
```

`meta/modality.json` is the GR00T-specific addition to stock LeRobot v2 and is
mandatory — start from `data_config/modality.json.template` and reconcile it
against the real column layout of whatever dataset you use. Then:

1. Copy your `modality.json` into `<DATASET>/meta/modality.json`.
2. Reconcile `data_config/reachy2_data_config.py` with it (the key names and
   dimensions must agree, or training fails on a shape mismatch).
3. Register the config — see the docstring in that file.
4. `./gr00t_train_reachy2.sh train`

Reference: `Isaac-GR00T/getting_started/LeRobot_compatible_data_schema.md` and
`getting_started/3_new_embodiment_finetuning.ipynb`.
