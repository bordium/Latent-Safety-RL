# Latent-Safety-RL

## System Specs

This installation is intended for Ubuntu 24.04, but should also work for Ubuntu 22.04. Nvidia R580 drivers are also required.

## Environment Setup
The environment is split into two tasks: nutpouring with the GR1T2 humanoid and pick-and-place with Reachy2.

We begin by initializing isaaclab. The required driver version is R580. All commands assume your relative directory is Latent-Safety-RL/.

If you have R580 drivers, run:
```bash
cd baselines
./isaaclab_setup.sh bootstrap
```

If you do not have R580 drivers, run:
```bash
cd baselines
AUTO_INSTALL_DRIVER=1 ./isaaclab_setup.sh bootstrap
```

This will initialize an isaac/ and conda/ at Latent-Safety-RL/

Note that `ubuntu-drivers` may install R595, which Isaac Sim does not support and which crashes on startup. Check with `nvidia-smi` and install `nvidia-driver-580-open` if you get R595.

Every task below assumes the conda environment is active:
```bash
source conda/etc/profile.d/conda.sh
conda activate env_isaaclab
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Nutpouring

### PPO

Install the extension, then confirm the task registers:
```bash
cd baselines/nutpouring/ppo
./ppo_setup.sh bootstrap
python scripts/list_envs.py
```

To train, run:
```bash
python scripts/skrl/train.py --task Template-Nut-Pour-Ppo-v0 \
  --enable_cameras --headless --num_envs 32 --max_iterations 1000
```

`--enable_cameras` is required. The policy has a vision branch, and Isaac Sim will not render without it.

To watch a trained checkpoint, drop `--headless`:
```bash
python scripts/skrl/play.py --task Template-Nut-Pour-Ppo-v0 --enable_cameras \
  --checkpoint logs/skrl/nut_pour_ppo/<run>/checkpoints/best_agent.pt
```

To record the state, vision, and per-term reward vectors to an .npz, run:
```bash
python scripts/collect_state_reward_dataset.py --headless --num_steps 400 \
  --checkpoint logs/skrl/nut_pour_ppo/<run>/checkpoints/best_agent.pt
```

### GR00T

GR00T needs its own conda environment, cloned from env_isaaclab so its Torch pin does not disturb the shared base:
```bash
cd baselines/nutpouring/gr00t
./gr00t_setup.sh bootstrap
```

To evaluate NVIDIA's published checkpoint, run:
```bash
./gr00t_run_nutpouring.sh eval-smoke   # 30 cycles, 1 rollout
./gr00t_run_nutpouring.sh eval-full    # 1000 cycles, 20 rollouts
```

To fine-tune your own instead, run:
```bash
./gr00t_train_nutpouring.sh download-assets
./gr00t_train_nutpouring.sh train
```

Then evaluate it with `MODEL_PATH=<run-dir>/checkpoint-N ./gr00t_run_nutpouring.sh eval-smoke`.

## Reachy2

### Assets

The converted Reachy2 USD is committed at baselines/reachy2/assets/, so there is nothing to download or build. It is fully self-contained: the meshes are baked in at conversion time, and it references no external files.

If you want to regenerate it -- to change import-time settings such as joint drive gains, fixed-joint merging, or collision setup -- clone the mesh source beside the URDF and re-run the pipeline:
```bash
git -C reachy2_assets clone --depth 1 --filter=blob:none --sparse https://github.com/pollen-robotics/reachy2_core.git
git -C reachy2_assets/reachy2_core sparse-checkout init --cone
git -C reachy2_assets/reachy2_core sparse-checkout set reachy_description reachy_controllers/dynamixel_control/dynamixel_description

python3 reachy2_assets/reachy2_usd_scripts/prepare_for_isaac.py
python reachy2_assets/reachy2_usd_scripts/convert_reachy2_usd.py --headless --force
python reachy2_assets/reachy2_usd_scripts/inspect_reachy2_usd.py --headless
```

The inspector should report 34 DOF, 35 bodies, and all checks passed.

### PPO

Install the extension, then confirm the task registers:
```bash
cd baselines/reachy2/ppo
./reachy2_ppo_setup.sh bootstrap
python scripts/list_envs.py
```

To train, run:
```bash
python scripts/skrl/train.py --task Template-Reachy2-Pick-Place-v0 \
  --enable_cameras --headless --num_envs 32 --max_iterations 1000
```

The action space is 16 DOF: seven joints per arm plus one gripper per side. The head, antennas, and torso lift are held by their drives.

`play.py` and `collect_state_reward_dataset.py` work the same as for nutpouring, with the task name and log directory changed.

### GR00T

This baseline is a scaffold. The setup, data config, and training script are in place, but GR00T is imitation learning and there is no Reachy2 pick-and-place demonstration set, so `train` stops with instructions until one exists. See baselines/reachy2/gr00t/README.md for the required dataset layout and two known blockers.

```bash
cd baselines/reachy2/gr00t
./gr00t_setup.sh bootstrap
```
