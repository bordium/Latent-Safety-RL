#!/usr/bin/env bash
# Bootstrap and run the pinned Isaac Lab + IsaacLabEvalTasks + Isaac GR00T stack.
#
# Tested target:
#   Ubuntu 22.04 x86_64, Python 3.11, CUDA Toolkit 12.8
#   Isaac Sim 5.0.0, Isaac Lab v2.2.0
#   IsaacLabEvalTasks 460f2878... and its pinned Isaac-GR00T submodule
#
# The script is idempotent: rerunning it preserves existing repositories and
# environments, verifies pinned revisions, and only downloads missing assets.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"

CONDA_ENV="${CONDA_ENV:-isaac-eval}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$ISAAC_ROOT/IsaacLab}"
EVAL_REPO="${EVAL_REPO:-$ISAAC_ROOT/IsaacLabEvalTasks}"
GROOT_DIR="${GROOT_DIR:-$EVAL_REPO/submodules/Isaac-GR00T}"

ISAACLAB_URL="${ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.2.0}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-46dff135f44683f031edf346e544fcfd8456b2bb}"
EVAL_REPO_URL="${EVAL_REPO_URL:-https://github.com/isaac-sim/IsaacLabEvalTasks.git}"
EVAL_REPO_REF="${EVAL_REPO_REF:-460f2878bdcb4db2d21913db789174fb316b73e2}"
EVAL_REPO_COMMIT="${EVAL_REPO_COMMIT:-460f2878bdcb4db2d21913db789174fb316b73e2}"
GROOT_URL="${GROOT_URL:-https://github.com/NVIDIA/Isaac-GR00T.git}"
GROOT_COMMIT="${GROOT_COMMIT:-755876a9afdb41ca6eb6383b36f4a0adb085c73f}"

CUDA_VERSION="${CUDA_VERSION:-12.8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-12-8}"
MIN_DRIVER_MAJOR="${MIN_DRIVER_MAJOR:-570}"
MAX_JOBS="${MAX_JOBS:-2}"
AUTO_INSTALL_DRIVER="${AUTO_INSTALL_DRIVER:-0}"

[[ "$AUTO_INSTALL_DRIVER" =~ ^[01]$ ]] || {
  printf 'ERROR: AUTO_INSTALL_DRIVER must be 0 or 1 (received: %s)\n' \
    "$AUTO_INSTALL_DRIVER" >&2
  exit 1
}

DATASETS_ROOT="${DATASETS_ROOT:-$ISAAC_ROOT/datasets/PhysicalAI-GR00T-Tuned-Tasks}"
DATASET_PATH="${DATASET_PATH:-$DATASETS_ROOT/Nut-Pouring-task}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ISAAC_ROOT/checkpoints/GR00T-N1-2B}"
TUNED_MODEL_PATH="${TUNED_MODEL_PATH:-$ISAAC_ROOT/checkpoints/GR00T-N1-2B-tuned-Nut-Pouring-task}"
RUN_DIR="${RUN_DIR:-$ISAAC_ROOT/training_runs/nutpour_lora_smoke}"
RESULTS_DIR="${RESULTS_DIR:-$ISAAC_ROOT/eval_results}"

BASE_MODEL_REPO="${BASE_MODEL_REPO:-nvidia/GR00T-N1-2B}"
TUNED_MODEL_REPO="${TUNED_MODEL_REPO:-nvidia/GR00T-N1-2B-tuned-Nut-Pouring-task}"
DATASET_REPO="${DATASET_REPO:-nvidia/PhysicalAI-GR00T-Tuned-Tasks}"

TASK_NAME="${TASK_NAME:-nutpouring}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_FEEDBACK_ACTIONS="${NUM_FEEDBACK_ACTIONS:-16}"
SEED="${SEED:-10}"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$GROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  elif [[ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
  else
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  fi
fi

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

Fresh-computer setup:
  bootstrap                 Install system packages, NVIDIA/CUDA, Conda, repos,
                            Python dependencies, patches, and tuned checkpoint
  bootstrap-software        Same, but skip the tuned checkpoint download
  setup-system              Install/check Ubuntu packages, driver, CUDA 12.8
  setup-software            Set up Conda, repos, Python packages, and patches
  apply-training-patches    Reapply the two RTX 4060 training compatibility fixes
  download-tuned-model      Download NVIDIA's tuned Nut Pouring checkpoint
  download-training-assets  Download the base model and Nut Pouring dataset

Run and verify:
  self-test        Check this Bash file without requiring Isaac Lab
  doctor           Check OS, pinned repos, packages, CUDA, and GPU access
  sim              Launch the basic Isaac Lab GUI simulation
  sim-headless     Launch the basic Isaac Lab simulation without a GUI
  eval-smoke       Run GR00T Nut Pouring (30 inference cycles, 1 rollout)
  eval-full        Run GR00T Nut Pouring (1000 cycles, 20 rollouts)
  eval-latest      Download NVIDIA's latest tuned checkpoint (if not already
                   present) from Hugging Face and evaluate it directly,
                   bypassing any local training-run checkpoint. See
                   https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/doc/checkpoints.md
  offline-eval     Compare policy actions with the demonstration dataset
  train-smoke      Attempt one local LoRA optimizer step on GPU 0
  show-config      Print resolved paths, versions, and settings

Fresh Ubuntu example:
  chmod +x $SCRIPT_NAME
  HF_TOKEN=hf_your_read_token ./$SCRIPT_NAME bootstrap
  ./$SCRIPT_NAME doctor
  ./$SCRIPT_NAME sim-headless
  ./$SCRIPT_NAME eval-smoke

Useful overrides:
  SKIP_TUNED_MODEL=1        Do not download the tuned checkpoint in bootstrap
  FORCE_DOWNLOAD=1          Re-download the tuned checkpoint (or other Hugging
                            Face assets) even if already present locally
  AUTO_INSTALL_DRIVER=1     Allow installation of the recommended NVIDIA driver
                            (default: 0; check only, never install automatically)
  MODEL_PATH=/abs/path      Evaluate a specific checkpoint
  HEADLESS=1                Run GR00T evaluation headlessly
  ROLLOUT_LENGTH=100 MAX_NUM_ROLLOUTS=2
  ISAAC_ROOT=/abs/path CONDA_ENV=another-env

Notes:
  * A newly installed NVIDIA driver requires a reboot. Re-run bootstrap after it.
  * Hugging Face assets require a read token via HF_TOKEN on gated repositories.
  * The Nut Pouring policy is GR00T imitation learning, not PPO/SAC reward-based RL.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

info() {
  printf '\n[%s] %s\n' "$SCRIPT_NAME" "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Directory not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "File not found: $1"
}

assert_supported_host() {
  [[ "$(uname -m)" == "x86_64" ]] || die "This pinned stack expects Linux x86_64."
  [[ -f /etc/os-release ]] || die "Cannot identify the operating system."

  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] ||
    die "This installer targets Ubuntu 22.04; detected ${PRETTY_NAME:-unknown}."

  if grep -qi microsoft /proc/version 2>/dev/null; then
    die "Use native Ubuntu 22.04. Isaac Sim GUI/GR00T is not supported here through WSL."
  fi

  local glibc_version
  glibc_version="$(ldd --version | head -n 1 | awk '{print $NF}')"
  printf 'Host: %s; GLIBC: %s; architecture: %s\n' \
    "${PRETTY_NAME:-Ubuntu}" "$glibc_version" "$(uname -m)"
}

sudo_prefix() {
  if (( EUID == 0 )); then
    SUDO=()
  else
    require_command sudo
    SUDO=(sudo)
  fi
}

install_system_packages() {
  assert_supported_host
  sudo_prefix
  info "Installing Ubuntu build, graphics, and media dependencies"
  "${SUDO[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libvulkan1 \
    ninja-build \
    pciutils \
    pkg-config \
    ubuntu-drivers-common \
    unzip \
    vulkan-tools \
    wget
}

driver_is_usable() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

ensure_nvidia_driver() {
  require_command lspci
  lspci | grep -qi 'NVIDIA' || die "No NVIDIA GPU was detected by lspci."

  if driver_is_usable; then
    local driver_version driver_major
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
    driver_major="${driver_version%%.*}"
    [[ "$driver_major" =~ ^[0-9]+$ ]] || die "Could not parse NVIDIA driver version: $driver_version"
    if (( driver_major >= MIN_DRIVER_MAJOR )); then
      printf 'NVIDIA driver %s is usable (minimum major version: %s).\n' \
        "$driver_version" "$MIN_DRIVER_MAJOR"
      return
    fi
    warn "NVIDIA driver $driver_version is older than the CUDA 12.8 target."
  else
    warn "nvidia-smi is unavailable; the NVIDIA kernel driver is not usable yet."
  fi

  if [[ "$AUTO_INSTALL_DRIVER" != "1" ]]; then
    die "Install Ubuntu's recommended NVIDIA driver, reboot, and rerun this command."
  fi

  sudo_prefix
  info "Installing Ubuntu's recommended NVIDIA driver"
  "${SUDO[@]}" ubuntu-drivers install
  cat >&2 <<EOF

The NVIDIA driver was installed or upgraded. Reboot the computer, then run:
  ./$SCRIPT_NAME bootstrap

Setup intentionally stops here because the new kernel module cannot be validated
until after reboot.
EOF
  exit 42
}

cuda_128_is_installed() {
  [[ -x "$CUDA_HOME/bin/nvcc" ]] &&
    "$CUDA_HOME/bin/nvcc" --version | grep -Fq "release $CUDA_VERSION"
}

ensure_cuda_toolkit() {
  if cuda_128_is_installed; then
    printf 'CUDA Toolkit %s found at %s.\n' "$CUDA_VERSION" "$CUDA_HOME"
    return
  fi

  sudo_prefix
  info "Installing NVIDIA CUDA Toolkit $CUDA_VERSION without replacing the driver"
  local keyring_deb
  keyring_deb="$(mktemp /tmp/cuda-keyring.XXXXXX.deb)"
  curl -fsSL \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -o "$keyring_deb"
  "${SUDO[@]}" dpkg -i "$keyring_deb"
  rm -f "$keyring_deb"
  "${SUDO[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y "$CUDA_TOOLKIT_PACKAGE"
  cuda_128_is_installed || die "CUDA Toolkit $CUDA_VERSION installation could not be verified at $CUDA_HOME."
}

find_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
  elif [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    printf '%s\n' "$CONDA_ROOT"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    printf '%s\n' "$HOME/anaconda3"
  else
    return 1
  fi
}

ensure_conda() {
  local conda_base=""
  conda_base="$(find_conda_base 2>/dev/null || true)"
  if [[ -z "$conda_base" ]]; then
    [[ ! -e "$CONDA_ROOT" ]] || die "$CONDA_ROOT exists but does not contain a usable Conda installation."
    info "Installing Miniconda under $CONDA_ROOT"
    local installer
    installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
    curl -fsSL "$MINICONDA_URL" -o "$installer"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
    conda_base="$CONDA_ROOT"
  fi

  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV"; then
    info "Creating Conda environment $CONDA_ENV with Python 3.11"
    conda create -y -n "$CONDA_ENV" --override-channels -c conda-forge python=3.11 pip
  fi
  conda activate "$CONDA_ENV"
  [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] ||
    die "Conda environment $CONDA_ENV does not use Python 3.11."
}

activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return
  fi
  local conda_base=""
  conda_base="$(find_conda_base 2>/dev/null || true)"
  [[ -n "$conda_base" ]] || die "Conda was not found. Run '$SCRIPT_NAME setup-software' first."
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

repo_is_dirty() {
  [[ -n "$(git -C "$1" status --porcelain --untracked-files=normal)" ]]
}

clone_and_pin() {
  local url="$1" destination="$2" ref="$3" commit="$4" label="$5"

  if [[ ! -d "$destination/.git" ]]; then
    [[ ! -e "$destination" || -z "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] ||
      die "$destination exists and is not an empty Git repository."
    info "Cloning $label"
    git clone --filter=blob:none --no-checkout "$url" "$destination"
  fi

  local current
  current="$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current" != "$commit" ]]; then
    if repo_is_dirty "$destination"; then
      die "$label has local changes at $destination; preserve them before switching to $commit."
    fi
    info "Pinning $label to $ref"
    git -C "$destination" fetch --depth 1 origin "$ref"
    git -C "$destination" checkout --detach "$commit"
  fi

  current="$(git -C "$destination" rev-parse HEAD)"
  [[ "$current" == "$commit" ]] || die "$label resolved to $current instead of $commit."
  printf '%s: %s\n' "$label" "$current"
}

setup_repositories() {
  require_command git
  mkdir -p "$ISAAC_ROOT"
  clone_and_pin "$ISAACLAB_URL" "$ISAACLAB_DIR" "$ISAACLAB_REF" "$ISAACLAB_COMMIT" "Isaac Lab"
  clone_and_pin "$EVAL_REPO_URL" "$EVAL_REPO" "$EVAL_REPO_REF" "$EVAL_REPO_COMMIT" "IsaacLabEvalTasks"

  if [[ -e "$GROOT_DIR/.git" ]]; then
    local groot_current
    groot_current="$(git -C "$GROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$groot_current" != "$GROOT_COMMIT" ]] && repo_is_dirty "$GROOT_DIR"; then
      die "Isaac-GR00T has local changes; preserve them before switching revisions."
    fi
  fi

  info "Cloning/updating the Isaac-GR00T submodule over HTTPS"
  git -C "$EVAL_REPO" config submodule.submodules/Isaac-GR00T.url "$GROOT_URL"
  git -C "$EVAL_REPO" \
    -c url."https://github.com/".insteadOf="git@github.com:" \
    submodule update --init --recursive

  local groot_current
  groot_current="$(git -C "$GROOT_DIR" rev-parse HEAD)"
  [[ "$groot_current" == "$GROOT_COMMIT" ]] ||
    die "Isaac-GR00T resolved to $groot_current instead of $GROOT_COMMIT."
  printf 'Isaac-GR00T: %s\n' "$groot_current"
}

install_python_stack() {
  activate_conda
  require_dir "$ISAACLAB_DIR"
  require_dir "$GROOT_DIR"
  require_dir "$EVAL_REPO"
  cuda_128_is_installed || die "CUDA Toolkit $CUDA_VERSION is required to build FlashAttention."

  info "Installing Isaac Sim 5.0.0 and Isaac Lab v2.2.0"
  python -m pip install --upgrade pip wheel ninja "setuptools==80.9.0"
  python -m pip install \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
  python -m pip install \
    "isaacsim[all,extscache]==5.0.0" \
    --extra-index-url https://pypi.nvidia.com
  (
    cd "$ISAACLAB_DIR"
    ./isaaclab.sh --install none
  )

  info "Installing the pinned Isaac-GR00T stack"
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -e "$GROOT_DIR[base]"
  MAX_JOBS="$MAX_JOBS" python -m pip install \
    --no-build-isolation "flash-attn==2.7.1.post4"

  # These pins are part of Isaac Lab v2.2.0 and also fix the import failures
  # encountered on the working RTX 4060 installation.
  python -m pip install \
    "setuptools==80.9.0" \
    "flatdict==4.0.1" \
    "prettytable==3.3.0" \
    "hidapi==0.14.0.post2" \
    "lxml>=5.2.2" \
    "trimesh>=4.4.0"
  python -m pip install -e "$EVAL_REPO/source/isaaclab_eval_tasks"

  info "Installed core package versions"
  python - <<'PY'
import torch
import torchvision
import transformers

print("Torch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)
print("Transformers:", transformers.__version__)
PY

  if ! python -m pip check; then
    warn "pip reports the expected Isaac Sim/Isaac Lab versus GR00T Torch-version conflicts."
    warn "IsaacLabEvalTasks documents these conflicts for its GR00T integration."
  fi
}

apply_training_patches() {
  activate_conda
  require_dir "$GROOT_DIR"
  info "Applying the 8 GB GPU BF16 and FP32 Beta-sampling compatibility patches"
  PATCH_GROOT_DIR="$GROOT_DIR" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["PATCH_GROOT_DIR"])
train_script = root / "scripts/gr00t_finetune.py"
flow_head = root / "gr00t/model/action_head/flow_matching_action_head.py"

train_text = train_script.read_text()
if "torch_dtype=torch.bfloat16" not in train_text:
    needle = "        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT\n    )"
    replacement = (
        "        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT\n"
        "        torch_dtype=torch.bfloat16,\n"
        "    )"
    )
    if needle not in train_text:
        raise SystemExit("Could not locate GR00T model-loading block; refusing an unsafe patch.")
    train_script.write_text(train_text.replace(needle, replacement, 1))
    print("Applied BF16 model-loading patch.")
else:
    print("BF16 model-loading patch already present.")

flow_text = flow_head.read_text()
if "fp32_beta_dist" not in flow_text:
    needle = "        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)"
    replacement = "\n".join(
        [
            "        fp32_beta_dist = torch.distributions.Beta(",
            "            self.beta_dist.concentration1.float(),",
            "            self.beta_dist.concentration0.float(),",
            "        )",
            "        sample = fp32_beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)",
        ]
    )
    if needle not in flow_text:
        raise SystemExit("Could not locate GR00T Beta sampling line; refusing an unsafe patch.")
    flow_head.write_text(flow_text.replace(needle, replacement, 1))
    print("Applied FP32 Beta-sampling patch.")
else:
    print("FP32 Beta-sampling patch already present.")

compile(train_script.read_text(), str(train_script), "exec")
compile(flow_head.read_text(), str(flow_head), "exec")
print("Patched Python files compile successfully.")
PY
}

download_hf_snapshot() {
  local repo_id="$1" repo_type="$2" destination="$3" marker="$4"
  activate_conda
  local incomplete_downloads=()
  if [[ -d "$destination/.cache/huggingface/download" ]]; then
    mapfile -t incomplete_downloads < <(
      find "$destination/.cache/huggingface/download" -name '*.incomplete' -print
    )
  fi
  if [[ -e "$destination/$marker" && ${#incomplete_downloads[@]} -eq 0 &&
        "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
    printf 'Hugging Face asset already present: %s\n' "$destination"
    return
  fi
  python -c "import huggingface_hub" >/dev/null 2>&1 ||
    python -m pip install -q "huggingface_hub[cli]"
  mkdir -p "$destination"
  if [[ ${#incomplete_downloads[@]} -gt 0 ]]; then
    info "Resuming ${#incomplete_downloads[@]} interrupted file(s) at $destination"
  else
    info "Downloading $repo_id to $destination"
  fi
  HF_REPO_ID="$repo_id" HF_REPO_TYPE="$repo_type" HF_DESTINATION="$destination" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type=os.environ["HF_REPO_TYPE"],
    local_dir=os.environ["HF_DESTINATION"],
    token=os.environ.get("HF_TOKEN"),
)
PY
}

download_tuned_model() {
  download_hf_snapshot "$TUNED_MODEL_REPO" model "$TUNED_MODEL_PATH" config.json
}

download_training_assets() {
  download_hf_snapshot "$BASE_MODEL_REPO" model "$BASE_MODEL_PATH" config.json
  download_hf_snapshot "$DATASET_REPO" dataset "$DATASETS_ROOT" Nut-Pouring-task/meta/info.json
}

setup_system() {
  install_system_packages
  ensure_nvidia_driver
  ensure_cuda_toolkit
  info "System prerequisites passed"
}

setup_software() {
  assert_supported_host
  ensure_nvidia_driver
  ensure_cuda_toolkit
  ensure_conda
  setup_repositories
  install_python_stack
  apply_training_patches
  mkdir -p "$ISAAC_ROOT/checkpoints" "$ISAAC_ROOT/training_runs" "$RESULTS_DIR" "$DATASETS_ROOT"
  info "Software setup completed"
}

bootstrap() {
  local include_model="$1"
  setup_system
  setup_software
  if [[ "$include_model" == "1" && "${SKIP_TUNED_MODEL:-0}" != "1" ]]; then
    download_tuned_model
  fi
  doctor
  info "Bootstrap completed"
}

show_config() {
  cat <<EOF
CONDA_ENV=$CONDA_ENV
CONDA_ROOT=$CONDA_ROOT
ISAAC_ROOT=$ISAAC_ROOT
ISAACLAB_DIR=$ISAACLAB_DIR
ISAACLAB_REF=$ISAACLAB_REF
ISAACLAB_COMMIT=$ISAACLAB_COMMIT
EVAL_REPO=$EVAL_REPO
EVAL_REPO_COMMIT=$EVAL_REPO_COMMIT
GROOT_DIR=$GROOT_DIR
GROOT_COMMIT=$GROOT_COMMIT
CUDA_HOME=$CUDA_HOME
CUDA_VERSION=$CUDA_VERSION
AUTO_INSTALL_DRIVER=$AUTO_INSTALL_DRIVER
DATASET_PATH=$DATASET_PATH
BASE_MODEL_PATH=$BASE_MODEL_PATH
TUNED_MODEL_PATH=$TUNED_MODEL_PATH
RUN_DIR=$RUN_DIR
RESULTS_DIR=$RESULTS_DIR
MODEL_PATH=${MODEL_PATH:-<auto: latest RUN_DIR checkpoint, then TUNED_MODEL_PATH>}
TASK_NAME=$TASK_NAME
NUM_ENVS=$NUM_ENVS
NUM_FEEDBACK_ACTIONS=$NUM_FEEDBACK_ACTIONS
SEED=$SEED
VK_ICD_FILENAMES=$VK_ICD_FILENAMES
EOF
}

resolve_model_path() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    require_dir "$MODEL_PATH"
    printf '%s\n' "$MODEL_PATH"
    return
  fi

  local latest_checkpoint=""
  if [[ -d "$RUN_DIR" ]]; then
    latest_checkpoint="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
  fi

  if [[ -n "$latest_checkpoint" ]]; then
    printf '%s\n' "$latest_checkpoint"
  elif [[ -d "$TUNED_MODEL_PATH" ]]; then
    printf '%s\n' "$TUNED_MODEL_PATH"
  else
    die "No model found. Run '$SCRIPT_NAME download-tuned-model' or set MODEL_PATH."
  fi
}

check_repo_commit() {
  local directory="$1" expected="$2" label="$3"
  if [[ ! -d "$directory/.git" && ! -f "$directory/.git" ]]; then
    printf 'MISSING repository: %s (%s)\n' "$label" "$directory" >&2
    return 1
  fi
  local actual
  actual="$(git -C "$directory" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'WRONG revision: %s is %s; expected %s\n' "$label" "$actual" "$expected" >&2
    return 1
  fi
  printf 'OK revision: %s %s\n' "$label" "$actual"
}

doctor() {
  local failures=0
  assert_supported_host || failures=$((failures + 1))
  activate_conda

  check_repo_commit "$ISAACLAB_DIR" "$ISAACLAB_COMMIT" "Isaac Lab" || failures=$((failures + 1))
  check_repo_commit "$EVAL_REPO" "$EVAL_REPO_COMMIT" "IsaacLabEvalTasks" || failures=$((failures + 1))
  check_repo_commit "$GROOT_DIR" "$GROOT_COMMIT" "Isaac-GR00T" || failures=$((failures + 1))

  require_file "$ISAACLAB_DIR/isaaclab.sh"
  require_file "$EVAL_REPO/scripts/evaluate_gn1.py"

  printf '\nNVIDIA driver and CUDA Toolkit:\n'
  if driver_is_usable; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
  else
    printf 'FAILED: nvidia-smi cannot access the NVIDIA GPU.\n' >&2
    failures=$((failures + 1))
  fi
  if cuda_128_is_installed; then
    "$CUDA_HOME/bin/nvcc" --version | tail -n 1
  else
    printf 'FAILED: CUDA Toolkit %s was not found at %s.\n' "$CUDA_VERSION" "$CUDA_HOME" >&2
    failures=$((failures + 1))
  fi
  if [[ -f "$VK_ICD_FILENAMES" ]]; then
    printf 'OK Vulkan ICD: %s\n' "$VK_ICD_FILENAMES"
  else
    warn "NVIDIA Vulkan ICD not found at $VK_ICD_FILENAMES"
  fi

  printf '\nPython and CUDA imports:\n'
  if ! python - <<'PY'
import importlib
import sys

modules = (
    "torch",
    "pinocchio",
    "pink",
    "isaaclab",
    "isaaclab_eval_tasks",
    "gr00t",
    "flash_attn",
)
failed = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f"OK import: {name}")
    except Exception as exc:
        failed.append(name)
        print(f"FAILED import: {name}: {exc}", file=sys.stderr)

import torch
print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"Torch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

raise SystemExit(bool(failed) or not torch.cuda.is_available())
PY
  then
    failures=$((failures + 1))
  fi

  local train_script="$GROOT_DIR/scripts/gr00t_finetune.py"
  local flow_head="$GROOT_DIR/gr00t/model/action_head/flow_matching_action_head.py"
  if grep -Eq 'torch_dtype[[:space:]]*=[[:space:]]*torch\.bfloat16' "$train_script"; then
    printf 'OK training patch: BF16 model loading\n'
  else
    printf 'FAILED training patch: BF16 model loading\n' >&2
    failures=$((failures + 1))
  fi
  if grep -q 'fp32_beta_dist' "$flow_head"; then
    printf 'OK training patch: FP32 Beta sampling\n'
  else
    printf 'FAILED training patch: FP32 Beta sampling\n' >&2
    failures=$((failures + 1))
  fi

  printf '\nData and model assets:\n'
  if [[ -f "$DATASET_PATH/meta/info.json" && -f "$DATASET_PATH/meta/modality.json" ]]; then
    printf 'OK training dataset: %s\n' "$DATASET_PATH"
  else
    warn "Training dataset not downloaded (not required for simulation evaluation)."
  fi
  local detected_model=""
  detected_model="$(resolve_model_path 2>/dev/null || true)"
  if [[ -n "$detected_model" ]]; then
    printf 'OK evaluation model: %s\n' "$detected_model"
  else
    warn "No evaluation model found; download one before eval-smoke."
  fi

  if (( failures > 0 )); then
    die "$failures required check(s) failed."
  fi
  printf '\nAll required checks passed.\n'
}

run_sim() {
  local headless="$1"
  activate_conda
  require_file "$ISAACLAB_DIR/isaaclab.sh"
  require_file "$ISAACLAB_DIR/scripts/tutorials/00_sim/create_empty.py"
  cd "$ISAACLAB_DIR"
  if [[ "$headless" == "1" ]]; then
    exec ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
  else
    exec ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
  fi
}

run_eval() {
  local default_rollout_length="$1" default_rollouts="$2"
  activate_conda
  require_file "$EVAL_REPO/scripts/evaluate_gn1.py"

  local model_path rollout_length max_rollouts timestamp eval_file
  model_path="$(resolve_model_path)"
  rollout_length="${ROLLOUT_LENGTH:-$default_rollout_length}"
  max_rollouts="${MAX_NUM_ROLLOUTS:-$default_rollouts}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$RESULTS_DIR"
  eval_file="${EVAL_FILE_PATH:-$RESULTS_DIR/${TASK_NAME}_${timestamp}.json}"

  printf 'Model: %s\nResult: %s\n' "$model_path" "$eval_file"
  printf 'Rollout length: %s; rollouts: %s; environments: %s\n' \
    "$rollout_length" "$max_rollouts" "$NUM_ENVS"

  cd "$EVAL_REPO"
  local args=(
    python scripts/evaluate_gn1.py
    --num_feedback_actions "$NUM_FEEDBACK_ACTIONS"
    --num_envs "$NUM_ENVS"
    --task_name "$TASK_NAME"
    --eval_file_path "$eval_file"
    --model_path "$model_path"
    --rollout_length "$rollout_length"
    --seed "$SEED"
    --max_num_rollouts "$max_rollouts"
  )
  if [[ "${HEADLESS:-0}" == "1" ]]; then
    args+=(--headless)
  fi
  "${args[@]}"

  if [[ -f "$eval_file" ]]; then
    printf '\nEvaluation result:\n'
    python -m json.tool "$eval_file" || true
  fi
}

eval_latest_checkpoint() {
  local default_rollout_length="$1" default_rollouts="$2"
  download_tuned_model
  local MODEL_PATH="$TUNED_MODEL_PATH"
  run_eval "$default_rollout_length" "$default_rollouts"
}

offline_eval() {
  activate_conda
  require_file "$GROOT_DIR/scripts/eval_policy.py"
  require_dir "$DATASET_PATH"
  local model_path
  model_path="$(resolve_model_path)"
  cd "$GROOT_DIR"
  python scripts/eval_policy.py \
    --plot \
    --model_path "$model_path" \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag gr1 \
    --data_config gr1_arms_only \
    --video_backend torchvision_av
}

pick_flag() {
  local help_text="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if grep -Fq -- "$candidate" <<<"$help_text"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

train_smoke() {
  activate_conda
  require_file "$GROOT_DIR/scripts/gr00t_finetune.py"
  require_dir "$DATASET_PATH"
  require_dir "$BASE_MODEL_PATH"
  mkdir -p "$RUN_DIR"

  cd "$GROOT_DIR"
  local help_text
  help_text="$(python scripts/gr00t_finetune.py --help 2>&1 || true)"

  local dataset_flag output_flag data_config_flag embodiment_flag base_model_flag
  local gpu_flag batch_flag steps_flag save_flag workers_flag video_flag
  local lora_rank_flag lora_alpha_flag report_flag no_resume_flag
  dataset_flag="$(pick_flag "$help_text" --dataset-path --dataset_path)" || die "Cannot detect dataset option."
  output_flag="$(pick_flag "$help_text" --output-dir --output_dir)" || die "Cannot detect output option."
  data_config_flag="$(pick_flag "$help_text" --data-config --data_config)" || die "Cannot detect data config option."
  embodiment_flag="$(pick_flag "$help_text" --embodiment-tag --embodiment_tag)" || die "Cannot detect embodiment option."
  base_model_flag="$(pick_flag "$help_text" --base-model-path --base_model_path)" || die "Cannot detect base model option."
  gpu_flag="$(pick_flag "$help_text" --num-gpus --num_gpus)" || die "Cannot detect GPU option."
  batch_flag="$(pick_flag "$help_text" --batch-size --batch_size)" || die "Cannot detect batch option."
  steps_flag="$(pick_flag "$help_text" --max-steps --max_steps)" || die "Cannot detect max steps option."
  save_flag="$(pick_flag "$help_text" --save-steps --save_steps)" || die "Cannot detect save option."
  workers_flag="$(pick_flag "$help_text" --dataloader-num-workers --dataloader_num_workers)" || die "Cannot detect worker option."
  video_flag="$(pick_flag "$help_text" --video-backend --video_backend)" || die "Cannot detect video option."
  lora_rank_flag="$(pick_flag "$help_text" --lora-rank --lora_rank)" || die "Cannot detect LoRA rank option."
  lora_alpha_flag="$(pick_flag "$help_text" --lora-alpha --lora_alpha)" || die "Cannot detect LoRA alpha option."
  report_flag="$(pick_flag "$help_text" --report-to --report_to)" || die "Cannot detect reporting option."
  no_resume_flag="$(pick_flag "$help_text" --no-resume --no_resume)" || die "Cannot detect resume option."

  printf 'Attempting one optimizer step on GPU 0. Isaac Sim is not used during training.\n'
  CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    "$dataset_flag=$DATASET_PATH" \
    "$output_flag=$RUN_DIR" \
    "$data_config_flag=gr1_arms_only" \
    "$embodiment_flag=gr1" \
    "$base_model_flag=$BASE_MODEL_PATH" \
    "$gpu_flag=1" \
    "$batch_flag=1" \
    "$steps_flag=1" \
    "$save_flag=1" \
    "$workers_flag=1" \
    "$video_flag=torchvision_av" \
    "$lora_rank_flag=4" \
    "$lora_alpha_flag=8" \
    "$report_flag=none" \
    "$no_resume_flag" \
    2>&1 | tee "$RUN_DIR/smoke.log"
}

self_test() {
  bash -n "$SCRIPT_PATH"
  [[ "$ISAACLAB_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$EVAL_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$GROOT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$CUDA_VERSION" == "12.8" ]]
  [[ "$AUTO_INSTALL_DRIVER" =~ ^[01]$ ]]
  printf 'Bash syntax and pinned-configuration checks passed.\n'
  if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$SCRIPT_PATH"
    printf 'ShellCheck passed.\n'
  else
    printf 'ShellCheck is not installed; static lint was skipped.\n'
  fi
}

main() {
  case "${1:-}" in
    bootstrap) bootstrap 1 ;;
    bootstrap-software) bootstrap 0 ;;
    setup-system) setup_system ;;
    setup-software) setup_software ;;
    apply-training-patches) apply_training_patches ;;
    download-tuned-model) download_tuned_model ;;
    download-training-assets) download_training_assets ;;
    doctor) doctor ;;
    sim) run_sim 0 ;;
    sim-headless) run_sim 1 ;;
    eval-smoke) run_eval 30 1 ;;
    eval-full) run_eval 1000 20 ;;
    eval-latest) eval_latest_checkpoint 30 1 ;;
    offline-eval) offline_eval ;;
    train-smoke) train_smoke ;;
    show-config) show_config ;;
    self-test) self_test ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
