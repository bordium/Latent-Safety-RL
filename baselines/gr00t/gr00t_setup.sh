#!/usr/bin/env bash
# Install and pin the Isaac Lab + IsaacLabEvalTasks + Isaac GR00T stack.
#
# Tested target:
#   Ubuntu 22.04/24.04 x86_64, Python 3.11, CUDA Toolkit 12.8
#   Isaac Sim 5.0.0, Isaac Lab v2.2.0
#   IsaacLabEvalTasks 460f2878... and its pinned Isaac-GR00T submodule
#
# This script only installs the environment (system packages, driver, CUDA,
# Conda, repos, Python dependencies, compatibility patches). It does not
# download any checkpoints or datasets -- use gr00t_run_nutpouring.sh for the
# published Nut Pouring checkpoint, or gr00t_train_nutpouring.sh to fine-tune
# your own.
#
# Idempotent: rerunning preserves existing repositories and environments,
# verifies pinned revisions, and only installs what's missing.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# shellcheck source=./gr00t_common.sh
source "$SCRIPT_DIR/gr00t_common.sh"

MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

ISAACLAB_URL="${ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.2.0}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-46dff135f44683f031edf346e544fcfd8456b2bb}"
EVAL_REPO_URL="${EVAL_REPO_URL:-https://github.com/isaac-sim/IsaacLabEvalTasks.git}"
EVAL_REPO_REF="${EVAL_REPO_REF:-460f2878bdcb4db2d21913db789174fb316b73e2}"
EVAL_REPO_COMMIT="${EVAL_REPO_COMMIT:-460f2878bdcb4db2d21913db789174fb316b73e2}"
GROOT_URL="${GROOT_URL:-https://github.com/NVIDIA/Isaac-GR00T.git}"
GROOT_COMMIT="${GROOT_COMMIT:-755876a9afdb41ca6eb6383b36f4a0adb085c73f}"

CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-12-8}"
MIN_DRIVER_MAJOR="${MIN_DRIVER_MAJOR:-570}"
AUTO_INSTALL_DRIVER="${AUTO_INSTALL_DRIVER:-0}"

[[ "$AUTO_INSTALL_DRIVER" =~ ^[01]$ ]] || {
  printf 'ERROR: AUTO_INSTALL_DRIVER must be 0 or 1 (received: %s)\n' \
    "$AUTO_INSTALL_DRIVER" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

Fresh-computer setup:
  bootstrap                 Install system packages, NVIDIA/CUDA, Conda, repos,
                            Python dependencies, and compatibility patches
  setup-system              Install/check Ubuntu packages, driver, CUDA 12.8
  setup-software             Set up Conda, repos, Python packages, and patches
  setup-repositories         Just clone/pin Isaac Lab, IsaacLabEvalTasks, and
                            the GR00T submodule (no Conda/Python install)
  apply-training-patches    Reapply the two RTX 4060 training compatibility fixes
  apply-warp-patch          Reapply the omni.warp/omni.warp.core R580+ driver fix

Verify:
  doctor            Check OS, pinned repos, packages, CUDA, and GPU access
  self-test         Check this Bash file without requiring Isaac Lab
  show-config       Print resolved paths, versions, and settings

Next steps once setup passes:
  ./gr00t_run_nutpouring.sh    Download and run the published Nut Pouring checkpoint
  ./gr00t_train_nutpouring.sh  Fine-tune your own Nut Pouring checkpoint

Fresh Ubuntu example:
  chmod +x $SCRIPT_NAME
  ./$SCRIPT_NAME bootstrap
  ./$SCRIPT_NAME doctor

Useful overrides:
  AUTO_INSTALL_DRIVER=1     Allow installation of the recommended NVIDIA driver
                            (default: 0; check only, never install automatically)
  ISAAC_ROOT=/abs/path CONDA_ROOT=/abs/path CONDA_ENV=another-env
                            (default: project-local ./isaac and ./conda)

Notes:
  * A newly installed NVIDIA driver requires a reboot. Re-run bootstrap after it.
  * Everything installed by this script lives under \$ISAAC_ROOT and \$CONDA_ROOT,
    which default to directories next to this script.
EOF
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
  local pci_devices
  pci_devices="$(lspci)"
  grep -qi 'NVIDIA' <<<"$pci_devices" || die "No NVIDIA GPU was detected by lspci."

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
    "https://developer.download.nvidia.com/compute/cuda/repos/$CUDA_APT_REPO_SUFFIX/x86_64/cuda-keyring_1.1-1_all.deb" \
    -o "$keyring_deb"
  "${SUDO[@]}" dpkg -i "$keyring_deb"
  rm -f "$keyring_deb"
  "${SUDO[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y "$CUDA_TOOLKIT_PACKAGE"
  cuda_128_is_installed || die "CUDA Toolkit $CUDA_VERSION installation could not be verified at $CUDA_HOME."
}

ensure_conda() {
  if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    [[ ! -e "$CONDA_ROOT" ]] || die "$CONDA_ROOT exists but does not contain a usable Conda installation."
    info "Installing Miniconda under $CONDA_ROOT"
    local installer
    installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
    curl -fsSL "$MINICONDA_URL" -o "$installer"
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
  fi

  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV"; then
    info "Creating Conda environment $CONDA_ENV with Python 3.11"
    conda create -y -n "$CONDA_ENV" --override-channels -c conda-forge python=3.11 pip
  fi
  conda activate "$CONDA_ENV"
  [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] ||
    die "Conda environment $CONDA_ENV does not use Python 3.11."
}

repo_is_dirty() {
  [[ -n "$(git -C "$1" status --porcelain --untracked-files=normal)" ]]
}

clone_and_pin() {
  local url="$1" destination="$2" ref="$3" commit="$4" label="$5"
  local freshly_cloned=0

  if [[ ! -d "$destination/.git" ]]; then
    [[ ! -e "$destination" || -z "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] ||
      die "$destination exists and is not an empty Git repository."
    info "Cloning $label"
    git clone --filter=blob:none --no-checkout "$url" "$destination"
    freshly_cloned=1
  fi

  local current worktree_populated=1
  current="$(git -C "$destination" rev-parse HEAD 2>/dev/null || true)"
  [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 ! -name .git -print -quit 2>/dev/null)" ]] ||
    worktree_populated=0

  # A --no-checkout clone can leave HEAD already equal to $commit (e.g. when
  # $ref is the tip of the default branch) despite having checked out no
  # files at all -- guard on the working tree, not just the commit hash.
  if [[ "$current" != "$commit" || "$worktree_populated" != "1" ]]; then
    # A clone we just made ourselves has an empty index against a populated
    # HEAD, which `git status` reports as every tracked file being deleted.
    # That's the normal --no-checkout state, not real local changes, so only
    # run the dirty check against a repo that already existed beforehand.
    if [[ "$freshly_cloned" != "1" && "$current" != "$commit" ]] && repo_is_dirty "$destination"; then
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
  # Isaac Lab's setup.py depends on unpinned "warp-lang"; pin it here to the
  # last release before warp-lang 1.16.0 started requiring a CUDA 13 pip
  # stack, which conflicts with the CUDA 12.8 toolkit/torch cu124/cu128
  # wheels this script installs everywhere else.
  python -m pip install "warp-lang==1.15.0"
  (
    cd "$ISAACLAB_DIR"
    ./isaaclab.sh --install none
  )
  # isaaclab.sh installs each source/* extension via `find -exec`, which
  # swallows a failed `pip install` for any one of them without propagating
  # it back here -- verify the core package actually landed.
  python -c "import isaaclab" ||
    die "isaaclab.sh --install did not install the core 'isaaclab' package; rerun setup-software."

  info "Installing the pinned Isaac-GR00T stack"
  # GR00T's own [base] extras pins torch==2.5.1, which predates Blackwell
  # (sm_120) kernel support entirely -- torch.cuda ops fail outright on an
  # RTX 50-series GPU under that build. Install [base] first for GR00T's
  # other pinned deps, then overwrite torch/torchvision/flash-attn with the
  # same cu128-era build Isaac Sim uses; GR00T's model code and this exact
  # checkpoint format run fine on it despite the newer version.
  python -m pip install -e "$GROOT_DIR[base]"
  python -m pip install \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
  python -m pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

  # These pins are part of Isaac Lab v2.2.0 and also fix the import failures
  # encountered on the working RTX 4060 installation.
  python -m pip install \
    "setuptools==80.9.0" \
    "prettytable==3.3.0" \
    "hidapi==0.14.0.post2" \
    "lxml>=5.2.2" \
    "trimesh>=4.4.0"
  # flatdict's setup.py needs pkg_resources, which newer setuptools no longer
  # ships; --no-build-isolation makes its build use the pin above instead of
  # pip fetching the latest setuptools into an isolated build env.
  python -m pip install --no-build-isolation "flatdict==4.0.1"
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

WARP_FIX_VERSION="${WARP_FIX_VERSION:-1.8.2}"
WARP_FIX_SOURCE_PACKAGE="${WARP_FIX_SOURCE_PACKAGE:-isaacsim-extscache-kit==5.1.0.0}"

# Isaac Sim 5.0.0 bundles the omni.warp/omni.warp.core Kit extension at
# version 1.7.1, whose CUDA driver-entry-point lookup for cuDeviceGetUuid
# fails against NVIDIA R580+ drivers ("Warp CUDA error 36: API call is not
# supported in the installed CUDA driver"). Fixed upstream in Warp v1.8.1;
# NVIDIA has not backported it into an Isaac Sim 5.0.0 point release, and no
# standalone extension package exists outside of a full Isaac Sim release --
# see https://github.com/isaac-sim/IsaacLab/issues/3477. Isaac Sim 5.1.0
# bundles the fixed 1.8.2 build, so pull just those two extension folders out
# of its extscache wheel and drop them in over the broken 1.7.1 ones, leaving
# everything else (IsaacLab, IsaacLabEvalTasks, Isaac Sim itself) at their
# tested pins.
patch_warp_extension() {
  activate_conda
  local extscache
  extscache="$(python -c 'import isaacsim, os; print(os.path.dirname(isaacsim.__file__))')/extscache"
  require_dir "$extscache"

  if [[ -d "$extscache/omni.warp-$WARP_FIX_VERSION" &&
        -d "$extscache/omni.warp.core-$WARP_FIX_VERSION+lx64" ]]; then
    printf 'omni.warp/omni.warp.core %s already installed at %s.\n' "$WARP_FIX_VERSION" "$extscache"
    return
  fi

  info "Replacing the broken omni.warp/omni.warp.core 1.7.1 Kit extension with $WARP_FIX_VERSION"
  local tmp
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" RETURN

  python -m pip download --no-deps -d "$tmp" "$WARP_FIX_SOURCE_PACKAGE" \
    --extra-index-url https://pypi.nvidia.com
  local whl
  whl="$(find "$tmp" -maxdepth 1 -name '*.whl' -print -quit)"
  [[ -n "$whl" ]] || die "Failed to download $WARP_FIX_SOURCE_PACKAGE."

  (
    cd "$tmp"
    unzip -q "$whl" \
      "isaacsim/extscache/omni.warp-$WARP_FIX_VERSION/*" \
      "isaacsim/extscache/omni.warp.core-$WARP_FIX_VERSION+lx64/*"
  )
  require_dir "$tmp/isaacsim/extscache/omni.warp-$WARP_FIX_VERSION"
  require_dir "$tmp/isaacsim/extscache/omni.warp.core-$WARP_FIX_VERSION+lx64"

  local old
  for old in "omni.warp-1.7.1" "omni.warp.core-1.7.1+lx64"; do
    if [[ -e "$extscache/$old" && ! -e "$extscache/$old.bak" ]]; then
      mv "$extscache/$old" "$extscache/$old.bak"
    fi
  done

  cp -r "$tmp/isaacsim/extscache/omni.warp-$WARP_FIX_VERSION" "$extscache/"
  cp -r "$tmp/isaacsim/extscache/omni.warp.core-$WARP_FIX_VERSION+lx64" "$extscache/"
  info "Installed omni.warp/omni.warp.core $WARP_FIX_VERSION at $extscache"
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
  patch_warp_extension
  apply_training_patches
  mkdir -p "$ISAAC_ROOT/checkpoints" "$ISAAC_ROOT/training_runs" "$RESULTS_DIR" "$DATASETS_ROOT"
  info "Software setup completed"
}

bootstrap() {
  setup_system
  setup_software
  doctor
  info "Bootstrap completed"
}

show_config() {
  cat <<EOF
PROJECT_ROOT=$PROJECT_ROOT
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
VK_ICD_FILENAMES=$VK_ICD_FILENAMES
EOF
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

  local extscache
  extscache="$(python -c 'import isaacsim, os; print(os.path.dirname(isaacsim.__file__))' 2>/dev/null)/extscache"
  if [[ -d "$extscache/omni.warp-$WARP_FIX_VERSION" &&
        -d "$extscache/omni.warp.core-$WARP_FIX_VERSION+lx64" ]]; then
    printf 'OK patch: omni.warp/omni.warp.core %s (R580+ driver fix)\n' "$WARP_FIX_VERSION"
  else
    printf 'FAILED patch: omni.warp/omni.warp.core %s not found at %s\n' "$WARP_FIX_VERSION" "$extscache" >&2
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
    warn "Training dataset not downloaded. Run './gr00t_train_nutpouring.sh download-assets' if you plan to train."
  fi
  if [[ -d "$TUNED_MODEL_PATH" ]]; then
    printf 'OK published checkpoint: %s\n' "$TUNED_MODEL_PATH"
  else
    warn "Published checkpoint not downloaded. Run './gr00t_run_nutpouring.sh download-checkpoint' if you plan to evaluate it."
  fi

  if (( failures > 0 )); then
    die "$failures required check(s) failed."
  fi
  printf '\nAll required checks passed.\n'
}

self_test() {
  bash -n "$SCRIPT_PATH"
  bash -n "$SCRIPT_DIR/gr00t_common.sh"
  [[ "$ISAACLAB_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$EVAL_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$GROOT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$CUDA_VERSION" == "12.8" ]]
  [[ "$AUTO_INSTALL_DRIVER" =~ ^[01]$ ]]
  printf 'Bash syntax and pinned-configuration checks passed.\n'
  if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$SCRIPT_PATH" "$SCRIPT_DIR/gr00t_common.sh"
    printf 'ShellCheck passed.\n'
  else
    printf 'ShellCheck is not installed; static lint was skipped.\n'
  fi
}

main() {
  case "${1:-}" in
    bootstrap) bootstrap ;;
    setup-system) setup_system ;;
    setup-software) setup_software ;;
    setup-repositories) mkdir -p "$ISAAC_ROOT" && setup_repositories ;;
    apply-training-patches) apply_training_patches ;;
    apply-warp-patch) patch_warp_extension ;;
    doctor) doctor ;;
    show-config) show_config ;;
    self-test) self_test ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
