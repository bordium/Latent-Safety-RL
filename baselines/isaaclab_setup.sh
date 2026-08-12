#!/usr/bin/env bash
# Install and pin Isaac Sim + Isaac Lab, standalone -- no dependency on any
# particular baseline (gr00t, ppo, ...). Baselines that need Isaac Lab should
# document this script as a prerequisite and layer their own Python packages
# into the same Conda environment afterward.
#
# Tested target:
#   Ubuntu 22.04/24.04 x86_64, Python 3.11, CUDA Toolkit 12.8
#   Isaac Sim 5.0.0, Isaac Lab v2.2.0
#
# Idempotent: rerunning preserves the existing repository and environment,
# verifies the pinned revision, and only installs what's missing.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$SCRIPT_DIR")"

CONDA_ENV="${CONDA_ENV:-isaac-eval}"
CONDA_ROOT="${CONDA_ROOT:-$PROJECT_ROOT/conda}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

ISAAC_ROOT="${ISAAC_ROOT:-$PROJECT_ROOT/isaac}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$ISAAC_ROOT/IsaacLab}"

ISAACLAB_URL="${ISAACLAB_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.2.0}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-46dff135f44683f031edf346e544fcfd8456b2bb}"

CUDA_VERSION="${CUDA_VERSION:-12.8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-12-8}"
MIN_DRIVER_MAJOR="${MIN_DRIVER_MAJOR:-570}"
AUTO_INSTALL_DRIVER="${AUTO_INSTALL_DRIVER:-0}"

WARP_FIX_VERSION="${WARP_FIX_VERSION:-1.8.2}"
WARP_FIX_SOURCE_PACKAGE="${WARP_FIX_SOURCE_PACKAGE:-isaacsim-extscache-kit==5.1.0.0}"

[[ "$AUTO_INSTALL_DRIVER" =~ ^[01]$ ]] || {
  printf 'ERROR: AUTO_INSTALL_DRIVER must be 0 or 1 (received: %s)\n' \
    "$AUTO_INSTALL_DRIVER" >&2
  exit 1
}

# NVIDIA's CUDA apt repo is keyed by Ubuntu release (e.g. "ubuntu2204",
# "ubuntu2404"); resolve it once here rather than hardcoding it.
_os_release_version_id=""
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  _os_release_version_id="$(. /etc/os-release && printf '%s' "$VERSION_ID")"
fi
case "$_os_release_version_id" in
  22.04) CUDA_APT_REPO_SUFFIX="${CUDA_APT_REPO_SUFFIX:-ubuntu2204}" ;;
  24.04) CUDA_APT_REPO_SUFFIX="${CUDA_APT_REPO_SUFFIX:-ubuntu2404}" ;;
  *) CUDA_APT_REPO_SUFFIX="${CUDA_APT_REPO_SUFFIX:-ubuntu2204}" ;;
esac
unset _os_release_version_id

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
  bootstrap           Install system packages, NVIDIA/CUDA, Conda, Isaac Lab,
                      and the Isaac Sim Python stack
  setup-system        Install/check Ubuntu packages, driver, CUDA 12.8
  setup-software       Set up Conda, clone/pin Isaac Lab, install Isaac Sim
  setup-repositories   Just clone/pin Isaac Lab (no Conda/Python install)
  apply-warp-patch     Reapply the omni.warp/omni.warp.core R580+ driver fix

Verify:
  doctor        Check OS, pinned repo, packages, CUDA, and GPU access
  self-test     Check this Bash file without requiring Isaac Lab
  show-config   Print resolved paths, versions, and settings

Next steps once setup passes, from a baseline directory, e.g.:
  ./gr00t/gr00t_setup.sh bootstrap

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
    which default to directories at the repo root.
  * Baselines (gr00t, ppo, ...) share this same Conda environment and Isaac Lab
    checkout; they layer their own Python packages on top.
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
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04|ubuntu:24.04) ;;
    *) die "This installer targets Ubuntu 22.04 or 24.04; detected ${PRETTY_NAME:-unknown}." ;;
  esac

  if grep -qi microsoft /proc/version 2>/dev/null; then
    die "Use native Ubuntu 22.04/24.04. Isaac Sim GUI is not supported here through WSL."
  fi
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

find_conda_base() {
  [[ -x "$CONDA_ROOT/bin/conda" ]] && printf '%s\n' "$CONDA_ROOT"
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

activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return
  fi
  local conda_base=""
  conda_base="$(find_conda_base 2>/dev/null || true)"
  [[ -n "$conda_base" ]] || die "Conda was not found at $CONDA_ROOT. Run './$SCRIPT_NAME setup-software' first."
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  # Isaac Sim's bundled extensions need a newer libstdc++ (CXXABI_1.3.15+)
  # than Ubuntu 22.04 ships. Conda's own copy has it; put it ahead of the
  # system one so the linker finds it first.
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
}

install_isaac_python_stack() {
  activate_conda
  require_dir "$ISAACLAB_DIR"
  cuda_128_is_installed || die "CUDA Toolkit $CUDA_VERSION is required."

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
  # wheels this stack installs everywhere else.
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

  info "Installed core package versions"
  python - <<'PY'
import torch
import torchvision

print("Torch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)
PY
}

# Isaac Sim 5.0.0 bundles the omni.warp/omni.warp.core Kit extension at
# version 1.7.1, whose CUDA driver-entry-point lookup for cuDeviceGetUuid
# fails against NVIDIA R580+ drivers ("Warp CUDA error 36: API call is not
# supported in the installed CUDA driver"). Fixed upstream in Warp v1.8.1;
# NVIDIA has not backported it into an Isaac Sim 5.0.0 point release, and no
# standalone extension package exists outside of a full Isaac Sim release --
# see https://github.com/isaac-sim/IsaacLab/issues/3477. Isaac Sim 5.1.0
# bundles the fixed 1.8.2 build, so pull just those two extension folders out
# of its extscache wheel and drop them in over the broken 1.7.1 ones, leaving
# everything else (Isaac Sim itself, Isaac Lab) at their tested pins.
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
  install_isaac_python_stack
  patch_warp_extension
  info "Isaac Lab software setup completed"
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
CUDA_HOME=$CUDA_HOME
CUDA_VERSION=$CUDA_VERSION
WARP_FIX_VERSION=$WARP_FIX_VERSION
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
  require_file "$ISAACLAB_DIR/isaaclab.sh"

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

modules = ("torch", "pinocchio", "pink", "isaaclab", "isaacsim")
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

  if (( failures > 0 )); then
    die "$failures required check(s) failed."
  fi
  printf '\nAll required checks passed.\n'
}

self_test() {
  bash -n "$SCRIPT_PATH"
  [[ "$ISAACLAB_COMMIT" =~ ^[0-9a-f]{40}$ ]]
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
    bootstrap) bootstrap ;;
    setup-system) setup_system ;;
    setup-software) setup_software ;;
    setup-repositories) mkdir -p "$ISAAC_ROOT" && setup_repositories ;;
    apply-warp-patch) patch_warp_extension ;;
    doctor) doctor ;;
    show-config) show_config ;;
    self-test) self_test ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
