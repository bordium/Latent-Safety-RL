#!/usr/bin/env bash
# Install a generic, reusable Isaac Lab base environment -- no dependency on
# any particular baseline (gr00t, ppo, ...). Baselines that need Isaac Lab
# should document this script as a prerequisite; ones with no conflicting
# package pins can activate this Conda environment directly, and ones that
# need different pins (e.g. gr00t needs a different Torch build) should clone
# it into their own environment rather than installing into this one.
#
# Tested target:
#   Ubuntu 22.04/24.04 x86_64, Python 3.11, CUDA Toolkit 12.8
#   isaaclab[isaacsim,all] 2.3.2.post1 (pip), Torch 2.7.0/cu128
#
# Idempotent: rerunning preserves the existing environment and only installs
# what's missing.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$SCRIPT_DIR")"

# The base environment baselines share; each baseline that needs its own
# package pins (e.g. gr00t) clones this rather than installing into it
# directly, so it stays reusable and unmodified.
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
CONDA_ROOT="${CONDA_ROOT:-$PROJECT_ROOT/conda}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"

ISAACLAB_VERSION="${ISAACLAB_VERSION:-2.3.2.post1}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
RL_GAMES_GIT="${RL_GAMES_GIT:-git+https://github.com/isaac-sim/rl_games.git@python3.11}"

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

# The first time `isaacsim` is imported, it interactively prompts to accept
# the NVIDIA Omniverse License Agreement, which hangs a non-interactive
# script. Running this script is an implicit agreement to that EULA
# (https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

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
  bootstrap        Install system packages, NVIDIA/CUDA, Conda, and the
                   Isaac Lab/Isaac Sim Python stack
  setup-system     Install/check Ubuntu packages, driver, CUDA 12.8
  setup-software    Set up Conda and install the Isaac Lab Python stack
  apply-warp-patch  Reapply the omni.warp/omni.warp.core R580+ driver fix

Verify:
  doctor        Check OS, packages, CUDA, and GPU access
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
  CONDA_ROOT=/abs/path CONDA_ENV=another-env
                            (default: project-local ./conda and env_isaaclab)
  ISAACLAB_VERSION=2.3.2.post1 TORCH_VERSION=2.7.0 TORCHVISION_VERSION=0.22.0

Notes:
  * A newly installed NVIDIA driver requires a reboot. Re-run bootstrap after it.
  * Everything installed by this script lives under \$CONDA_ROOT, which
    defaults to a directory at the repo root.
  * Baselines with no conflicting package pins can activate $CONDA_ENV
    directly; ones with conflicting pins (e.g. gr00t, which needs a
    different Torch build) should \`conda create --clone $CONDA_ENV\` into
    their own environment, so installing them never mutates this shared base.
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
    conda create -y -n "$CONDA_ENV" python=3.11
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
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV" ||
    die "Conda environment '$CONDA_ENV' not found. Run './$SCRIPT_NAME setup-software' first."
  conda activate "$CONDA_ENV"
  # Isaac Sim's bundled extensions need a newer libstdc++ (CXXABI_1.3.15+)
  # than Ubuntu 22.04 ships. Conda's own copy has it; put it ahead of the
  # system one so the linker finds it first.
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

install_isaac_python_stack() {
  activate_conda
  cuda_128_is_installed || die "CUDA Toolkit $CUDA_VERSION is required."

  info "Installing Isaac Lab $ISAACLAB_VERSION (bundles Isaac Sim) and rl_games"
  python -m pip install --upgrade pip
  python -m pip install \
    "isaaclab[isaacsim,all]==$ISAACLAB_VERSION" \
    --extra-index-url https://pypi.nvidia.com
  # isaaclab[isaacsim,all] may pull in a different Torch build than the one
  # this stack is validated against; reassert it explicitly.
  python -m pip install -U \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX_URL"
  python -m pip install "$RL_GAMES_GIT"

  info "Installed core package versions"
  python - <<'PY'
import torch
import torchvision
import isaaclab

print("Torch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)
print("Isaac Lab:", getattr(isaaclab, "__version__", "unknown"))
PY
}

# Isaac Sim bundles the omni.warp/omni.warp.core Kit extension at version
# 1.7.1 in some releases, whose CUDA driver-entry-point lookup for
# cuDeviceGetUuid fails against NVIDIA R580+ drivers ("Warp CUDA error 36:
# API call is not supported in the installed CUDA driver"). Fixed upstream in
# Warp v1.8.1; not every Isaac Sim point release bundles the fix, and no
# standalone extension package exists outside of a full Isaac Sim release --
# see https://github.com/isaac-sim/IsaacLab/issues/3477. Isaac Sim 5.1.0
# bundles the fixed 1.8.2 build, so pull just those two extension folders out
# of its extscache wheel and drop them in over the broken 1.7.1 ones, leaving
# everything else at its installed pin. Skipped entirely if the broken 1.7.1
# build isn't present (e.g. a newer Isaac Sim already ships the fix).
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
  if ! compgen -G "$extscache/omni.warp-1.7.*" >/dev/null; then
    printf 'omni.warp 1.7.x (the broken build) not present at %s; skipping the R580+ patch.\n' "$extscache"
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
ISAACLAB_VERSION=$ISAACLAB_VERSION
TORCH_VERSION=$TORCH_VERSION
TORCHVISION_VERSION=$TORCHVISION_VERSION
CUDA_HOME=$CUDA_HOME
CUDA_VERSION=$CUDA_VERSION
WARP_FIX_VERSION=$WARP_FIX_VERSION
AUTO_INSTALL_DRIVER=$AUTO_INSTALL_DRIVER
VK_ICD_FILENAMES=$VK_ICD_FILENAMES
EOF
}

doctor() {
  local failures=0
  assert_supported_host || failures=$((failures + 1))
  activate_conda

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

required = ("torch", "isaaclab", "isaacsim", "rl_games")
optional = ("pinocchio", "pink")
failed = []
for name in required:
    try:
        importlib.import_module(name)
        print(f"OK import: {name}")
    except Exception as exc:
        failed.append(name)
        print(f"FAILED import: {name}: {exc}", file=sys.stderr)
for name in optional:
    try:
        importlib.import_module(name)
        print(f"OK import: {name}")
    except Exception as exc:
        print(f"NOTE optional import unavailable: {name}: {exc}")

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
  elif compgen -G "$extscache/omni.warp-1.7.*" >/dev/null 2>&1; then
    printf 'FAILED patch: omni.warp/omni.warp.core %s not found at %s\n' "$WARP_FIX_VERSION" "$extscache" >&2
    failures=$((failures + 1))
  else
    printf 'OK patch: omni.warp 1.7.x (the broken build) not present; patch not needed\n'
  fi

  if (( failures > 0 )); then
    die "$failures required check(s) failed."
  fi
  printf '\nAll required checks passed.\n'
}

self_test() {
  bash -n "$SCRIPT_PATH"
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
    apply-warp-patch) patch_warp_extension ;;
    doctor) doctor ;;
    show-config) show_config ;;
    self-test) self_test ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
