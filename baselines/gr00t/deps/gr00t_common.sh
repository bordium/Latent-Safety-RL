#!/usr/bin/env bash
# Shared configuration and helpers for the gr00t_*.sh scripts.
# Source this file; it is not meant to be executed directly.
#
# Everything these scripts download or generate -- cloned repos, checkpoints,
# datasets, training runs, eval results, and the Conda install itself -- lives
# under $ISAAC_ROOT / $CONDA_ROOT, which default to directories at the repo
# root (regardless of how deeply these scripts are nested). Nothing is
# written outside this project tree unless you override a path variable
# yourself.

set -Eeuo pipefail

_gr00t_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$_gr00t_common_dir" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$_gr00t_common_dir")"
unset _gr00t_common_dir

# isaac-gr00t is a clone of ../isaaclab_setup.sh's base "env_isaaclab"
# environment (see gr00t_setup.sh's ensure_gr00t_conda_env), so GR00T's own
# package pins never leak into the shared base other baselines (ppo, ...) use.
CONDA_ENV="${CONDA_ENV:-isaac-gr00t}"
CONDA_ROOT="${CONDA_ROOT:-$PROJECT_ROOT/conda}"

ISAAC_ROOT="${ISAAC_ROOT:-$PROJECT_ROOT/isaac}"
EVAL_REPO="${EVAL_REPO:-$ISAAC_ROOT/IsaacLabEvalTasks}"
GROOT_DIR="${GROOT_DIR:-$EVAL_REPO/submodules/Isaac-GR00T}"

CUDA_VERSION="${CUDA_VERSION:-12.8}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"

DATASETS_ROOT="${DATASETS_ROOT:-$ISAAC_ROOT/datasets/PhysicalAI-GR00T-Tuned-Tasks}"
DATASET_PATH="${DATASET_PATH:-$DATASETS_ROOT/Nut-Pouring-task}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ISAAC_ROOT/checkpoints/GR00T-N1-2B}"
TUNED_MODEL_PATH="${TUNED_MODEL_PATH:-$ISAAC_ROOT/checkpoints/GR00T-N1-2B-tuned-Nut-Pouring-task}"
RUN_DIR="${RUN_DIR:-$ISAAC_ROOT/training_runs/nutpouring_lora}"
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

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

info() {
  printf '\n[%s] %s\n' "${SCRIPT_NAME:-gr00t}" "$*"
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
    die "Use native Ubuntu 22.04/24.04. Isaac Sim GUI/GR00T is not supported here through WSL."
  fi
}

find_conda_base() {
  [[ -x "$CONDA_ROOT/bin/conda" ]] && printf '%s\n' "$CONDA_ROOT"
}

activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return
  fi
  local conda_base=""
  conda_base="$(find_conda_base 2>/dev/null || true)"
  [[ -n "$conda_base" ]] || die "Conda was not found at $CONDA_ROOT. Run '../isaaclab_setup.sh setup-software' first."
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV" ||
    die "Conda environment '$CONDA_ENV' not found. Run './gr00t_setup.sh setup-software' first."
  conda activate "$CONDA_ENV"
  # Isaac Sim's bundled extensions (e.g. omni.kit.test's coverage/sqlite3
  # chain) need a newer libstdc++ (CXXABI_1.3.15+) than Ubuntu 22.04 ships.
  # Conda's own copy has it; put it ahead of the system one so the linker
  # finds it first.
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

# Downloads a Hugging Face repo snapshot into $destination, skipping the
# download if $destination/$marker already exists and no interrupted files
# are left over from a previous run (resumable via Hugging Face's own
# .incomplete markers).
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

# Resolves which checkpoint directory to use: an explicit $MODEL_PATH always
# wins; otherwise the newest checkpoint-* directory under $RUN_DIR; otherwise
# $TUNED_MODEL_PATH.
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
    die "No model found. Download a checkpoint or set MODEL_PATH."
  fi
}
