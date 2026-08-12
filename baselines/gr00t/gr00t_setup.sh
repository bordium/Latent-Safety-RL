#!/usr/bin/env bash
# Install and pin the IsaacLabEvalTasks + Isaac GR00T stack on top of an
# existing Isaac Lab installation.
#
# Prerequisite: ../isaaclab_setup.sh bootstrap (installs the shared Conda
# environment, system packages, driver, CUDA, and Isaac Lab itself). This
# script only layers GR00T-specific pieces on top of that: the
# IsaacLabEvalTasks repo, the pinned Isaac-GR00T submodule, GR00T's Python
# dependencies, and compatibility patches.
#
# Tested target:
#   Ubuntu 22.04/24.04 x86_64, Python 3.11, CUDA Toolkit 12.8
#   IsaacLabEvalTasks 460f2878... and its pinned Isaac-GR00T submodule
#
# This script does not download any checkpoints or datasets -- use
# gr00t_run_nutpouring.sh for the published Nut Pouring checkpoint, or
# gr00t_train_nutpouring.sh to fine-tune your own.
#
# Idempotent: rerunning preserves existing repositories and environments,
# verifies pinned revisions, and only installs what's missing.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# shellcheck source=./deps/gr00t_common.sh
source "$SCRIPT_DIR/deps/gr00t_common.sh"

EVAL_REPO_URL="${EVAL_REPO_URL:-https://github.com/isaac-sim/IsaacLabEvalTasks.git}"
EVAL_REPO_REF="${EVAL_REPO_REF:-460f2878bdcb4db2d21913db789174fb316b73e2}"
EVAL_REPO_COMMIT="${EVAL_REPO_COMMIT:-460f2878bdcb4db2d21913db789174fb316b73e2}"
GROOT_URL="${GROOT_URL:-https://github.com/NVIDIA/Isaac-GR00T.git}"
GROOT_COMMIT="${GROOT_COMMIT:-755876a9afdb41ca6eb6383b36f4a0adb085c73f}"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

Prerequisite (run once, shared across baselines):
  ../isaaclab_setup.sh bootstrap

Fresh-computer setup:
  bootstrap                 Clone/pin repos, install GR00T's Python
                            dependencies, and apply compatibility patches
  setup-software             Same as bootstrap (kept for symmetry)
  setup-repositories         Just clone/pin IsaacLabEvalTasks and the GR00T
                            submodule (no Python install)
  apply-training-patches    Reapply the two RTX 4060 training compatibility fixes

Verify:
  doctor            Check pinned repos, packages, and downloaded assets
  self-test         Check this Bash file without requiring Isaac Lab
  show-config       Print resolved paths, versions, and settings

Next steps once setup passes:
  ./gr00t_run_nutpouring.sh    Download and run the published Nut Pouring checkpoint
  ./gr00t_train_nutpouring.sh  Fine-tune your own Nut Pouring checkpoint

Fresh Ubuntu example:
  ../isaaclab_setup.sh bootstrap
  chmod +x $SCRIPT_NAME
  ./$SCRIPT_NAME bootstrap
  ./$SCRIPT_NAME doctor

Useful overrides:
  ISAAC_ROOT=/abs/path CONDA_ROOT=/abs/path CONDA_ENV=another-env
                            (default: project-local ./isaac and ./conda,
                            shared with ../isaaclab_setup.sh)

Notes:
  * Everything installed by this script lives under \$ISAAC_ROOT and \$CONDA_ROOT,
    which default to directories at the repo root.
  * 'pip check' may report a Torch build-tag mismatch between Isaac Sim and
    GR00T's own [base] extras; this script reinstalls the cu128 build GR00T
    is validated against right after, per IsaacLabEvalTasks' own
    documentation of the GR00T integration.
EOF
}

cuda_128_is_installed() {
  [[ -x "$CUDA_HOME/bin/nvcc" ]] &&
    "$CUDA_HOME/bin/nvcc" --version | grep -Fq "release $CUDA_VERSION"
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

require_isaaclab() {
  [[ -f "$ISAACLAB_DIR/isaaclab.sh" ]] ||
    die "Isaac Lab not found at $ISAACLAB_DIR. Run '../isaaclab_setup.sh bootstrap' first."
}

setup_repositories() {
  require_command git
  require_isaaclab
  mkdir -p "$ISAAC_ROOT"
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
  require_isaaclab
  require_dir "$GROOT_DIR"
  require_dir "$EVAL_REPO"
  cuda_128_is_installed || die "CUDA Toolkit $CUDA_VERSION is required to build FlashAttention."

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

setup_software() {
  assert_supported_host
  require_isaaclab
  setup_repositories
  install_python_stack
  apply_training_patches
  mkdir -p "$ISAAC_ROOT/checkpoints" "$ISAAC_ROOT/training_runs" "$RESULTS_DIR" "$DATASETS_ROOT"
  info "GR00T software setup completed"
}

bootstrap() {
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
EVAL_REPO=$EVAL_REPO
EVAL_REPO_COMMIT=$EVAL_REPO_COMMIT
GROOT_DIR=$GROOT_DIR
GROOT_COMMIT=$GROOT_COMMIT
CUDA_HOME=$CUDA_HOME
CUDA_VERSION=$CUDA_VERSION
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

  if [[ -f "$ISAACLAB_DIR/isaaclab.sh" ]]; then
    printf 'OK Isaac Lab present at: %s (run ../isaaclab_setup.sh doctor for full detail)\n' "$ISAACLAB_DIR"
  else
    printf 'FAILED: Isaac Lab not found at %s. Run ../isaaclab_setup.sh bootstrap first.\n' "$ISAACLAB_DIR" >&2
    failures=$((failures + 1))
  fi
  check_repo_commit "$EVAL_REPO" "$EVAL_REPO_COMMIT" "IsaacLabEvalTasks" || failures=$((failures + 1))
  check_repo_commit "$GROOT_DIR" "$GROOT_COMMIT" "Isaac-GR00T" || failures=$((failures + 1))
  require_file "$EVAL_REPO/scripts/evaluate_gn1.py"

  printf '\nPython imports:\n'
  if ! python - <<'PY'
import importlib
import sys

modules = ("torch", "isaaclab_eval_tasks", "gr00t", "flash_attn")
failed = []
for name in modules:
    try:
        importlib.import_module(name)
        print(f"OK import: {name}")
    except Exception as exc:
        failed.append(name)
        print(f"FAILED import: {name}: {exc}", file=sys.stderr)

import torch
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

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
  bash -n "$SCRIPT_DIR/deps/gr00t_common.sh"
  [[ "$EVAL_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$GROOT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
  [[ "$CUDA_VERSION" == "12.8" ]]
  printf 'Bash syntax and pinned-configuration checks passed.\n'
  if command -v shellcheck >/dev/null 2>&1; then
    shellcheck "$SCRIPT_PATH" "$SCRIPT_DIR/deps/gr00t_common.sh"
    printf 'ShellCheck passed.\n'
  else
    printf 'ShellCheck is not installed; static lint was skipped.\n'
  fi
}

main() {
  case "${1:-}" in
    bootstrap) bootstrap ;;
    setup-software) setup_software ;;
    setup-repositories) mkdir -p "$ISAAC_ROOT" && setup_repositories ;;
    apply-training-patches) apply_training_patches ;;
    doctor) doctor ;;
    show-config) show_config ;;
    self-test) self_test ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
