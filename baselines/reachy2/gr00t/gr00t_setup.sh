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
# gr00t_run_reachy2.sh for a Reachy2 checkpoint (none published -- see README), or
# gr00t_train_reachy2.sh to fine-tune your own.
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

# ../isaaclab_setup.sh's base environment. GR00T needs a different Torch
# build than that shared base, so rather than installing into it directly
# (which would mutate it out from under other baselines, e.g. ppo), this
# script clones it into $CONDA_ENV (isaac-gr00t) and installs there instead.
BASE_CONDA_ENV="${BASE_CONDA_ENV:-env_isaaclab}"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

Prerequisite (run once, shared across baselines):
  ../isaaclab_setup.sh bootstrap

Fresh-computer setup:
  bootstrap                 Clone the base Conda env into isaac-gr00t, clone/pin
                            repos, install GR00T's Python dependencies, and
                            apply compatibility patches
  setup-software             Same as bootstrap (kept for symmetry)
  setup-repositories         Just clone/pin IsaacLabEvalTasks and the GR00T
                            submodule (no Conda/Python install)
  apply-training-patches    Reapply the two RTX 4060 training compatibility fixes

Verify:
  doctor            Check pinned repos, packages, and downloaded assets
  self-test         Check this Bash file without requiring Isaac Lab
  show-config       Print resolved paths, versions, and settings

Next steps once setup passes:
  ./gr00t_run_reachy2.sh    Download and run a Reachy2 checkpoint (none published -- see README)
  ./gr00t_train_reachy2.sh  Fine-tune a Reachy2 checkpoint (needs demos -- see README)

Fresh Ubuntu example:
  ../isaaclab_setup.sh bootstrap
  chmod +x $SCRIPT_NAME
  ./$SCRIPT_NAME bootstrap
  ./$SCRIPT_NAME doctor

Useful overrides:
  ISAAC_ROOT=/abs/path CONDA_ROOT=/abs/path
                            (default: project-local ./isaac and ./conda,
                            shared with ../isaaclab_setup.sh)
  CONDA_ENV=another-env       Name of GR00T's own environment (default:
                            isaac-gr00t)
  BASE_CONDA_ENV=another-env  Name of the base environment to clone from
                            (default: env_isaaclab, see ../isaaclab_setup.sh)

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

# Creates $CONDA_ENV (isaac-gr00t) by cloning $BASE_CONDA_ENV (env_isaaclab)
# the first time, then activates it. A clone, not a fresh env, so GR00T
# doesn't have to reinstall Isaac Sim/Isaac Lab and their multi-GB extension
# cache; a clone, not installing into the base directly, so GR00T's own
# Torch pin never mutates the environment other baselines share.
ensure_gr00t_conda_env() {
  local conda_base=""
  conda_base="$(find_conda_base 2>/dev/null || true)"
  [[ -n "$conda_base" ]] || die "Conda was not found at $CONDA_ROOT. Run '../isaaclab_setup.sh setup-software' first."
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"

  if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV"; then
    conda activate "$CONDA_ENV"
  else
    conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$BASE_CONDA_ENV" ||
      die "Conda environment '$BASE_CONDA_ENV' not found. Run '../isaaclab_setup.sh setup-software' first."
    info "Cloning Conda environment $BASE_CONDA_ENV into $CONDA_ENV for GR00T"
    conda create -y -n "$CONDA_ENV" --clone "$BASE_CONDA_ENV"
    conda activate "$CONDA_ENV"
    # `conda create --clone` of a pip-managed env can leave pip's own files
    # inconsistent (e.g. two dist-info dirs for different versions, one
    # missing symbols the other's modules import) -- self-heal via Python's
    # bundled ensurepip, which doesn't depend on the (possibly broken)
    # existing pip at all.
    if ! python -m pip --version >/dev/null 2>&1; then
      warn "pip is broken after cloning $BASE_CONDA_ENV; reinstalling it via ensurepip"
      local site_packages
      site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
      rm -rf "$site_packages"/pip "$site_packages"/pip-*.dist-info
      python -m ensurepip --upgrade
      python -m pip install --upgrade pip
    fi
  fi
  # Isaac Sim's bundled extensions need a newer libstdc++ (CXXABI_1.3.15+)
  # than Ubuntu 22.04 ships. Conda's own copy has it; put it ahead of the
  # system one so the linker finds it first.
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

setup_repositories() {
  require_command git
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
  # evaluate_gn1.py imports pinocchio/pink directly for GR1 IK/joint
  # remapping. The old git-checkout install of Isaac Lab pulled these in
  # transitively; the isaaclab[isaacsim,all] pip package does not, so pin
  # them here explicitly.
  python -m pip install "pin==2.7.0" "pin-pink==3.1.0"
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
  ensure_gr00t_conda_env
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
BASE_CONDA_ENV=$BASE_CONDA_ENV
CONDA_ROOT=$CONDA_ROOT
ISAAC_ROOT=$ISAAC_ROOT
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

  local base_conda_base=""
  base_conda_base="$(find_conda_base 2>/dev/null || true)"
  if [[ -n "$base_conda_base" ]] &&
     "$base_conda_base/bin/conda" env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$BASE_CONDA_ENV"; then
    printf 'OK base environment present: %s (run ../isaaclab_setup.sh doctor for full detail)\n' "$BASE_CONDA_ENV"
  else
    printf 'FAILED: base environment %s not found. Run ../isaaclab_setup.sh bootstrap first.\n' "$BASE_CONDA_ENV" >&2
    failures=$((failures + 1))
  fi
  check_repo_commit "$EVAL_REPO" "$EVAL_REPO_COMMIT" "IsaacLabEvalTasks" || failures=$((failures + 1))
  check_repo_commit "$GROOT_DIR" "$GROOT_COMMIT" "Isaac-GR00T" || failures=$((failures + 1))
  require_file "$EVAL_REPO/scripts/evaluate_gn1.py"

  printf '\nPython imports:\n'
  if ! python - <<'PY'
import importlib
import sys

modules = ("torch", "pinocchio", "pink", "isaaclab_eval_tasks", "gr00t", "flash_attn")
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
    warn "No Reachy2 demonstration dataset present -- this baseline is a scaffold (see README)."
  fi
  if [[ -d "$TUNED_MODEL_PATH" ]]; then
    printf 'OK published checkpoint: %s\n' "$TUNED_MODEL_PATH"
  else
    warn "No Reachy2 checkpoint present. None is published; train one first (see README)."
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
