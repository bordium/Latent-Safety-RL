#!/usr/bin/env bash
# Download and run NVIDIA's published GR00T-N1-2B Nut Pouring checkpoint in
# the Isaac Lab simulation. See:
#   https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/doc/checkpoints.md
#   https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/doc/evaluation.md
#
# Requires ./gr00t_setup.sh bootstrap (or setup-software) to have completed.
# To evaluate your own fine-tuned checkpoint instead, set MODEL_PATH.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# shellcheck source=./deps/gr00t_common.sh
source "$SCRIPT_DIR/deps/gr00t_common.sh"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

  download-checkpoint  Download NVIDIA's tuned Nut Pouring checkpoint from
                        Hugging Face (resumable; skipped if already present)
  download-dataset      Download the Nut Pouring demonstration dataset
                        (only needed for offline-eval)
  sim                   Launch the basic Isaac Lab GUI simulation
  sim-headless          Launch the basic Isaac Lab simulation without a GUI
  eval-smoke            Run GR00T Nut Pouring (30 inference cycles, 1 rollout)
  eval-full              Run GR00T Nut Pouring (1000 cycles, 20 rollouts)
  offline-eval          Compare policy actions with the demonstration dataset
  show-config           Print resolved paths and settings

Quick start:
  ./gr00t_setup.sh bootstrap
  ./$SCRIPT_NAME eval-smoke

Useful overrides:
  MODEL_PATH=/abs/path      Evaluate a specific checkpoint instead of the
                            published one (e.g. your own fine-tuned model)
  FORCE_DOWNLOAD=1          Re-download the checkpoint/dataset even if present
  HF_TOKEN=hf_xxx           Hugging Face read token, for gated repos
  HEADLESS=1                Run GR00T evaluation headlessly
  ROLLOUT_LENGTH=100 MAX_NUM_ROLLOUTS=2
  ISAAC_ROOT=/abs/path       (default: project-local ./isaac)

Notes:
  * The Nut Pouring policy is GR00T imitation learning, not PPO/SAC reward-based RL.
  * All downloaded files live under \$ISAAC_ROOT, next to this script.
EOF
}

download_checkpoint() {
  download_hf_snapshot "$TUNED_MODEL_REPO" model "$TUNED_MODEL_PATH" config.json
}

download_dataset() {
  download_hf_snapshot "$DATASET_REPO" dataset "$DATASETS_ROOT" Nut-Pouring-task/meta/info.json
}

show_config() {
  cat <<EOF
ISAAC_ROOT=$ISAAC_ROOT
TUNED_MODEL_PATH=$TUNED_MODEL_PATH
DATASET_PATH=$DATASET_PATH
RESULTS_DIR=$RESULTS_DIR
MODEL_PATH=${MODEL_PATH:-<auto: $TUNED_MODEL_PATH>}
TASK_NAME=$TASK_NAME
NUM_ENVS=$NUM_ENVS
NUM_FEEDBACK_ACTIONS=$NUM_FEEDBACK_ACTIONS
SEED=$SEED
VK_ICD_FILENAMES=$VK_ICD_FILENAMES
EOF
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
  download_checkpoint
  : "${MODEL_PATH:=$TUNED_MODEL_PATH}"

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

offline_eval() {
  download_checkpoint
  download_dataset
  : "${MODEL_PATH:=$TUNED_MODEL_PATH}"

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

main() {
  case "${1:-}" in
    download-checkpoint) download_checkpoint ;;
    download-dataset) download_dataset ;;
    sim) run_sim 0 ;;
    sim-headless) run_sim 1 ;;
    eval-smoke) run_eval 30 1 ;;
    eval-full) run_eval 1000 20 ;;
    offline-eval) offline_eval ;;
    show-config) show_config ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
