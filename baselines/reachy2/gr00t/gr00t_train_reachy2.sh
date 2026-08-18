#!/usr/bin/env bash
# Fine-tune a GR00T-N1-2B Reachy2 pick-and-place checkpoint (LoRA).
#
# SCAFFOLD: this will NOT run until a Reachy2 demonstration dataset exists at
# $DATASET_PATH. See README.md in this directory for the required layout and
# for two important caveats (the lab's existing erl-hub Reachy2 checkpoints are
# likely N1.5/LeRobot-native and won't load under this N1 pin; and the
# IsaacLabEvalTasks closed-loop eval layer is hardcoded to GR1 joint naming and
# would silently emit all-zero actions on Reachy2).
#
# Reachy2 is not a GR00T-pretrained embodiment, so training uses the generic
# --embodiment-tag new_embodiment plus the bespoke reachy2_bimanual data config
# in data_config/. Requires ./gr00t_setup.sh bootstrap to have completed.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# shellcheck source=./deps/gr00t_common.sh
source "$SCRIPT_DIR/deps/gr00t_common.sh"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME COMMAND

  download-assets  Download the GR00T-N1-2B base model (dataset is BYO -- see README)
  train-smoke      Attempt one local LoRA optimizer step, to verify trainability
  train            Run LoRA fine-tuning on the Reachy2 pick-and-place dataset
  show-config      Print resolved paths and settings

Quick start:
  ./gr00t_setup.sh bootstrap
  ./$SCRIPT_NAME download-assets
  ./$SCRIPT_NAME train-smoke
  ./$SCRIPT_NAME train

Useful overrides for 'train':
  NUM_GPUS=1 BATCH_SIZE=1 MAX_STEPS=10000 SAVE_STEPS=500
  DATALOADER_WORKERS=2 LORA_RANK=4 LORA_ALPHA=8
  REPORT_TO=none            (or wandb/tensorboard, if configured)
  NO_RESUME=1                Restart from scratch instead of resuming from
                            the newest checkpoint already in \$RUN_DIR
  TRAIN_RUN_DIR=/abs/path    (default: \$RUN_DIR)
  DATASET_PATH=/abs/path BASE_MODEL_PATH=/abs/path
  FORCE_DOWNLOAD=1           Re-download assets even if present
  HF_TOKEN=hf_xxx            Hugging Face read token, for gated repos

Notes:
  * All downloaded/generated files live under \$ISAAC_ROOT, next to this script.
  * This is GR00T imitation-learning fine-tuning, not PPO/SAC reward-based RL.
EOF
}

download_assets() {
  download_hf_snapshot "$BASE_MODEL_REPO" model "$BASE_MODEL_PATH" config.json
  download_hf_snapshot "$DATASET_REPO" dataset "$DATASETS_ROOT" Nut-Pouring-task/meta/info.json
}

show_config() {
  cat <<EOF
ISAAC_ROOT=$ISAAC_ROOT
BASE_MODEL_PATH=$BASE_MODEL_PATH
DATASET_PATH=$DATASET_PATH
RUN_DIR=$RUN_DIR
EOF
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

train() {
  if [[ ! -d "$DATASET_PATH" ]]; then
    cat >&2 <<EOM
ERROR: No Reachy2 demonstration dataset at:
  $DATASET_PATH

This GR00T baseline is a SCAFFOLD -- the plumbing is in place but the
demonstrations are not. GR00T is imitation learning, so there is no
reward-function shortcut; it needs recorded episodes.

To proceed you need a GR00T-LeRobot dataset laid out as:
  meta/{episodes.jsonl,modality.json,info.json,tasks.jsonl}
  videos/chunk-000/observation.images.<name>/episode_000000.mp4
  data/chunk-000/episode_000000.parquet

See README.md in this directory for the full checklist, and
data_config/modality.json.template for the expected state/action layout.
EOM
    exit 1
  fi

  activate_conda
  require_file "$GROOT_DIR/scripts/gr00t_finetune.py"
  require_dir "$DATASET_PATH"
  require_dir "$BASE_MODEL_PATH"
  local output_dir="${TRAIN_RUN_DIR:-$RUN_DIR}"
  mkdir -p "$output_dir"

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

  local num_gpus="${NUM_GPUS:-1}" batch_size="${BATCH_SIZE:-1}" max_steps="${MAX_STEPS:-10000}"
  local save_steps="${SAVE_STEPS:-500}" workers="${DATALOADER_WORKERS:-2}"
  local lora_rank="${LORA_RANK:-4}" lora_alpha="${LORA_ALPHA:-8}" report_to="${REPORT_TO:-none}"

  local args=(
    python scripts/gr00t_finetune.py
    "$dataset_flag=$DATASET_PATH"
    "$output_flag=$output_dir"
    "$data_config_flag=$DATA_CONFIG"
    "$embodiment_flag=$EMBODIMENT_TAG"
    "$base_model_flag=$BASE_MODEL_PATH"
    "$gpu_flag=$num_gpus"
    "$batch_flag=$batch_size"
    "$steps_flag=$max_steps"
    "$save_flag=$save_steps"
    "$workers_flag=$workers"
    "$video_flag=torchvision_av"
    "$lora_rank_flag=$lora_rank"
    "$lora_alpha_flag=$lora_alpha"
    "$report_flag=$report_to"
  )
  [[ "${NO_RESUME:-0}" == "1" ]] && args+=("$no_resume_flag")

  printf 'Training for %s steps (batch %s, %s GPU(s)); checkpoints every %s steps in %s\n' \
    "$max_steps" "$batch_size" "$num_gpus" "$save_steps" "$output_dir"
  "${args[@]}" 2>&1 | tee -a "$output_dir/train.log"
}

train_smoke() {
  printf 'Attempting one optimizer step on GPU 0. Isaac Sim is not used during training.\n'
  TRAIN_RUN_DIR="$ISAAC_ROOT/training_runs/reachy2_pickplace_smoke" \
    NUM_GPUS=1 BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 DATALOADER_WORKERS=1 NO_RESUME=1 \
    train
}

main() {
  case "${1:-}" in
    download-assets) download_assets ;;
    train-smoke) train_smoke ;;
    train) train ;;
    show-config) show_config ;;
    help|-h|--help|'') usage ;;
    *) die "Unknown command '$1'. Run '$SCRIPT_NAME help'." ;;
  esac
}

main "$@"
