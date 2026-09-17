#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run.sh [options]

Run one stage or the complete ODPS-to-prediction pipeline.

Options:
  --stage NAME         all, fetch, preprocess, pretrain, finetune, or predict (default: all)
  --dataset NAME       Dataset name (currently lianhua; default: lianhua)
  --work-dir DIR       Raw data, processed data, checkpoints, and results (default: /ml/output)
  --plm-path DIR       Local text encoder directory (default: ./bert-base-uncased)
  --env-file FILE      ODPS credentials file (default: /ml/output/.env)
  --python EXECUTABLE  Python executable (default: python3)
  --top-k NUMBER       Number of recommended items (default: 50)
  --output-table NAME  ODPS result table (default: lianhua_tmp_Unisrec_uid2simitem)
  --pretrained-checkpoint FILE  Weight file for a standalone finetune stage
  --finetuned-checkpoint FILE   Weight file for a standalone predict stage
  -h, --help           Show this help
EOF
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=all
dataset=lianhua
work_dir=/ml/output
plm_path=$repo_dir/bert-base-uncased
env_file=/ml/output/.env
python=python3
top_k=50
output_table=lianhua_tmp_Unisrec_uid2simitem
pretrained_checkpoint=
finetuned_checkpoint=

while (($#)); do
  case "$1" in
    --stage|--dataset|--work-dir|--plm-path|--env-file|--python|--top-k|--output-table|--pretrained-checkpoint|--finetuned-checkpoint)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --stage) stage=$2 ;;
        --dataset) dataset=$2 ;;
        --work-dir) work_dir=$2 ;;
        --plm-path) plm_path=$2 ;;
        --env-file) env_file=$2 ;;
        --python) python=$2 ;;
        --top-k) top_k=$2 ;;
        --output-table) output_table=$2 ;;
        --pretrained-checkpoint) pretrained_checkpoint=$2 ;;
        --finetuned-checkpoint) finetuned_checkpoint=$2 ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$stage" in
  all|fetch|preprocess|pretrain|finetune|predict) ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
if [[ -n "$pretrained_checkpoint" && "$stage" != finetune ]]; then
  echo "--pretrained-checkpoint is only used with --stage finetune" >&2
  exit 2
fi
if [[ -n "$finetuned_checkpoint" && "$stage" != predict ]]; then
  echo "--finetuned-checkpoint is only used with --stage predict" >&2
  exit 2
fi
if [[ "$dataset" != lianhua ]]; then
  echo "Only the lianhua ODPS source is configured" >&2
  exit 2
fi
if [[ ! "$top_k" =~ ^[1-9][0-9]*$ ]]; then
  echo "--top-k must be a positive integer" >&2
  exit 2
fi
if ! command -v "$python" >/dev/null 2>&1; then
  echo "Python executable not found: $python" >&2
  exit 2
fi
if [[ "$stage" == all || "$stage" == preprocess ]]; then
  if [[ ! -d "$plm_path" ]]; then
    echo "Text encoder directory does not exist: $plm_path" >&2
    exit 2
  fi
  plm_path=$(cd "$plm_path" && pwd)
fi
if [[ "$stage" == all || "$stage" == fetch || "$stage" == predict ]]; then
  if [[ ! -f "$env_file" ]]; then
    echo "ODPS credentials file does not exist: $env_file" >&2
    exit 2
  fi
  env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")
fi
if [[ "$stage" == finetune ]]; then
  if [[ -z "$pretrained_checkpoint" || ! -f "$pretrained_checkpoint" ]]; then
    echo "--stage finetune requires --pretrained-checkpoint FILE" >&2
    exit 2
  fi
  pretrained_checkpoint=$(cd "$(dirname "$pretrained_checkpoint")" && pwd)/$(basename "$pretrained_checkpoint")
fi
if [[ "$stage" == predict && -n "$finetuned_checkpoint" ]]; then
  if [[ ! -f "$finetuned_checkpoint" ]]; then
    echo "Fine-tuned checkpoint does not exist: $finetuned_checkpoint" >&2
    exit 2
  fi
  finetuned_checkpoint=$(cd "$(dirname "$finetuned_checkpoint")" && pwd)/$(basename "$finetuned_checkpoint")
fi

work_dir=$(mkdir -p "$work_dir" && cd "$work_dir" && pwd)
raw_dir=$work_dir/raw
data_dir=$work_dir/downstream
pretrain_dir=$work_dir/checkpoints/pretrain
finetune_dir=$work_dir/checkpoints/finetune
result_dir=$work_dir/results

cd "$repo_dir"

fetch() {
  mkdir -p "$raw_dir"
  "$python" dataset/raw/get_data_from_odps.py --output-dir "$raw_dir" --env-file "$env_file"
}

preprocess() {
  if [[ ! -f "$raw_dir/$dataset.csv" ]]; then
    echo "Raw data does not exist: $raw_dir/$dataset.csv" >&2
    exit 1
  fi
  mkdir -p "$data_dir"
  "$python" dataset/preprocessing/process_or.py --dataset "$dataset" --input_path "$raw_dir" \
    --output_path "$data_dir" --plm_name "$plm_path" --word_drop_ratio 0.2
}

check_data() {
  for suffix in "$@"; do
    if [[ ! -f "$data_dir/$dataset/$dataset.$suffix" ]]; then
      echo "Processed data does not exist: $data_dir/$dataset/$dataset.$suffix" >&2
      exit 1
    fi
  done
}

pretrain() {
  check_data train.inter feat1CLS feat2CLS
  mkdir -p "$pretrain_dir"
  checkpoint_path_file=$(mktemp "$work_dir/.pretrain-path.XXXXXX")
  trap 'rm -f "$checkpoint_path_file"' EXIT
  "$python" pretrain.py -d "$dataset" --data-path "$data_dir" --checkpoint-dir "$pretrain_dir" \
    --checkpoint-path-file "$checkpoint_path_file"
  if [[ ! -s "$checkpoint_path_file" ]]; then
    echo "Pretraining did not report a checkpoint" >&2
    exit 1
  fi
  pretrained_file=$(cat "$checkpoint_path_file")
  if [[ ! -f "$pretrained_file" ]]; then
    echo "Pretraining checkpoint does not exist: $pretrained_file" >&2
    exit 1
  fi
  echo "Pretrained checkpoint: $pretrained_file"
}

finetune() {
  local checkpoint=$1
  check_data train.inter valid.inter test.inter feat1CLS
  mkdir -p "$finetune_dir"
  "$python" finetune.py -d "$dataset" -p "$checkpoint" --data-path "$data_dir" \
    --checkpoint-dir "$finetune_dir"
  finetuned_file=$finetune_dir/UniSRec-${dataset}-finetuned.pth
  if [[ ! -f "$finetuned_file" ]]; then
    echo "Fine tuning did not produce $finetuned_file" >&2
    exit 1
  fi
  echo "Fine-tuned checkpoint: $finetuned_file"
}

predict() {
  local checkpoint=$1
  check_data train.inter valid.inter test.inter feat1CLS
  if [[ ! -f "$data_dir/$dataset/index2user.json" || ! -f "$data_dir/$dataset/index2item.json" ]]; then
    echo "Prediction ID mapping files are missing in $data_dir/$dataset" >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Fine-tuned checkpoint does not exist: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$result_dir"
  "$python" predict.py -d "$dataset" -fp "$checkpoint" -t "$top_k" -sp "$result_dir" \
    --data-path "$data_dir" --env-file "$env_file" --output-table "$output_table"
}

case "$stage" in
  all) fetch; preprocess; pretrain; finetune "$pretrained_file"; predict "$finetuned_file" ;;
  fetch) fetch ;;
  preprocess) preprocess ;;
  pretrain) pretrain ;;
  finetune) finetune "$pretrained_checkpoint" ;;
  predict) predict "${finetuned_checkpoint:-$finetune_dir/UniSRec-${dataset}-finetuned.pth}" ;;
esac
