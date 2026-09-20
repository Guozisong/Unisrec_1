#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash run.sh [options]

Run one stage or the complete interaction-to-recommendation pipeline.

Options:
  --stage NAME         all, fetch, preprocess, pretrain, finetune, or predict (default: all)
  --dataset NAME       Dataset name used for intermediate files (required)
  --input-csv FILE     Interaction CSV for fetch (alternative to --input-table)
  --input-table NAME   ODPS interaction table for fetch
  --input-query-file FILE  ODPS SQL for fetch (alternative to table)
  --item-metadata-csv FILE   CSV with item_id, category_id for prediction
  --eligible-items-csv FILE CSV with item_id for prediction
  --item-metadata-table NAME ODPS item/category table for prediction
  --eligible-items-table NAME ODPS eligible item table for prediction
  --work-dir DIR       Raw data, processed data, checkpoints, and results (default: project outputs/)
  --plm-path DIR       Local text encoder directory (default: ./bert-base-uncased)
  --max-seq-length N   Recent interactions retained per user in preprocess (default: 50; 3-100)
  --env-file FILE      ODPS connection file (default: project .env)
  --python EXECUTABLE  Python executable (default: python3)
  --top-k NUMBER       Number of recommended items (default: 50)
  --output-table NAME  Optional ODPS result table; predictions are always saved as CSV
  --resume-checkpoint FILE     Checkpoint for resuming a standalone pretrain stage
  --pretrained-checkpoint FILE  Weight file for a standalone finetune stage
  --finetuned-checkpoint FILE   Weight file for a standalone predict stage
  -h, --help           Show this help
EOF
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=all
dataset=
input_csv=
input_table=
input_query_file=
item_metadata_csv=
eligible_items_csv=
item_metadata_table=
eligible_items_table=
work_dir=$repo_dir/outputs
plm_path=$repo_dir/bert-base-uncased
max_seq_length=50
env_file=$repo_dir/.env
python=python3
top_k=50
output_table=
resume_checkpoint=
pretrained_checkpoint=
finetuned_checkpoint=

while (($#)); do
  case "$1" in
    --stage|--dataset|--input-csv|--input-table|--input-query-file|--item-metadata-csv|--eligible-items-csv|--item-metadata-table|--eligible-items-table|--work-dir|--plm-path|--max-seq-length|--env-file|--python|--top-k|--output-table|--resume-checkpoint|--pretrained-checkpoint|--finetuned-checkpoint)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --stage) stage=$2 ;;
        --dataset) dataset=$2 ;;
        --input-csv) input_csv=$2 ;;
        --input-table) input_table=$2 ;;
        --input-query-file) input_query_file=$2 ;;
        --item-metadata-csv) item_metadata_csv=$2 ;;
        --eligible-items-csv) eligible_items_csv=$2 ;;
        --item-metadata-table) item_metadata_table=$2 ;;
        --eligible-items-table) eligible_items_table=$2 ;;
        --work-dir) work_dir=$2 ;;
        --plm-path) plm_path=$2 ;;
        --max-seq-length) max_seq_length=$2 ;;
        --env-file) env_file=$2 ;;
        --python) python=$2 ;;
        --top-k) top_k=$2 ;;
        --output-table) output_table=$2 ;;
        --resume-checkpoint) resume_checkpoint=$2 ;;
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
if [[ -n "$resume_checkpoint" && "$stage" != pretrain ]]; then
  echo "--resume-checkpoint is only used with --stage pretrain" >&2
  exit 2
fi
if [[ -n "$pretrained_checkpoint" && "$stage" != finetune ]]; then
  echo "--pretrained-checkpoint is only used with --stage finetune" >&2
  exit 2
fi
if [[ -n "$finetuned_checkpoint" && "$stage" != predict ]]; then
  echo "--finetuned-checkpoint is only used with --stage predict" >&2
  exit 2
fi
if [[ ! "$dataset" =~ ^[A-Za-z][A-Za-z0-9_]*$ ]]; then
  echo "--dataset is required and must contain letters, digits, or underscores, starting with a letter" >&2
  exit 2
fi
if [[ "$stage" == all || "$stage" == fetch ]]; then
  sources=0
  for source in "$input_csv" "$input_table" "$input_query_file"; do if [[ -n "$source" ]]; then ((sources+=1)); fi; done
  if ((sources != 1)); then
    echo "Choose exactly one of --input-csv, --input-table, or --input-query-file for fetch" >&2
    exit 2
  fi
fi
if [[ "$stage" == all || "$stage" == predict ]]; then
  if [[ -z "$item_metadata_csv" && -z "$item_metadata_table" ]] || [[ -z "$eligible_items_csv" && -z "$eligible_items_table" ]] ||
     [[ -n "$item_metadata_csv" && -n "$item_metadata_table" ]] || [[ -n "$eligible_items_csv" && -n "$eligible_items_table" ]]; then
    echo "Prediction requires one metadata source and one eligible-item source (CSV or ODPS table)" >&2
    exit 2
  fi
fi
if [[ ! "$top_k" =~ ^[1-9][0-9]*$ ]]; then
  echo "--top-k must be a positive integer" >&2
  exit 2
fi
if [[ ! "$max_seq_length" =~ ^[0-9]+$ ]] || ((10#$max_seq_length < 3 || 10#$max_seq_length > 100)); then
  echo "--max-seq-length must be an integer from 3 to 100" >&2
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
if [[ ( "$stage" == all || "$stage" == fetch ) && -n "$input_csv" && ! -f "$input_csv" ]] ||
   [[ ( "$stage" == all || "$stage" == fetch ) && -n "$input_query_file" && ! -f "$input_query_file" ]]; then
  echo "Interaction input file does not exist" >&2
  exit 2
fi
if [[ ( "$stage" == all || "$stage" == predict ) && -n "$item_metadata_csv" && ! -f "$item_metadata_csv" ]] ||
   [[ ( "$stage" == all || "$stage" == predict ) && -n "$eligible_items_csv" && ! -f "$eligible_items_csv" ]]; then
  echo "Prediction metadata CSV does not exist" >&2
  exit 2
fi
if [[ ( "$stage" == all || "$stage" == fetch ) && ( -n "$input_table" || -n "$input_query_file" ) ]] ||
   [[ ( "$stage" == all || "$stage" == predict ) && ( -n "$item_metadata_table" || -n "$eligible_items_table" || -n "$output_table" ) ]]; then
  if [[ ! -f "$env_file" ]]; then
    echo "ODPS credentials file does not exist: $env_file" >&2
    exit 2
  fi
  env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")
fi
for source in input_csv input_query_file item_metadata_csv eligible_items_csv; do
  if [[ -n "${!source}" ]]; then
    printf -v "$source" '%s/%s' "$(cd "$(dirname "${!source}")" && pwd)" "$(basename "${!source}")"
  fi
done
if [[ "$stage" == pretrain && -n "$resume_checkpoint" ]]; then
  if [[ ! -f "$resume_checkpoint" ]]; then
    echo "Pretraining checkpoint does not exist: $resume_checkpoint" >&2
    exit 2
  fi
  resume_checkpoint=$(cd "$(dirname "$resume_checkpoint")" && pwd)/$(basename "$resume_checkpoint")
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
  local source=(--dataset "$dataset" --output-dir "$raw_dir")
  if [[ -n "$input_csv" ]]; then
    source+=(--input-csv "$input_csv")
  elif [[ -n "$input_query_file" ]]; then
    source+=(--input-query-file "$input_query_file" --env-file "$env_file")
  else
    source+=(--input-table "$input_table" --env-file "$env_file")
  fi
  "$python" data_pipeline/raw/prepare_interactions.py "${source[@]}"
}

preprocess() {
  if [[ ! -f "$raw_dir/$dataset.csv" ]]; then
    echo "Raw data does not exist: $raw_dir/$dataset.csv" >&2
    exit 1
  fi
  mkdir -p "$data_dir"
  "$python" data_pipeline/preprocessing/preprocess.py --dataset "$dataset" --input_path "$raw_dir" \
    --output_path "$data_dir" --plm_name "$plm_path" --word_drop_ratio 0.2 \
    --max_seq_length "$max_seq_length"
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
  local args=(-d "$dataset" --data-path "$data_dir" --checkpoint-dir "$pretrain_dir"
    --checkpoint-path-file "$checkpoint_path_file")
  if [[ -n "$resume_checkpoint" ]]; then args+=(--resume-checkpoint "$resume_checkpoint"); fi
  "$python" pretrain.py "${args[@]}"
  if [[ ! -s "$checkpoint_path_file" ]]; then
    echo "Pretraining did not report a checkpoint" >&2
    exit 1
  fi
  pretrained_file=$(cat "$checkpoint_path_file")
  if [[ ! -f "$pretrained_file" ]]; then
    echo "Pretraining checkpoint does not exist: $pretrained_file" >&2
    exit 1
  fi
  rm -f "$checkpoint_path_file"
  checkpoint_path_file=
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
  check_data train.inter valid.inter predict.inter feat1CLS
  if [[ ! -f "$data_dir/$dataset/index2user.json" || ! -f "$data_dir/$dataset/index2item.json" ]]; then
    echo "Prediction ID mapping files are missing in $data_dir/$dataset" >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Fine-tuned checkpoint does not exist: $checkpoint" >&2
    exit 1
  fi
  mkdir -p "$result_dir"
  local sources=(--data-path "$data_dir")
  if [[ -n "$item_metadata_csv" ]]; then sources+=(--item-metadata-csv "$item_metadata_csv"); else sources+=(--item-metadata-table "$item_metadata_table"); fi
  if [[ -n "$eligible_items_csv" ]]; then sources+=(--eligible-items-csv "$eligible_items_csv"); else sources+=(--eligible-items-table "$eligible_items_table"); fi
  if [[ -n "$item_metadata_table" || -n "$eligible_items_table" || -n "$output_table" ]]; then sources+=(--env-file "$env_file"); fi
  if [[ -n "$output_table" ]]; then sources+=(--output-table "$output_table"); fi
  "$python" predict.py -d "$dataset" -fp "$checkpoint" -t "$top_k" -sp "$result_dir" "${sources[@]}"
}

stage_info() {
  case "$1" in
    fetch)
      local input=${input_csv:-${input_query_file:-$input_table}}
      printf 'input=%s output=%s' "$input" "$raw_dir/$dataset.csv" ;;
    preprocess)
      printf 'input=%s output=%s max_seq_length=%s encoder=%s' \
        "$raw_dir/$dataset.csv" "$data_dir/$dataset" "$max_seq_length" "$plm_path" ;;
    pretrain)
      printf 'input=%s output=%s' "$data_dir/$dataset" "$pretrain_dir"
      if [[ -n "$resume_checkpoint" ]]; then printf ' resume=%s' "$resume_checkpoint"; fi ;;
    finetune)
      printf 'input=%s checkpoint=%s output=%s' \
        "$data_dir/$dataset" "$2" "$finetune_dir" ;;
    predict)
      printf 'input=%s checkpoint=%s output=%s' \
        "$data_dir/$dataset/$dataset.predict.inter" "$2" "$result_dir" ;;
  esac
}

current_stage=
stage_started=0
checkpoint_path_file=
on_exit() {
  local status=$1
  if [[ -n "$checkpoint_path_file" ]]; then
    rm -f "$checkpoint_path_file"
  fi
  if ((status != 0)) && [[ -n "$current_stage" ]]; then
    printf '[%s] stage=%s FAILED exit=%s elapsed=%ss\n' \
      "$(date '+%Y-%m-%d %H:%M:%S%z')" "$current_stage" "$status" "$((SECONDS - stage_started))" >&2
  fi
}
trap 'on_exit $?' EXIT

run_stage() {
  current_stage=$1
  shift
  stage_started=$SECONDS
  printf '[%s] stage=%s START dataset=%s work_dir=%s ' \
    "$(date '+%Y-%m-%d %H:%M:%S%z')" "$current_stage" "$dataset" "$work_dir"
  stage_info "$current_stage" "$@"
  printf '\n'
  "$current_stage" "$@"
  printf '[%s] stage=%s COMPLETE elapsed=%ss\n' \
    "$(date '+%Y-%m-%d %H:%M:%S%z')" "$current_stage" "$((SECONDS - stage_started))"
  current_stage=
}

case "$stage" in
  all) run_stage fetch; run_stage preprocess; run_stage pretrain; run_stage finetune "$pretrained_file"; run_stage predict "$finetuned_file" ;;
  fetch) run_stage fetch ;;
  preprocess) run_stage preprocess ;;
  pretrain) run_stage pretrain ;;
  finetune) run_stage finetune "$pretrained_checkpoint" ;;
  predict) run_stage predict "${finetuned_checkpoint:-$finetune_dir/UniSRec-${dataset}-finetuned.pth}" ;;
esac
