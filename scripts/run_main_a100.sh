#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/main_qwen25coder_1p5b_L18_a100_autodl.yaml}"
STAGE="${STAGE:-all}"
: "${NLA_STORAGE_ROOT:?Set NLA_STORAGE_ROOT to the cache/checkpoint/dataset root}"

PY_SITE="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
export LD_LIBRARY_PATH="$PY_SITE/nvidia/cusparselt/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/nccl/lib:$PY_SITE/nvidia/cublas/lib:$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cuda_runtime/lib:$PY_SITE/nvidia/cuda_nvrtc/lib:$PY_SITE/nvidia/curand/lib:$PY_SITE/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-${NLA_STORAGE_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-${NLA_STORAGE_ROOT}/torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

STAGES=(preflight cache_hf cache_model cache_data data extract bootstrap ar_train ar_eval av_sft av_generate rerank av_rerank roundtrip baselines token_roles figures)

fmt_seconds() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  if (( h > 0 )); then
    printf "%dh%02dm%02ds" "$h" "$m" "$s"
  elif (( m > 0 )); then
    printf "%dm%02ds" "$m" "$s"
  else
    printf "%ds" "$s"
  fi
}

estimate_stage() {
  local stage="$1"
  "$PYTHON_BIN" - "$CONFIG" "$stage" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(cfg.get("runtime", {}).get("stage_estimates_seconds", {}).get(sys.argv[2], 0)))
PY
}

remaining_estimate() {
  local seen_current=0
  local total=0
  for s in "${STAGES[@]}"; do
    if [[ "$STAGE" != "all" && "$STAGE" != "$s" ]]; then
      continue
    fi
    if [[ "$STAGE" == "all" ]]; then
      if (( seen_current == 0 )); then
        total=$((total + $(estimate_stage "$s")))
      fi
    else
      total=$((total + $(estimate_stage "$s")))
    fi
  done
  echo "$total"
}

print_progress() {
  local label="$1"
  local stage="$2"
  local estimate="$3"
  local remaining="$4"
  printf '[%s] %s stage=%s stage_eta=%s remaining_eta=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$label" "$stage" "$(fmt_seconds "$estimate")" "$(fmt_seconds "$remaining")"
}

completed_estimate=0

run_stage() {
  local name="$1"
  shift
  if [[ "$STAGE" == "all" || "$STAGE" == "$name" ]]; then
    local est
    est="$(estimate_stage "$name")"
    local total_remaining
    if [[ "$STAGE" == "all" ]]; then
      local total=0
      local include=0
      for s in "${STAGES[@]}"; do
        [[ "$s" == "$name" ]] && include=1
        if (( include == 1 )); then
          total=$((total + $(estimate_stage "$s")))
        fi
      done
      total_remaining="$total"
    else
      total_remaining="$est"
    fi
    print_progress "start" "$name" "$est" "$total_remaining"
    local start
    start="$(date +%s)"
    "$@"
    local elapsed=$(( $(date +%s) - start ))
    completed_estimate=$((completed_estimate + est))
    local remaining_after=$((total_remaining - est))
    (( remaining_after < 0 )) && remaining_after=0
    printf '[%s] done stage=%s elapsed=%s next_remaining_eta=%s\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$name" "$(fmt_seconds "$elapsed")" "$(fmt_seconds "$remaining_after")"
  fi
}

run_stage preflight "$PYTHON_BIN" scripts/check_gpu_ready.py --config "$CONFIG"
run_stage cache_hf "$PYTHON_BIN" scripts/99_cache_hf_assets.py --config "$CONFIG"
run_stage cache_model "$PYTHON_BIN" scripts/98_cache_model_weights.py --config "$CONFIG"
run_stage cache_data "$PYTHON_BIN" scripts/97_download_codesearchnet_python_raw.py --config "$CONFIG"
run_stage data "$PYTHON_BIN" scripts/00_prepare_data.py --config "$CONFIG"
run_stage extract "$PYTHON_BIN" scripts/01_extract_activations.py --config "$CONFIG"
run_stage bootstrap "$PYTHON_BIN" scripts/02_make_bootstrap_texts.py --config "$CONFIG"
run_stage ar_train "$PYTHON_BIN" scripts/03_train_ar_train.py --config "$CONFIG"
run_stage ar_eval "$PYTHON_BIN" scripts/04_train_ar_eval.py --config "$CONFIG"
run_stage av_sft "$PYTHON_BIN" scripts/05_train_av_sft.py --config "$CONFIG"
run_stage av_generate "$PYTHON_BIN" scripts/06_generate_av_outputs.py --config "$CONFIG"
run_stage rerank "$PYTHON_BIN" scripts/07_rerank_av_outputs.py --config "$CONFIG"
run_stage av_rerank "$PYTHON_BIN" scripts/08_train_av_rerank_sft.py --config "$CONFIG"
run_stage roundtrip "$PYTHON_BIN" scripts/09_eval_roundtrip.py --config "$CONFIG"
run_stage baselines "$PYTHON_BIN" scripts/10_eval_baselines.py --config "$CONFIG"
run_stage token_roles "$PYTHON_BIN" scripts/11_eval_token_roles.py --config "$CONFIG"
run_stage figures "$PYTHON_BIN" scripts/13_make_figures.py --config "$CONFIG"
