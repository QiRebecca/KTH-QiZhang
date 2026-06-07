#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/main_qwen25coder_1p5b_L18_autodl.yaml}"
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

run_stage() {
  local name="$1"
  shift
  if [[ "$STAGE" == "all" || "$STAGE" == "$name" ]]; then
    echo "===== stage: $name ====="
    "$@"
  fi
}

run_stage preflight "$PYTHON_BIN" scripts/check_h800_ready.py --config "$CONFIG"
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
run_stage tests "$PYTHON_BIN" -m pytest -q
