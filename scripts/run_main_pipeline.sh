#!/usr/bin/env bash
set -e
PYTHON_BIN="${PYTHON:-python3}"
PY_SITE="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
export LD_LIBRARY_PATH="$PY_SITE/nvidia/cusparselt/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/nccl/lib:$PY_SITE/nvidia/cublas/lib:$PY_SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON_BIN" scripts/00_prepare_data.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/01_extract_activations.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/02_make_bootstrap_texts.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/03_train_ar_train.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/04_train_ar_eval.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/05_train_av_sft.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/06_generate_av_outputs.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/07_rerank_av_outputs.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/08_train_av_rerank_sft.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/09_eval_roundtrip.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/10_eval_baselines.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/11_eval_token_roles.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" scripts/13_make_figures.py --config configs/main_qwen25coder_1p5b_L18.yaml
"$PYTHON_BIN" -m pytest -q
