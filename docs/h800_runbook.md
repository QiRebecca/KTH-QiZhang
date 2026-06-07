# H800 Runbook

The H800 profile is `configs/main_qwen25coder_1p5b_L18_h800.yaml`.

## Strategy

- Use one H800 as a single large GPU rather than model parallelism. Qwen2.5-Coder-1.5B is small enough for single-device bf16.
- Keep target sequence length at 384 and use LoRA only. Do not full-finetune.
- Use bf16, TF32, SDPA attention, `use_cache=False` during hidden-state extraction, and batched extraction.
- Cache large assets under `${NLA_STORAGE_ROOT}`, not the root disk. The H800 config currently points to a local ModelScope model path and a local parquet CodeSearchNet-style dataset path.
- Start conservatively: extraction batch size 16, AV micro-batch 4 with grad accumulation 4, AR micro-batch 8 with grad accumulation 2.
- If H800 memory is underused, increase only one knob at a time:
  - extraction `activation.extraction_batch_size`: 16 -> 24 -> 32
  - AV `micro_batch_size`: 4 -> 6 -> 8
  - AR `micro_batch_size`: 8 -> 12 -> 16
- If OOM occurs, lower the last changed knob, keep gradient checkpointing enabled, and rerun the failed stage using `STAGE=<stage>`.

## Commands

Preflight only:

```bash
cd ${NLA_STORAGE_ROOT}/nla-codescope
PYTHON=python CONFIG=configs/main_qwen25coder_1p5b_L18_h800.yaml STAGE=preflight bash scripts/run_main_h800.sh
```

Validate local tokenizer/config:

```bash
PYTHON=python STAGE=cache_hf bash scripts/run_main_h800.sh
```

Download the mirrored CodeSearchNet Python parquet files:

```bash
PYTHON=python STAGE=cache_data bash scripts/run_main_h800.sh
```

Validate/load model weights:

```bash
PYTHON=python STAGE=cache_model bash scripts/run_main_h800.sh
```

Run extraction only:

```bash
PYTHON=python STAGE=extract bash scripts/run_main_h800.sh
```

Run all stages:

```bash
PYTHON=python bash scripts/run_main_h800.sh
```

## Current Known Constraint

The current remote host has no `/dev/nvidia*`, so H800 preflight must fail until the GPU is allocated. Qwen weights were pulled from ModelScope into `${NLA_STORAGE_ROOT}/modelscope_cache`; CodeSearchNet-style parquet files are pulled from `hf-mirror.com` into `${NLA_STORAGE_ROOT}/datasets`.
