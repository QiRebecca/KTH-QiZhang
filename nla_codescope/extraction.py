from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from .token_roles import choose_token_positions, python_tokens
from .utils import l2_normalize, stable_int


def _smoke_activation(code: str, token_text: str, role: str, d_model: int) -> np.ndarray:
    rng = np.random.default_rng(stable_int(code + "\n" + token_text + "\n" + role))
    base = rng.normal(size=d_model).astype(np.float32)
    role_rng = np.random.default_rng(stable_int(role))
    role_vec = role_rng.normal(size=d_model).astype(np.float32)
    text_rng = np.random.default_rng(stable_int(token_text))
    text_vec = text_rng.normal(size=d_model).astype(np.float32)
    return (0.60 * l2_normalize(base[None, :])[0] + 0.25 * l2_normalize(role_vec[None, :])[0] + 0.15 * l2_normalize(text_vec[None, :])[0]).astype(np.float32)


def extract_smoke(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    d_model = int(cfg["model"].get("d_model", 64))
    per_fn = int(cfg["data"]["activations_per_function"])
    meta: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for row in rows:
        toks = python_tokens(row["code"])
        positions = choose_token_positions(toks, per_fn, stable_int(row["function_id"]))
        for j, pos in enumerate(positions):
            tok = toks[pos]
            lines = row["code"].splitlines()
            context = lines[min(len(lines) - 1, max(0, pos % max(1, len(lines))))] if lines else ""
            h = _smoke_activation(row["code"], tok.text, tok.role, d_model)
            activation_id = f"{row['function_id']}:{j}:{pos}"
            meta.append({
                "activation_id": activation_id,
                "function_id": row["function_id"],
                "code_hash": row["code_hash"],
                "split": row["split"],
                "token_index": int(pos),
                "token_text": tok.text,
                "token_role": tok.role,
                "summary_text": row["summary"],
                "local_context_for_bootstrap_only": context,
                "context_excerpt_for_human_display_only": context,
            })
            vectors.append(h)
    return meta, np.vstack(vectors)


def extract_hf(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"].get("local_path") or cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=bool(cfg["model"].get("local_files_only", False)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if cfg["model"].get("dtype") == "bfloat16" and torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": bool(cfg["model"].get("low_cpu_mem_usage", True)),
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = cfg["model"].get("device_map", "auto")
    attn_impl = cfg["model"].get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if cfg["model"].get("local_files_only", False):
        model_kwargs["local_files_only"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    model.config.use_cache = False
    hidden_idx = int(cfg["model"]["hidden_states_index"])
    expected = int(cfg["model"]["target_block_index_zero_based"]) + 1
    if hidden_idx != expected:
        raise ValueError(f"hidden_states_index must equal target_block_index_zero_based + 1, got {hidden_idx} vs {expected}")
    per_fn = int(cfg["data"]["activations_per_function"])
    max_length = int(cfg["model"]["max_length"])
    batch_size = int(cfg.get("activation", {}).get("extraction_batch_size", 1))
    meta: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    device = next(model.parameters()).device
    prepared: list[tuple[dict[str, Any], list[Any], list[int]]] = []
    for row in rows:
        toks = python_tokens(row["code"])
        positions = choose_token_positions(toks, per_fn, stable_int(row["function_id"]))
        prepared.append((row, toks, positions))

    start = 0
    pbar = tqdm(total=len(prepared), desc=f"extract activations bs={batch_size}", unit="fn", dynamic_ncols=True)
    while start < len(prepared):
        current_bs = min(batch_size, len(prepared) - start)
        batch = prepared[start : start + current_bs]
        codes = [row["code"] for row, _, _ in batch]
        enc = tokenizer(codes, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        try:
            with torch.inference_mode():
                outputs = model(**enc, output_hidden_states=True, use_cache=False)
                hs_batch = outputs.hidden_states[hidden_idx].detach().float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            del enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if current_bs <= 1:
                raise
            batch_size = max(1, current_bs // 2)
            pbar.set_description(f"extract activations bs={batch_size}")
            print(f"CUDA OOM during extraction; reducing batch size to {batch_size} and retrying from row {start}.")
            continue
        seq_lens = enc["attention_mask"].sum(dim=1).detach().cpu().numpy().astype(int)
        del outputs, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for b, (row, toks, positions) in enumerate(batch):
            hs = hs_batch[b]
            seq_len = int(seq_lens[b])
            for j, pos in enumerate(positions):
                tok_idx = min(pos, seq_len - 1, hs.shape[0] - 1)
                tok = toks[min(pos, len(toks) - 1)]
                activation_id = f"{row['function_id']}:{j}:{tok_idx}"
                meta.append({
                    "activation_id": activation_id,
                    "function_id": row["function_id"],
                    "code_hash": row["code_hash"],
                    "split": row["split"],
                    "token_index": int(tok_idx),
                    "token_text": tok.text,
                    "token_role": tok.role,
                    "summary_text": row["summary"],
                    "local_context_for_bootstrap_only": row["code"][:300],
                    "context_excerpt_for_human_display_only": row["code"][:300],
                })
                vectors.append(hs[tok_idx].astype(np.float32))
        start += current_bs
        pbar.update(current_bs)
        if torch.cuda.is_available() and start % max(batch_size * 5, 1) == 0:
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            pbar.set_postfix_str(f"cuda_alloc={alloc:.1f}GB reserved={reserved:.1f}GB")
    pbar.close()
    return meta, np.vstack(vectors)


def extract_activations(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    if cfg.get("mode") == "smoke":
        return extract_smoke(rows, cfg)
    return extract_hf(rows, cfg)
