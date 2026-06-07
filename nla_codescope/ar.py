from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .utils import l2_normalize, stable_int


def _tokenize_text(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\s]", text.lower())


def hashed_features(texts: list[str], dim: int) -> np.ndarray:
    x = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _tokenize_text(text):
            idx = stable_int(tok) % dim
            sign = 1.0 if stable_int("sign:" + tok) % 2 == 0 else -1.0
            x[i, idx] += sign
    return l2_normalize(x)


@dataclass
class HashedRidgeAR:
    feature_dim: int
    weights: np.ndarray
    bias: np.ndarray
    train_mean: np.ndarray

    def predict(self, texts: list[str]) -> np.ndarray:
        x = hashed_features(texts, self.feature_dim)
        return (x @ self.weights + self.bias).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, feature_dim=self.feature_dim, weights=self.weights, bias=self.bias, train_mean=self.train_mean)

    @classmethod
    def load(cls, path: Path) -> "HashedRidgeAR":
        obj = np.load(path)
        return cls(int(obj["feature_dim"]), obj["weights"], obj["bias"], obj["train_mean"])


def train_hashed_ridge(texts: list[str], targets: np.ndarray, cfg: dict[str, Any]) -> HashedRidgeAR:
    feature_dim = int(cfg["ar"].get("feature_dim", 512))
    lam = float(cfg["ar"].get("ridge_lambda", 1.0))
    x = hashed_features(texts, feature_dim)
    y = targets.astype(np.float32)
    bias = y.mean(axis=0)
    y0 = y - bias
    if x.shape[0] < feature_dim:
        gram = x @ x.T + lam * np.eye(x.shape[0], dtype=np.float32)
        alpha = np.linalg.solve(gram, y0)
        w = (x.T @ alpha).astype(np.float32)
    else:
        xtx = x.T @ x + lam * np.eye(feature_dim, dtype=np.float32)
        w = np.linalg.solve(xtx, x.T @ y0).astype(np.float32)
    return HashedRidgeAR(feature_dim=feature_dim, weights=w, bias=bias.astype(np.float32), train_mean=bias.astype(np.float32))


def load_or_train_ar(texts: list[str], targets: np.ndarray, cfg: dict[str, Any], path: Path) -> HashedRidgeAR:
    if path.exists():
        return HashedRidgeAR.load(path)
    model = train_hashed_ridge(texts, targets, cfg)
    model.save(path)
    return model


def _model_name_or_path(cfg: dict[str, Any]) -> str:
    return cfg["model"].get("local_path") or cfg["model"]["name"]


def _torch_dtype(cfg: dict[str, Any], torch: Any) -> Any:
    if cfg["model"].get("dtype") == "bfloat16" and torch.cuda.is_available():
        return torch.bfloat16
    if cfg["model"].get("dtype") == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _truncate_qwen_layers(model: Any, cfg: dict[str, Any]) -> None:
    import torch

    keep = int(cfg["ar"].get("truncate_to_layers", 0) or 0)
    if keep <= 0:
        return
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Could not find model.model.layers for AR truncation.")
    if keep > len(layers):
        raise ValueError(f"truncate_to_layers={keep} exceeds model layer count {len(layers)}")
    backbone.layers = torch.nn.ModuleList(list(layers[:keep]))
    if hasattr(backbone, "config"):
        backbone.config.num_hidden_layers = keep
    if hasattr(model, "config"):
        model.config.num_hidden_layers = keep


def _load_lora_backbone(cfg: dict[str, Any], adapter_path: Path | None = None, train: bool = False) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _model_name_or_path(cfg)
    local_only = bool(cfg["model"].get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(cfg, torch),
        "trust_remote_code": True,
        "local_files_only": local_only,
        "low_cpu_mem_usage": bool(cfg["model"].get("low_cpu_mem_usage", True)),
    }
    attn_impl = cfg["model"].get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": int(cfg.get("runtime", {}).get("cuda_device", 0))}
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    _truncate_qwen_layers(model, cfg)
    model.config.use_cache = False
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train)
    elif train:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg["ar"]["lora_r"]),
            lora_alpha=int(cfg["ar"]["lora_alpha"]),
            lora_dropout=float(cfg["ar"]["lora_dropout"]),
            target_modules=cfg["ar"].get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    if train and bool(cfg["ar"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train(train)
    return model, tokenizer


@dataclass
class LoRAAR:
    cfg: dict[str, Any]
    path: Path
    model: Any
    tokenizer: Any
    value_head: Any

    def predict(self, texts: list[str]) -> np.ndarray:
        import torch

        device = next(self.model.parameters()).device
        max_len = int(self.cfg["ar"].get("max_text_length", 160))
        batch_size = int(self.cfg["ar"].get("eval_batch_size", max(1, int(self.cfg["ar"].get("micro_batch_size", 4)))))
        outs: list[np.ndarray] = []
        self.model.eval()
        self.value_head.eval()
        for start in tqdm(range(0, len(texts), batch_size), desc="AR predict", unit="text", dynamic_ncols=True):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self.model(**enc, output_hidden_states=True, use_cache=False)
                lengths = enc["attention_mask"].sum(dim=1) - 1
                states = out.hidden_states[-1][torch.arange(len(batch), device=device), lengths]
                pred = self.value_head(states).float().cpu().numpy()
            outs.append(pred.astype(np.float32))
        return np.vstack(outs) if outs else np.zeros((0, int(self.cfg.get("model", {}).get("d_model", 0))), dtype=np.float32)


def train_lora_ar(texts: list[str], targets: np.ndarray, cfg: dict[str, Any], path: Path, seed: int) -> LoRAAR:
    import json
    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_lora_backbone(cfg, train=True)
    device = next(model.parameters()).device
    d_model = int(targets.shape[1])
    value_head = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=_torch_dtype(cfg, torch))
    value_head.train()

    class TextTargetDataset(Dataset):
        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
            return texts[idx], targets[idx].astype(np.float32)

    max_len = int(cfg["ar"].get("max_text_length", 160))

    def collate(rows: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
        batch_texts = [r[0] for r in rows]
        y = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32)
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc["target"] = y
        return enc

    loader = DataLoader(TextTargetDataset(), batch_size=int(cfg["ar"]["micro_batch_size"]), shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["ar"].get("learning_rate", 2e-4)), weight_decay=float(cfg["ar"].get("weight_decay", 0.0)))
    accum = int(cfg["ar"].get("gradient_accumulation_steps", 1))
    epochs = int(cfg["ar"].get("epochs", 1))
    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"AR train epoch {epoch + 1}/{epochs}", unit="batch", dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            y = batch.pop("target").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, output_hidden_states=True, use_cache=False)
            lengths = batch["attention_mask"].sum(dim=1) - 1
            states = out.hidden_states[-1][torch.arange(y.shape[0], device=device), lengths]
            pred = value_head(states).float()
            loss = torch.nn.functional.mse_loss(pred, y)
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, float(cfg["ar"].get("max_grad_norm", 1.0)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            if torch.cuda.is_available():
                pbar.set_postfix(loss=f"{loss.item():.4g}", mem=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4g}")
    model.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "tokenizer")
    torch.save(value_head.state_dict(), path / "value_head.pt")
    meta = {
        "kind": "lora_ar",
        "d_model": d_model,
        "seed": seed,
        "global_steps": global_step,
        "truncate_to_layers": int(cfg["ar"].get("truncate_to_layers", 0) or 0),
    }
    (path / "ar_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return LoRAAR.load(cfg, path)


def load_lora_ar(cfg: dict[str, Any], path: Path) -> LoRAAR:
    return LoRAAR.load(cfg, path)


def _load_value_head(cfg: dict[str, Any], path: Path, model: Any) -> Any:
    import torch

    d_model = int(getattr(model.config, "hidden_size", 0))
    device = next(model.parameters()).device
    value_head = torch.nn.Linear(d_model, d_model, bias=False, device=device, dtype=_torch_dtype(cfg, torch))
    state = torch.load(path / "value_head.pt", map_location=device)
    value_head.load_state_dict(state)
    value_head.eval()
    return value_head


@classmethod
def _lora_ar_load(cls: type[LoRAAR], cfg: dict[str, Any], path: Path) -> LoRAAR:
    model, tokenizer = _load_lora_backbone(cfg, adapter_path=path / "adapter", train=False)
    value_head = _load_value_head(cfg, path, model)
    return cls(cfg=cfg, path=path, model=model, tokenizer=tokenizer, value_head=value_head)


LoRAAR.load = _lora_ar_load  # type: ignore[attr-defined]


def load_ar(cfg: dict[str, Any], path: Path) -> Any:
    if cfg["ar"].get("backend") == "hashed_ridge_smoke":
        return HashedRidgeAR.load(path)
    return load_lora_ar(cfg, path)
