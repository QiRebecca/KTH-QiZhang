from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .injection import compute_injection_scale
from .utils import l2_normalize, read_jsonl, stable_int, write_jsonl


GENERIC_TEXT = "This activation describes generic Python code structure without sample-specific context."


@dataclass
class NearestTextAV:
    train_h: np.ndarray
    train_texts: list[str]
    injection_scale: float
    rerank_texts: dict[str, str] | None = None

    def generate(self, h: np.ndarray, activation_ids: list[str], mode: str = "sft", no_injection: bool = False) -> list[str]:
        if no_injection:
            return [GENERIC_TEXT for _ in activation_ids]
        h_norm = l2_normalize(h.astype(np.float32))
        tr_norm = l2_normalize(self.train_h.astype(np.float32))
        sims = h_norm @ tr_norm.T
        idxs = np.argmax(sims, axis=1)
        out = []
        for row_i, idx in enumerate(idxs):
            aid = activation_ids[row_i]
            if mode == "rerank" and self.rerank_texts and aid in self.rerank_texts:
                out.append(self.rerank_texts[aid])
            else:
                out.append(self.train_texts[int(idx)])
        return out

    def sample_candidates(self, h: np.ndarray, activation_ids: list[str], n: int) -> list[list[str]]:
        h_norm = l2_normalize(h.astype(np.float32))
        tr_norm = l2_normalize(self.train_h.astype(np.float32))
        sims = h_norm @ tr_norm.T
        order = np.argsort(-sims, axis=1)
        candidates: list[list[str]] = []
        for i, aid in enumerate(activation_ids):
            row: list[str] = []
            for idx in order[i, : max(n, 1)]:
                text = self.train_texts[int(idx)]
                if text not in row:
                    row.append(text)
            while len(row) < n:
                row.append(f"{self.train_texts[int(order[i, 0])]} Candidate {len(row)} for activation {stable_int(aid) % 97}.")
            candidates.append(row[:n])
        return candidates

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train_h=self.train_h.astype(np.float16), injection_scale=np.array([self.injection_scale], dtype=np.float32))
        write_jsonl(path.with_suffix(".texts.jsonl"), [{"text": t} for t in self.train_texts])
        if self.rerank_texts:
            write_jsonl(path.with_suffix(".rerank.jsonl"), [{"activation_id": k, "text": v} for k, v in sorted(self.rerank_texts.items())])

    @classmethod
    def load(cls, path: Path) -> "NearestTextAV":
        obj = np.load(path)
        texts = [r["text"] for r in read_jsonl(path.with_suffix(".texts.jsonl"))]
        rerank_rows = read_jsonl(path.with_suffix(".rerank.jsonl"))
        rerank = {r["activation_id"]: r["text"] for r in rerank_rows} if rerank_rows else None
        return cls(obj["train_h"].astype(np.float32), texts, float(obj["injection_scale"][0]), rerank)


def train_nearest_text_av(train_h: np.ndarray, train_texts: list[str], cfg: dict[str, Any]) -> NearestTextAV:
    token_norms = np.ones(128, dtype=np.float32)
    scale = compute_injection_scale(token_norms, train_h)
    return NearestTextAV(train_h=train_h.astype(np.float32), train_texts=train_texts, injection_scale=scale)


def _model_name_or_path(cfg: dict[str, Any]) -> str:
    return cfg["model"].get("local_path") or cfg["model"]["name"]


def _torch_dtype(cfg: dict[str, Any], torch: Any) -> Any:
    if cfg["model"].get("dtype") == "bfloat16" and torch.cuda.is_available():
        return torch.bfloat16
    if cfg["model"].get("dtype") == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_lora_lm(cfg: dict[str, Any], adapter_path: Path | None = None, train: bool = False) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _model_name_or_path(cfg)
    local_only = bool(cfg["model"].get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
    injection_token = cfg["av"]["injection_token"]
    if injection_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [injection_token]})
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
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train)
    elif train:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg["av"]["lora_r"]),
            lora_alpha=int(cfg["av"]["lora_alpha"]),
            lora_dropout=float(cfg["av"]["lora_dropout"]),
            target_modules=cfg["av"].get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    if train and bool(cfg["av"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train(train)
    return model, tokenizer


def _activation_scale(model: Any, activations: np.ndarray, cfg: dict[str, Any]) -> float:
    import torch

    emb = model.get_input_embeddings().weight.detach()
    if emb.shape[0] > 8192:
        idx = torch.linspace(0, emb.shape[0] - 1, 8192, device=emb.device).long()
        norms = emb[idx].float().norm(dim=1).cpu().numpy()
    else:
        norms = emb.float().norm(dim=1).cpu().numpy()
    base = compute_injection_scale(norms, activations.astype(np.float32))
    return float(base * float(cfg["av"].get("injection_scale_multiplier", 1.0)))


def _prompt_ids(tokenizer: Any, cfg: dict[str, Any]) -> list[int]:
    prompt = cfg["av"]["prompt_template"]
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    act_id = tokenizer.convert_tokens_to_ids(cfg["av"]["injection_token"])
    if act_id not in ids:
        raise ValueError(f"Prompt template does not contain injection token {cfg['av']['injection_token']!r}")
    return ids


@dataclass
class LoRAAV:
    cfg: dict[str, Any]
    path: Path
    model: Any
    tokenizer: Any
    injection_scale: float
    rerank_texts: dict[str, str] | None = None

    def _make_prompt_embeds(self, h: np.ndarray, no_injection: bool = False) -> tuple[Any, Any]:
        import torch

        device = next(self.model.parameters()).device
        prompt_ids = _prompt_ids(self.tokenizer, self.cfg)
        input_ids = torch.tensor([prompt_ids] * len(h), dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        embeds = self.model.get_input_embeddings()(input_ids)
        act_id = self.tokenizer.convert_tokens_to_ids(self.cfg["av"]["injection_token"])
        pos = (input_ids == act_id).nonzero(as_tuple=False)
        if len(pos) != len(h):
            raise ValueError("Each AV prompt must contain exactly one injection token.")
        if no_injection:
            inj = torch.zeros((len(h), embeds.shape[-1]), device=device, dtype=embeds.dtype)
        else:
            inj = torch.tensor(h.astype(np.float32), device=device, dtype=embeds.dtype) * float(self.injection_scale)
        embeds[pos[:, 0], pos[:, 1], :] = inj
        return embeds, attn

    def generate(self, h: np.ndarray, activation_ids: list[str], mode: str = "sft", no_injection: bool = False) -> list[str]:
        import torch

        if mode == "rerank" and self.rerank_texts:
            cached = [self.rerank_texts.get(aid) for aid in activation_ids]
            if all(text is not None for text in cached):
                return [str(text) for text in cached]
        self.model.eval()
        batch_size = int(self.cfg["av"].get("generation_batch_size", 4))
        max_new = int(self.cfg["av"].get("max_new_tokens", 80))
        temp = float(self.cfg["av"].get("generation_temperature_eval", 0.0))
        texts: list[str] = []
        start = 0
        pbar = tqdm(total=len(h), desc=f"AV generate {mode} bs={batch_size}", unit="act", dynamic_ncols=True)
        while start < len(h):
            current_bs = min(batch_size, len(h) - start)
            h_b = h[start : start + current_bs]
            try:
                embeds, attn = self._make_prompt_embeds(h_b, no_injection=no_injection)
                do_sample = temp > 0
                with torch.inference_mode():
                    out = self.model.generate(
                        inputs_embeds=embeds,
                        attention_mask=attn,
                        max_new_tokens=max_new,
                        do_sample=do_sample,
                        temperature=temp if do_sample else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_bs <= 1:
                    raise
                batch_size = max(1, current_bs // 2)
                pbar.set_description(f"AV generate {mode} bs={batch_size}")
                print(f"CUDA OOM during AV generation; reducing batch size to {batch_size} and retrying from row {start}.")
                continue
            for row in out:
                texts.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
            start += current_bs
            pbar.update(current_bs)
        pbar.close()
        return texts

    def sample_candidates(self, h: np.ndarray, activation_ids: list[str], n: int) -> list[list[str]]:
        import torch

        self.model.eval()
        batch_size = int(self.cfg["av"].get("generation_batch_size", 4))
        max_new = int(self.cfg["av"].get("max_new_tokens", 80))
        temp = float(self.cfg["av"].get("generation_temperature_rerank", 0.7))
        out_all: list[list[str]] = [[] for _ in activation_ids]
        start = 0
        pbar = tqdm(total=len(h), desc=f"AV sample candidates bs={batch_size} n={n}", unit="act", dynamic_ncols=True)
        while start < len(h):
            current_bs = min(batch_size, len(h) - start)
            h_b = h[start : start + current_bs]
            try:
                embeds, attn = self._make_prompt_embeds(h_b, no_injection=False)
                embeds = embeds.repeat_interleave(n, dim=0)
                attn = attn.repeat_interleave(n, dim=0)
                with torch.inference_mode():
                    out = self.model.generate(
                        inputs_embeds=embeds,
                        attention_mask=attn,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=temp,
                        top_p=float(self.cfg["av"].get("generation_top_p", 0.95)),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_bs <= 1:
                    raise
                batch_size = max(1, current_bs // 2)
                pbar.set_description(f"AV sample candidates bs={batch_size} n={n}")
                print(f"CUDA OOM during AV candidate generation; reducing batch size to {batch_size} and retrying from row {start}.")
                continue
            decoded = [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in out]
            for i in range(len(h_b)):
                row = decoded[i * n : (i + 1) * n]
                out_all[start + i] = row
            start += current_bs
            pbar.update(current_bs)
        pbar.close()
        return out_all

    def save(self, path: Path) -> None:
        raise RuntimeError("LoRAAV objects are saved during training with adapter metadata.")

    @classmethod
    def load(cls, cfg: dict[str, Any], path: Path) -> "LoRAAV":
        import json

        model, tokenizer = _load_lora_lm(cfg, adapter_path=path / "adapter", train=False)
        meta = json.loads((path / "av_meta.json").read_text(encoding="utf-8"))
        rerank_file = path / "rerank_texts.jsonl"
        rerank_rows = read_jsonl(rerank_file)
        rerank = {r["activation_id"]: r["text"] for r in rerank_rows} if rerank_rows else None
        return cls(cfg=cfg, path=path, model=model, tokenizer=tokenizer, injection_scale=float(meta["injection_scale"]), rerank_texts=rerank)


def train_lora_av(
    train_h: np.ndarray,
    train_texts: list[str],
    cfg: dict[str, Any],
    path: Path,
    seed: int,
    *,
    base_adapter_path: Path | None = None,
    rerank_texts: dict[str, str] | None = None,
) -> LoRAAV:
    import json
    import torch
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = _load_lora_lm(cfg, adapter_path=(base_adapter_path / "adapter") if base_adapter_path else None, train=True)
    device = next(model.parameters()).device
    injection_scale = _activation_scale(model, train_h, cfg) if base_adapter_path is None else json.loads((base_adapter_path / "av_meta.json").read_text())["injection_scale"]
    prompt_ids = _prompt_ids(tokenizer, cfg)
    act_id = tokenizer.convert_tokens_to_ids(cfg["av"]["injection_token"])
    act_positions = [i for i, tok in enumerate(prompt_ids) if tok == act_id]
    if len(act_positions) != 1:
        raise ValueError("AV prompt must contain exactly one <ACT> token.")
    act_pos = act_positions[0]

    class AVTextDataset(Dataset):
        def __len__(self) -> int:
            return len(train_texts)

        def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
            return train_h[idx].astype(np.float32), train_texts[idx]

    max_target = int(cfg["av"].get("max_target_tokens", 96))
    max_len = int(cfg["model"].get("max_length", 384))

    def collate(rows: list[tuple[np.ndarray, str]]) -> dict[str, Any]:
        hs = np.stack([r[0] for r in rows])
        target_lists = tokenizer([r[1] for r in rows], add_special_tokens=False, truncation=True, max_length=max_target)["input_ids"]
        eos = tokenizer.eos_token_id
        ids_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for target in target_lists:
            target = list(target) + ([eos] if eos is not None else [])
            ids = (prompt_ids + target)[:max_len]
            labels = ([-100] * len(prompt_ids) + target)[:max_len]
            ids_rows.append(ids)
            label_rows.append(labels)
        max_batch_len = max(len(x) for x in ids_rows)
        pad = tokenizer.pad_token_id
        input_ids = torch.full((len(rows), max_batch_len), pad, dtype=torch.long)
        labels = torch.full((len(rows), max_batch_len), -100, dtype=torch.long)
        attn = torch.zeros((len(rows), max_batch_len), dtype=torch.long)
        for i, (ids, labels_i) in enumerate(zip(ids_rows, label_rows)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(labels_i)] = torch.tensor(labels_i, dtype=torch.long)
            attn[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels, "h": torch.tensor(hs, dtype=torch.float32)}

    loader = DataLoader(AVTextDataset(), batch_size=int(cfg["av"]["micro_batch_size"]), shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["av"].get("learning_rate", 2e-4)), weight_decay=float(cfg["av"].get("weight_decay", 0.0)))
    accum = int(cfg["av"].get("gradient_accumulation_steps", 1))
    epochs = int(cfg["av"].get("epochs_rerank_sft" if base_adapter_path else "epochs_sft", 1))
    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"AV train epoch {epoch + 1}/{epochs}", unit="batch", dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            h = batch["h"].to(device, dtype=model.get_input_embeddings().weight.dtype)
            embeds = model.get_input_embeddings()(input_ids).clone()
            embeds[:, act_pos, :] = h * float(injection_scale)
            out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels, use_cache=False)
            loss = out.loss
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, float(cfg["av"].get("max_grad_norm", 1.0)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            if torch.cuda.is_available():
                pbar.set_postfix(loss=f"{loss.item():.4g}", mem=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4g}")
    model.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "tokenizer")
    meta = {
        "kind": "lora_av",
        "injection_scale": float(injection_scale),
        "seed": seed,
        "global_steps": global_step,
        "prompt_template": cfg["av"]["prompt_template"],
        "injection_token": cfg["av"]["injection_token"],
        "base_adapter_path": str(base_adapter_path) if base_adapter_path else None,
    }
    (path / "av_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if rerank_texts:
        write_jsonl(path / "rerank_texts.jsonl", [{"activation_id": k, "text": v} for k, v in sorted(rerank_texts.items())])
    return LoRAAV.load(cfg, path)


def load_av(cfg: dict[str, Any], path: Path) -> Any:
    if cfg["av"].get("backend") == "nearest_text_smoke":
        return NearestTextAV.load(path)
    return LoRAAV.load(cfg, path)
