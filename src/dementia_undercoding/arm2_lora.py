"""Arm 2: LoRA fine-tune of Bio_ClinicalBERT.

Trains on train (validation already carved out), scores on validation, logs
parameters and metrics to MLflow. Test rows are dropped before anything else
runs. Settings come from the `lora:` block in config.yaml.

The trained adapter is saved under data/ (gitignored), never to MLflow: it is
learned from note text, so it stays on this machine with the notes.

Run:
  smoke test:  .venv/bin/python -m dementia_undercoding.arm2_lora --smoke
  full run:    caffeinate -i .venv/bin/python -m dementia_undercoding.arm2_lora
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .cohort import load_config
from .evaluate import report, save_scores, score
from .splits import load_notes


def load_base(model_name: str) -> nn.Module:
    """Pretrained BERT with a fresh one-output head: the head's score goes into a sigmoid."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)


def add_lora(base: nn.Module, lora: dict) -> nn.Module:
    """Freeze every base weight; add LoRA adapters; keep the new head trainable."""
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,           # also marks the classifier head as trainable
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
    )
    return get_peft_model(base, config)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) weights."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_training_step(
    model: nn.Module,
    device: str,
    batch_size: int,
    max_len: int = 512,
    steps: int = 10,
    warmup: int = 2,
) -> float:
    """Seconds per training step on made-up token ids (no note text involved).

    Every sequence is the full max_len, so this is the worst case: real notes
    shorter than 512 tokens are padded up to it under truncation.
    """
    model.to(device).train()
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch_size, max_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 2, (batch_size, 1), device=device).float()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def step():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    _sync(device)
    return (time.perf_counter() - start) / steps


def _encode(tokenizer, texts: list[str], max_len: int, device: str) -> dict:
    """Text -> token ids, cut at max_len, padded only to the longest note in this batch."""
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def train(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    labels: np.ndarray,
    *,
    device: str,
    batch_size: int,
    learning_rate: float,
    warmup_frac: float,
    max_grad_norm: float,
    max_len: int,
    seed: int,
    log_every: int = 100,
    log: Callable[[int, int, float, float], None] | None = None,
) -> None:
    """One pass over the training notes, in a seeded random order.

    log(step, total_steps, mean_loss_since_last_log, elapsed_seconds) is called
    every log_every steps and at the end. It only ever receives numbers.
    """
    model.to(device).train()
    order = np.random.default_rng(seed).permutation(len(texts))
    total_steps = math.ceil(len(order) / batch_size)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warmup_frac * total_steps), num_training_steps=total_steps
    )
    loss_fn = nn.BCEWithLogitsLoss()

    running = torch.zeros((), device=device)
    since_log = 0
    start = time.perf_counter()
    for step in range(total_steps):
        idx = order[step * batch_size:(step + 1) * batch_size]
        batch = _encode(tokenizer, [texts[i] for i in idx], max_len, device)
        y = torch.tensor(labels[idx], dtype=torch.float32, device=device).unsqueeze(1)

        logits = model(**batch).logits                  # 1. predict
        loss = loss_fn(logits, y)                        # 2. measure the error
        optimizer.zero_grad()
        loss.backward()                                  # 3. gradients
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()                                 # 4. nudge
        scheduler.step()                                 # move the learning rate along its schedule

        running += loss.detach()
        since_log += 1
        if log is not None and ((step + 1) % log_every == 0 or step + 1 == total_steps):
            log(step + 1, total_steps, (running / since_log).item(), time.perf_counter() - start)
            running.zero_()
            since_log = 0


@torch.no_grad()
def predict(
    model: nn.Module, tokenizer, texts: list[str], *, device: str, max_len: int, batch_size: int = 64
) -> np.ndarray:
    """P(label = 1) for each note, in the order given."""
    model.to(device).eval()                              # dropout off
    out = []
    for i in range(0, len(texts), batch_size):
        batch = _encode(tokenizer, texts[i:i + batch_size], max_len, device)
        out.append(torch.sigmoid(model(**batch).logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def run(
    config_path: str | Path = "config/config.yaml",
    variant: str = "main",
    smoke: bool = False,
) -> dict:
    """Train, save the adapter under data/, score validation, log to MLflow.

    smoke=True trains on 800 notes and scores 2,000 validation notes, to check
    the whole path end to end in a couple of minutes. Smoke runs are tagged in
    MLflow and save nothing.
    """
    cfg = load_config(config_path)
    lcfg, tcfg = cfg["lora"], cfg["lora"]["train"]
    label, seed, device = tcfg["label"], tcfg["seed"], "mps" if torch.backends.mps.is_available() else "cpu"

    notes = load_notes(config_path, columns=[label, "text"])
    notes = notes[notes["split"].isin(["train", "validation"])]   # drop test first
    train_df = notes[notes["split"] == "train"]
    val_df = notes[notes["split"] == "validation"]
    if smoke:
        train_df = train_df.sample(n=min(800, len(train_df)), random_state=seed)
        val_df = val_df.sample(n=min(2_000, len(val_df)), random_state=seed)
    health_df = train_df.sample(n=min(tcfg["health_check_train_n"], len(train_df)), random_state=seed)

    torch.manual_seed(seed)          # fixes the head and LoRA starting values, and dropout
    tokenizer = AutoTokenizer.from_pretrained(lcfg["model"])
    model = add_lora(load_base(lcfg["model"]), lcfg[variant])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run_name = f"arm2-lora-{variant}-{label}" + ("-smoke" if smoke else "")
    with mlflow.start_run(run_name=run_name, tags={"smoke": str(smoke)}):
        mlflow.log_params({
            "arm": "2", "variant": variant, "label": label, "scored_on": "validation",
            "model": lcfg["model"], "max_len": lcfg["max_len"],
            **{f"lora_{k}": v for k, v in lcfg[variant].items()},
            **tcfg, "n_train": len(train_df), "device": device,
        })

        def log(step, total, loss, elapsed):
            eta_h = elapsed / step * (total - step) / 3600
            print(f"step {step:>6,}/{total:,}  loss {loss:.4f}  "
                  f"elapsed {elapsed / 3600:.2f} h  eta {eta_h:.2f} h", flush=True)
            mlflow.log_metric("train_loss", loss, step=step)

        start = time.perf_counter()
        train(
            model, tokenizer, train_df["text"].tolist(), train_df[label].to_numpy(),
            device=device, batch_size=tcfg["batch_size"], learning_rate=tcfg["learning_rate"],
            warmup_frac=tcfg["warmup_frac"], max_grad_norm=tcfg["max_grad_norm"],
            max_len=lcfg["max_len"], seed=seed, log=log,
        )
        fit_seconds = time.perf_counter() - start
        if not smoke:                    # save before scoring, so a crash there loses no training
            out = Path(lcfg["out_dir"]) / variant
            model.save_pretrained(out)
            print(f"adapter saved to {out}", flush=True)

        print("scoring validation ...", flush=True)
        val_scores = predict(model, tokenizer, val_df["text"].tolist(),
                             device=device, max_len=lcfg["max_len"])
        val_result = score(val_df[label], val_scores)
        if not smoke:
            save_scores(val_df["hadm_id"], val_scores,
                        Path(cfg["paths"]["scores"]) / f"arm2-lora-{variant}-{label}.parquet")

        print("scoring a train sample (health check) ...", flush=True)
        health_result = score(health_df[label], predict(model, tokenizer, health_df["text"].tolist(),
                                                        device=device, max_len=lcfg["max_len"]))

        print(report(f"arm 2 LoRA {variant}, {label}, validation  ", val_result))
        print(report(f"arm 2 LoRA {variant}, {label}, train sample", health_result))
        mlflow.log_metrics({f"val_{k}": v for k, v in val_result.items()})
        mlflow.log_metrics({f"train_sample_{k}": v for k, v in health_result.items()})
        mlflow.log_metric("fit_seconds", fit_seconds)

    return {"val": val_result, "train_sample": health_result, "fit_seconds": fit_seconds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm 2: LoRA fine-tune of Bio_ClinicalBERT.")
    parser.add_argument("--variant", default="main", choices=["main", "fallback"])
    parser.add_argument("--smoke", action="store_true", help="800 train notes, 2,000 validation notes")
    args = parser.parse_args()
    run(variant=args.variant, smoke=args.smoke)