"""Arm 3: a prompted local LLM, few-shot, scored by its Yes/No probability.

No training. The model reads each validation note's baseline text in full (no
512-token limit) and answers one question: does this note document that the
patient has dementia? Nothing is generated beyond the first answer token: the
score is P(Yes) / (P(Yes) + P(No)) at that position, read from llama.cpp's
log-probabilities.

The few-shot examples are real TRAIN notes drawn by a fixed rule, never chosen
by eye (config arm3.examples): 2 coded positive, 2 coded negative, all at most
max_chars long, negatives with no explicit dementia or drug mention (arm 0
score 0), shown in the order pos, neg, neg, pos.

The model runs in llama.cpp's llama-server, bound to 127.0.0.1. Its output is
discarded and it saves nothing to disk, so note text never leaves this machine.

Run:
  timing on N real notes:  .venv/bin/python -m dementia_undercoding.arm3_llm --model medgemma-1.5-4b --n 200
  full validation:         caffeinate -i .venv/bin/python -m dementia_undercoding.arm3_llm --model qwen3.5-9b
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .arm0_regex import score_note
from .cohort import load_config
from .evaluate import report, save_scores, score
from .splits import load_notes

SYSTEM = (
    "You read the history sections of a hospital discharge summary: history of present illness, "
    "past medical history and social history. Dementia includes Alzheimer's disease, vascular "
    "dementia, Lewy body dementia, frontotemporal dementia and dementia of unspecified type. "
    "Answer No if the note describes only mild cognitive impairment, only delirium or acute "
    "confusion, dementia that is denied or ruled out, or dementia in someone other than the "
    "patient. Answer with one word: Yes or No."
)
QUESTION = "Does this note document that the patient has dementia? Answer Yes or No."


# --- the prompt -----------------------------------------------------------------------


def pick_examples(
    train: pd.DataFrame, label: str, *, n_pos: int, n_neg: int, max_chars: int, seed: int,
    window: int, uncertain_weight: float,
) -> list[tuple[str, bool]]:
    """Few-shot examples from TRAIN notes, by rule. Returns [(text, is_positive)] in prompt order.

    Candidates are notes of at most max_chars characters. Negatives must also
    have arm 0 score 0 for both diagnosis terms and drugs: an uncoded note that
    mentions dementia is the very undercoding this project looks for, and
    showing it to the model as "No" would teach the wrong answer.
    Candidates are sorted by hadm_id before the draw, so row order cannot change it.
    Order: positives and negatives interleaved pos, neg, neg, pos (for 2 + 2).
    """
    short = train[train["text"].str.len().between(1, max_chars)].sort_values("hadm_id")
    pos = short[short[label] == 1]
    neg = short[short[label] == 0]
    explicit = neg["text"].map(lambda t: sum(score_note(t, window, uncertain_weight)) > 0)
    neg = neg[~explicit]

    rng = np.random.default_rng(seed)
    pos_texts = pos["text"].to_numpy()[rng.choice(len(pos), size=n_pos, replace=False)]
    neg_texts = neg["text"].to_numpy()[rng.choice(len(neg), size=n_neg, replace=False)]

    pos_list = [(t, True) for t in pos_texts]
    neg_list = [(t, False) for t in neg_texts]
    order = []
    while pos_list or neg_list:                    # pos, neg, neg, pos, pos, neg, ...
        if pos_list:
            order.append(pos_list.pop(0))
        order.extend(neg_list.pop(0) for _ in range(2) if neg_list)
        if pos_list:
            order.append(pos_list.pop(0))
    return order


def build_messages(examples: list[tuple[str, bool]], note: str, prefill: str = "") -> list[dict]:
    """Chat messages: instructions, the examples as earlier turns, then the note.

    Everything before the final note is identical for every note, so
    llama-server reuses it from its cache instead of reading it again.
    prefill starts the model's answer: for a model that "thinks" first, an
    empty, closed thought block, so the next token is the answer.
    """
    messages = [{"role": "system", "content": SYSTEM}]
    for text, positive in examples:
        messages.append({"role": "user", "content": f"Note:\n{text}\n\n{QUESTION}"})
        messages.append({"role": "assistant", "content": "Yes" if positive else "No"})
    messages.append({"role": "user", "content": f"Note:\n{note}\n\n{QUESTION}"})
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def yes_probability(top_logprobs: list[dict]) -> float:
    """P(Yes) / (P(Yes) + P(No)) from the first answer token's top candidates.

    "Yes", " yes", "YES" all count as Yes. NaN if neither answer is among the
    candidates: the model did not treat this as a yes/no question at all.
    """
    p = {"yes": 0.0, "no": 0.0}
    for cand in top_logprobs:
        word = cand["token"].strip().lower()
        if word in p:
            p[word] += math.exp(cand["logprob"])
    total = p["yes"] + p["no"]
    return p["yes"] / total if total > 0 else float("nan")


# --- the local server -----------------------------------------------------------------


class LlamaServer:
    """llama-server on 127.0.0.1 for the length of a `with` block. Prints nothing, saves nothing."""

    def __init__(self, model_path: str, *, port: int, ctx: int, gpu_layers: int = 99):
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [
            "llama-server", "-m", str(Path(model_path).expanduser()),
            "--host", "127.0.0.1", "--port", str(port),   # this machine only
            "-c", str(ctx), "-ngl", str(gpu_layers), "-np", "1",
            "--no-webui", "--offline", "--log-disable",
        ]

    def __enter__(self) -> LlamaServer:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError):
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become ready within 10 minutes")

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=60)

    def first_token(self, messages: list[dict], top: int = 20) -> tuple[list[dict], dict]:
        """(top candidates for the first answer token, llama-server timings) for one prompt."""
        body = {
            "messages": messages, "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": top,
            "chat_template_kwargs": {"enable_thinking": False},     # Qwen3.5: answer, don't think
        }
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        return out["choices"][0]["logprobs"]["content"][0]["top_logprobs"], out.get("timings", {})


def score_notes(server: LlamaServer, examples, texts: list[str], prefill: str, max_note_chars: int,
                progress_every: int = 500) -> tuple[np.ndarray, np.ndarray, int]:
    """(scores, seconds per note, notes cut to max_note_chars), one note at a time, in order."""
    scores, seconds, cut = np.empty(len(texts)), np.empty(len(texts)), 0
    start = time.perf_counter()
    for i, text in enumerate(texts):
        if len(text) > max_note_chars:
            text, cut = text[:max_note_chars], cut + 1
        t = time.perf_counter()
        top, _ = server.first_token(build_messages(examples, text, prefill))
        seconds[i] = time.perf_counter() - t
        scores[i] = yes_probability(top)
        if progress_every and (i + 1) % progress_every == 0:
            done = time.perf_counter() - start
            print(f"{i + 1:>6,}/{len(texts):,} notes  {done / 3600:.2f} h  "
                  f"eta {done / (i + 1) * (len(texts) - i - 1) / 3600:.2f} h", flush=True)
    return scores, seconds, cut


# --- the run --------------------------------------------------------------------------


def run(config_path: str | Path = "config/config.yaml", model: str = "medgemma-1.5-4b",
        n: int | None = None, label: str = "dementia_core") -> dict:
    import mlflow  # imported here, not at the top: CI installs only core packages

    cfg = load_config(config_path)
    acfg, mcfg, ecfg = cfg["arm3"], cfg["arm3"]["models"][model], cfg["arm3"]["examples"]

    notes = load_notes(config_path, columns=[label, "text"])
    notes = notes[notes["split"].isin(["train", "validation"])]   # drop test first
    examples = pick_examples(
        notes[notes["split"] == "train"], label, n_pos=ecfg["n_pos"], n_neg=ecfg["n_neg"],
        max_chars=ecfg["max_chars"], seed=ecfg["seed"],
        window=cfg["arm0"]["window"], uncertain_weight=cfg["arm0"]["uncertain_weight"],
    )
    val = notes[notes["split"] == "validation"].sort_values("hadm_id")
    if n is not None:                                  # fixed random sample, same for every model
        val = val.sample(n=n, random_state=acfg["sample_seed"]).sort_values("hadm_id")

    with LlamaServer(mcfg["path"], port=acfg["port"], ctx=acfg["ctx"]) as server:
        scores, seconds, cut = score_notes(server, examples, val["text"].tolist(),
                                           mcfg["prefill"], acfg["max_note_chars"])

    no_answer = int(np.isnan(scores).sum())
    result = score(val[label], np.nan_to_num(scores, nan=0.5))   # no answer ranks in the middle
    name = f"arm3-{model}" + (f"-n{n}" if n is not None else "")
    save_scores(val["hadm_id"], scores, Path(cfg["paths"]["scores"]) / f"{name}-{label}.parquet")

    full = int((notes["split"] == "validation").sum())
    print(report(f"{name}, {label}, validation", result))
    print(f"seconds per note: median {np.median(seconds):.2f}, p90 {np.quantile(seconds, 0.9):.2f}, "
          f"total {seconds.sum() / 3600:.2f} h; all {full:,} validation notes would take about "
          f"{seconds.mean() * full / 3600:.1f} h")
    print(f"no Yes/No among the top answers: {no_answer:,} notes; cut to {acfg['max_note_chars']:,} "
          f"characters: {cut:,} notes")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"{name}-{label}"):
        mlflow.log_params({"arm": "3", "model": model, "label": label, "scored_on": "validation",
                           "n_notes": len(val), "ctx": acfg["ctx"], "prefill": mcfg["prefill"],
                           **{f"examples_{k}": v for k, v in ecfg.items()}})
        mlflow.log_metrics({f"val_{k}": v for k, v in result.items()})
        mlflow.log_metrics({"seconds_per_note_median": float(np.median(seconds)),
                            "seconds_total": float(seconds.sum()),
                            "no_answer": no_answer, "cut_notes": cut})
    return {"val": result, "seconds": seconds, "no_answer": no_answer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm 3: prompted local LLM via llama.cpp.")
    parser.add_argument("--model", required=True, help="a key under arm3.models in config.yaml")
    parser.add_argument("--n", type=int, default=None, help="score a fixed random sample of n notes")
    args = parser.parse_args()
    run(model=args.model, n=args.n)