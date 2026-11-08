"""Arm 2 setup on a tiny, untrained BERT: no download, no notes.

Skipped where torch / peft are not installed (CI installs only the core deps).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

# One thread: pytest runs every test file in one process, and LightGBM's tests
# load a second copy of OpenMP. Two copies both starting worker threads
# deadlock on macOS. The real runs (no LightGBM in the process) are unaffected.
torch.set_num_threads(1)

from transformers import BertConfig, BertForSequenceClassification  # noqa: E402

from dementia_undercoding.arm2_lora import add_lora, count_parameters, time_training_step  # noqa: E402

HIDDEN, LAYERS = 32, 2
LORA = {"r": 4, "alpha": 8, "dropout": 0.1, "target_modules": ["query", "value"]}


def tiny_bert():
    config = BertConfig(
        vocab_size=100, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=64, num_labels=1,
    )
    return BertForSequenceClassification(config)


def test_only_adapters_and_head_train():
    model = add_lora(tiny_bert(), LORA)
    trainable, total = count_parameters(model)
    adapters = LAYERS * 2 * (2 * HIDDEN * LORA["r"])   # 2 targets per layer, A and B each HIDDEN x r
    head = HIDDEN + 1                                   # one weight per input, plus bias
    assert trainable == adapters + head
    assert trainable < total


def test_base_weights_frozen():
    model = add_lora(tiny_bert(), LORA)
    for name, p in model.named_parameters():
        if "lora_" not in name and "classifier" not in name:
            assert not p.requires_grad, name


def test_timing_runs_on_cpu():
    model = add_lora(tiny_bert(), LORA)
    seconds = time_training_step(model, "cpu", batch_size=2, max_len=16, steps=2, warmup=1)
    assert seconds > 0



WORDS = ["history", "of", "dementia", "no", "cognitive", "issues", "chest", "pain", "fever", "fall"]


@pytest.fixture
def tokenizer():
    from transformers import BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    tok = BertTokenizerFast(vocab={word: i for i, word in enumerate(vocab)})
    assert tok.unk_token_id not in tok("history of dementia")["input_ids"]   # every word known
    return tok


def fake_notes(n=200, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.3
    filler = ["chest pain", "fever", "fall"]
    texts = [f"{rng.choice(filler)} {'history of dementia' if y else 'no cognitive issues'}" for y in labels]
    return texts, labels


def test_train_then_predict(tokenizer):
    import torch

    from dementia_undercoding.arm2_lora import predict, train
    from dementia_undercoding.evaluate import score

    texts, labels = fake_notes()
    torch.manual_seed(0)
    model = add_lora(tiny_bert(), LORA)
    logged = []
    # Five passes: a tiny untrained BERT needs more than one pass of 25 steps.
    train(model, tokenizer, texts, labels, device="cpu", batch_size=8, learning_rate=1e-2,
          warmup_frac=0.1, max_grad_norm=1.0, max_len=16, seed=0, passes=5, log_every=5,
          log=lambda *args: logged.append(args))
    probs = predict(model, tokenizer, texts, device="cpu", max_len=16)

    assert probs.shape == (len(texts),)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert logged and all(isinstance(v, int | float) for row in logged for v in row)
    assert logged[-1][:2] == (125, 125)              # 5 passes x 25 steps, counted across passes
    assert score(labels, probs)["pr_auc"] > 0.9      # the words give the answer away    



@pytest.mark.parametrize(
    "status, pr_auc, expect_fallback",
    [("FINISHED", 0.70, True), ("FINISHED", 0.80, False), ("FAILED", 0.70, False)],
)
def test_fallback_rule(tmp_path, status, pr_auc, expect_fallback):
    import mlflow
    import yaml

    from dementia_undercoding.arm2_lora import fallback_if_needed

    db = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = {
        "mlflow": {"tracking_uri": db, "experiment": "test"},
        "lora": {"fallback_trigger": 0.76, "train": {"label": "dementia_core"}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    mlflow.set_tracking_uri(db)
    mlflow.set_experiment("test")
    with mlflow.start_run(run_name="arm2-lora-main-dementia_core", tags={"smoke": "False"}) as r:
        mlflow.log_metric("val_pr_auc", pr_auc)
    mlflow.MlflowClient().set_terminated(r.info.run_id, status=status)

    started = []
    ran = fallback_if_needed(config_path, poll_seconds=0, start=lambda *a, **k: started.append(k))
    assert ran == expect_fallback
    assert started == ([{"variant": "fallback"}] if expect_fallback else [])