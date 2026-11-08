"""Arm 2 setup on a tiny, untrained BERT: no download, no notes.

Skipped where torch / peft are not installed (CI installs only the core deps).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

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
    import numpy as np
    import torch

    from dementia_undercoding.arm2_lora import predict, train
    from dementia_undercoding.evaluate import score

    texts, labels = fake_notes()
    torch.manual_seed(0)
    model = add_lora(tiny_bert(), LORA)
    logged = []
    # Five copies of the notes: a tiny untrained BERT needs more than 25 steps.
    train(model, tokenizer, texts * 5, np.tile(labels, 5), device="cpu", batch_size=8, learning_rate=1e-2,
          warmup_frac=0.1, max_grad_norm=1.0, max_len=16, seed=0, log_every=5,
          log=lambda *args: logged.append(args))
    probs = predict(model, tokenizer, texts, device="cpu", max_len=16)

    assert probs.shape == (len(texts),)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert logged and all(isinstance(v, int | float) for row in logged for v in row)
    assert score(labels, probs)["pr_auc"] > 0.9      # the words give the answer away    