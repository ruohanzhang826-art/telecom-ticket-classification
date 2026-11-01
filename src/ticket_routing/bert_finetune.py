"""Extension: fine-tune a BERT-family model for ticket routing.

Motivation: the TF-IDF baselines reach ~98% when test tickets are phrased like
training tickets, but drop to ~61% on phrasings they have never seen
(`unseen_phrasing_split`). A pretrained language model knows that "balance not
updated" and "credit didn't arrive" mean similar things, so it should close
part of that gap.

Usage (GPU recommended - e.g. Google Colab):
    pip install -r requirements-bert.txt
    python -m src.ticket_routing.bert_finetune --split unseen_phrasing_split
    python -m src.ticket_routing.bert_finetune --model-name distilbert-base-uncased --epochs 3

Offline check that the training loop works (tiny randomly-initialised model,
no download, runs on CPU in seconds - NOT a meaningful accuracy number):
    python -m src.ticket_routing.bert_finetune --smoke-test
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .data import ROOT, load_tickets, random_split, time_split, unseen_phrasing_split

SPLITS = {"random_split": random_split, "time_split": time_split,
          "unseen_phrasing_split": unseen_phrasing_split}


def tiny_offline_model(texts, n_labels):
    """Build a 2-layer random BERT + word-level vocab for --smoke-test."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    vocab = sorted({w for t in texts for w in t.split()})
    tmp = Path(tempfile.mkdtemp()) / "vocab.txt"
    tmp.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *vocab]), encoding="utf-8")
    tok = BertTokenizer(str(tmp), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128, num_labels=n_labels)
    return tok, BertForSequenceClassification(cfg)


def encode(tok, texts, labels, max_len):
    enc = tok(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="bert-base-uncased")
    ap.add_argument("--split", choices=list(SPLITS), default="unseen_phrasing_split")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(42)
    df = load_tickets()
    train, test = SPLITS[a.split](df)
    if a.smoke_test:
        train, test = train.sample(400, random_state=0), test.sample(200, random_state=0)
        a.epochs, a.lr = 1, 1e-3
    labels = sorted(df["department"].unique())
    lid = {l: i for i, l in enumerate(labels)}

    if a.smoke_test:
        tok, model = tiny_offline_model(df["text"], len(labels))
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=lid)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tr = DataLoader(encode(tok, train["text"], train["department"].map(lid).values, a.max_len),
                    batch_size=a.batch_size, shuffle=True)
    te = DataLoader(encode(tok, test["text"], test["department"].map(lid).values, a.max_len),
                    batch_size=128)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(tr) * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * total + 1)) * max(0.0, 1 - s / total))

    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        model.train()
        running = 0.0
        for ids, mask, y in tr:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += out.loss.item()
        print(f"epoch {epoch + 1}/{a.epochs}  train loss {running / len(tr):.4f}")

    y_true = test["department"].map(lid).values
    y_pred = predict(model, te, device)
    res = {"model": "tiny-random-bert (smoke test)" if a.smoke_test else a.model_name,
           "split": a.split, "accuracy": round(accuracy_score(y_true, y_pred), 4),
           "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
           "train_minutes": round((time.perf_counter() - t0) / 60, 2), "device": device}
    print(json.dumps(res, indent=2))
    if not a.smoke_test:
        out = ROOT / "reports" / f"bert_{a.split}.json"
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
