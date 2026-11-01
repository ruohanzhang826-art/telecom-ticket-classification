"""Train and evaluate the baseline ticket-routing models.

Run from the repository root:
    python -m src.ticket_routing.train

Outputs:
    models/best_model.joblib        - fitted pipeline used by predict.py
    reports/metrics.json            - all metrics for all splits
    reports/figures/*.png           - confusion matrix, coverage curve
    reports/top_features.md         - most predictive n-grams per department
    reports/error_examples.md       - misrouted tickets for error analysis
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score

from .data import ROOT, load_tickets, random_split, time_split, unseen_phrasing_split
from .models import build_models
from .plotting import confusion_matrix_plot, coverage_plot

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS = ROOT / "models"


def coverage_curve(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray):
    """For each confidence threshold: share of tickets above it, and accuracy on them."""
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    order = np.argsort(-conf)
    correct = (pred[order] == y_true[order]).astype(float)
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), np.cumsum(correct) / n, conf[order]


def evaluate(model, test: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    pred = model.predict(test["text"])
    latency_ms = (time.perf_counter() - t0) / len(test) * 1000
    return {
        "accuracy": round(accuracy_score(test["department"], pred), 4),
        "macro_f1": round(f1_score(test["department"], pred, average="macro"), 4),
        "weighted_f1": round(f1_score(test["department"], pred, average="weighted"), 4),
        "ms_per_ticket": round(latency_ms, 3),
        "n_test": int(len(test)),
    }


def top_features(model, k: int = 8) -> str:
    """Highest-weight word n-grams per class from the logistic regression."""
    union = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    word_vec = dict(union.transformer_list)["word"]
    n_word = len(word_vec.vocabulary_)
    names = word_vec.get_feature_names_out()
    lines = ["# Most predictive word n-grams per department (TF-IDF + LogReg)\n"]
    for cls, coefs in zip(clf.classes_, clf.coef_):
        idx = np.argsort(-coefs[:n_word])[:k]
        lines.append(f"- **{cls}**: " + ", ".join(f"`{names[i]}`" for i in idx))
    return "\n".join(lines) + "\n"


def select_C(train: pd.DataFrame, grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0)) -> tuple[float, dict]:
    """Pick the LogReg regularisation strength with *grouped* CV on the training set.

    Groups = phrasing templates, so each validation fold contains phrasings the
    fold's model has not seen. Plain k-fold rewards memorising phrasings (every C
    scores ~0.97); grouped CV rewards generalising, which is what matters when a
    new kind of complaint shows up.
    """
    scores = {}
    for C in grid:
        model = build_models(C=C)["tfidf_logreg"]
        scores[C] = round(cross_val_score(model, train["text"], train["department"],
                                          groups=train["template_id"], cv=GroupKFold(4),
                                          scoring="f1_macro").mean(), 4)
    return max(scores, key=scores.get), scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "tickets.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df = load_tickets(args.data)
    splits = {
        "random_split": random_split(df, seed=args.seed),
        "time_split": time_split(df),
        "unseen_phrasing_split": unseen_phrasing_split(df, seed=args.seed),
    }

    best_C, c_scores = select_C(splits["random_split"][0])
    print(f"grouped-CV macro-F1 by C: {c_scores} -> C={best_C}")
    results: dict = {"model_selection": {"logreg_C_grouped_cv_macro_f1": c_scores, "chosen_C": best_C}}
    fitted: dict = {}
    for split_name, (train, test) in splits.items():
        results[split_name] = {}
        for name, model in build_models(args.seed, C=best_C).items():
            t0 = time.perf_counter()
            model.fit(train["text"], train["department"])
            m = evaluate(model, test)
            m["train_seconds"] = round(time.perf_counter() - t0, 2)
            m["n_train"] = int(len(train))
            results[split_name][name] = m
            fitted[(split_name, name)] = model
            print(f"[{split_name:>22}] {name:<20} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")

    # Detailed analysis on the standard split with the logistic regression.
    train, test = splits["random_split"]
    best = fitted[("random_split", "tfidf_logreg")]
    pred = best.predict(test["text"])
    labels = sorted(df["department"].unique())
    results["random_split"]["tfidf_logreg"]["per_class"] = classification_report(
        test["department"], pred, labels=labels, output_dict=True, zero_division=0)
    confusion_matrix_plot(confusion_matrix(test["department"], pred, labels=labels), labels,
                          "TF-IDF + Logistic Regression - random split (counts)",
                          FIGS / "confusion_matrix_logreg.png")

    # Unseen-phrasing confusion matrix shows where generalisation breaks.
    tr_u, te_u = splits["unseen_phrasing_split"]
    m_u = fitted[("unseen_phrasing_split", "tfidf_logreg")]
    confusion_matrix_plot(confusion_matrix(te_u["department"], m_u.predict(te_u["text"]), labels=labels),
                          labels, "TF-IDF + Logistic Regression - unseen phrasing split (counts)",
                          FIGS / "confusion_matrix_unseen_phrasing.png")

    # Human-in-the-loop routing: auto-route only confident predictions.
    curves = {}
    thresholds = {}
    for split_name in ("random_split", "unseen_phrasing_split"):
        model = fitted[(split_name, "tfidf_logreg")]
        _, te = splits[split_name]
        proba = model.predict_proba(te["text"])
        cov, acc, conf = coverage_curve(proba, te["department"].to_numpy(), model.classes_)
        curves[split_name.replace("_", " ")] = (cov, acc)
        thresholds[split_name] = {
            f"coverage_at_{int(t*100)}pct_accuracy": round(float(cov[acc >= t].max()) if (acc >= t).any() else 0.0, 4)
            for t in (0.90, 0.95)
        }
    coverage_plot(curves, FIGS / "coverage_vs_accuracy.png")
    results["human_in_the_loop"] = thresholds

    (REPORTS / "top_features.md").write_text(top_features(best), encoding="utf-8")

    errors = test.assign(predicted=pred)[pred != test["department"]]
    # The generator knows which department actually produced each text (template
    # prefix). When that matches the *prediction*, the historical label was the
    # mistake, not the model - i.e. the model surfaces mis-routed tickets.
    origin = errors["template_id"].str.rsplit("_", n=1).str[0]
    label_was_wrong = int((origin == errors["predicted"]).sum())
    results["random_split"]["tfidf_logreg"]["errors_total"] = int(len(errors))
    results["random_split"]["tfidf_logreg"]["errors_where_historical_label_was_wrong"] = label_was_wrong
    lines = ["# Misrouted tickets (random split, TF-IDF + LogReg)\n",
             "| ticket | true | predicted | description |", "|---|---|---|---|"]
    for _, r in errors.head(25).iterrows():
        lines.append(f"| {r.ticket_id} | {r.department} | {r.predicted} | {r.description} |")
    (REPORTS / "error_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Ship the model trained on all data for inference.
    final = build_models(args.seed, C=best_C)["tfidf_logreg"].fit(df["text"], df["department"])
    joblib.dump(final, MODELS / "best_model.joblib")
    print(json.dumps(thresholds, indent=2))
    print(f"errors: {len(errors)}, of which historical label wrong: {label_was_wrong}")
    print(f"saved model -> {MODELS / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
