"""Route new tickets with the trained model.

    python -m src.ticket_routing.predict "topped up 50 via the app but balance not updated"
    python -m src.ticket_routing.predict --threshold 0.7 "bill too high and internet slow"

Tickets whose top probability is below --threshold are flagged MANUAL_TRIAGE,
mirroring how a service desk keeps a human in the loop for ambiguous cases.
"""

from __future__ import annotations

import argparse

import joblib

from .data import ROOT, clean_text


def route(texts: list[str], threshold: float = 0.6, model_path=ROOT / "models" / "best_model.joblib"):
    model = joblib.load(model_path)
    proba = model.predict_proba([clean_text(t) for t in texts])
    out = []
    for text, p in zip(texts, proba):
        top = p.argsort()[::-1][:2]
        conf = float(p[top[0]])
        out.append({
            "ticket": text,
            "route_to": model.classes_[top[0]] if conf >= threshold else "MANUAL_TRIAGE",
            "top1": (model.classes_[top[0]], round(conf, 3)),
            "top2": (model.classes_[top[1]], round(float(p[top[1]]), 3)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickets", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.6)
    a = ap.parse_args()
    for r in route(a.tickets, a.threshold):
        print(f"{r['route_to']:<18} top1={r['top1']}  top2={r['top2']}  | {r['ticket']}")


if __name__ == "__main__":
    main()
