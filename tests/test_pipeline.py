import importlib.util
from pathlib import Path

import pandas as pd

from src.ticket_routing.data import clean_text, random_split, unseen_phrasing_split
from src.ticket_routing.models import build_models

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen", ROOT / "data" / "generate_tickets.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def small_df(n=600):
    df = pd.DataFrame(gen.generate(n, seed=1, label_noise=0.0))
    df["text"] = df["description"].map(clean_text)
    return df


def test_generator_is_deterministic():
    assert gen.generate(50, 7, 0.03) == gen.generate(50, 7, 0.03)


def test_clean_text():
    assert clean_text("  Cust CHARGED 30 extra!!  ") == "cust charged 30 extra"


def test_unseen_phrasing_split_has_no_template_overlap():
    tr, te = unseen_phrasing_split(small_df())
    assert set(tr.template_id).isdisjoint(set(te.template_id))
    assert te.department.nunique() == len(gen.DEPARTMENTS)


def test_logreg_beats_majority_baseline():
    tr, te = random_split(small_df())
    models = build_models()
    accs = {}
    for name in ("majority_class", "tfidf_logreg"):
        m = models[name].fit(tr.text, tr.department)
        accs[name] = (m.predict(te.text) == te.department).mean()
    assert accs["tfidf_logreg"] > 0.9 > accs["majority_class"]
