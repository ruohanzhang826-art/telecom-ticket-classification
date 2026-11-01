"""Baseline model definitions (scikit-learn pipelines)."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def tfidf_features() -> FeatureUnion:
    """Word n-grams capture key phrases ("not activated", "topped up");
    character n-grams make the model robust to agent typos ("custmer", "recieved")."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=30000)),
    ])


def build_models(seed: int = 42, C: float = 0.3) -> dict[str, Pipeline]:
    return {
        "majority_class": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "tfidf_logreg": Pipeline([
            ("tfidf", tfidf_features()),
            ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
        ]),
        "tfidf_random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           n_jobs=-1, random_state=seed)),
        ]),
    }
