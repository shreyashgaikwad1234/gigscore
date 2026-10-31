"""Load saved model bundles and turn raw rows into model inputs and probabilities of default.

Shared by evaluate.py, explain.py, fairness.py and (step 6) score.py, so every caller featurizes
and predicts the same way.
"""
import joblib
import numpy as np
import pandas as pd

from src.encode import apply_encoders
from src.features import build_features
from src.load import RAW_TEST, load

ENCODERS = "models/encoders.pkl"


def load_bundle(path: str) -> dict:
    return joblib.load(path)


def align_categories(X: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Give categorical columns exactly the training categories (unseen levels become missing)."""
    X = X.copy()
    for c, cats in bundle.get("categories", {}).items():
        if c in X:
            X[c] = pd.Categorical(X[c].astype(object), categories=cats)
    return X


def apply_offset(p: np.ndarray, offset: float) -> np.ndarray:
    """Shift probabilities by `offset` in log-odds (recalibration-in-the-large; ranking unchanged)."""
    if not offset:
        return p
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + offset)))


def predict_pd(bundle: dict, X: pd.DataFrame, raw: bool = False, offset: float = 0.0) -> np.ndarray:
    """Probability of default. `offset` is the log-odds recalibration from config/thresholds.yaml."""
    feats = bundle["features"]
    if bundle["kind"] == "scorecard":
        return apply_offset(bundle["model"].predict_proba(X[feats]), offset)
    Xa = align_categories(X[feats], bundle)
    p = bundle["model"].predict(Xa, num_iteration=bundle["best_iteration"])
    return p if raw else apply_offset(bundle["cal"].predict(p), offset)


def featurize(df: pd.DataFrame, enc: dict | None = None) -> pd.DataFrame:
    """Rows already passed through load.load() -> full feature frame including ID encodings."""
    enc = enc or joblib.load(ENCODERS)
    return build_features(df).join(apply_encoders(df, enc))


def featurize_raw_file(path: str = RAW_TEST) -> pd.DataFrame:
    enc = joblib.load(ENCODERS)
    df = load(path, bounds=enc["winsor_bounds"])
    return df[["UniqueID", "disb"]].join(featurize(df, enc))
