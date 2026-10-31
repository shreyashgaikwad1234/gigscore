"""Weight-of-evidence logistic scorecard and the points scale (build guide Section 6.1).

Scale: 600 points at 4:1 good:bad odds (PD 20%, about this portfolio's default rate), 50 points to
double the odds. The guide's example (600 at 20:1, 20 points) squeezed every applicant into roughly
500-580 because PDs here run 5-45%; this anchor spreads them over roughly 450-750. Only the
presentation changes: ranking, bands and reasons are identical under any such scale.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.encode import WoEBinner

BASE_SCORE, BASE_ODDS, PDO = 600, 4, 50
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)
SCORE_MIN, SCORE_MAX = 300, 900


def to_points(pd_: np.ndarray | float) -> np.ndarray:
    """Probability of default -> points on the 300-900 scale."""
    p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    return np.clip(np.round(OFFSET + FACTOR * np.log((1 - p) / p)), SCORE_MIN, SCORE_MAX).astype(int)


def from_points(score: np.ndarray | float) -> np.ndarray:
    """Points -> probability of default (inverse of to_points before rounding)."""
    return 1 / (1 + np.exp((np.asarray(score, dtype=float) - OFFSET) / FACTOR))


class Scorecard:
    """WoE bins -> L2 logistic regression -> points table a credit officer can read."""

    def __init__(self, features: list[str], categorical=(), C: float = 1.0, max_bins: int = 10):
        self.features = list(features)
        self.categorical = [c for c in categorical if c in self.features]
        self.C = C
        self.max_bins = max_bins

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binner_ = WoEBinner(self.max_bins).fit(X[self.features], y, categorical=self.categorical)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(self.binner_.transform(X[self.features]), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self.binner_.transform(X[self.features]))[:, 1]

    def points_table(self) -> pd.DataFrame:
        """Points per bin. Score = base_points + sum of each feature's bin points (before clipping).

        WoE = ln(%good / %bad), so a well-behaved coefficient on WoE is negative for P(bad)."""
        rows = []
        for f, coef in zip(self.features, self.lr_.coef_[0]):
            t = self.binner_.tables_[f]
            for b, r in t.iterrows():
                rows.append({"feature": f, "bin": b, "rows": int(r["rows"]), "bad_rate": r["bad_rate"],
                             "woe": r["woe"], "coef": coef, "points": -FACTOR * coef * r["woe"]})
        return pd.DataFrame(rows)

    @property
    def base_points(self) -> float:
        return float(OFFSET - FACTOR * self.lr_.intercept_[0])

    @property
    def sign_flips(self) -> list[str]:
        """Features whose coefficient has the wrong sign (a scorecard red flag, usually collinearity)."""
        return [f for f, c in zip(self.features, self.lr_.coef_[0]) if c > 0]
