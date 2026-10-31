"""Leakage-safe encodings and screening statistics (build guide Section 5.1).

Out-of-fold target encoding and volume counts for high-cardinality IDs, weight-of-evidence
binning, information value (IV) and population stability index (PSI).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TE_SMOOTHING = 50
N_FOLDS, SEED = 5, 42

# raw ID column -> feature name. State_ID is protected (CLAUDE.md rule 3) and never encoded;
# Employee_code_ID is dropped (operational artefact, rule 4).
TE_COLUMNS = {"branch_id": "branch_te", "supplier_id": "supplier_te",
              "manufacturer_id": "manufacturer_te", "Current_pincode_ID": "pincode_te"}
COUNT_COLUMNS = {"branch_id": "branch_n", "supplier_id": "supplier_n", "Current_pincode_ID": "pincode_n"}


def fold_ids(n: int, folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    f = np.empty(n, dtype=int)
    for k, (_, va) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(np.arange(n))):
        f[va] = k
    return f


def oof_target_encode(keys: pd.Series, y: pd.Series, folds: np.ndarray, m: int = TE_SMOOTHING):
    """Each row is encoded from the other folds only (prior included); `mapping` (all rows) is for new data."""
    k, t = keys.to_numpy(), y.to_numpy()
    out = np.full(len(keys), np.nan)
    for f in np.unique(folds):
        tr = folds != f
        fold_prior = t[tr].mean()
        stats = pd.Series(t[tr]).groupby(k[tr]).agg(["sum", "count"])
        enc = (stats["sum"] + fold_prior * m) / (stats["count"] + m)
        out[~tr] = pd.Series(k[~tr]).map(enc).fillna(fold_prior).to_numpy()
    prior = float(y.mean())
    full = y.groupby(keys).agg(["sum", "count"])
    mapping = ((full["sum"] + prior * m) / (full["count"] + m)).to_dict()
    return pd.Series(out, index=keys.index), mapping, prior


def fit_encoders(train: pd.DataFrame, y: pd.Series, folds: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Fit on the training window. Returns OOF-encoded columns for those rows and the encoder state."""
    cols, enc = {}, {"te": {}, "counts": {}, "smoothing": TE_SMOOTHING}
    for raw, name in TE_COLUMNS.items():
        cols[name], mapping, prior = oof_target_encode(train[raw], y, folds)
        enc["te"][raw] = {"name": name, "mapping": mapping, "prior": prior}
    for raw, name in COUNT_COLUMNS.items():
        counts = train[raw].value_counts().to_dict()
        enc["counts"][raw] = {"name": name, "mapping": counts}
        cols[name] = train[raw].map(counts)
    return pd.DataFrame(cols, index=train.index), enc


def apply_encoders(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Encode new rows (out-of-time test, test.csv, a single application) with the fitted maps."""
    cols = {}
    for raw, e in enc["te"].items():
        cols[e["name"]] = df[raw].map(e["mapping"]).fillna(e["prior"]).astype(float)
    for raw, e in enc["counts"].items():
        cols[e["name"]] = df[raw].map(e["mapping"]).fillna(0).astype(float)
    return pd.DataFrame(cols, index=df.index)


# --- weight of evidence ---------------------------------------------------------------------

MISSING = "Missing"


def _cat_labels(x: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(x):
        lab = x.astype(float).map(lambda v: f"{v:g}")
    else:
        lab = x.astype(object).astype(str)
    return lab.where(x.notna(), MISSING)


def _num_labels(x: pd.Series, edges: np.ndarray) -> pd.Series:
    idx = np.searchsorted(edges[1:-1], x.to_numpy(dtype=float), side="left")
    names = [f"({edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(len(edges) - 1)]
    lab = pd.Series(np.array(names, dtype=object)[np.minimum(idx, len(names) - 1)], index=x.index)
    return lab.where(x.notna(), MISSING)


def woe_table(labels: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"bin": labels.to_numpy(), "y": y.to_numpy()}).groupby("bin")["y"].agg(["size", "sum"])
    t.columns = ["rows", "bads"]
    goods = t["rows"] - t["bads"]
    n = len(t)
    dist_good = (goods + 0.5) / (goods.sum() + 0.5 * n)
    dist_bad = (t["bads"] + 0.5) / (t["bads"].sum() + 0.5 * n)
    t["bad_rate"] = t["bads"] / t["rows"]
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t


class WoEBinner:
    """Quantile bins for numerics (missing gets its own bin), one bin per level for categoricals
    and low-cardinality numerics. Fit on the training window only.

    Zero-inflated numerics (e.g. delinquencies: mostly 0) would collapse to a single quantile bin,
    so the most common value gets its own bin and the rest are cut at their own quantiles; bins
    under `min_share` of rows are merged into a neighbour.
    """

    def __init__(self, max_bins: int = 10, min_share: float = 0.02):
        self.max_bins = max_bins
        self.min_share = min_share
        self.specs_: dict = {}
        self.tables_: dict = {}

    def _edges(self, v: np.ndarray) -> np.ndarray:
        qs = np.linspace(0, 1, self.max_bins + 1)
        mode = pd.Series(v).mode().iloc[0]
        rest = v[v != mode]
        cuts = set(np.quantile(v, qs)) | {mode}
        if len(rest):
            cuts |= set(np.quantile(rest, qs))
        below = v[v < mode]
        if len(below):
            cuts.add(below.max())
        cuts = sorted(c for c in cuts if c < v.max())
        # merge bins that are too small, removing the cut shared with the smaller neighbour
        while cuts:
            edges = np.array([-np.inf, *cuts, np.inf])
            share = np.bincount(np.searchsorted(edges[1:-1], v, side="left"), minlength=len(edges) - 1) / len(v)
            i = int(np.argmin(share))
            if share[i] >= self.min_share:
                break
            if i == 0:
                cuts.pop(0)
            elif i == len(share) - 1:
                cuts.pop(-1)
            else:
                cuts.pop(i - 1 if share[i - 1] <= share[i + 1] else i)
        return np.array([-np.inf, *cuts, np.inf])

    def _labels(self, x: pd.Series, spec) -> pd.Series:
        return _cat_labels(x) if spec is None else _num_labels(x, spec)

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical=()) -> "WoEBinner":
        for c in X.columns:
            x = X[c]
            if c in categorical or not pd.api.types.is_numeric_dtype(x) or x.nunique() <= self.max_bins:
                spec = None
            else:
                v = x.to_numpy(dtype=float)
                spec = self._edges(v[~np.isnan(v)])
            self.specs_[c] = spec
            self.tables_[c] = woe_table(self._labels(x, spec), y)
        return self

    def labels(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self._labels(X[c], self.specs_[c]) for c in self.specs_}, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        lab = self.labels(X)
        return pd.DataFrame({c: lab[c].map(self.tables_[c]["woe"]).fillna(0.0) for c in self.specs_}, index=X.index)

    @property
    def iv(self) -> pd.Series:
        return pd.Series({c: t["iv"].sum() for c, t in self.tables_.items()}).sort_values(ascending=False)


def psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    """PSI between two series of bin labels (use WoEBinner.labels so both share the training bins)."""
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx, fill_value=0) + eps, a.reindex(idx, fill_value=0) + eps
    return float(((a - e) * np.log(a / e)).sum())
