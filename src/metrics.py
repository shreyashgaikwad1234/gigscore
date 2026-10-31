"""Discrimination metrics with paired bootstrap confidence intervals (train.py, evaluate.py)."""
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

N_BOOT, SEED = 500, 42


def auc_ks(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(y, p)
    return float(auc(fpr, tpr)), float(np.max(tpr - fpr))


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], groups: dict[str, np.ndarray],
              n_boot: int = N_BOOT, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """AUC, Gini (= 2·AUC − 1) and KS for every model on every subgroup, with 95% percentile CIs.

    The same resampled rows are used for all models within a group (paired bootstrap), so the
    returned draws can also give CIs on differences between models.
    """
    rng = np.random.default_rng(seed)
    rows, draws = [], {}
    for g, mask in groups.items():
        idx = np.flatnonzero(mask)
        yg = y[idx]
        point = {m: auc_ks(yg, p[idx]) for m, p in preds.items()}
        boot = {m: np.empty((n_boot, 2)) for m in preds}
        for b in range(n_boot):
            s = rng.choice(idx, len(idx), replace=True)
            for m, p in preds.items():
                boot[m][b] = auc_ks(y[s], p[s])
        for m in preds:
            a, k = point[m]
            (a_lo, k_lo), (a_hi, k_hi) = np.percentile(boot[m], [2.5, 97.5], axis=0)
            rows.append({"model": m, "group": g, "n": len(idx), "bad_rate": yg.mean(),
                         "auc": a, "auc_lo": a_lo, "auc_hi": a_hi,
                         "gini": 2 * a - 1, "gini_lo": 2 * a_lo - 1, "gini_hi": 2 * a_hi - 1,
                         "ks": k, "ks_lo": k_lo, "ks_hi": k_hi})
            draws[(m, g)] = boot[m][:, 0]
    return pd.DataFrame(rows), draws


def diff_ci(draws: dict, a: tuple, b: tuple) -> tuple[float, float]:
    """95% CI of AUC(a) − AUC(b) from paired bootstrap draws; a, b are (model, group) keys."""
    d = draws[a] - draws[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)
