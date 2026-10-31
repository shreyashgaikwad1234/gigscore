"""Model-improvement experiments, judged ONLY by 5-fold cross-validation inside Aug-Sep 2018.

October (the out-of-time test) is never touched here, so choices made from this file do not
leak into the headline result.

Run:  python -m src.experiments      (writes reports/experiments.md)
"""
import json
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.encode import oof_target_encode
from src.load import TARGET, load
from src.train import FEATURE_LIST, FEATURES, MONOTONE

LR = 0.05
MIN_GAIN = 0.002  # a candidate must beat the baseline CV AUC by at least this much


def cv_auc(X: pd.DataFrame, y: pd.Series, folds: np.ndarray, features: list[str], params: dict,
           seeds=(42,)) -> tuple[float, np.ndarray]:
    oof = np.zeros(len(X))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        for s in seeds:
            p = {**params, "learning_rate": LR, "seed": s,
                 "monotone_constraints": [MONOTONE.get(f, 0) for f in features]}
            m = lgb.train(p, lgb.Dataset(X.loc[tr, features], y[tr]), 3000,
                          valid_sets=[lgb.Dataset(X.loc[va, features], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] += m.predict(X.loc[va, features], num_iteration=m.best_iteration) / len(seeds)
    return roc_auc_score(y, oof), oof


def candidate_features(win: pd.DataFrame, raw: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Extra features built inside the training window (OOF where labels are involved)."""
    y = win[TARGET]
    c = pd.DataFrame(index=win.index)
    r = raw.loc[win.index]
    for a, b, name in [("supplier_id", "manufacturer_id", "supplier_mfr_te"),
                       ("branch_id", "manufacturer_id", "branch_mfr_te"),
                       ("branch_id", "supplier_id", "branch_supplier_te")]:
        key = r[a].astype(str) + "_" + r[b].astype(str)
        c[name], _, _ = oof_target_encode(key, y, folds)
    # label-free dealer profile: how this applicant compares with the dealer's usual customer
    sup = r["supplier_id"]
    c["ltv_vs_dealer"] = r["ltv"] - r.groupby("supplier_id")["ltv"].transform("mean")
    c["dealer_noscore_share"] = r.groupby("supplier_id")["no_score"].transform("mean")
    c["asset_vs_dealer"] = np.log(r["asset_cost"]) - np.log(r["asset_cost"]).groupby(sup).transform("mean")
    return c


def main() -> pd.DataFrame:
    t = pd.read_parquet(FEATURES)
    fl = json.loads(Path(FEATURE_LIST).read_text())
    win = t[t["split"] != "oot"].copy()  # Aug-Sep only
    y = win[TARGET]
    folds = win["fold"].to_numpy()
    raw = load()
    raw.index = t.index  # same row order as the feature table
    cand = candidate_features(win, raw, folds)
    X = win.join(cand)
    params = joblib.load("models/lgbm_combined.pkl")["params"]
    base = fl["sets"]["combined"]

    rows = []

    def run(name, feats, seeds=(42,), note=""):
        auc, oof = cv_auc(X, y, folds, feats, params, seeds)
        rows.append({"experiment": name, "features": len(feats), "cv_auc": auc, "note": note})
        print(f"{name:32s} {auc:.4f}")
        return oof

    oof_base = run("baseline (current combined set)", base)
    for f in cand.columns:
        run(f"+ {f}", base + [f])
    keep = [r["experiment"][2:] for r in rows[1:] if r["cv_auc"] - rows[0]["cv_auc"] >= MIN_GAIN]
    if keep:
        run("+ all candidates that helped", base + keep, note=", ".join(keep))
    run("all candidates together", base + list(cand.columns))
    run("baseline, 3-seed bagging", base, seeds=(42, 7, 2024))

    # blend with the scorecard (logit average), weight chosen on the same OOF predictions
    sc = joblib.load("models/lr_scorecard_combined.pkl")
    oof_sc = np.zeros(len(win))
    from src.scorecard import Scorecard
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = Scorecard(sc["features"], fl["categorical"], C=sc["C"]).fit(win[tr], y[tr])
        oof_sc[va] = m.predict_proba(win[va])
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    best_w, best_auc = 1.0, roc_auc_score(y, oof_base)
    for w in np.linspace(0.5, 1.0, 11):
        a = roc_auc_score(y, w * logit(oof_base) + (1 - w) * logit(oof_sc))
        if a > best_auc:
            best_w, best_auc = w, a
    rows.append({"experiment": f"blend LightGBM {best_w:.2f} + scorecard {1 - best_w:.2f}", "features": len(base),
                 "cv_auc": best_auc, "note": "weight chosen on OOF"})
    print(f"blend w={best_w:.2f}                     {best_auc:.4f}")

    res = pd.DataFrame(rows)
    res["gain_vs_baseline"] = res["cv_auc"] - res.loc[0, "cv_auc"]
    Path("reports/experiments.md").write_text(md(res), encoding="utf-8")
    return res


def md(res: pd.DataFrame) -> str:
    return "\n".join([
        "# Improvement experiments (Aug–Sep cross-validation only)",
        "",
        f"Produced by `src/experiments.py` on {date.today()}. 5-fold CV inside Aug–Sep 2018 using the same folds as "
        "the out-of-fold target encoding; LightGBM with the step 3 combined-model parameters at learning rate "
        f"{LR}. **October was not used.** A change is adopted only if it gains at least {MIN_GAIN} CV AUC.",
        "",
        "| experiment | features | CV AUC | gain | note |",
        "|:--|--:|--:|--:|:--|",
        *[f"| {r.experiment} | {r.features} | {r.cv_auc:.4f} | {r.gain_vs_baseline:+.4f} | {r.note} |"
          for r in res.itertuples()],
        "",
    ])


if __name__ == "__main__":
    main()
