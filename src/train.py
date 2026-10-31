"""Fit the WoE scorecard and the LightGBM challenger on three feature sets (build guide Section 6).

Everything is decided inside Aug-Sep 2018 with 5-fold cross-validation (the same folds as the
out-of-fold target encoding): Optuna tuning, the number of trees, the scorecard's C, and isotonic
calibration (fitted on out-of-fold predictions). Final models are then fit on all of Aug-Sep and
tested once, out-of-time, on October ('oot').

Run:  python -m src.train [--trials 40] [--boot 500]
Writes models/{lr_scorecard,lgbm}_{bureau,alternative,combined}.pkl (+ lgbm.pkl / lr_scorecard.pkl
= combined), data/processed/{oot,oof}_predictions.parquet, reports/model_results.md,
reports/model_card.md.
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.load import TARGET
from src.metrics import bootstrap, diff_ci
from src.scorecard import BASE_ODDS, BASE_SCORE, PDO, Scorecard, to_points

FEATURES = "data/processed/features.parquet"
FEATURE_LIST = "data/processed/feature_list.json"
SETS = ["bureau", "alternative", "combined", "fair"]
# Shipped pair (models/lgbm.pkl + models/lr_scorecard.pkl). 'fair' leaves out the features that stand
# in for state; 'combined' is the documented higher-accuracy alternative (see reports/model_card.md).
PRODUCTION_SET = "fair"
MODEL_VERSION = "2026.10"
LEAKAGE_AUC = 0.75  # CLAUDE.md rule 2

BASE_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                   scale_pos_weight=3.5, metric="auc", verbose=-1, seed=42, deterministic=True,
                   force_row_wise=True)
# More delinquency can never lower risk; more documents can never raise it (guide Section 6.2).
MONOTONE = {"ltv": 1, "delinq_6m": 1, "overdue_ratio": 1, "inquiries": 1, "kyc_count": -1,
            "down_payment_share": -1}
TUNE_LR, MAX_ROUNDS, EARLY_STOP = 0.05, 3000, 100
SCORECARD_C = [0.01, 0.1, 1.0]

GROUPS = {"all": "all borrowers", "no_score": "no bureau score (score == 0)",
          "no_usable_score": "no usable score (0 or 'Not Scored')", "self_employed": "self-employed"}


def load_data():
    t = pd.read_parquet(FEATURES)
    fl = json.loads(Path(FEATURE_LIST).read_text())
    return t, fl


def fit_lgbm(params: dict, X_fit, y_fit, X_va, y_va, features: list[str]):
    p = {**params, "monotone_constraints": [MONOTONE.get(f, 0) for f in features]}
    return lgb.train(p, lgb.Dataset(X_fit[features], y_fit), MAX_ROUNDS,
                     valid_sets=[lgb.Dataset(X_va[features], y_va)],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def cv_lgbm(params: dict, X: pd.DataFrame, y: pd.Series, folds: np.ndarray, features: list[str],
            trial=None) -> tuple[float, np.ndarray, list[int]]:
    """5-fold CV: returns OOF AUC, OOF raw predictions and each fold's early-stopped tree count."""
    oof, iters = np.zeros(len(X)), []
    for i, k in enumerate(np.unique(folds)):
        tr, va = folds != k, folds == k
        m = fit_lgbm(params, X[tr], y[tr], X[va], y[va], features)
        oof[va] = m.predict(X.loc[va, features], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        if trial is not None:
            trial.report(roc_auc_score(y[va], oof[va]), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return roc_auc_score(y, oof), oof, iters


def tune_lgbm(X: pd.DataFrame, y: pd.Series, folds: np.ndarray, features: list[str], n_trials: int) -> dict:
    """Optuna on 5-fold CV AUC inside Aug-Sep (median pruning across folds)."""
    def objective(trial):
        params = {**BASE_PARAMS, "learning_rate": TUNE_LR,
                  "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                  "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                  "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
                  "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 100, log=True)}
        return cv_lgbm(params, X, y, folds, features, trial)[0]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.enqueue_trial({"num_leaves": 31, "min_child_samples": 200, "feature_fraction": 0.8,
                         "bagging_fraction": 0.8, "lambda_l1": 1e-3, "lambda_l2": 5.0})  # guide defaults first
    study.optimize(objective, n_trials=n_trials)
    return {**BASE_PARAMS, **study.best_params}


def final_lgbm(params: dict, X: pd.DataFrame, y: pd.Series, folds: np.ndarray, features: list[str]):
    """CV at the final learning rate for OOF predictions and tree count, then one model on all rows.
    Isotonic calibration is fitted on the OOF predictions (no row calibrates its own prediction)."""
    cv_auc, oof_raw, iters = cv_lgbm(params, X, y, folds, features)
    n_rounds = int(round(np.mean(iters) * 1.1))  # final model sees 25% more rows than each fold model
    p = {**params, "monotone_constraints": [MONOTONE.get(f, 0) for f in features]}
    model = lgb.train(p, lgb.Dataset(X[features], y), n_rounds)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof_raw, y)
    return model, cal, oof_raw, n_rounds, cv_auc, iters


def fit_scorecard(features, categorical, C, X_fit, y_fit) -> tuple[Scorecard, list[str]]:
    """Refit without any feature whose coefficient has the wrong sign, until none is left, so every
    feature's points move in the direction its bad rates do."""
    features, removed = list(features), []
    while True:
        sc = Scorecard(features, categorical, C=C).fit(X_fit, y_fit)
        if not sc.sign_flips:
            return sc, removed
        worst = max(sc.sign_flips, key=lambda f: sc.lr_.coef_[0][sc.features.index(f)])
        features.remove(worst)
        removed.append(worst)


def cv_scorecard(features, categorical, C, X, y, folds) -> np.ndarray:
    oof = np.zeros(len(X))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        oof[va] = Scorecard(features, categorical, C=C).fit(X[tr], y[tr]).predict_proba(X[va])
    return oof


def tune_scorecard(features, categorical, X, y, folds) -> tuple[Scorecard, list[str], np.ndarray, float]:
    """C by CV AUC; final fit on all rows with wrong-sign features removed; OOF with the final features."""
    best_C = max(SCORECARD_C, key=lambda C: roc_auc_score(y, cv_scorecard(features, categorical, C, X, y, folds)))
    sc, removed = fit_scorecard(features, categorical, best_C, X, y)
    oof = cv_scorecard(sc.features, categorical, best_C, X, y, folds)
    return sc, removed, oof, roc_auc_score(y, oof)


TUNED = ("num_leaves", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2")


def main(n_trials: int = 40, n_boot: int = 500) -> dict:
    t, fl = load_data()
    win, oot = t[t["split"] != "oot"], t[t["split"] == "oot"]  # Aug-Sep (fit + valid) / October
    y_win, y_oot = win[TARGET], oot[TARGET].to_numpy()
    folds = win["fold"].to_numpy()
    Path("models").mkdir(exist_ok=True)

    preds, oof, saved, info = {}, {}, {}, {}
    for s in SETS:
        feats = fl["sets"][s]
        sc_feats = [f for f in fl["scorecard"] if f in feats]

        t0 = time.time()
        sc, sign_removed, oof_sc, cv_sc = tune_scorecard(sc_feats, fl["categorical"], win, y_win, folds)
        sc_feats = sc.features
        name = f"scorecard_{s}"
        preds[name], oof[name] = sc.predict_proba(oot), oof_sc
        bundle = {"kind": "scorecard", "set": s, "model": sc, "features": sc_feats, "C": sc.C,
                  "points_table": sc.points_table(), "base_points": sc.base_points,
                  "version": f"scorecard-{s}-{MODEL_VERSION}"}
        joblib.dump(bundle, f"models/lr_scorecard_{s}.pkl")
        saved[name] = f"models/lr_scorecard_{s}.pkl"
        info[name] = {"features": sc_feats, "C": sc.C, "sign_removed": sign_removed, "cv_auc": cv_sc,
                      "secs": time.time() - t0}

        t0 = time.time()
        params = tune_lgbm(win, y_win, folds, feats, n_trials)
        m, cal, oof_raw, n_rounds, cv_auc, iters = final_lgbm(params, win, y_win, folds, feats)
        raw_oot = m.predict(oot[feats])
        name = f"lgbm_{s}"
        preds[name], preds[f"{name}_raw"] = cal.predict(raw_oot), raw_oot
        oof[name], oof[f"{name}_raw"] = cal.predict(oof_raw), oof_raw
        cats = {c: list(t[c].cat.categories) for c in fl["categorical"] if c in feats}
        bundle = {"kind": "lgbm", "set": s, "model": m, "cal": cal, "features": feats, "params": params,
                  "best_iteration": n_rounds, "categories": cats, "monotone": MONOTONE,
                  "version": f"lgbm-{s}-{MODEL_VERSION}"}
        joblib.dump(bundle, f"models/lgbm_{s}.pkl")
        saved[name] = f"models/lgbm_{s}.pkl"
        info[name] = {"features": feats, "params": {k: round(params[k], 4) for k in TUNED},
                      "best_iteration": n_rounds, "fold_iters": iters, "cv_auc": cv_auc,
                      "secs": time.time() - t0,
                      "top_gain": pd.Series(m.feature_importance("gain"), index=feats)
                      .sort_values(ascending=False).head(10).round(0).to_dict()}
        print(f"{s:12s} scorecard CV {cv_sc:.4f} | lgbm CV {cv_auc:.4f} "
              f"({info[name]['secs']:.0f}s, {n_rounds} trees)", flush=True)

    o = win[["UniqueID", TARGET, "no_score", "no_usable_score", "employment_self", "age_band", "State_ID"]].copy()
    for k, v in oof.items():
        o[f"pd_{k}"] = v
    o.to_parquet("data/processed/oof_predictions.parquet", index=False)

    # production pair: LightGBM main model with its scorecard as the kill-switch fallback
    joblib.dump(joblib.load(f"models/lgbm_{PRODUCTION_SET}.pkl"), "models/lgbm.pkl")
    joblib.dump(joblib.load(f"models/lr_scorecard_{PRODUCTION_SET}.pkl"), "models/lr_scorecard.pkl")

    groups = {"all": np.ones(len(oot), bool), "no_score": (oot["no_score"] == 1).to_numpy(),
              "no_usable_score": (oot["no_usable_score"] == 1).to_numpy(),
              "self_employed": (oot["employment_self"] == 1).to_numpy()}
    models = [m for m in preds if not m.endswith("_raw")]
    res, draws = bootstrap(y_oot, {m: preds[m] for m in models}, groups, n_boot=n_boot)

    calib = pd.DataFrame({m: {"mean_pd": float(preds[m].mean()), "observed": float(y_oot.mean())}
                          for m in models}).T

    out = oot[["UniqueID", TARGET, "no_score", "no_usable_score", "employment_self", "age", "age_band",
               "State_ID"]].copy()
    for m, p in preds.items():
        out[f"pd_{m}"] = p
    out["score_lgbm_combined"] = to_points(preds["lgbm_combined"])
    out["score_scorecard_combined"] = to_points(preds["scorecard_combined"])
    out.to_parquet("data/processed/oot_predictions.parquet", index=False)

    # gain from alternative data: AUC of each set minus the bureau-only model of the same type
    auc_of = res.set_index(["model", "group"])["auc"]
    lifts = {}
    for g in ("no_score", "no_usable_score", "all", "self_employed"):
        for kind in ("lgbm", "scorecard"):
            for s in ("alternative", "combined", "fair"):
                a, b = (f"{kind}_{s}", g), (f"{kind}_bureau", g)
                lifts[(kind, s, g)] = (auc_of[a] - auc_of[b], *diff_ci(draws, a, b))

    leak = res[res["auc"] > LEAKAGE_AUC]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_results.md").write_text(results_md(res, calib, info, lifts, leak, n_trials, n_boot),
                                                encoding="utf-8")
    Path("reports/model_card.md").write_text(model_card_md(res, calib, info, saved, fl, win, oot),
                                             encoding="utf-8")
    return {"results": res, "calib": calib, "info": info, "lifts": lifts, "leak": leak}


def _ci(r, k, pct=False):
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(r[k])} ({f(r[k + '_lo'])}–{f(r[k + '_hi'])})"


def results_table(res: pd.DataFrame, group: str) -> list[str]:
    lines = ["| model | AUC (95% CI) | Gini (95% CI) | KS (95% CI) |", "|:--|--:|--:|--:|"]
    for _, r in res[res["group"] == group].iterrows():
        lines.append(f"| {r['model']} | {_ci(r, 'auc')} | {_ci(r, 'gini')} | {_ci(r, 'ks')} |")
    return lines


def results_md(res, calib, info, lifts, leak, n_trials, n_boot) -> str:
    lines = [
        "# Model results — step 3",
        "",
        f"Produced by `src/train.py` (`python -m src.train --trials {n_trials} --boot {n_boot}`) on {date.today()}. "
        "Real L&T data. Tuning (Optuna on 5-fold CV), tree counts, scorecard C and isotonic calibration (on "
        "out-of-fold predictions) are all decided inside Aug–Sep 2018; final models are fit on all of Aug–Sep and "
        "tested once, **out-of-time on October 2018** (98,364 loans). CIs are 95% percentile "
        f"intervals from {n_boot} paired bootstrap resamples of the October rows.",
        "",
    ]
    if len(leak):
        lines += ["**LEAKAGE WARNING:** AUC above 0.75 for: " + ", ".join(
            f"{r.model}/{r.group} {r.auc:.3f}" for r in leak.itertuples()) + ". Do not use these results.", ""]
    else:
        lines += ["Leakage check (CLAUDE.md rule 2): no AUC above 0.75.", ""]
    for g, label in GROUPS.items():
        r0 = res[res["group"] == g].iloc[0]
        lines += [f"## {label} — {r0['n']:,} loans, default rate {r0['bad_rate']:.1%}", "", *results_table(res, g), ""]
    lines += ["## Gain from alternative data (AUC difference vs the bureau-only model, paired 95% CI)", "",
              "| model type | feature set | subgroup | AUC gain (95% CI) |", "|:--|:--|:--|--:|"]
    for (kind, s, g), (d, lo, hi) in lifts.items():
        lines.append(f"| {kind} | {s} vs bureau | {g} | {d:+.3f} ({lo:+.3f} to {hi:+.3f}) |")
    lines += ["", "## Calibration-in-the-large on October", "",
              "Mean predicted PD vs observed default rate. LightGBM is isotonic-calibrated on Aug–Sep "
              "out-of-fold predictions; October's default rate is higher (23.5% vs 20.4%), so some under-prediction "
              "is expected. Full calibration analysis is step 4.", "",
              "| model | mean predicted PD | observed |", "|:--|--:|--:|"]
    for m, r in calib.iterrows():
        lines.append(f"| {m} | {r['mean_pd']:.1%} | {r['observed']:.1%} |")
    lines += ["", "## Training details", ""]
    for m, d in info.items():
        extra = (f"C = {d['C']}; removed for a wrong-sign coefficient: {d['sign_removed'] or 'none'}"
                 if m.startswith("scorecard")
                 else f"params {d['params']}; {d['best_iteration']} trees (fold early-stops {d['fold_iters']})")
        lines.append(f"- **{m}**: {len(d['features'])} features; Aug–Sep 5-fold CV AUC {d['cv_auc']:.4f}; {extra}.")
    lines.append("")
    for m, d in info.items():
        if "top_gain" in d:
            lines.append(f"- {m} top features by gain: " + ", ".join(f"`{k}`" for k in d["top_gain"]))
    lines.append("")
    return "\n".join(lines)


def model_card_md(res, calib, info, saved, fl, win, oot) -> str:
    def window(d):
        return f"{d['disb'].min():%d %b %Y} – {d['disb'].max():%d %b %Y}"

    lines = [
        "# Model card — GigScore",
        "",
        f"Generated by `src/train.py` on {date.today()}; metrics from `reports/model_results.md` (same run). "
        "Evaluation, calibration, fairness and explainability sections are added in steps 4–5.",
        "",
        "## Purpose and use",
        "",
        "- **Predicts:** probability that a two-wheeler borrower misses the **first EMI** on its due date "
        "(L&T `loan_default`), at origination.",
        "- **Intended use:** decision support for small-ticket two-wheeler loans to self-employed and "
        "new-to-credit applicants; a 300–900 score, approve/refer/decline band and reasons; shadow-mode pilots.",
        "- **Not intended for:** fully automated declines without human review; loan types other than "
        "two-wheeler; pricing without recalibration on the lender's own book; any use of age or state as an input.",
        "",
        "## Data",
        "",
        f"- L&T Vehicle Loan Default (Kaggle `mamtadhaker/lt-vehicle-loan-default-prediction`), real Indian NBFC loans.",
        f"- Training window: {window(win)} ({len(win):,} loans); tuning, tree counts and calibration by 5-fold CV "
        "inside this window.",
        f"- Out-of-time test: {window(oot)} ({len(oot):,} rows).",
        "- Protected attributes (age, age band, state) are **not** model inputs; used only for fairness checks.",
        "- `Employee_code_ID` not used. IDs (branch, supplier, manufacturer, pincode) are out-of-fold target encoded.",
        "",
        "## Models",
        "",
    ]
    for m, path in saved.items():
        d = info[m]
        lines += [f"### {m} (`{path}`)", ""]
        if m.startswith("scorecard"):
            lines.append(f"WoE-binned L2 logistic regression, C = {d['C']} (chosen by CV). Scale: {BASE_SCORE} "
                         f"points at {BASE_ODDS}:1 good:bad odds, {PDO} points to double the odds.")
        else:
            lines.append(f"LightGBM, monotone constraints {MONOTONE} (where present), Optuna-tuned on 5-fold CV "
                         f"{d['params']}, {d['best_iteration']} trees, isotonic calibration on out-of-fold predictions.")
        lines += ["", "Features: " + ", ".join(f"`{f}`" for f in d["features"]), "",
                  "| subgroup | n | default rate | AUC (95% CI) | Gini (95% CI) | KS (95% CI) |",
                  "|:--|--:|--:|--:|--:|--:|"]
        for _, r in res[res["model"] == m].iterrows():
            lines.append(f"| {GROUPS[r['group']]} | {r['n']:,} | {r['bad_rate']:.1%} | {_ci(r, 'auc')} | "
                         f"{_ci(r, 'gini')} | {_ci(r, 'ks')} |")
        lines += ["", f"Mean predicted PD on October {calib.loc[m, 'mean_pd']:.1%} vs observed "
                      f"{calib.loc[m, 'observed']:.1%}.", ""]
    lines += [
        "## Limitations",
        "",
        "- Data is from 2018 and from one lender; default levels and ticket sizes are dated. Retrain on the "
        "deploying lender's book before use.",
        "- Borrowers are self-employed / new-to-credit two-wheeler buyers ('gig-like'), not confirmed gig workers.",
        "- The label is first-EMI default, which is noisy; expected out-of-time AUC on this data is 0.66–0.68.",
        "- October's default rate (23.5%) is above the training window's (20.4%); probabilities calibrated on "
        "Aug–Sep under-predict October.",
        "- No income or cash-flow data is available in L&T; the Watch layer and Account Aggregator data address that.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    r = main(a.trials, a.boot)
    pd.set_option("display.width", 200)
    print(r["results"][["model", "group", "n", "auc", "auc_lo", "auc_hi", "gini", "ks"]].round(3).to_string())
    print(r["calib"].round(3))
    for k, v in r["lifts"].items():
        print(k, tuple(round(x, 3) for x in v))
    if len(r["leak"]):
        print("LEAKAGE WARNING: AUC > 0.75", r["leak"][["model", "group", "auc"]])
