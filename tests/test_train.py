"""SYNTHETIC data only: checks the scoring scale, scorecard, metrics and LightGBM wiring."""
import numpy as np
import pandas as pd

from src.metrics import auc_ks, bootstrap, diff_ci
from src.scorecard import FACTOR, OFFSET, Scorecard, from_points, to_points
from src.train import BASE_PARAMS, MONOTONE, cv_lgbm, final_lgbm, fit_lgbm


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"ltv": rng.uniform(40, 95, n), "noise": rng.normal(size=n),
                      "grp": pd.Categorical(rng.choice(["LowRisk", "High", "NoHistory"], n))})
    logit = -4 + 0.04 * X["ltv"] + np.where(X["grp"] == "High", 0.8, 0)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_points_scale():
    # 600 points at 4:1 good:bad odds (PD 20%); +50 points doubles the odds
    assert to_points(1 / 5) == 600
    assert to_points(1 / 9) == 650
    assert to_points(1 / 3) == 550
    assert to_points(0.999999) == 300 and to_points(1e-9) == 900
    assert abs(from_points(600) - 1 / 5) < 1e-12


def test_scorecard_points_add_up_and_signs():
    X, y = _synthetic()
    sc = Scorecard(["ltv", "noise", "grp"], categorical=["grp"]).fit(X, y)
    pt = sc.points_table()
    lab = sc.binner_.labels(X[sc.features])
    total = sc.base_points + sum(lab[f].map(pt[pt.feature == f].set_index("bin")["points"]) for f in sc.features)
    p = sc.predict_proba(X)
    assert np.allclose(total, OFFSET + FACTOR * np.log((1 - p) / p))
    assert "ltv" not in sc.sign_flips


def test_scorecard_drops_wrong_sign_features():
    from src.train import fit_scorecard
    X, y = _synthetic()
    # an exact copy of ltv shifted slightly: collinear, so one of the pair tends to get a wrong sign
    X["ltv_copy"] = X["ltv"] + np.random.default_rng(9).normal(0, 0.5, len(X))
    sc, removed = fit_scorecard(["ltv", "ltv_copy", "noise", "grp"], ["grp"], 1.0, X, y)
    assert not sc.sign_flips
    assert set(removed).isdisjoint(sc.features)


def test_metrics_and_bootstrap():
    y = np.array([0, 0, 1, 1])
    assert auc_ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == (1.0, 1.0)
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 3000)
    good, bad = y + rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)
    res, draws = bootstrap(y, {"good": good, "bad": bad}, {"all": np.ones(3000, bool)}, n_boot=50)
    r = res.set_index("model")
    assert r.loc["good", "auc_lo"] <= r.loc["good", "auc"] <= r.loc["good", "auc_hi"]
    assert abs(r.loc["good", "gini"] - (2 * r.loc["good", "auc"] - 1)) < 1e-12
    lo, hi = diff_ci(draws, ("good", "all"), ("bad", "all"))
    assert lo > 0


def test_cv_and_final_model():
    from src.encode import fold_ids
    X, y = _synthetic()
    folds = fold_ids(len(X))
    params = {**BASE_PARAMS, "learning_rate": 0.1}
    auc, oof, iters = cv_lgbm(params, X, y, folds, ["ltv", "noise", "grp"])
    assert 0.55 < auc < 0.9 and len(iters) == 5 and (oof > 0).all()
    m, cal, oof_raw, n_rounds, cv_auc, _ = final_lgbm(params, X, y, folds, ["ltv", "noise", "grp"])
    assert m.num_trees() == n_rounds and np.allclose(oof_raw, oof)
    p = cal.predict(oof_raw)
    assert abs(p.mean() - y.mean()) < 0.01  # isotonic on OOF reproduces the base rate


def test_lgbm_monotone_in_ltv():
    X, y = _synthetic()
    feats = ["ltv", "noise", "grp"]
    params = {**BASE_PARAMS, "learning_rate": 0.1}
    m = fit_lgbm(params, X, y, X, y, feats)
    grid = pd.DataFrame({"ltv": np.linspace(40, 95, 50), "noise": 0.0,
                         "grp": pd.Categorical(["High"] * 50, categories=X["grp"].cat.categories)})
    p = m.predict(grid[feats])
    assert MONOTONE["ltv"] == 1 and np.all(np.diff(p) >= -1e-12)
