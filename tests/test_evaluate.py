"""SYNTHETIC data only: business curve, cut-offs, calibration, reason codes and fairness tables."""
import numpy as np
import pandas as pd
import yaml

from src import evaluate
from src.evaluate import (approval_at_bad_rate, approval_curve, assign_bands, band_table, cutoff_for_share,
                          reliability, score_psi)
from src.explain import DIRECTION_TEXT, NOT_CUSTOMER_FACING, VALUE_TEXT, reason_text, reasons_for
from src.fairness import group_table, option_metrics
from src.features import FEATURE_SETS


def test_approval_curve_and_bad_rate_target():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.arange(8) / 10  # already sorted safest first
    c = approval_curve(y, p, [0.25, 0.5, 1.0])
    assert list(c["bad_rate"]) == [0.0, 0.25, 0.5]
    assert approval_at_bad_rate(y, p, 0.25) == 5 / 8  # first five rows: 1 bad of 5 = 20%
    assert approval_at_bad_rate(y, p, -1) == 0.0


def test_ties_are_not_split():
    # one tied block of 10 (3 bads, the bads sorted first by row order) then 10 clean rows at higher PD
    y = np.array([1, 1, 1] + [0] * 7 + [0] * 10)
    p = np.array([0.1] * 10 + [0.2] * 10)
    # inside the tie the bad rate is the block average (30%), not the row-order accident (100%)
    assert abs(approval_curve(y, p, [0.1])["bad_rate"].iloc[0] - 0.3) < 1e-12
    # cut-offs can only fall between blocks: first block alone is 3/10 = 30%, both blocks 3/20 = 15%.
    # Row order would allow "approve 0 of the first 3"; a 14% target must therefore approve nobody.
    assert approval_at_bad_rate(y, p, 0.14) == 0.0
    assert approval_at_bad_rate(y, p, 0.15) == 1.0
    y2 = np.r_[y, [1] * 10]  # add a third, all-bad block: whole book is 13/30 = 43%
    p2 = np.r_[p, [0.3] * 10]
    assert approval_at_bad_rate(y2, p2, 0.30) == 20 / 30


def test_cutoffs_and_bands():
    scores = np.arange(300, 400)  # 100 distinct scores
    a = cutoff_for_share(scores, 0.6)
    r = cutoff_for_share(scores, 0.8)
    assert (scores >= a).mean() == 0.6 and (scores >= r).mean() == 0.8
    b = assign_bands(scores, a, r)
    assert (b == "approve").sum() == 60 and (b == "refer").sum() == 20 and (b == "decline").sum() == 20
    t = band_table(np.zeros(100, int), b)
    assert list(t.index) == ["approve", "refer", "decline"] and abs(t["share"].sum() - 1) < 1e-12


def test_reliability_and_score_psi():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.5, 20000)
    y = (rng.random(20000) < p).astype(int)
    rel = reliability(y, p)
    assert len(rel) == 10 and (rel["diff_pp"].abs() < 3).all()
    s = rng.normal(600, 40, 20000)
    assert score_psi(s, s) < 1e-9 and score_psi(s, s + 40) > 0.25


def test_offset_recalibration():
    from src.evaluate import fit_offset, recalibration_backtest
    from src.predict import apply_offset
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.4, 20000)
    y = (rng.random(20000) < apply_offset(p, 0.3)).astype(int)  # truth is 0.3 log-odds riskier
    d = fit_offset(p, y)
    assert abs(d - 0.3) < 0.05 and abs(apply_offset(p, d).mean() - y.mean()) < 1e-9
    assert np.array_equal(np.argsort(apply_offset(p, d)), np.argsort(p))  # ranking unchanged
    oot = pd.DataFrame({"disb": pd.to_datetime("2018-10-01") + pd.to_timedelta(rng.integers(0, 31, 20000), "D")})
    r = recalibration_backtest(oot, y, p)
    assert r["auc_unchanged"] and abs(r["mean_after"] - r["observed"]) < abs(r["mean_before"] - r["observed"])


def test_write_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "THRESHOLDS", tmp_path / "thresholds.yaml")
    bt = pd.DataFrame({"rows": [6, 2, 2], "share": [.6, .2, .2], "bad_rate": [.1, .2, .4]},
                      index=["approve", "refer", "decline"])
    evaluate.write_thresholds({"lgbm_combined": {"approve": 640, "refer": 600},
                               "scorecard_combined": {"approve": 630, "refer": 590}}, bt)
    cfg = yaml.safe_load((tmp_path / "thresholds.yaml").read_text())
    assert cfg["bands"]["lgbm_combined"] == {"approve": 640, "refer": 600}
    assert cfg["kill_switch"]["psi_max"] == 0.25 and cfg["calibration_offset"] == 0.0


def test_every_feature_has_reason_text():
    for f in FEATURE_SETS:
        assert f in VALUE_TEXT or f in DIRECTION_TEXT, f
    assert reason_text("has_pan", 0, True) == "PAN was not provided"
    assert reason_text("score_desc_grp", "NoHistory", True) == "No credit-bureau history is available"


def test_reasons_follow_band_and_skip_process_features():
    feats = ["disb_day", "ltv", "supplier_te", "has_pan", "kyc_count"]
    contrib = np.array([0.9, 0.5, 0.3, -0.2, -0.4])
    vals = pd.Series([15, 90.0, 0.3, 0, 2])
    decline = reasons_for(contrib, vals, feats, "decline")
    assert decline == [DIRECTION_TEXT["ltv"][0], DIRECTION_TEXT["supplier_te"][0]]  # only positives, no disb_day
    approve = reasons_for(contrib, vals, feats, "approve")
    assert approve == [DIRECTION_TEXT["kyc_count"][1], "PAN was not provided"]
    assert "disb_day" in NOT_CUSTOMER_FACING


def test_group_table_gaps():
    rng = np.random.default_rng(1)
    n = 4000
    attr = pd.Series(np.where(np.arange(n) < 2000, "A", "B"))
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    bands = np.where(attr == "A", "approve", "decline")  # A always approved, B never
    tab = group_table(attr, y, p, bands, min_rows=100)
    assert abs(tab.loc["A", "approval_gap"] - 0.5) < 1e-12 and tab.loc["A", "flag"]
    assert tab.loc["B", "approval"] == 0 and tab.loc["B", "fpr"] == 1


def test_option_metrics_state_gaps():
    rng = np.random.default_rng(2)
    n = 4000
    oot = pd.DataFrame({"State_ID": np.where(np.arange(n) < 2000, 1, 2), "no_score": rng.integers(0, 2, n)})
    y = rng.integers(0, 2, n)
    p = np.where(oot["State_ID"] == 1, 0.1, 0.5) + rng.random(n) * 0.01  # state 1 always safer
    r = option_metrics(oot, y, p, 0.5)
    assert abs(r["max_state_gap"] - 0.5) < 1e-9 and r["states_flagged"] == 2
