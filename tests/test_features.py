import numpy as np
import pandas as pd

from src.encode import WoEBinner, apply_encoders, fit_encoders, fold_ids, oof_target_encode, psi
from src.features import FAIRNESS_ONLY, FEATURE_SETS, build_features, new_emi, score_desc_group
from src.load import load


def test_new_emi_matches_annuity_formula():
    # 65,000 at 1.5%/month over 36 months: 975 * 1.015^36 / (1.015^36 - 1) = 2,349.91
    assert round(float(new_emi(pd.Series([65000.0]))[0]), 2) == 2349.91


def test_score_desc_group():
    s = pd.Series(["No Bureau History Available", "Not Scored: Only a Guarantor",
                   "Not Scored: Sufficient History Not Available", "Not Scored: More than 50 active Accounts found",
                   "A-Very Low Risk", "D-Very Low Risk", "E-Low Risk", "H-Medium Risk", "I-Medium Risk",
                   "M-Very High Risk"])
    assert list(score_desc_group(s)) == ["NoHistory", "NotScored_GuarantorOnly", "NotScored_ThinHistory",
                                         "NotScored_Other", "LowRisk", "LowRisk", "Medium", "Medium", "High", "High"]


def test_build_features_shape_and_protected_attributes(synthetic_csv):
    f = build_features(load(synthetic_csv))
    for c in FEATURE_SETS:
        if not c.endswith(("_te", "_n")):
            assert c in f.columns, c
    # protected attributes are never part of any model feature set
    assert not set(FAIRNESS_ONLY) & set(FEATURE_SETS)
    assert "Employee_code_ID" not in FEATURE_SETS and "disb_month" not in f.columns
    assert np.allclose(f["total_emi"], f["existing_emi"] + f["new_emi_est"])


def test_oof_target_encoding_never_sees_own_label():
    rng = np.random.default_rng(1)
    keys = pd.Series(rng.integers(0, 5, 500))
    y = pd.Series(rng.integers(0, 2, 500))
    folds = fold_ids(500)
    enc1, _, _ = oof_target_encode(keys, y, folds)
    y2 = y.copy()
    y2.iloc[0] = 1 - y2.iloc[0]  # flip one label: its own encoding must not move
    enc2, _, _ = oof_target_encode(keys, y2, folds)
    assert enc1.iloc[0] == enc2.iloc[0]


def test_encoders_handle_unseen_ids(synthetic_csv):
    df = load(synthetic_csv)
    cols, enc = fit_encoders(df, df["loan_default"], fold_ids(len(df)))
    new = df.head(3).copy()
    new["supplier_id"] = -999
    out = apply_encoders(new, enc)
    assert (out["supplier_te"] == enc["te"]["supplier_id"]["prior"]).all()
    assert (out["supplier_n"] == 0).all()


def test_woe_iv_and_psi():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=5000))
    y = pd.Series((rng.random(5000) < 1 / (1 + np.exp(-2 * x))).astype(int))
    noise = pd.Series(rng.normal(size=5000))
    b = WoEBinner().fit(pd.DataFrame({"x": x, "noise": noise}), y)
    assert b.iv["x"] > 0.3 and b.iv["noise"] < 0.02
    lab = b.labels(pd.DataFrame({"x": x, "noise": noise}))
    assert psi(lab["x"], lab["x"]) < 1e-9
    shifted = b.labels(pd.DataFrame({"x": x + 1, "noise": noise}))
    assert psi(lab["x"], shifted["x"]) > 0.25
    # a zero-inflated count (85% zeros) must still get more than one bin and a real IV
    z = pd.Series(np.where(rng.random(5000) < 0.85, 0, rng.integers(1, 30, 5000)).astype(float))
    yz = pd.Series((rng.random(5000) < np.where(z > 0, 0.45, 0.18)).astype(int))
    bz = WoEBinner().fit(pd.DataFrame({"z": z}), yz)
    assert len(bz.tables_["z"]) >= 2 and bz.iv["z"] > 0.1
    assert (bz.tables_["z"]["rows"] / 5000 >= 0.02).all()
    # missing values get their own bin and unseen categories map to WoE 0
    xm = x.copy(); xm[:100] = np.nan
    b2 = WoEBinner().fit(pd.DataFrame({"xm": xm, "c": pd.Series(["a", "b"] * 2500)}), y, categorical=["c"])
    assert "Missing" in b2.tables_["xm"].index
    assert b2.transform(pd.DataFrame({"xm": [np.nan], "c": ["zzz"]}))["c"].iloc[0] == 0.0
