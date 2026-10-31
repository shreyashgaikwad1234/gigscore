"""Score one loan application: 300-900 score, probability of default, band and three reasons.

score(application) is what the Streamlit app and the API call. The application is turned into a raw
L&T-style row and passed through exactly the same cleaning, feature building and encoders as the
training data. Decision thresholds and the recalibration offset come from config/thresholds.yaml.

Kill switch: if the main model errors, or `force_fallback` is set in the config (e.g. after a PSI
alarm), the scorecard fallback is used and the response says so.
"""
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.explain import reason_text, reasons_for, shap_values
from src.load import clean
from src.predict import ENCODERS, featurize, load_bundle, predict_pd
from src.scorecard import to_points

CONFIG = Path("config/thresholds.yaml")
MODEL_PATHS = {"lgbm_fair": "models/lgbm_fair.pkl", "lgbm_combined": "models/lgbm_combined.pkl",
               "scorecard_fair": "models/lr_scorecard_fair.pkl",
               "scorecard_combined": "models/lr_scorecard_combined.pkl"}

# Bureau score -> L&T risk-grade description (score ranges observed in train.csv, reports/profile.md).
GRADES = [(806, "A-Very Low Risk"), (761, "B-Very Low Risk"), (736, "C-Very Low Risk"), (706, "D-Very Low Risk"),
          (681, "E-Low Risk"), (651, "F-Low Risk"), (631, "G-Low Risk"), (601, "H-Medium Risk"),
          (571, "I-Medium Risk"), (521, "J-High Risk"), (351, "K-High Risk"), (301, "L-Very High Risk"),
          (0, "M-Very High Risk")]
# Why there is no bureau score (bureau_status) -> L&T description and placeholder score code.
NO_SCORE_STATUS = {
    "no_history": ("No Bureau History Available", 0),
    "thin_history": ("Not Scored: Sufficient History Not Available", 15),
    "not_enough_info": ("Not Scored: Not Enough Info available on the customer", 17),
    "inactive": ("Not Scored: No Activity seen on the customer (Inactive)", 16),
    "no_updates_36m": ("Not Scored: No Updates available in last 36 months", 18),
    "guarantor_only": ("Not Scored: Only a Guarantor", 14),
}
EMPLOYMENT = {"Salaried", "Self employed", "Unknown"}
DOCS = {"mobile": "MobileNo_Avl_Flag", "aadhar": "Aadhar_flag", "pan": "PAN_flag", "voterid": "VoterID_flag",
        "driving": "Driving_flag", "passport": "Passport_flag"}


@lru_cache(maxsize=1)
def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _bundle(name: str) -> dict:
    return load_bundle(MODEL_PATHS[name])


@lru_cache(maxsize=1)
def _encoders() -> dict:
    return joblib.load(ENCODERS)


def grade_for(score: float) -> str:
    return next(g for lo, g in GRADES if score >= lo)


def _months(m) -> str:
    m = int(m or 0)
    return f"{m // 12}yrs {m % 12}mon"


def to_raw_row(app: dict) -> pd.DataFrame:
    """Application dict (API contract, guide Section 9.3) -> one raw L&T-style row."""
    disb = pd.Timestamp(app.get("disbursal_date") or date.today())
    if app.get("date_of_birth"):
        dob = pd.Timestamp(app["date_of_birth"])
    else:  # only the age is known: an exact (not 1 January) birth date that gives that age
        dob = disb - pd.DateOffset(years=int(app["age"])) - pd.DateOffset(days=1)
    docs = {"mobile": 1, **(app.get("docs") or {})}
    bs = app.get("bureau_score")
    if bs:
        perform, desc = float(bs), grade_for(float(bs))
    else:
        desc, perform = NO_SCORE_STATUS[app.get("bureau_status") or "no_history"]
    disbursed, asset = float(app["disbursed_amount"]), float(app["asset_cost"])
    g = lambda k: float(app.get(k) or 0)  # noqa: E731
    row = {
        "UniqueID": app.get("application_id", 0), "disbursed_amount": disbursed, "asset_cost": asset,
        "ltv": float(app.get("ltv") or round(disbursed / asset * 100, 2)),
        # IDs the shipped model does not use; unseen values fall back to the training prior
        "branch_id": app.get("branch_id", -1), "supplier_id": app.get("supplier_id", -1),
        "manufacturer_id": app.get("manufacturer_id", -1), "Current_pincode_ID": app.get("pincode_id", -1),
        "Date.of.Birth": dob.strftime("%d-%m-%y"), "Employment.Type": app.get("employment_type"),
        "DisbursalDate": disb.strftime("%d-%m-%y"), "State_ID": app.get("state_id", -1), "Employee_code_ID": -1,
        **{col: int(docs.get(k, 0)) for k, col in DOCS.items()},
        "PERFORM_CNS.SCORE": perform, "PERFORM_CNS.SCORE.DESCRIPTION": desc,
        "PRI.NO.OF.ACCTS": g("pri_no_of_accts"), "PRI.ACTIVE.ACCTS": g("pri_active_accts"),
        "PRI.OVERDUE.ACCTS": g("pri_overdue_accts"), "PRI.CURRENT.BALANCE": g("pri_current_balance"),
        "PRI.SANCTIONED.AMOUNT": g("pri_sanctioned_amount"), "PRI.DISBURSED.AMOUNT": g("pri_disbursed_amount")
        or g("pri_sanctioned_amount"),
        "SEC.NO.OF.ACCTS": g("sec_no_of_accts"), "SEC.ACTIVE.ACCTS": g("sec_active_accts"),
        "SEC.OVERDUE.ACCTS": g("sec_overdue_accts"), "SEC.CURRENT.BALANCE": g("sec_current_balance"),
        "SEC.SANCTIONED.AMOUNT": g("sec_sanctioned_amount"), "SEC.DISBURSED.AMOUNT": g("sec_disbursed_amount")
        or g("sec_sanctioned_amount"),
        "PRIMARY.INSTAL.AMT": g("primary_instal_amt"), "SEC.INSTAL.AMT": g("sec_instal_amt"),
        "NEW.ACCTS.IN.LAST.SIX.MONTHS": g("new_accts_6m"), "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS": g("delinq_6m"),
        "AVERAGE.ACCT.AGE": _months(app.get("avg_account_age_months", app.get("credit_history_months"))),
        "CREDIT.HISTORY.LENGTH": _months(app.get("credit_history_months")),
        "NO.OF_INQUIRIES": g("inquiries"),
    }
    return pd.DataFrame([row])


def validate(app: dict) -> list[str]:
    """Input problems that make a score meaningless; empty list = OK."""
    errors = []
    for k in ("disbursed_amount", "asset_cost", "age", "employment_type"):
        if app.get(k) in (None, ""):
            errors.append(f"{k} is required")
    if errors:
        return errors
    if float(app["asset_cost"]) <= 0 or float(app["disbursed_amount"]) <= 0:
        errors.append("amounts must be positive")
    elif float(app["disbursed_amount"]) > float(app["asset_cost"]) * 1.2:
        errors.append("loan amount is more than 120% of the vehicle price")
    if not 18 <= float(app["age"]) <= 75:
        errors.append("age must be between 18 and 75")
    if app["employment_type"] not in EMPLOYMENT:
        errors.append(f"employment_type must be one of {sorted(EMPLOYMENT)}")
    bs = app.get("bureau_score")
    if bs and not 300 <= float(bs) <= 900:
        errors.append("bureau_score must be 300-900, or empty when there is none")
    if not bs and (app.get("bureau_status") or "no_history") not in NO_SCORE_STATUS:
        errors.append(f"bureau_status must be one of {sorted(NO_SCORE_STATUS)}")
    return errors


def featurize_application(app: dict) -> pd.DataFrame:
    enc = _encoders()
    return featurize(clean(to_raw_row(app), bounds=enc["winsor_bounds"]), enc)


def band_for(score: int, cuts: dict) -> str:
    return "approve" if score >= cuts["approve"] else "refer" if score >= cuts["refer"] else "decline"


def scorecard_reasons(bundle: dict, X: pd.DataFrame, band: str) -> list[str]:
    """Reasons from the points table: each feature's points vs that feature's average points."""
    sc, pt = bundle["model"], bundle["points_table"]
    lab = sc.binner_.labels(X[sc.features]).iloc[0]
    rel = {}
    for f in sc.features:
        t = pt[pt.feature == f].set_index("bin")
        avg = (t["points"] * t["rows"]).sum() / t["rows"].sum()
        rel[f] = float(t["points"].get(lab[f], avg) - avg)  # negative = fewer points = more risk
    contrib = -np.array([rel[f] for f in sc.features])  # positive = raises risk, like SHAP
    return reasons_for(contrib, X[sc.features].iloc[0], sc.features, band)


def score(app: dict, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    errors = validate(app)
    if errors:
        raise ValueError("; ".join(errors))
    X = featurize_application(app)
    offset = float(cfg.get("calibration_offset") or 0.0)
    main, fallback = cfg["model"], cfg["fallback_model"]
    used, note = main, None
    try:
        if cfg.get("force_fallback"):
            raise RuntimeError("force_fallback is set in config/thresholds.yaml")
        b = _bundle(main)
        pd_ = float(predict_pd(b, X, offset=offset)[0])
    except Exception as e:  # kill switch: scorecard fallback, decision flagged for review
        used, note = fallback, f"main model unavailable ({e}); scorecard fallback used"
        b = _bundle(fallback)
        pd_ = float(predict_pd(b, X, offset=offset)[0])
    s = int(to_points(pd_))
    band = band_for(s, cfg["bands"][used])
    if b["kind"] == "lgbm":
        contrib, Xa = shap_values(b, X)
        reasons = reasons_for(contrib[0, :-1], Xa.iloc[0], b["features"], band)
    else:
        reasons = scorecard_reasons(b, X, band)
    out = {"score": s, "pd": round(pd_, 4), "band": band, "reasons": reasons, "model_version": b["version"],
           "model_used": used}
    if note:
        out["note"] = note
    return out


APP_REFERENCE = Path("models/app_reference.json")


@lru_cache(maxsize=1)
def app_reference() -> dict:
    """October out-of-time reference written by src/evaluate.py (curve, score quantiles, metrics)."""
    import json
    return json.loads(APP_REFERENCE.read_text(encoding="utf-8"))


def percentile(s: int) -> float:
    """Approximate share of October 2018 test applicants with a lower score than `s`."""
    q = np.array(app_reference()["score_quantiles"])
    return float(np.interp(s, q, np.linspace(0, 1, len(q)), left=0.0, right=1.0))


# reason_text is re-exported for the app's 'what would change this' hints
__all__ = ["score", "validate", "percentile", "reason_text", "to_raw_row", "featurize_application"]
