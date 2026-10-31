"""Feature table (build guide Section 5).

build_features(df) works on the output of load.load() and on a single application alike; the
target encodings and volume counts for IDs come from src/encode.py.

Run:  python -m src.features
Writes data/processed/features.parquet, feature_list.json, splits.json, models/encoders.pkl
and reports/features.md.
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.encode import COUNT_COLUMNS, TE_COLUMNS, WoEBinner, apply_encoders, fit_encoders, fold_ids, psi
from src.load import OOT_START, RAW_TEST, RAW_TRAIN, TARGET, load

# new_emi_est assumption (guide Section 5): 1.5% a month over 36 months.
EMI_RATE, EMI_TERM = 0.015, 36

DOC_FLAGS = ["MobileNo_Avl_Flag", "Aadhar_flag", "PAN_flag", "VoterID_flag", "Driving_flag", "Passport_flag"]
ACCOUNT_RENAME = {f"{p}.{k}": f"{p.lower()}_{v}" for p in ("PRI", "SEC") for k, v in [
    ("NO.OF.ACCTS", "no_accts"), ("ACTIVE.ACCTS", "active_accts"), ("OVERDUE.ACCTS", "overdue_accts"),
    ("CURRENT.BALANCE", "current_balance"), ("SANCTIONED.AMOUNT", "sanctioned_amount"),
    ("DISBURSED.AMOUNT", "disbursed_amount")]}

AGE_BANDS = [0, 25, 35, 45, 55, np.inf]
AGE_BAND_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Feature -> feature set (guide Section 6.3). "combined" features only enter the combined model.
FEATURE_SETS = {
    # bureau
    "no_score": "bureau", "not_scored": "bureau", "bureau_score": "bureau", "score_desc_grp": "bureau",
    "overdue_ratio": "bureau", "delinq_6m": "bureau", "active_ratio": "bureau", "leverage": "bureau",
    "inquiries": "bureau", "new_accts_6m": "bureau", "hist_len_m": "bureau", "acct_age_m": "bureau",
    "first_loan": "bureau", "existing_emi": "bureau",
    **{v: "bureau" for v in ACCOUNT_RENAME.values()},
    # alternative
    "ltv": "alternative", "down_payment_share": "alternative", "new_emi_est": "alternative",
    "asset_cost_log": "alternative", "kyc_count": "alternative", "has_aadhar": "alternative",
    "has_pan": "alternative", "has_voterid": "alternative", "has_driving": "alternative",
    "employment_self": "alternative", "employment_unknown": "alternative", "dob_jan1": "alternative",
    "disb_weekday": "alternative", "disb_day": "alternative",
    **{v: "alternative" for v in TE_COLUMNS.values()},
    **{v: "alternative" for v in COUNT_COLUMNS.values()},
    # needs both bureau (existing EMIs) and application data (new EMI)
    "total_emi": "combined",
}
CATEGORICAL = ["score_desc_grp"]
# Features that stand in for state (proxy check and mitigation test, step 4). The 'fair' feature set
# leaves them out; it is the shipped model (decision recorded in reports/model_card.md).
LOCATION_FEATURES = ["pincode_te", "pincode_n", "branch_te", "branch_n"]
DEALER_FEATURES = ["supplier_te", "supplier_n"]
DOCUMENT_FEATURES = ["has_aadhar", "has_voterid", "has_pan", "has_driving", "kyc_count"]
FAIR_EXCLUDE = LOCATION_FEATURES + DEALER_FEATURES + DOCUMENT_FEATURES
# Protected attributes: kept in the table for src/fairness.py, never model inputs (CLAUDE.md rule 3).
FAIRNESS_ONLY = ["age", "age_band", "State_ID"]
# Keep these when a correlated pair has to lose one: the more readable feature for a credit officer.
PREFER = ["no_score", "ltv", "delinq_6m", "overdue_ratio", "bureau_score", "kyc_count", "total_emi", "new_emi_est",
          "hist_len_m", "inquiries", "supplier_te", "branch_te", "pri_no_accts", "pri_overdue_accts"]

# Scorecard: IV >= 0.02 and no pair above |Spearman| 0.8. LightGBM copes with correlated inputs,
# so it only loses constants and near-duplicates (|Spearman| >= 0.99).
IV_MIN, CORR_MAX, DUP_MAX, PSI_FLAG = 0.02, 0.8, 0.99, 0.25


# Why a score is missing carries risk information (profile.md: 18.4%-25.6% default by reason),
# so each 'Not Scored' reason keeps its own group instead of one NotScored bucket.
NOT_SCORED_REASONS = {
    "Sufficient History Not Available": "NotScored_ThinHistory",
    "Not Enough Info available": "NotScored_NotEnoughInfo",
    "No Activity seen": "NotScored_Inactive",
    "No Updates available": "NotScored_NoUpdates36m",
    "Only a Guarantor": "NotScored_GuarantorOnly",
}


def score_desc_group(desc: pd.Series) -> pd.Series:
    desc = desc.fillna("")
    letter = desc.str.extract(r"^([A-M])-", expand=False)
    grp = pd.Series(np.select(
        [desc.str.startswith("No Bureau"), desc.str.startswith("Not Scored"),
         letter.isin(list("ABCD")), letter.isin(list("EFGH")), letter.isin(list("IJKLM"))],
        ["NoHistory", "NotScored_Other", "LowRisk", "Medium", "High"], default="Other"), index=desc.index)
    for key, name in NOT_SCORED_REASONS.items():
        grp[desc.str.startswith("Not Scored") & desc.str.contains(key, regex=False)] = name
    return grp


def new_emi(principal: pd.Series, r: float = EMI_RATE, n: int = EMI_TERM) -> pd.Series:
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    acc = df.rename(columns=ACCOUNT_RENAME)

    # Capacity
    f["ltv"] = df["ltv"]
    f["down_payment_share"] = 1 - df["disbursed_amount"] / df["asset_cost"]
    f["new_emi_est"] = new_emi(df["disbursed_amount"])
    f["existing_emi"] = df["PRIMARY.INSTAL.AMT"] + df["SEC.INSTAL.AMT"]
    f["total_emi"] = f["existing_emi"] + f["new_emi_est"]
    f["leverage"] = acc["pri_current_balance"] / (acc["pri_sanctioned_amount"] + 1)

    # Asset quality
    f["no_score"] = df["no_score"]
    f["not_scored"] = df["not_scored"]
    f["bureau_score"] = df["bureau_score"]
    f["score_desc_grp"] = score_desc_group(df["PERFORM_CNS.SCORE.DESCRIPTION"]).astype("category")
    f["overdue_ratio"] = acc["pri_overdue_accts"] / (acc["pri_no_accts"] + 1)
    f["delinq_6m"] = df["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"]
    f["active_ratio"] = acc["pri_active_accts"] / (acc["pri_no_accts"] + 1)
    for c in ACCOUNT_RENAME.values():
        f[c] = acc[c]

    # Management
    f["kyc_count"] = df[DOC_FLAGS].sum(axis=1)
    f["has_aadhar"] = df["Aadhar_flag"]
    f["has_pan"] = df["PAN_flag"]
    f["has_voterid"] = df["VoterID_flag"]
    f["has_driving"] = df["Driving_flag"]
    f["inquiries"] = df["NO.OF_INQUIRIES"]
    f["new_accts_6m"] = df["NEW.ACCTS.IN.LAST.SIX.MONTHS"]

    # Earnings
    f["employment_self"] = (df["Employment.Type"] == "Self employed").astype(int)
    f["employment_unknown"] = (df["Employment.Type"] == "Unknown").astype(int)
    f["asset_cost_log"] = np.log(df["asset_cost"])

    # Liquidity
    f["hist_len_m"] = df["hist_len_m"]
    f["acct_age_m"] = df["acct_age_m"]
    f["first_loan"] = (acc["pri_no_accts"] == 0).astype(int)

    # Context. disb_month is not built: training has only Aug-Sep, so October would be out of range.
    f["disb_weekday"] = df["disb"].dt.weekday
    f["disb_day"] = df["disb"].dt.day
    # 1 January birth dates are mostly placeholders (23.7% of rows, profile.md).
    f["dob_jan1"] = df["Date.of.Birth"].astype(str).str.startswith("01-01-").astype(int)

    # Fairness-only attributes
    f["age"] = df["age"]
    f["age_band"] = pd.cut(df["age"], AGE_BANDS, labels=AGE_BAND_LABELS, right=False).astype(str)
    f["State_ID"] = df["State_ID"]
    return f


def correlated_pairs(X: pd.DataFrame, candidates: list[str], threshold: float) -> list[tuple]:
    num = [c for c in candidates if c not in CATEGORICAL]
    corr = X[num].sample(min(len(X), 60_000), random_state=0).corr(method="spearman").abs()
    return [(a, b, corr.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if corr.loc[a, b] > threshold]


def screen(iv: pd.Series, pairs: list[tuple]) -> dict:
    """Keep the preferred / higher-IV feature of each correlated pair, strongest pairs first."""
    dropped = {}
    rank = lambda c: (PREFER.index(c) if c in PREFER else len(PREFER), -iv.get(c, 0))  # noqa: E731
    for a, b, r in sorted(pairs, key=lambda p: -p[2]):
        if a in dropped or b in dropped:
            continue
        keep, lose = (a, b) if rank(a) <= rank(b) else (b, a)
        dropped[lose] = f"|Spearman| {r:.2f} with `{keep}` (kept: {'preferred for readability' if keep in PREFER else 'higher IV'})"
    return dropped


def main(train_path: str = RAW_TRAIN, test_path: str = RAW_TEST) -> dict:
    out_dir, models_dir = Path("data/processed"), Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    df = load(train_path)
    bounds = df.attrs["winsor_bounds"]
    feats = build_features(df)

    # Splits: Aug-Sep = training window (random 20% held out for early stopping / calibration),
    # Oct = out-of-time test. Folds drive OOF target encoding and CV tuning.
    in_window = (df["disb"] < OOT_START).to_numpy()
    tr_idx = df.index[in_window]
    folds = fold_ids(len(tr_idx))
    valid_ids = df.loc[tr_idx].sample(frac=0.2, random_state=42)["UniqueID"]
    split = pd.Series("oot", index=df.index)
    split[tr_idx] = "fit"
    split[df["UniqueID"].isin(valid_ids) & in_window] = "valid"
    fold = pd.Series(-1, index=df.index)
    fold[tr_idx] = folds

    te_tr, enc = fit_encoders(df.loc[tr_idx], df.loc[tr_idx, TARGET], folds)
    te_oot = apply_encoders(df.loc[~in_window], enc)
    feats = feats.join(pd.concat([te_tr, te_oot]))
    enc["winsor_bounds"] = bounds

    candidates = list(FEATURE_SETS)
    y_tr = df.loc[tr_idx, TARGET]
    binner = WoEBinner().fit(feats.loc[tr_idx, candidates], y_tr, categorical=CATEGORICAL)
    iv = binner.iv

    # Stability compares populations, so the training window is encoded with the same full mapping
    # as October; its OOF values differ by fold and would show up as spurious drift.
    stab_tr = feats.loc[tr_idx, candidates].copy()
    full_tr = apply_encoders(df.loc[tr_idx], enc)
    stab_tr[full_tr.columns] = full_tr
    stab_binner = WoEBinner().fit(stab_tr, y_tr, categorical=CATEGORICAL)
    lab_tr = stab_binner.labels(stab_tr)
    lab_oot = stab_binner.labels(feats.loc[~in_window, candidates])
    psi_oct = pd.Series({c: psi(lab_tr[c], lab_oot[c]) for c in candidates})

    psi_nov = None
    if Path(test_path).exists():  # Nov 2018, unlabelled: pipeline check + drift only
        te = load(test_path, bounds=bounds)
        ft = build_features(te).join(apply_encoders(te, enc))
        lab_nov = stab_binner.labels(ft[candidates])
        psi_nov = pd.Series({c: psi(lab_tr[c], lab_nov[c]) for c in candidates})

    dropped = {c: "constant in the training window" for c in candidates
               if feats.loc[tr_idx, c].nunique(dropna=False) <= 1}
    pairs = correlated_pairs(feats.loc[tr_idx], [c for c in candidates if c not in dropped], CORR_MAX)
    dropped.update(screen(iv, [p for p in pairs if p[2] >= DUP_MAX]))
    kept = [c for c in candidates if c not in dropped]

    sc_candidates = [c for c in kept if iv[c] >= IV_MIN]
    sc_dropped = screen(iv, [p for p in pairs if p[0] in sc_candidates and p[1] in sc_candidates])
    scorecard = [c for c in sc_candidates if c not in sc_dropped]
    psi_flagged = [c for c in kept if psi_oct[c] > PSI_FLAG]

    sets = {
        "bureau": [c for c in kept if FEATURE_SETS[c] == "bureau"],
        "alternative": [c for c in kept if FEATURE_SETS[c] == "alternative"],
        "combined": kept,
        "fair": [c for c in kept if c not in FAIR_EXCLUDE],
    }
    feature_list = {
        "features": kept, "scorecard": scorecard, "sets": sets, "categorical": CATEGORICAL,
        "fairness_only": FAIRNESS_ONLY, "dropped": dropped,
        "scorecard_excluded_low_iv": [c for c in kept if iv[c] < IV_MIN],
        "scorecard_excluded_correlated": sc_dropped, "psi_flagged": psi_flagged,
        "assumptions": {"new_emi_rate_month": EMI_RATE, "new_emi_term_months": EMI_TERM,
                        "te_smoothing": enc["smoothing"]},
    }

    keys = ["UniqueID", TARGET, "disb", "no_score", "no_usable_score"]
    table = df[keys].join(feats.drop(columns=["no_score"])).assign(split=split, fold=fold)
    table.to_parquet(out_dir / "features.parquet", index=False)
    (out_dir / "feature_list.json").write_text(json.dumps(feature_list, indent=2))
    splits = {
        "rule": "fit/valid = Aug-Sep 2018 disbursals (valid = random 20%, seed 42); oot = Oct 2018",
        "fit": df.loc[split == "fit", "UniqueID"].tolist(),
        "valid": df.loc[split == "valid", "UniqueID"].tolist(),
        "oot": df.loc[split == "oot", "UniqueID"].tolist(),
        "folds": {str(k): df.loc[fold == k, "UniqueID"].tolist() for k in range(folds.max() + 1)},
    }
    (out_dir / "splits.json").write_text(json.dumps(splits))
    joblib.dump(enc, models_dir / "encoders.pkl")

    stats = pd.DataFrame({"set": pd.Series(FEATURE_SETS), "iv": iv, "psi_oct": psi_oct})
    if psi_nov is not None:
        stats["psi_nov"] = psi_nov
    stats["status"] = [
        "dropped" if c in dropped else "lgbm only (IV < 0.02)" if iv[c] < IV_MIN
        else "lgbm only (correlated)" if c in sc_dropped else "scorecard + lgbm"
        for c in stats.index]
    stats = stats.loc[iv.index]
    stats.rename_axis("feature").to_csv(out_dir / "feature_stats.csv")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/features.md").write_text(
        report(stats, dropped, pairs, split, df, psi_flagged, feature_list), encoding="utf-8")
    return {"stats": stats, "dropped": dropped, "pairs": pairs, "feature_list": feature_list, "split": split}


def report(stats, dropped, pairs, split, df, psi_flagged, fl) -> str:
    counts = split.value_counts()
    rates = df.groupby(split)[TARGET].mean()
    s = stats.copy()
    for c in [c for c in s.columns if c.startswith(("iv", "psi"))]:
        s[c] = s[c].map(lambda v: f"{v:.3f}")
    lines = [
        "# Features — step 2",
        "",
        f"Produced by `src/features.py` (`python -m src.features`) on {date.today()} from `data/raw/train.csv`"
        " (real L&T data). IV and bins are fitted on the Aug–Sep training window only; PSI compares the"
        " training window with October (out-of-time test) and with test.csv (November, unlabelled).",
        "",
        "## Splits",
        "",
        "| split | rows | default rate |",
        "|:--|--:|--:|",
        *[f"| {k} | {counts[k]:,} | {rates[k]:.1%} |" for k in ["fit", "valid", "oot"]],
        "",
        "## Decisions",
        "",
        "- `bureau_score` is missing for score 0 *and* for the 'Not Scored' placeholder codes 11–18; "
        "`not_scored` flags the latter. The headline no-score subgroup stays `PERFORM_CNS.SCORE == 0`; "
        "`no_usable_score` (both groups) is reported alongside.",
        "- Protected attributes `age`, `age_band`, `State_ID` are in the table for fairness only and are "
        "**not** model inputs (CLAUDE.md rule 3). The guide lists `age` and `state_te` as features; "
        "CLAUDE.md takes precedence.",
        "- `Employee_code_ID` is not used (CLAUDE.md rule 4). `disb_month` is not built: the training "
        "window has only Aug and Sep, so October would be an unseen value.",
        "- `score_desc_grp` keeps each 'Not Scored' reason as its own group (thin history, not enough info, "
        "inactive, no updates in 36 months, guarantor only), since the reason carries risk information.",
        "- Winsorising (`src/load.py`) clips only PRI/SEC *amount* columns, at the 1st/99th percentile of their "
        "non-zero values. Clipping every PRI/SEC column at plain percentiles, as the guide's snippet does, made "
        "`SEC.OVERDUE.ACCTS` constant and capped co-applicant amounts at ₹50,000, because 98–99% of SEC values are 0.",
        f"- `new_emi_est` assumes {EMI_RATE:.1%} a month over {EMI_TERM} months.",
        "- IDs are target-encoded out-of-fold (5 folds, smoothing m = 50) on the training window; October and "
        "November use the full training-window mapping. `*_n` columns are loan counts per ID in the training window.",
        "",
        f"## Dropped from both models (constant, or near-duplicate with |Spearman| ≥ {DUP_MAX})",
        "",
        *([f"- `{c}`: {r}" for c, r in dropped.items()] or ["- none"]),
        "",
        f"## Excluded from the scorecard only (LightGBM keeps them)",
        "",
        f"IV < {IV_MIN}: " + (", ".join(f"`{c}`" for c in fl["scorecard_excluded_low_iv"]) or "none") + ".",
        "",
        f"Correlated above {CORR_MAX} with a kept scorecard feature:",
        "",
        *([f"- `{c}`: {r}" for c, r in fl["scorecard_excluded_correlated"].items()] or ["- none"]),
        "",
        f"PSI > {PSI_FLAG} between Aug–Sep and Oct (investigate): "
        + (", ".join(f"`{c}`" for c in psi_flagged) or "**none**") + ".",
        "",
        "## Correlated pairs (|Spearman| > 0.8)",
        "",
        *([f"- `{a}` – `{b}`: {r:.2f}" for a, b, r in sorted(pairs, key=lambda p: -p[2])] or ["- none"]),
        "",
        "## Information value and stability (sorted by IV)",
        "",
        "IV guide: < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, > 0.3 strong. "
        "PSI guide: < 0.10 stable, 0.10–0.25 watch, > 0.25 investigate.",
        "",
        s.to_markdown(),
        "",
        f"Final counts: {len(fl['features'])} features for LightGBM "
        f"(bureau {len(fl['sets']['bureau'])}, alternative {len(fl['sets']['alternative'])}), "
        f"{len(fl['scorecard'])} for the scorecard.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 160)
    print(r["stats"].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nDropped:", json.dumps(r["dropped"], indent=1))
