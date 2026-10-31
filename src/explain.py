"""SHAP explanations and reason codes (build guide Sections 7 and 7.1).

SHAP values come from LightGBM's own TreeSHAP (`pred_contrib=True`, log-odds of the uncalibrated
model; isotonic calibration is monotone, so directions carry over). Each applicant gets three
plain-language reasons: the strongest risk-raising factors for refer/decline, the strongest
risk-lowering factors for approve.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.predict import align_categories

FIG = Path("reports/figures")
N_REASONS = 3

# Features whose meaning depends on the value (flags, categories): text per value.
VALUE_TEXT = {
    "no_score": {1: "No credit-bureau history is available", 0: "A credit-bureau history is available"},
    "not_scored": {1: "The credit bureau could not score the applicant", 0: "The credit bureau was able to score the applicant"},
    "has_aadhar": {1: "Aadhaar was provided", 0: "Aadhaar was not provided"},
    "has_pan": {1: "PAN was provided", 0: "PAN was not provided"},
    "has_voterid": {1: "Voter ID was provided", 0: "Voter ID was not provided"},
    "has_driving": {1: "A driving licence was provided", 0: "No driving licence was provided"},
    "employment_self": {1: "Applicant is self-employed", 0: "Applicant is not self-employed"},
    "employment_unknown": {1: "Employment type was not recorded", 0: "Employment type was recorded"},
    "dob_jan1": {1: "Date of birth is recorded only approximately", 0: "Exact date of birth is recorded"},
    "first_loan": {1: "This would be the applicant's first loan", 0: "The applicant has borrowed before"},
    "score_desc_grp": {
        "NoHistory": "No credit-bureau history is available",
        "NotScored_ThinHistory": "Credit history is too thin for a bureau score",
        "NotScored_NotEnoughInfo": "The bureau has too little information to score",
        "NotScored_Inactive": "No recent activity on credit accounts",
        "NotScored_NoUpdates36m": "No credit account updates in the last 3 years",
        "NotScored_GuarantorOnly": "Credit history is only as a guarantor",
        "NotScored_Other": "The credit bureau could not score the applicant",
        "LowRisk": "Credit-bureau risk grade is low (A–D)",
        "Medium": "Credit-bureau risk grade is medium (E–H)",
        "High": "Credit-bureau risk grade is high (I–M)",
    },
}
# Continuous features: (text when it raises risk, text when it lowers risk).
DIRECTION_TEXT = {
    "ltv": ("The loan is a high share of the vehicle price", "A good share of the vehicle price was paid upfront"),
    "down_payment_share": ("The down payment is small", "A larger down payment was made"),
    "pri_disbursed_amount": ("The size of past loans adds to risk", "Past loans of a similar size support repayment"),
    "pri_overdue_accts": ("Some existing loans are overdue", "No existing loans are overdue"),
    "sec_sanctioned_amount": ("Loans as co-applicant or guarantor add to risk", "Co-applicant or guarantor record supports repayment"),
    "new_emi_est": ("The new EMI is large", "The new EMI is modest"),
    "total_emi": ("Total monthly EMIs, including this loan, are high", "Total monthly EMIs are manageable"),
    "existing_emi": ("Existing EMIs are high", "Existing EMIs are low"),
    "leverage": ("Much of the existing credit is still outstanding", "Most existing credit has been repaid"),
    "bureau_score": ("The credit-bureau score is low", "The credit-bureau score is good"),
    "overdue_ratio": ("Some existing loans are overdue", "No existing loans are overdue"),
    "delinq_6m": ("A loan was overdue in the last six months", "No loan was overdue in the last six months"),
    "active_ratio": ("Many existing loans are still running", "Most past loans are closed"),
    "pri_no_accts": ("The number of existing loans adds to risk", "A record of past loans supports repayment"),
    "pri_active_accts": ("The number of existing loans adds to risk", "A record of past loans supports repayment"),
    "pri_current_balance": ("Outstanding balances on existing loans add to risk", "Outstanding balances on existing loans are low"),
    "pri_sanctioned_amount": ("The size of past loans adds to risk", "Past loans of a similar size support repayment"),
    "hist_len_m": ("The credit history is short", "The credit history is long"),
    "acct_age_m": ("The credit history is short", "The credit history is long"),
    "inquiries": ("Several recent loan enquiries", "Few loan enquiries"),
    "new_accts_6m": ("Several new loans were opened in the last six months", "No rush of new loans recently"),
    "sec_no_accts": ("Loans as co-applicant or guarantor add to risk", "Co-applicant or guarantor record supports repayment"),
    "sec_active_accts": ("Loans as co-applicant or guarantor add to risk", "Co-applicant or guarantor record supports repayment"),
    "sec_overdue_accts": ("Loans as co-applicant or guarantor add to risk", "Co-applicant or guarantor record supports repayment"),
    "sec_current_balance": ("Loans as co-applicant or guarantor add to risk", "Co-applicant or guarantor record supports repayment"),
    "sec_disbursed_amount": ("Loans as co-applicant or guarantor add to risk", "Co-applicant or guarantor record supports repayment"),
    "kyc_count": ("Fewer identity documents were provided", "Several identity documents were provided"),
    "supplier_te": ("Loans through this dealer have defaulted more often than average", "Loans through this dealer have repaid better than average"),
    "branch_te": ("Loans from this branch have defaulted more often than average", "Loans from this branch have repaid better than average"),
    "pincode_te": ("Loans from this area have defaulted more often than average", "Loans from this area have repaid better than average"),
    "manufacturer_te": ("This vehicle make has a weaker repayment history", "This vehicle make has a better repayment history"),
    "supplier_n": ("The dealer has few loans with this lender", "The dealer has many loans with this lender"),
    "branch_n": ("The branch has few loans of this type", "The branch has many loans of this type"),
    "pincode_n": ("Few loans have come from this area", "Many loans have come from this area"),
    "asset_cost_log": ("The vehicle price band is linked to higher risk", "The vehicle price band is linked to lower risk"),
    "disb_day": ("Disbursal timing", "Disbursal timing"),
    "disb_weekday": ("Disbursal timing", "Disbursal timing"),
}
# Process and data-entry features: in the model, but not a reason a customer can be given or act on
# (dob_jan1 marks a branch recording a placeholder birth date, not anything about the applicant).
NOT_CUSTOMER_FACING = {"disb_day", "disb_weekday", "dob_jan1"}


def reason_text(feature: str, value, raises_risk: bool) -> str:
    if feature in VALUE_TEXT:
        key = value if feature == "score_desc_grp" else int(value) if pd.notna(value) else None
        return VALUE_TEXT[feature].get(key, feature)
    up, down = DIRECTION_TEXT[feature]
    return up if raises_risk else down


def reasons_for(contrib: np.ndarray, values: pd.Series, features: list[str], band: str, k: int = N_REASONS) -> list[str]:
    """Top-k distinct reasons: risk-raising (positive SHAP) for refer/decline, risk-lowering for approve."""
    raise_risk = band != "approve"
    order = np.argsort(-contrib) if raise_risk else np.argsort(contrib)
    out = []
    for i in order:
        f = features[i]
        if f in NOT_CUSTOMER_FACING or (contrib[i] <= 0 if raise_risk else contrib[i] >= 0):
            continue
        txt = reason_text(f, values.iloc[i], raise_risk)
        if txt not in out:
            out.append(txt)
        if len(out) == k:
            break
    return out


def shap_values(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    feats = bundle["features"]
    Xa = align_categories(X[feats], bundle)
    c = bundle["model"].predict(Xa, num_iteration=bundle["best_iteration"], pred_contrib=True)
    return c, Xa


def run(bundle: dict, X: pd.DataFrame, bands: np.ndarray, sample: int = 5000, seed: int = 0) -> dict:
    import matplotlib  # plotting only here, so the API can import this module without matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    feats = bundle["features"]
    contrib, Xa = shap_values(bundle, X)
    sv = contrib[:, :-1]
    margin = bundle["model"].predict(Xa, num_iteration=bundle["best_iteration"], raw_score=True)
    additivity = float(np.max(np.abs(contrib.sum(axis=1) - margin)))

    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    share_process = float(imp[[f for f in NOT_CUSTOMER_FACING if f in imp]].sum() / imp.sum())

    reasons = [reasons_for(sv[i], Xa.iloc[i], feats, bands[i]) for i in range(len(Xa))]
    n_reasons = np.array([len(r) for r in reasons])
    freq = {}
    for band in ("approve", "refer", "decline"):
        idx = np.flatnonzero(bands == band)
        s = pd.Series([r for i in idx for r in reasons[i]]).value_counts()
        freq[band] = (s / len(idx)).head(6)

    # figures: global importance, beeswarm-style summary on a sample, dependence for the top numerics
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(15)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title(f"{bundle['version']} — global feature importance (October)")
    fig.tight_layout()
    fig.savefig(FIG / "shap_importance.png", dpi=120)
    plt.close(fig)

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(Xa), min(sample, len(Xa)), replace=False)
    top = list(imp.index[:12])
    fig, ax = plt.subplots(figsize=(8, 6))
    for j, f in enumerate(top[::-1]):
        col = feats.index(f)
        v = Xa[f].iloc[pick]
        v = v.cat.codes.astype(float) if hasattr(v, "cat") else v.astype(float)
        rank = v.rank(pct=True).to_numpy()
        ax.scatter(sv[pick, col], j + rng.normal(0, 0.12, len(pick)), c=rank, cmap="coolwarm", s=3, alpha=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[::-1])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds; right = higher risk). Colour: feature value, blue low → red high")
    ax.set_title("SHAP summary, 5,000 October applicants")
    fig.tight_layout()
    fig.savefig(FIG / "shap_summary.png", dpi=120)
    plt.close(fig)

    numeric_top = [f for f in imp.index if f != "score_desc_grp"][:4]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for ax, f in zip(axes, numeric_top):
        ax.scatter(Xa[f].iloc[pick], sv[pick, feats.index(f)], s=3, alpha=0.4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(f)
        ax.set_ylabel("SHAP")
    fig.suptitle("SHAP dependence, top numeric features (October sample)")
    fig.tight_layout()
    fig.savefig(FIG / "shap_dependence.png", dpi=120)
    plt.close(fig)

    examples = []
    for band in ("approve", "refer", "decline"):
        i = int(np.flatnonzero(bands == band)[0])
        examples.append((band, reasons[i]))

    full = float((n_reasons == N_REASONS).mean())
    md = [
        f"TreeSHAP from LightGBM (`pred_contrib`) for all {len(Xa):,} October applicants; additivity check "
        f"(max |sum of SHAP − model log-odds|) = {additivity:.1e}.",
        "",
        "![SHAP importance](figures/shap_importance.png)",
        "",
        "![SHAP summary](figures/shap_summary.png)",
        "",
        "![SHAP dependence](figures/shap_dependence.png)",
        "",
        "Top features by mean |SHAP|: " + ", ".join(f"`{f}` {v:.3f}" for f, v in imp.head(10).items()) + ".",
        "",
        f"**Reason codes.** {full:.1%} of applicants get three distinct reasons "
        f"({(n_reasons >= 1).mean():.1%} get at least one). Process features "
        f"({', '.join(sorted(NOT_CUSTOMER_FACING))}) are never shown as reasons; they carry {share_process:.1%} of "
        "total |SHAP|. Every model feature has a reason text (`src/explain.py`).",
        "",
        "Most frequent reasons by band (share of applicants in the band who get the reason):",
        "",
    ]
    for band, s in freq.items():
        md += [f"*{band}*: " + "; ".join(f"{r} ({v:.0%})" for r, v in s.items()), ""]
    md += ["Examples (first October applicant in each band):", ""]
    md += [f"- {band}: " + " · ".join(r) for band, r in examples]
    card = [f"TreeSHAP reason codes; {full:.1%} of October applicants receive three distinct reasons. "
            "Most frequent decline reasons: " + "; ".join(f"{r} ({v:.0%})" for r, v in freq["decline"].head(3).items())
            + "."]
    return {"md": md, "card": card, "importance": imp, "reasons": reasons, "freq": freq, "full": full,
            "additivity": additivity}
