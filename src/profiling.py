"""Data profile for step 1: the numbers the kickoff prompt asks for, plus checks against CLAUDE.md.

Run:  python -m src.profiling      (writes reports/profile.md)
"""
from datetime import date
from pathlib import Path

import pandas as pd

from src.load import RAW_TRAIN, TARGET, account_columns, load

EXPECTED_ROWS, EXPECTED_COLS, EXPECTED_DEFAULT_RATE = 233_154, 41, 0.217
DOC_FLAGS = ["MobileNo_Avl_Flag", "Aadhar_flag", "PAN_flag", "VoterID_flag", "Driving_flag", "Passport_flag"]


def _rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    t = df.groupby(by, dropna=False, observed=True)[TARGET].agg(rows="size", default_rate="mean")
    t["share"] = t["rows"] / len(df)
    return t[["rows", "share", "default_rate"]].sort_values("rows", ascending=False)


def profile(path: str = RAW_TRAIN) -> dict:
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    df = load(path, winsorise=False)

    p = {"path": path, "rows": len(raw), "cols": raw.shape[1], "default_rate": raw[TARGET].mean()}

    p["no_score"] = _rate_table(df.assign(group=df["no_score"].map({1: "score == 0", 0: "score > 0"})), "group")
    # Broader view: 'Not Scored: ...' rows carry placeholder scores 11-18, not real bureau scores.
    not_scored = df["PERFORM_CNS.SCORE.DESCRIPTION"].str.startswith("Not Scored")
    p["score_groups"] = _rate_table(df.assign(group=pd.Series(
        pd.Categorical.from_codes(df["no_score"] * 2 + not_scored.astype(int),
                                  ["Real bureau score", "Not Scored (score 11-18)", "No Bureau History (score 0)"]),
        index=df.index)), "group")
    scored = df[(df["no_score"] == 0) & ~not_scored]
    from sklearn.metrics import roc_auc_score
    p["bureau_auc_scored"] = roc_auc_score(scored[TARGET], -scored["PERFORM_CNS.SCORE"])
    p["first_loan_share"] = (df["PRI.NO.OF.ACCTS"] == 0).mean()
    p["dob_jan1_share"] = raw["Date.of.Birth"].str.startswith("01-01-").mean()
    p["score_desc"] = (
        df.groupby("PERFORM_CNS.SCORE.DESCRIPTION")
        .agg(rows=(TARGET, "size"), default_rate=(TARGET, "mean"),
             score_min=("PERFORM_CNS.SCORE", "min"), score_max=("PERFORM_CNS.SCORE", "max"))
        .sort_values("rows", ascending=False)
    )

    p["employment_blank"] = int(raw["Employment.Type"].isna().sum())
    p["employment"] = _rate_table(df, "Employment.Type")

    p["disb_min"], p["disb_max"] = df["disb"].min(), df["disb"].max()
    p["by_month"] = _rate_table(df.assign(month=df["disb"].dt.to_period("M").astype(str)), "month").sort_index()

    miss = raw.isna().mean()
    p["missing_gt5"] = miss[miss > 0.05].sort_values(ascending=False)
    p["missing_any"] = miss[miss > 0].sort_values(ascending=False)

    # checks against CLAUDE.md / the guide
    dob_raw = pd.to_datetime(raw["Date.of.Birth"], format="%d-%m-%y", errors="coerce")
    p["dob_unparsed"] = int(df["dob"].isna().sum())
    p["disb_unparsed"] = int(df["disb"].isna().sum())
    p["dob_pivot_shifted"] = int((dob_raw.dt.year > 2025).sum())
    p["age_min"], p["age_max"] = df["age"].min(), df["age"].max()
    p["age_lt18"] = int((df["age"] < 18).sum())
    p["age_gt75"] = int((df["age"] > 75).sum())
    p["acct_age_unparsed"] = int(df["acct_age_m"].isna().sum())
    p["hist_len_unparsed"] = int(df["hist_len_m"].isna().sum())
    acc = raw[account_columns(raw)]
    p["negative_acct_values"] = (acc < 0).sum()[lambda s: s > 0]
    p["doc_flags"] = raw[DOC_FLAGS].mean()
    p["n_levels"] = raw[["branch_id", "supplier_id", "manufacturer_id", "Current_pincode_ID",
                         "State_ID", "Employee_code_ID"]].nunique()
    p["uniqueid_dupes"] = int(raw["UniqueID"].duplicated().sum())

    test_path = Path(path).with_name("test.csv")
    if test_path.exists():
        td = pd.to_datetime(pd.read_csv(test_path, usecols=["DisbursalDate"])["DisbursalDate"], format="%d-%m-%y")
        p["test_rows"], p["test_disb_min"], p["test_disb_max"] = len(td), td.min(), td.max()
    return p


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def _table(t: pd.DataFrame) -> str:
    t = t.copy()
    for c in ("share", "default_rate"):
        if c in t:
            t[c] = t[c].map(_fmt_pct)
    return t.to_markdown()


def contradictions(p: dict) -> list[str]:
    out = []
    if p["rows"] != EXPECTED_ROWS:
        out.append(f"Row count is {p['rows']:,}, CLAUDE.md says {EXPECTED_ROWS:,}.")
    if p["cols"] != EXPECTED_COLS:
        out.append(f"Column count is {p['cols']}, CLAUDE.md says {EXPECTED_COLS}.")
    if abs(p["default_rate"] - EXPECTED_DEFAULT_RATE) > 0.005:
        out.append(f"Default rate is {_fmt_pct(p['default_rate'])}, the guide says 21.7%.")
    nonzero_unscored = p["score_desc"][
        p["score_desc"].index.str.startswith(("Not Scored", "No Bureau")) & (p["score_desc"]["score_max"] > 0)
    ]
    if len(nonzero_unscored):
        out.append(
            f"{int(nonzero_unscored['rows'].sum()):,} rows have a 'Not Scored'/'No Bureau' description but a "
            f"non-zero PERFORM_CNS.SCORE (values {int(nonzero_unscored['score_min'].min())}–"
            f"{int(nonzero_unscored['score_max'].max())}). CLAUDE.md treats only score == 0 as no score; these "
            "rows would count as 'scored' under that rule. Decision needed before features."
        )
    months = set(p["by_month"].index)
    if not months <= {"2018-08", "2018-09", "2018-10"}:
        out.append(f"Disbursal months outside Aug–Oct 2018: {sorted(months - {'2018-08', '2018-09', '2018-10'})}.")
    tr = p["by_month"].loc[["2018-08", "2018-09"]] if {"2018-08", "2018-09"} <= months else None
    if tr is not None and "2018-10" in months:
        tr_rate = (tr["rows"] * tr["default_rate"]).sum() / tr["rows"].sum()
        oct_rate = p["by_month"].loc["2018-10", "default_rate"]
        if abs(oct_rate - tr_rate) > 0.02:
            out.append(f"October default rate is {_fmt_pct(oct_rate)} vs {_fmt_pct(tr_rate)} in Aug–Sep: a "
                       "base-rate shift in the out-of-time month. Probabilities calibrated on Aug–Sep will "
                       "under-predict October; check calibration-in-the-large in step 4.")
    if "test_rows" in p and p["test_disb_min"] > p["disb_max"]:
        out.append(f"test.csv is disbursed {p['test_disb_min']:%d %b} – {p['test_disb_max']:%d %b %Y}, after "
                   "train.csv, not Aug–Oct as the guide says. It has no label, so this changes nothing for "
                   "evaluation, but it is a later-period sample usable for score-drift (PSI) checks.")
    if p["age_lt18"]:
        out.append(f"{p['age_lt18']:,} applicants are under 18 at disbursal after the century fix.")
    if p["dob_unparsed"] or p["disb_unparsed"]:
        out.append(f"Unparsed dates: {p['dob_unparsed']} DOB, {p['disb_unparsed']} disbursal.")
    if p["acct_age_unparsed"] or p["hist_len_unparsed"]:
        out.append(f"Unparsed 'Xyrs Ymon' strings: {p['acct_age_unparsed']} AVERAGE.ACCT.AGE, "
                   f"{p['hist_len_unparsed']} CREDIT.HISTORY.LENGTH.")
    if p["uniqueid_dupes"]:
        out.append(f"{p['uniqueid_dupes']:,} duplicate UniqueID values.")
    return out or ["Nothing found that contradicts CLAUDE.md or the guide."]


def to_markdown(p: dict) -> str:
    ns = p["no_score"]
    emp = p["employment"]
    lines = [
        "# Data profile — L&T train.csv",
        "",
        f"Produced by `src/profiling.py` (called from `notebooks/01_profile.ipynb`) on {date.today()} "
        f"from `{p['path']}`. Real L&T data; nothing here is simulated.",
        "",
        "## Headline",
        "",
        f"- Rows: **{p['rows']:,}**, columns: **{p['cols']}**",
        f"- Default rate (first EMI missed): **{_fmt_pct(p['default_rate'])}**",
        f"- No bureau score (`PERFORM_CNS.SCORE == 0`): **{_fmt_pct(ns.loc['score == 0', 'share'])}** of rows, "
        f"default rate {_fmt_pct(ns.loc['score == 0', 'default_rate'])} vs "
        f"{_fmt_pct(ns.loc['score > 0', 'default_rate'])} for scored rows",
        f"- Self employed: **{_fmt_pct(emp.loc['Self employed', 'share'])}** of rows, default rate "
        f"{_fmt_pct(emp.loc['Self employed', 'default_rate'])}",
        f"- Disbursal dates: **{p['disb_min']:%d %b %Y} – {p['disb_max']:%d %b %Y}**",
        "",
        "## Rows by disbursal month",
        "",
        _table(p["by_month"]),
        "",
        "## No-score vs scored",
        "",
        _table(p["no_score"]),
        "",
        "## Score groups (treating 'Not Scored' placeholder scores as no score)",
        "",
        _table(p["score_groups"]),
        "",
        f"AUC of the raw bureau score on rows with a real score: {p['bureau_auc_scored']:.3f} "
        "(in-sample, all months, no model; a ceiling check for the bureau-only feature set).",
        "",
        f"First-time borrowers (`PRI.NO.OF.ACCTS == 0`): {_fmt_pct(p['first_loan_share'])}.",
        "",
        "## Bureau score description",
        "",
        _table(p["score_desc"]),
        "",
        "## Employment type",
        "",
        f"{p['employment_blank']:,} blank values in the raw file, filled as `Unknown`.",
        "",
        _table(p["employment"]),
        "",
        "## Missing values",
        "",
        "Columns with > 5% missing: " + (", ".join(f"`{c}` ({_fmt_pct(v)})" for c, v in p["missing_gt5"].items())
                                         or "**none**") + ".",
        "",
        "All columns with any missing: " + (", ".join(f"`{c}` ({_fmt_pct(v)})" for c, v in p["missing_any"].items())
                                            or "none") + ".",
        "",
        "## Data-quality checks",
        "",
        f"- Birth years moved back a century by the pivot rule (> 2025 → 19xx): {p['dob_pivot_shifted']:,}",
        f"- Age at disbursal after fixes: {p['age_min']:.1f} – {p['age_max']:.1f} years "
        f"({p['age_lt18']:,} under 18, {p['age_gt75']:,} over 75)",
        f"- Unparsed dates: DOB {p['dob_unparsed']}, disbursal {p['disb_unparsed']}",
        f"- Unparsed account-age strings: AVERAGE.ACCT.AGE {p['acct_age_unparsed']}, "
        f"CREDIT.HISTORY.LENGTH {p['hist_len_unparsed']}",
        f"- Duplicate UniqueID: {p['uniqueid_dupes']}",
        f"- Date of birth recorded as 1 January (likely a placeholder): {_fmt_pct(p['dob_jan1_share'])}",
        (f"- test.csv: {p['test_rows']:,} rows, disbursed {p['test_disb_min']:%d %b %Y} – {p['test_disb_max']:%d %b %Y}"
         if "test_rows" in p else "- test.csv not found"),
        "- PRI/SEC columns with negative values (before winsorising): "
        + (", ".join(f"`{c}` ({v:,})" for c, v in p["negative_acct_values"].items()) or "none"),
        "- Document-flag share = 1: " + ", ".join(f"`{c}` {_fmt_pct(v)}" for c, v in p["doc_flags"].items()),
        "- Distinct levels: " + ", ".join(f"`{c}` {v:,}" for c, v in p["n_levels"].items()),
        "",
        "## Anything that contradicts CLAUDE.md or the guide",
        "",
        *[f"- {c}" for c in contradictions(p)],
        "",
    ]
    return "\n".join(lines)


def main(path: str = RAW_TRAIN, out: str = "reports/profile.md") -> dict:
    p = profile(path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(to_markdown(p), encoding="utf-8")
    return p


if __name__ == "__main__":
    main()
    print(Path("reports/profile.md").read_text(encoding="utf-8"))
