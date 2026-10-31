"""Read the L&T vehicle-loan CSVs, parse dates and fix types (build guide Section 4.3)."""
import numpy as np
import pandas as pd

RAW_TRAIN = "data/raw/train.csv"
RAW_TEST = "data/raw/test.csv"
TARGET = "loan_default"

# Out-of-time split: train/validate on Aug-Sep 2018 disbursals, test on Oct 2018.
OOT_START = pd.Timestamp("2018-10-01")

# Two-digit years: pandas reads "84" as 1984 but "45" as 2045. Anything after this year is 19xx.
CENTURY_PIVOT = 2025

ACCOUNT_PREFIXES = ("PRI.", "SEC.")


def parse_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d-%m-%y", errors="coerce")
    return d.where(d.dt.year <= CENTURY_PIVOT, d - pd.DateOffset(years=100))


def to_months(s: pd.Series) -> pd.Series:
    """'1yrs 11mon' -> 23.0; unparseable -> NaN."""
    m = s.astype(str).str.extract(r"(\d+)yrs\s*(\d+)mon").astype(float)
    return m[0] * 12 + m[1]


def account_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ACCOUNT_PREFIXES)]


def amount_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in account_columns(df) if c.endswith(("BALANCE", "AMOUNT"))]


def winsor_bounds(df: pd.DataFrame, lo=0.01, hi=0.99) -> dict[str, tuple[float, float]]:
    """1st/99th percentile of the non-zero values of each PRI/SEC amount column.

    Only amounts are clipped (the guide's concern: negative or absurd balances such as 1e9).
    Counts are left alone, and zeros are excluded from the percentiles: SEC.* columns are 98-99%
    zero, so plain percentiles would flatten them (SEC.OVERDUE.ACCTS became constant).
    """
    out = {}
    for c in amount_columns(df):
        nz = df.loc[df[c] != 0, c]
        if len(nz):
            out[c] = (float(min(nz.quantile(lo), 0)), float(nz.quantile(hi)))
    return out


def apply_winsor(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)
    return df


def load(path: str = RAW_TRAIN, winsorise: bool = True, bounds: dict | None = None) -> pd.DataFrame:
    """Load a raw L&T CSV and add the cleaned columns used everywhere else.

    Winsorisation bounds come from the Aug-Sep training window only, so October (the
    out-of-time test month) never influences preprocessing. Pass `bounds` to reuse the
    training-window bounds on test.csv or in production; they are kept in df.attrs.
    """
    return clean(pd.read_csv(path), winsorise, bounds)


def clean(df: pd.DataFrame, winsorise: bool = True, bounds: dict | None = None) -> pd.DataFrame:
    """The cleaning in load(), for any frame with the raw L&T columns (a CSV or one application)."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    df["dob"] = parse_date(df["Date.of.Birth"])
    # Disbursal dates are recent, so no century pivot ("26" is 2026, not 1926).
    df["disb"] = pd.to_datetime(df["DisbursalDate"], format="%d-%m-%y", errors="coerce")
    # A birth date on or after the disbursal date can only be a century error.
    late = df["dob"] >= df["disb"]
    df.loc[late, "dob"] = df.loc[late, "dob"] - pd.DateOffset(years=100)
    df["age"] = ((df["disb"] - df["dob"]).dt.days / 365.25).round(1)

    df["acct_age_m"] = to_months(df["AVERAGE.ACCT.AGE"])
    df["hist_len_m"] = to_months(df["CREDIT.HISTORY.LENGTH"])

    df["Employment.Type"] = df["Employment.Type"].fillna("Unknown")

    # Score 0 means "no score", not a bad score: keep a flag and treat the score as missing.
    df["no_score"] = (df["PERFORM_CNS.SCORE"] == 0).astype(int)
    # 'Not Scored: ...' rows carry placeholder codes 11-18, not scores (profile.md): also missing.
    df["not_scored"] = df["PERFORM_CNS.SCORE.DESCRIPTION"].str.startswith("Not Scored").fillna(False).astype(int)
    df["bureau_score"] = df["PERFORM_CNS.SCORE"].where((df["no_score"] == 0) & (df["not_scored"] == 0))
    # Broader credit-invisible group, reported next to the score == 0 headline subgroup.
    df["no_usable_score"] = df["bureau_score"].isna().astype(int)

    if winsorise:
        if bounds is None:
            bounds = winsor_bounds(df[df["disb"] < OOT_START])
        df = apply_winsor(df, bounds)
        df.attrs["winsor_bounds"] = bounds
    return df


def is_oot(df: pd.DataFrame) -> pd.Series:
    return df["disb"] >= OOT_START
