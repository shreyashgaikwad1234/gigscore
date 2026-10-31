import numpy as np
import pandas as pd

from src.load import OOT_START, load, parse_date, to_months
from src.profiling import profile, to_markdown


def test_parse_date_century_rule():
    d = parse_date(pd.Series(["01-01-84", "15-06-01", "31-12-45", "03-08-18", "bad"]))
    assert list(d.dt.year[:4]) == [1984, 2001, 1945, 2018]
    assert pd.isna(d.iloc[4])


def test_disbursal_dates_are_not_pivoted(synthetic_csv):
    from src.load import clean
    raw = pd.read_csv(synthetic_csv).head(1)
    raw["DisbursalDate"] = "29-10-26"
    raw["Date.of.Birth"] = "15-06-95"
    df = clean(raw)
    assert df["disb"].iloc[0] == pd.Timestamp("2026-10-29") and 31 < df["age"].iloc[0] < 32


def test_to_months():
    m = to_months(pd.Series(["1yrs 11mon", "0yrs 0mon", "10yrs 2mon", None]))
    assert list(m[:3]) == [23, 0, 122]
    assert np.isnan(m.iloc[3])


def test_load_columns_and_flags(synthetic_csv):
    df = load(synthetic_csv)
    assert (df["age"] > 0).all()
    assert df["Employment.Type"].notna().all() and (df["Employment.Type"] == "Unknown").any()
    zero = df["PERFORM_CNS.SCORE"] == 0
    assert (df.loc[zero, "no_score"] == 1).all() and df.loc[zero, "bureau_score"].isna().all()
    placeholder = df["PERFORM_CNS.SCORE.DESCRIPTION"].str.startswith("Not Scored")
    assert (df.loc[~zero, "no_score"] == 0).all()
    assert df.loc[placeholder, "bureau_score"].isna().all() and (df.loc[placeholder, "not_scored"] == 1).all()
    assert df.loc[~zero & ~placeholder, "bureau_score"].notna().all()
    assert (df["no_usable_score"] == (zero | placeholder).astype(int)).all()


def test_winsor_bounds_use_training_window_only(synthetic_csv):
    df = load(synthetic_csv)
    raw = load(synthetic_csv, winsorise=False)
    train = raw[raw["disb"] < OOT_START]
    nz = train.loc[train["PRI.CURRENT.BALANCE"] != 0, "PRI.CURRENT.BALANCE"]
    lo, hi = df.attrs["winsor_bounds"]["PRI.CURRENT.BALANCE"]
    assert lo == min(nz.quantile(0.01), 0) and hi == nz.quantile(0.99)
    assert df["PRI.CURRENT.BALANCE"].between(lo, hi).all()
    # counts are never clipped
    assert "PRI.NO.OF.ACCTS" not in df.attrs["winsor_bounds"]
    assert (df["SEC.OVERDUE.ACCTS"] == raw["SEC.OVERDUE.ACCTS"]).all()


def test_profile_runs_on_synthetic(synthetic_csv):
    p = profile(synthetic_csv)
    assert p["rows"] == 200
    md = to_markdown(p)
    assert "Default rate" in md and "contradicts" in md
    # the synthetic frame has 'Not Scored' rows with score 15, which the check must catch
    assert "non-zero PERFORM_CNS.SCORE" in md
