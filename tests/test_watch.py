"""Early-warning rules on SYNTHETIC statements (make_synthetic_statement); not real borrower data."""
import copy

import pandas as pd
import pytest

from src.watch import combine, load_rules, make_synthetic_statement, watch, weekly_inflow


def _run(profile, **kw):
    df = make_synthetic_statement(profile, seed=1)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return watch(df.to_dict(orient="records"), emi=3300, due_day=5, **kw)


def test_synthetic_statement_is_labelled_and_twelve_weeks():
    df = make_synthetic_statement("steady")
    assert (df["description"] == "SYNTHETIC").all()
    assert (df["date"].max() - df["date"].min()).days == 83


def test_profiles_get_the_expected_flags():
    steady, slipping, stressed = _run("steady"), _run("slipping"), _run("stressed")
    assert steady["flag"] == "green" and steady["synthetic"]
    assert slipping["flag"] in ("amber", "red")
    assert stressed["flag"] == "red" and stressed["suggested_action"] == "restructuring_offer"
    assert {"inflow_trend", "inflow_gaps", "bills", "obligation_ratio"} <= {r["rule"] for r in stressed["rules"]}


def test_weekly_inflow_and_gap_rule():
    rows = [{"date": d, "credit": 0 if d.isocalendar()[1] % 4 == 0 else 1000, "debit": 0, "balance": 5000}
            for d in pd.date_range("2026-08-03", "2026-10-25")]
    df = pd.DataFrame(rows)
    wk = weekly_inflow(df)
    assert len(wk) == 12 and (wk >= 0).all()
    r = watch(df.to_dict(orient="records"), emi=1000, due_day=5)
    gaps = next(x for x in r["rules"] if x["rule"] == "inflow_gaps")
    assert gaps["value"] >= 1 and gaps["level"] in ("amber", "red")


def test_thresholds_come_from_config():
    rules = copy.deepcopy(load_rules())
    loose = copy.deepcopy(rules)
    loose["obligation_ratio"] = {"amber_above": 5.0, "red_above": 9.0}
    df = make_synthetic_statement("stressed", seed=1)
    strict = watch(df, 3300, 5, rules=rules)
    relaxed = watch(df, 3300, 5, rules=loose)
    lvl = lambda r: next(x["level"] for x in r["rules"] if x["rule"] == "obligation_ratio")  # noqa: E731
    assert lvl(relaxed) == "green" and lvl(strict) != "green"


def test_combination_logic():
    R = load_rules()
    mk = lambda *lv: [{"level": x} for x in lv]  # noqa: E731
    assert combine(mk("green", "amber"), R) == "green"
    assert combine(mk("amber", "amber"), R) == "amber"
    assert combine(mk("amber", "amber", "amber"), R) == "red"
    assert combine(mk("red"), R) == "red"


def test_short_statement_asks_for_more_data():
    df = make_synthetic_statement("steady", weeks=4)
    r = watch(df, 3300, 5)
    assert r["flag"] == "insufficient_data" and not r["rules"]


def test_bank_narration_gives_category_and_status():
    from src.watch import prepare
    rows = [{"date": "2026-10-01", "credit": 0, "debit": 450, "balance": 100, "description": "ACH RETURN - MSEDCL ELECTRICITY"},
            {"date": "2026-10-02", "credit": 0, "debit": 2500, "balance": 50, "description": "IMPS ROOM RENT OCT"},
            {"date": "2026-10-03", "credit": 900, "debit": 0, "balance": 950, "description": "UPI CURRENT A/C TRANSFER"},
            {"date": "2026-10-05", "credit": 0, "debit": 3300, "balance": 0, "description": "NACH DR TW LOAN EMI"}]
    df = prepare(rows)
    assert df["category"].tolist() == ["utility", "rent", "", "emi"]
    assert df["status"].tolist() == ["bounced", "ok", "ok", "ok"]
    given = prepare([{**rows[1], "category": "utility", "status": "late"}])  # explicit columns win
    assert given.loc[0, "category"] == "utility" and given.loc[0, "status"] == "late"


def test_bank_style_statement_flags_like_the_categorised_one():
    """The same SYNTHETIC statement, written as a bank export (narrations, no category / status),
    must get the same flag as the categorised original."""
    df = make_synthetic_statement("stressed")
    words = {"platform_payout": "UPI PLATFORM PAYOUT", "spend": "UPI PURCHASE", "emi": "NACH DR LOAN EMI",
             "rent": "IMPS ROOM RENT", "utility": "ELECTRICITY BILL PAY"}
    bank = df.assign(description=[("ACH RETURN " if s == "bounced" else "LATE FEE " if s == "late" else "") + words[c]
                                  for c, s in zip(df["category"], df["status"])]).drop(columns=["category", "status"])
    assert watch(bank, 3300, 5)["flag"] == watch(df, 3300, 5)["flag"] == "red"


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        watch([{"date": "2026-10-01", "credit": 10}], 3300, 5)
