"""Layer 2 early-warning monitor: transparent rules on bank / UPI statement rows (build guide Section 8).

watch(rows, emi, due_day) returns a green / amber / red flag, the rules that fired and a suggested
action. Thresholds live in config/watch_rules.yaml. The rule outputs are the same columns a trained
survival or discrete-time model would use later (guide Section 8, Step 3).

Statement rows: date, credit, debit, balance (end-of-row balance); optional category
(utility / rent / emi / ...) and status (ok / late / bounced).

make_synthetic_statement() builds SYNTHETIC statements for tests and the app demo; they are labelled
as such in every row and file name and are not evidence about real borrowers.
"""
import re
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RULES_FILE = Path("config/watch_rules.yaml")
LEVELS = {"green": 0, "amber": 1, "red": 2}


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or RULES_FILE).read_text(encoding="utf-8"))


# Real bank / UPI exports have a narration instead of category and status columns; these keywords
# recover what the rules need from it (checked in order, first match wins).
CATEGORY_WORDS = [
    ("emi", ("emi", "nach", "ecs", "loan")),
    ("rent", ("rent",)),
    ("utility", ("electricity", "msedcl", "mahavitaran", "bescom", "tneb", "power", "broadband", "gas", "water",
                 "mobile bill", "postpaid", "dth", "bill pay", "billpay")),
]
STATUS_WORDS = [("bounced", ("return", "bounce", "insufficient", "rtn", "dishonour", "unpaid")),
                ("late", ("late fee", "late payment", "penalty", "overdue"))]


def _from_text(text: str, table) -> str:
    # whole words only, so "current account" is not read as rent
    return next((label for label, words in table
                 if re.search(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", text)), "")


def prepare(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    missing = {"date", "credit", "debit", "balance"} - set(df.columns)
    if missing:
        raise ValueError(f"statement rows need columns {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    for c in ("credit", "debit", "balance"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ("category", "status"):
        df[c] = df[c].astype(str).str.lower().replace({"nan": "", "none": ""}) if c in df else ""
    if "description" in df:  # no category / status given: infer them from the narration
        text = df["description"].fillna("").astype(str).str.lower()
        if not (df["category"] != "").any():
            df["category"] = text.map(lambda t: _from_text(t, CATEGORY_WORDS))
        if not (df["status"] != "").any():
            df["status"] = text.map(lambda t: _from_text(t, STATUS_WORDS) or "ok")
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def weekly_inflow(df: pd.DataFrame, weeks: int = 12) -> pd.Series:
    """Credits per 7-day week ending on the last statement date; index 0 = oldest week."""
    end = df["date"].max()
    wk = ((end - df["date"]).dt.days // 7).clip(upper=weeks)
    s = df[wk < weeks].groupby(wk[wk < weeks])["credit"].sum().reindex(range(weeks), fill_value=0.0)
    return s[::-1].reset_index(drop=True)


def due_dates(start: pd.Timestamp, end: pd.Timestamp, due_day: int) -> list[pd.Timestamp]:
    out, m = [], pd.Timestamp(start.year, start.month, 1)
    while m <= end:
        d = m + pd.offsets.MonthEnd(0)
        out.append(pd.Timestamp(m.year, m.month, min(due_day, d.day)))
        m = m + pd.offsets.MonthBegin(1)
    return [d for d in out if start <= d <= end]


def _level(value: float, amber, red, higher_is_worse: bool) -> str:
    if higher_is_worse:
        return "red" if value >= red else "amber" if value >= amber else "green"
    return "red" if value < red else "amber" if value < amber else "green"


def evaluate_rules(df: pd.DataFrame, emi: float, due_day: int, other_emis: float, R: dict) -> list[dict]:
    end, start = df["date"].max(), df["date"].min()
    out = []

    wk = weekly_inflow(df)
    n_weeks = min(12, int((end - start).days // 7) + 1)
    w = wk.iloc[-n_weeks:]
    base, recent = w.mean(), w.iloc[-4:].mean()
    ratio = recent / base if base > 0 else 0.0
    r = R["inflow_trend"]
    lvl = _level(ratio, r["amber_below"], r["red_below"], higher_is_worse=False)
    out.append({"rule": "inflow_trend", "value": round(ratio, 3), "level": lvl,
                "trigger": f"inflow_trend_{int((r['red_below'] if lvl == 'red' else r['amber_below']) * 100)}",
                "detail": f"last 4 weeks' inflow is {ratio:.0%} of the {n_weeks}-week average"})

    zero = int((wk.iloc[-8:] == 0).sum())
    r = R["inflow_gaps"]
    lvl = "red" if zero >= r["red_at"] else "amber" if zero >= r["amber_at"] else "green"
    out.append({"rule": "inflow_gaps", "value": zero, "level": lvl, "trigger": f"inflow_gaps_{zero}",
                "detail": f"{zero} of the last 8 weeks had no inflow"})

    r = R["balance_before_emi"]
    dues = [d for d in due_dates(start + timedelta(days=r["days_before"]), end, due_day)]
    if dues and emi > 0:
        d = dues[-1]
        win = df[(df["date"] >= d - timedelta(days=r["days_before"])) & (df["date"] < d)]
        before = df[df["date"] < d - timedelta(days=r["days_before"])]
        bal = win["balance"].min() if len(win) else (before["balance"].iloc[-1] if len(before) else 0.0)
        mult = bal / emi
        lvl = _level(mult, r["amber_below"], r["red_below"], higher_is_worse=False)
        out.append({"rule": "balance_before_emi", "value": round(mult, 2), "level": lvl,
                    "trigger": f"balance_before_emi_{r['red_below'] if lvl == 'red' else r['amber_below']}",
                    "detail": f"lowest balance in the {r['days_before']} days before the {d:%d %b} EMI was "
                              f"₹{bal:,.0f} ({mult:.2f}× the EMI)"})

    r = R["bills"]
    recent_rows = df[df["date"] > end - timedelta(days=r["lookback_days"])]
    bad = int((recent_rows["category"].isin(["utility", "rent"]) & recent_rows["status"].isin(["late", "bounced"])).sum())
    lvl = "red" if bad >= r["red_at"] else "amber" if bad >= r["amber_at"] else "green"
    out.append({"rule": "bills", "value": bad, "level": lvl, "trigger": f"bills_late_{bad}",
                "detail": f"{bad} utility / rent payment(s) late or bounced in the last {r['lookback_days']} days"})

    r = R["obligation_ratio"]
    inflow_28 = df.loc[df["date"] > end - timedelta(days=28), "credit"].sum()
    obl = (emi + other_emis) / inflow_28 if inflow_28 > 0 else np.inf
    lvl = _level(obl, r["amber_above"], r["red_above"], higher_is_worse=True)
    out.append({"rule": "obligation_ratio", "value": round(float(obl), 3) if np.isfinite(obl) else None,
                "level": lvl, "trigger": f"obligation_{int((r['red_above'] if lvl == 'red' else r['amber_above']) * 100)}",
                "detail": f"monthly EMIs ₹{emi + other_emis:,.0f} vs last 28 days' inflow ₹{inflow_28:,.0f}"})

    r = R["cash_out_speed"]
    days = max(1, r["hours"] // 24)
    credits = df[df["credit"] > 0]
    if len(credits):
        daily_debit = df.groupby(df["date"].dt.normalize())["debit"].sum()
        shares = []
        for d, amt in zip(credits["date"].dt.normalize(), credits["credit"]):
            spent = daily_debit.reindex(pd.date_range(d, d + timedelta(days=days - 1))).fillna(0).sum()
            shares.append(min(1.0, spent / amt))
        speed = float(np.average(shares, weights=credits["credit"]))
        lvl = _level(speed, r["amber_above"], r["red_above"], higher_is_worse=True)
        out.append({"rule": "cash_out_speed", "value": round(speed, 3), "level": lvl,
                    "trigger": f"cash_out_{int((r['red_above'] if lvl == 'red' else r['amber_above']) * 100)}",
                    "detail": f"on average {speed:.0%} of each inflow is spent within {r['hours']} hours"})
    return out


def combine(results: list[dict], R: dict) -> str:
    reds = sum(r["level"] == "red" for r in results)
    ambers = sum(r["level"] == "amber" for r in results)
    c = R["combination"]
    if reds or ambers >= c["red_if_ambers"]:
        return "red"
    return "amber" if ambers >= c["amber_if_ambers"] else "green"


def watch(rows, emi: float, due_day: int, other_emis: float = 0.0, loan_id: str | None = None,
          rules: dict | None = None) -> dict:
    R = rules or load_rules()
    df = prepare(rows)
    end = df["date"].max()
    weeks = int((end - df["date"].min()).days // 7) + 1
    nxt = [d for d in due_dates(end, end + timedelta(days=62), due_day) if d > end]
    base = {"loan_id": loan_id, "weeks_of_data": weeks, "statement_end": f"{end:%Y-%m-%d}",
            "days_to_next_emi": int((nxt[0] - end).days) if nxt else None,
            "weekly_inflow": [round(float(v), 2) for v in weekly_inflow(df)],
            "synthetic": bool(df.get("description", pd.Series(dtype=str)).astype(str).str.contains("SYNTHETIC").any())}
    if weeks < R["min_weeks"]:
        return {**base, "flag": "insufficient_data", "triggers": [], "rules": [],
                "suggested_action": f"request at least {R['min_weeks']} weeks of statement"}
    results = evaluate_rules(df, float(emi), int(due_day), float(other_emis), R)
    flag = combine(results, R)
    fired = [r for r in results if r["level"] != "green"]
    action = R["actions"][flag]
    if flag == "amber" and {r["rule"] for r in fired} & {"balance_before_emi"} and not any(
            r["rule"] in ("inflow_trend", "inflow_gaps") and r["level"] != "green" for r in fired):
        action = R["actions"]["amber_timing"]  # money arrives, just not before the due date
    return {**base, "flag": flag, "triggers": [r["trigger"] for r in fired], "rules": results,
            "suggested_action": action}


# --- SYNTHETIC statements (tests and app demo only) -------------------------------------------

PROFILES = {
    # (payout multiplier in the last 4 weeks, zero-inflow weeks in the last 4, bills late/bounced, start balance)
    "steady": (1.00, 0, 0, 6000),
    "slipping": (0.72, 1, 1, 3500),
    "stressed": (0.45, 2, 2, 1200),
}


def make_synthetic_statement(profile: str = "steady", weeks: int = 12, emi: float = 3300, due_day: int = 5,
                             end: str = "2026-10-26", seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC 12-week statement for a delivery rider: daily platform payouts, spending, rent,
    electricity and the EMI. Not real data; for tests and the app demo."""
    mult, gaps, late_bills, bal = PROFILES[profile]
    rng = np.random.default_rng(seed)
    end_d = pd.Timestamp(end)
    days = pd.date_range(end_d - timedelta(days=7 * weeks - 1), end_d)
    gap_weeks = set(rng.choice(range(weeks - 4, weeks), gaps, replace=False)) if gaps else set()
    rows, late_left = [], late_bills
    for d in days:
        wk = (d - days[0]).days // 7
        recent = wk >= weeks - 4
        if d.weekday() != 6 and wk not in gap_weeks:  # paid six days a week
            pay = round(rng.normal(950, 120) * (mult if recent else 1.0), 0)
            bal += pay
            rows.append((d, pay, 0.0, bal, "platform_payout", "ok"))
            spend = round(pay * rng.uniform(0.35, 0.6), 0)  # fuel, food, top-ups
            bal -= spend
            rows.append((d, 0.0, spend, bal, "spend", "ok"))
        elif wk in gap_weeks:  # no income that week, but daily spending continues from savings
            spend = round(rng.uniform(120, 220), 0)
            bal -= spend
            rows.append((d, 0.0, spend, bal, "spend", "ok"))
        if d.day == due_day:
            bal -= emi
            rows.append((d, 0.0, emi, bal, "emi", "ok" if bal >= 0 else "bounced"))
        for day, cat, amt in ((1, "rent", 2500), (10, "utility", 450)):
            if d.day == day:
                late = recent and late_left > 0
                late_left -= int(late)
                bal -= amt
                rows.append((d, 0.0, amt, bal, cat, "late" if late else "ok"))
    df = pd.DataFrame(rows, columns=["date", "credit", "debit", "balance", "category", "status"])
    df["description"] = "SYNTHETIC"
    return df
