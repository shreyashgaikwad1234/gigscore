"""SYNTHETIC test fixtures: a tiny hand-made frame shaped like the L&T train.csv. Not real data."""
import numpy as np
import pandas as pd
import pytest


def synthetic_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    disb_day = rng.integers(1, 29, n)
    disb_month = rng.choice([8, 9, 10], n)
    score = rng.choice([0, 0, 15, 300, 650, 780], n)
    desc = np.where(score == 0, "No Bureau History Available",
                    np.where(score == 15, "Not Scored: Only a Guarantor", "C-Very Low Risk"))
    df = pd.DataFrame({
        "UniqueID": np.arange(n) + 400000,
        "disbursed_amount": rng.integers(30000, 90000, n),
        "asset_cost": rng.integers(60000, 120000, n),
        "ltv": rng.uniform(40, 95, n).round(2),
        "branch_id": rng.integers(1, 20, n),
        "supplier_id": rng.integers(1, 50, n),
        "manufacturer_id": rng.choice([45, 86, 120], n),
        "Current_pincode_ID": rng.integers(1, 100, n),
        "Date.of.Birth": [f"{d:02d}-{m:02d}-{y:02d}" for d, m, y in
                          zip(rng.integers(1, 29, n), rng.integers(1, 13, n), rng.choice([84, 90, 99, 0, 1], n))],
        "Employment.Type": rng.choice(["Salaried", "Self employed", None], n, p=[0.4, 0.55, 0.05]),
        "DisbursalDate": [f"{d:02d}-{m:02d}-18" for d, m in zip(disb_day, disb_month)],
        "State_ID": rng.integers(1, 10, n),
        "Employee_code_ID": rng.integers(1, 300, n),
        "MobileNo_Avl_Flag": 1,
        "Aadhar_flag": rng.integers(0, 2, n),
        "PAN_flag": rng.integers(0, 2, n),
        "VoterID_flag": rng.integers(0, 2, n),
        "Driving_flag": rng.integers(0, 2, n),
        "Passport_flag": 0,
        "PERFORM_CNS.SCORE": score,
        "PERFORM_CNS.SCORE.DESCRIPTION": desc,
    })
    for p in ("PRI", "SEC"):
        df[f"{p}.NO.OF.ACCTS"] = rng.integers(0, 5, n)
        df[f"{p}.ACTIVE.ACCTS"] = rng.integers(0, 3, n)
        df[f"{p}.OVERDUE.ACCTS"] = rng.integers(0, 2, n)
        df[f"{p}.CURRENT.BALANCE"] = rng.integers(-5000, 500000, n)
        df[f"{p}.SANCTIONED.AMOUNT"] = rng.integers(0, 600000, n)
        df[f"{p}.DISBURSED.AMOUNT"] = rng.integers(0, 600000, n)
    df["PRIMARY.INSTAL.AMT"] = rng.integers(0, 20000, n)
    df["SEC.INSTAL.AMT"] = rng.integers(0, 2000, n)
    df["NEW.ACCTS.IN.LAST.SIX.MONTHS"] = rng.integers(0, 3, n)
    df["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"] = rng.integers(0, 2, n)
    df["AVERAGE.ACCT.AGE"] = [f"{y}yrs {m}mon" for y, m in zip(rng.integers(0, 6, n), rng.integers(0, 12, n))]
    df["CREDIT.HISTORY.LENGTH"] = [f"{y}yrs {m}mon" for y, m in zip(rng.integers(0, 9, n), rng.integers(0, 12, n))]
    df["NO.OF_INQUIRIES"] = rng.integers(0, 4, n)
    df["loan_default"] = rng.binomial(1, 0.22, n)
    return df


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "synthetic_train.csv"
    synthetic_raw().to_csv(path, index=False)
    return str(path)
