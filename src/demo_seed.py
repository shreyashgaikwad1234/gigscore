"""Fill the workspace with SYNTHETIC demo cases for a pitch:  python -m src.demo_seed
Back to a clean demo before a pitch:                             python -m src.demo_seed --reset
(--reset copies the decision log to logs/ first, then removes only the APP-DEMO-… cases and re-creates them.)

Twelve made-up applicants (names are fictional, every record is flagged synthetic) across three branches:
six waiting in the inbox — one past the 24-hour target — and six already decided, some disbursed and
under early warning with collections calls recorded. Scores come from the real model; only the people,
the timestamps and the officers' actions are invented. Run once; it skips if the demo cases exist.
"""
import argparse
import shutil
from datetime import datetime, timedelta, timezone

from src import decision_log, score as scoring, watch as watching

NOW = datetime.now(timezone.utc)
BASE = dict(disbursed_amount=55000, asset_cost=72000, age=27, employment_type="Self employed", docs={"aadhar": 1},
            bureau_score=None, bureau_status="no_history")
PROFILES = {
    "refer": {}, "refer_b": dict(disbursed_amount=56000, age=31),
    "thin": dict(bureau_status="thin_history", disbursed_amount=52000),
    "high_ltv": dict(disbursed_amount=66000), "young": dict(age=21, disbursed_amount=68000, asset_cost=75000),
    "older": dict(age=41, disbursed_amount=45000, asset_cost=78000, docs={"aadhar": 1, "pan": 1}),
    "bureau_690": dict(bureau_score=690, bureau_status=None, pri_no_of_accts=2, pri_active_accts=1, credit_history_months=30,
                       pri_sanctioned_amount=80000, pri_current_balance=30000, disbursed_amount=50000),
    "salaried_750": dict(employment_type="Salaried", bureau_score=752, bureau_status=None, pri_no_of_accts=3,
                         pri_active_accts=1, credit_history_months=48, pri_sanctioned_amount=150000, pri_current_balance=20000),
    "docs": dict(disbursed_amount=48000, docs={"aadhar": 1, "pan": 1, "voterid": 1}),
    "overdue": dict(bureau_score=560, bureau_status=None, pri_no_of_accts=4, pri_active_accts=3, pri_overdue_accts=2,
                    delinq_6m=2, inquiries=5, credit_history_months=20, disbursed_amount=64000),
}
CO1, CO2, COL = "R. Patil (CO-114)", "S. Kulkarni (CO-120)", "A. Shaikh (Collections)"
HAD, WAK, AND = "Pune – Hadapsar", "Pune – Wakad", "Mumbai – Andheri"

# (application no., name, branch, profile, hours ago, story)
CASES = [
    ("APP-DEMO-1001", "Sanjay Jadhav", HAD, "refer", 0.4, "waiting"),
    ("APP-DEMO-1002", "Pooja Shinde", WAK, "thin", 1.2, ("assigned", CO1)),
    ("APP-DEMO-1003", "Imran Shaikh", AND, "high_ltv", 2.5, "waiting"),
    ("APP-DEMO-1004", "Kavita More", HAD, "refer_b", 5, ("assigned", CO2)),
    ("APP-DEMO-1005", "Arjun Nair", AND, "young", 27, "waiting"),
    ("APP-DEMO-1006", "Meena Gaikwad", WAK, "older", 0.2, "waiting"),
    ("APP-DEMO-0991", "Rahul Pawar", HAD, "bureau_690", 24 * 80, ("loan", ["steady", "slipping"], "promised_to_pay")),
    ("APP-DEMO-0992", "Sneha Kulkarni", WAK, "salaried_750", 24 * 70, ("loan", ["steady"], None)),
    ("APP-DEMO-0993", "Vikas Kamble", AND, "docs", 24 * 85, ("loan", ["slipping", "stressed"], "no_answer")),
    ("APP-DEMO-0994", "Nitin Bhosale", HAD, "overdue", 24 * 6, ("decline", "Two existing loans overdue and five recent enquiries; asked to reapply after clearing dues.")),
    ("APP-DEMO-0995", "Farhan Qureshi", AND, "high_ltv", 24 * 40, ("override", "Verified 8 months of delivery-platform payouts (about ₹24,000 a month) on his UPI statement.")),
    ("APP-DEMO-0996", "Anita Deshmukh", WAK, "refer", 24 * 2, ("approve", "Spoke to her employer; income steady. Approved after review.")),
]


def at(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat(timespec="seconds")


def scored(aid, name, branch, profile, hours):
    data = {**BASE, **PROFILES[profile], "application_id": aid, "applicant_name": name, "branch": branch, "synthetic": True}
    out = scoring.score(data)
    out["percentile"] = round(scoring.percentile(out["score"]), 3)
    out["application_id"] = aid
    return decision_log.log("score", data, out, synthetic=True, ts=at(hours)), out


def watch_check(loan_no, profile, hours_ago):
    end = (NOW - timedelta(hours=hours_ago)).date().isoformat()
    df = watching.make_synthetic_statement(profile, end=end)
    out = watching.watch(df, 3300, 5, loan_id=loan_no)
    out["model_version"] = "watch-rules-demo"
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    inputs = {"loan_id": loan_no, "emi": 3300, "due_day": 5, "other_emis": 0, "consent": True, "rows": df.to_dict("records")}
    return decision_log.log("watch", inputs, out, synthetic=True, ts=at(hours_ago)), out


def reset():
    db = decision_log.db_path()
    if db.exists():
        backup = db.with_name(f"{db.stem}.backup-{NOW:%Y%m%d-%H%M%S}{db.suffix}")
        shutil.copy2(db, backup)
        print(f"backed up {db} -> {backup}")
    print("removed", decision_log.delete_cases("APP-DEMO-"))


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--reset", action="store_true", help="remove the demo cases (after a backup) and create them again")
    if args.parse_args(argv).reset:
        reset()
    if any(c["application_id"].startswith("APP-DEMO-") for c in decision_log.cases(1000)):
        print("demo cases already present — nothing added (use --reset for a fresh demo)")
        return
    for aid, name, branch, profile, hours, story in CASES:
        did, out = scored(aid, name, branch, profile, hours)
        kind = story if isinstance(story, str) else story[0]
        if kind == "assigned":
            decision_log.add_action(did, "assign", "assigned", story[1], ts=at(hours * 0.6))
        elif kind == "decline":
            decision_log.add_action(did, "assign", "assigned", CO1, ts=at(hours - 0.3))
            decision_log.add_action(did, "final", "decline", CO1, story[1], ts=at(hours - 1.1))
        elif kind in ("override", "approve", "loan"):
            officer = CO2 if branch == WAK else CO1
            note = story[1] if kind != "loan" else "Within policy; documents verified."
            decision_log.add_action(did, "assign", "assigned", officer, ts=at(hours - 0.2))
            decision_log.add_action(did, "final", "approve", officer, note, ts=at(hours - 0.9))
            if kind in ("loan", "override"):
                loan_no = "TW-" + ("MUM" if branch == AND else "PUN") + "-" + aid[-4:]
                decision_log.add_action(did, "disbursed", "disbursed", officer, loan_no, ts=at(hours - 30))
                checks = story[1] if kind == "loan" else ["steady"]
                for i, prof in enumerate(checks):  # monthly checks, the latest about 10 days ago
                    wid, w = watch_check(loan_no, prof, 24 * (10 + 30 * (len(checks) - 1 - i)))
                    if i == len(checks) - 1 and kind == "loan" and story[2]:
                        decision_log.add_action(wid, "followup", story[2], COL,
                                                "Will pay on the 10th after the weekly payout." if story[2] == "promised_to_pay"
                                                else "Phone switched off; try again tomorrow morning.", ts=at(24 * 9))
        print(f"{aid}  {name:16s} {branch:18s} score {out['score']}  {out['band']:8s} {kind}")


if __name__ == "__main__":
    main()
