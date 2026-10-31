"""Scoring and API tests. Need the trained models in models/ (skipped on a clean checkout without them).
Applications here are made-up test inputs, flagged synthetic when logged."""
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not Path("models/lgbm_fair.pkl").exists() or
                                not Path("models/app_reference.json").exists(),
                                reason="trained models / app reference not present")

BASE = {"disbursed_amount": 55000, "asset_cost": 72000, "age": 27, "employment_type": "Self employed",
        "docs": {"aadhar": 1}, "bureau_score": None, "bureau_status": "no_history"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Decision log, thresholds and watch rules go to a temp dir so tests never touch the real ones."""
    from src import score as scoring, watch as watching
    monkeypatch.setenv("DECISION_DB", str(tmp_path / "decisions.sqlite"))
    cfg, rules = tmp_path / "thresholds.yaml", tmp_path / "watch_rules.yaml"
    shutil.copy("config/thresholds.yaml", cfg)
    shutil.copy("config/watch_rules.yaml", rules)
    monkeypatch.setattr(scoring, "CONFIG", cfg)
    monkeypatch.setattr(watching, "RULES_FILE", rules)
    for f in (scoring.load_config, scoring._bundle, scoring._encoders, scoring.app_reference, watching.load_rules):
        f.cache_clear()
    yield
    for f in (scoring.load_config, scoring._bundle, scoring._encoders, scoring.app_reference, watching.load_rules):
        f.cache_clear()


def test_score_output_shape_and_band():
    from src.score import band_for, load_config, score
    r = score(BASE)
    cfg = load_config()
    assert 300 <= r["score"] <= 900 and 0 < r["pd"] < 1
    assert r["band"] == band_for(r["score"], cfg["bands"][cfg["model"]])
    assert len(r["reasons"]) == 3 and len(set(r["reasons"])) == 3
    assert r["model_used"] == cfg["model"]


def test_higher_loan_to_value_never_lowers_risk():
    from src.score import score
    low = score({**BASE, "disbursed_amount": 45000})["pd"]
    high = score({**BASE, "disbursed_amount": 68000})["pd"]
    assert high >= low


def test_state_does_not_change_the_score():
    from src.score import score
    assert score({**BASE, "state_id": 3})["score"] == score({**BASE, "state_id": 13})["score"]


def test_kill_switch_uses_fallback():
    from src.score import load_config, score
    cfg = {**load_config(), "force_fallback": True}
    r = score(BASE, cfg)
    assert r["model_used"] == cfg["fallback_model"] and "fallback" in r["note"] and len(r["reasons"]) >= 1


def test_validation_errors():
    from src.score import grade_for, validate
    assert validate({**BASE, "age": 12})
    assert validate({**BASE, "disbursed_amount": 200000})
    assert validate({**BASE, "bureau_score": 120})
    assert not validate(BASE)
    assert grade_for(812) == "A-Very Low Risk" and grade_for(300) == "M-Very High Risk"


def test_api_end_to_end(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"

    r = c.post("/score", json={**BASE, "synthetic": True})
    assert r.status_code == 200
    out = r.json()
    d = c.get(f"/decisions/{out['explain_id']}").json()
    assert d["output"]["score"] == out["score"] and d["synthetic"]
    assert c.post("/score", json={**BASE, "age": 10}).status_code == 422

    s = c.get("/samples/stressed").json()
    assert s["synthetic"] and s["rows"]
    body = {"emi": s["emi"], "due_day": s["due_day"], "rows": s["rows"]}
    assert c.post("/watch", json={**body, "consent": False}).status_code == 400
    w = c.post("/watch", json={**body, "consent": True}).json()
    assert w["flag"] == "red" and w["synthetic"]
    assert c.get("/portfolio/summary").json()["live"]["scored"] == 1
    assert c.get("/samples/stressed/check").json()["flag"] == "red"
    assert c.get("/portfolio/summary").json()["live"]["scored"] == 1  # the demo check is not logged
    assert "markdown" in c.get("/about/model_card").json()


def test_preview_is_not_logged_and_matches_score():
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    p = c.post("/score/preview", json=BASE).json()
    assert "explain_id" not in p and c.get("/portfolio/summary").json()["live"]["scored"] == 0
    assert c.post("/score", json=BASE).json()["score"] == p["score"]


def test_officer_final_decision_and_collections_queue():
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    out = c.post("/score", json={**BASE, "synthetic": True}).json()
    did = out["explain_id"]
    flip = "decline" if out["band"] == "approve" else "approve"
    assert c.post(f"/decisions/{did}/final", json={"decision": flip, "reason": "short", "officer": "A1"}).status_code == 422
    f = c.post(f"/decisions/{did}/final", json={"decision": flip, "reason": "Verified salary slips in person", "officer": "A1"}).json()
    assert f["override"] == (out["band"] != "refer")
    assert c.get(f"/decisions/{did}").json()["actions"][0]["status"] == flip
    assert c.get("/decisions?kind=score").json()[0]["final"]["status"] == flip
    assert c.get("/portfolio/summary").json()["live"]["officer_decided"] == 1

    s = c.get("/samples/stressed").json()
    body = {"emi": s["emi"], "due_day": s["due_day"], "rows": s["rows"], "consent": True}
    c.post("/watch", json={**body, "loan_id": "LN-1"})
    w = c.post("/watch", json={**body, "loan_id": "LN-1"}).json()  # re-checked: only the latest is queued
    c.post("/watch", json={**body, "loan_id": "LN-2", "rows": c.get("/samples/steady").json()["rows"]})  # green
    q = c.get("/collections/queue").json()
    assert [r["loan_id"] for r in q] == ["LN-1"] and q[0]["id"] == w["explain_id"] and q[0]["flag"] == "red"
    assert c.post(f"/decisions/{did}/followup", json={"status": "called", "officer": "C1"}).status_code == 404
    c.post(f"/decisions/{w['explain_id']}/followup", json={"status": "promised_to_pay", "note": "pays on 10th", "officer": "C1"})
    assert c.get("/collections/queue").json()[0]["followup"]["status"] == "promised_to_pay"


def test_workspace_inbox_case_and_search():
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    out = c.post("/score", json={**BASE, "applicant_name": "Test Applicant", "branch": "Pune – Wakad", "synthetic": True}).json()
    aid, did = out["application_id"], out["explain_id"]
    assert aid.startswith("APP-")
    inbox = c.get("/inbox").json()
    assert [r["application_id"] for r in inbox] == [aid] and not c.get("/inbox?branch=Elsewhere").json()
    assert c.get("/search?q=test appl").json()[0]["application_id"] == aid
    c.post(f"/decisions/{did}/assign", json={"officer": "CO-1"})
    assert c.post(f"/decisions/{did}/disburse", json={"officer": "CO-1"}).status_code == 409  # not approved yet
    c.post(f"/decisions/{did}/final", json={"decision": "approve", "reason": "Income verified on statement", "officer": "CO-1"})
    assert c.get("/inbox").json() == []
    loan = c.post(f"/decisions/{did}/disburse", json={"officer": "CO-1"}).json()["note"]
    s = c.get("/samples/stressed").json()
    c.post("/watch", json={"emi": 3300, "due_day": 5, "rows": s["rows"], "consent": True, "loan_id": loan})
    case = c.get(f"/cases/{aid}").json()
    assert case["loan_numbers"] == [loan] and case["watch"][0]["outcome"] == "red"
    assert [e["type"] for e in case["timeline"]] == ["scored", "assign", "final", "disbursed", "watch"]
    assert c.get("/collections/queue").json()[0]["application_id"] == aid
    assert c.get("/cases/APP-NOPE").status_code == 404


def test_demo_reset_touches_only_demo_cases(tmp_path):
    from src import decision_log, demo_seed
    from src.score import score
    demo_seed.main([])
    keep = decision_log.log("score", {**BASE, "application_id": "APP-REAL-1"}, score(BASE))
    decision_log.add_action(keep, "final", "approve", "CO-1", "kept")
    demo = [c for c in decision_log.cases() if c["application_id"].startswith("APP-DEMO-")]
    decision_log.add_action(demo[0]["decision_id"], "final", "decline", "CO-1", "changed during a rehearsal")
    demo_seed.main(["--reset"])
    after = decision_log.cases()
    assert len([c for c in after if c["application_id"].startswith("APP-DEMO-")]) == len(demo) == 12
    status = lambda f: f and f["status"]
    assert status(next(c for c in after if c["application_id"] == demo[0]["application_id"])["final"]) == status(demo[0]["final"])
    assert decision_log.case("APP-REAL-1")["final"]["note"] == "kept"
    assert list(tmp_path.glob("decisions.backup-*.sqlite"))


def test_admin_requires_token_and_updates_live(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app
    c = TestClient(app)
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert c.put("/admin/thresholds", json={"approve": 650, "reason": "test"}).status_code == 503
    monkeypatch.setenv("ADMIN_TOKEN", "test-token-for-pytest")
    bad = {"X-Admin-Token": "wrong"}
    good = {"X-Admin-Token": "test-token-for-pytest"}
    assert c.put("/admin/thresholds", json={"approve": 650, "reason": "test"}, headers=bad).status_code == 401
    assert c.put("/admin/thresholds", json={"approve": 500, "refer": 600, "reason": "bad"}, headers=good).status_code == 422

    before = c.post("/score", json=BASE).json()
    r = c.put("/admin/thresholds", json={"approve": 300, "refer": 300, "reason": "pytest: approve all"}, headers=good)
    assert r.status_code == 200
    assert c.post("/score", json=BASE).json()["band"] == "approve"

    r = c.put("/admin/thresholds", json={"force_fallback": True, "reason": "pytest kill switch"}, headers=good)
    assert c.post("/score", json=BASE).json()["model_used"].startswith("scorecard")
    assert c.get("/health").json()["force_fallback"]

    assert c.put("/admin/watch-rules", json={"rules": {"nope": {}}, "reason": "bad"}, headers=good).status_code == 422
    r = c.put("/admin/watch-rules", json={"rules": {"inflow_trend": {"amber_below": 0.9}}, "reason": "pytest"}, headers=good)
    assert r.json()["updated"]["inflow_trend"]["amber_below"] == 0.9
    assert len(c.get("/admin/changes", headers=good).json()) == 3
    assert before["band"] in ("approve", "refer", "decline")
