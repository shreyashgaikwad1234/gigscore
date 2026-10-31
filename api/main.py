"""GigScore API (FastAPI): live scoring, early warning, audit log, portfolio and runtime admin.

Run locally:   uvicorn api.main:app --reload            (from the repository root)
Render:        see render.yaml

Public endpoints serve the web front end (web/). Admin endpoints change decision thresholds, the
recalibration offset, the model in use, the kill switch and the watch rules at runtime; they need the
X-Admin-Token header to match $ADMIN_TOKEN and every change is written to the audit log.
"""
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import decision_log, score as scoring, watch as watching

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # models/, config/ and reports/ are resolved relative to the repository root

app = FastAPI(title="GigScore API", version="2026.10",
              description="Alternative-data credit scoring and early warning. Trained on real L&T 2018 loans; "
                          "demo statements are SYNTHETIC and labelled.")
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
if Path("reports/figures").exists():
    app.mount("/figures", StaticFiles(directory="reports/figures"), name="figures")


# --- schemas ----------------------------------------------------------------------------------

class Docs(BaseModel):
    aadhar: int = 0
    pan: int = 0
    voterid: int = 0
    driving: int = 0
    passport: int = 0
    mobile: int = 1


class Application(BaseModel):
    application_id: str | None = Field(None, max_length=40, description="Lender's application number; generated if empty")
    applicant_name: str | None = Field(None, max_length=80, description="Shown in the workspace only; never a model input")
    branch: str | None = Field(None, max_length=60)
    disbursed_amount: float = Field(gt=0, examples=[65000])
    asset_cost: float = Field(gt=0, examples=[82000])
    age: float = Field(ge=18, le=75, examples=[27])
    employment_type: Literal["Salaried", "Self employed", "Unknown"] = "Self employed"
    state_id: int | None = Field(None, description="Stored for the fairness report only; never a model input")
    docs: Docs = Docs()
    bureau_score: float | None = Field(None, description="300-900; null when there is no score")
    bureau_status: Literal["no_history", "thin_history", "not_enough_info", "inactive", "no_updates_36m",
                           "guarantor_only"] | None = None
    pri_no_of_accts: int = 0
    pri_active_accts: int = 0
    pri_overdue_accts: int = 0
    pri_current_balance: float = 0
    pri_sanctioned_amount: float = 0
    primary_instal_amt: float = 0
    sec_no_of_accts: int = 0
    sec_active_accts: int = 0
    sec_overdue_accts: int = 0
    sec_current_balance: float = 0
    sec_sanctioned_amount: float = 0
    sec_instal_amt: float = 0
    new_accts_6m: int = 0
    delinq_6m: int = 0
    inquiries: int = 0
    credit_history_months: int = 0
    avg_account_age_months: int | None = None
    synthetic: bool = Field(False, description="True for demo / test applications")


class Row(BaseModel):
    date: str
    credit: float = 0
    debit: float = 0
    balance: float
    category: str | None = None
    status: str | None = None
    description: str | None = None


class WatchRequest(BaseModel):
    loan_id: str | None = None
    emi: float = Field(gt=0, examples=[3300])
    due_day: int = Field(ge=1, le=31, examples=[5])
    other_emis: float = 0
    consent: bool = Field(description="Borrower consented to sharing this statement (RBI Digital Lending Directions)")
    rows: list[Row]


class FinalDecision(BaseModel):
    decision: Literal["approve", "decline"]
    reason: str = Field(min_length=10, description="Why — required, especially when it differs from GigScore")
    officer: str = Field(min_length=2, description="Name or employee id of the credit officer")


class FollowUp(BaseModel):
    status: Literal["called", "promised_to_pay", "no_answer", "emi_date_moved", "restructuring", "resolved"]
    note: str | None = None
    officer: str = Field(min_length=2)


class Officer(BaseModel):
    officer: str = Field(min_length=2)
    note: str | None = Field(None, max_length=60, description="disburse: the loan account number")


class ThresholdUpdate(BaseModel):
    model: Literal["lgbm_fair", "lgbm_combined"] | None = None
    approve: int | None = Field(None, ge=300, le=900)
    refer: int | None = Field(None, ge=300, le=900)
    calibration_offset: float | None = Field(None, ge=-3, le=3)
    force_fallback: bool | None = None
    reason: str = Field(min_length=3, description="Why the change is made (audit log)")


class RulesUpdate(BaseModel):
    rules: dict = Field(description="Partial watch_rules.yaml, e.g. {'inflow_trend': {'amber_below': 0.75}}")
    reason: str = Field(min_length=3)


# --- helpers ----------------------------------------------------------------------------------

def require_admin(x_admin_token: str | None = Header(None)):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(503, "admin is disabled: set the ADMIN_TOKEN environment variable on the server")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(401, "invalid admin token")


def _merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _reload():
    scoring.load_config.cache_clear()
    scoring._bundle.cache_clear()
    scoring._encoders.cache_clear()
    scoring.app_reference.cache_clear()
    watching.load_rules.cache_clear()


# --- public -----------------------------------------------------------------------------------

@app.get("/health")
def health():
    cfg = scoring.load_config()
    return {"status": "ok", "model": cfg["model"], "fallback_model": cfg["fallback_model"],
            "force_fallback": bool(cfg.get("force_fallback")), "calibration_offset": cfg.get("calibration_offset", 0),
            "admin_enabled": bool(os.environ.get("ADMIN_TOKEN"))}


@app.post("/score")
def score(app_in: Application):
    data = app_in.model_dump()
    data["application_id"] = (data["application_id"] or "").strip() or \
        f"APP-{datetime.now(timezone.utc):%y%m%d}-{secrets.token_hex(2).upper()}"
    try:
        out = scoring.score(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["percentile"] = round(scoring.percentile(out["score"]), 3)
    out["application_id"] = data["application_id"]
    out["explain_id"] = decision_log.log("score", data, out, synthetic=app_in.synthetic)
    return out


@app.post("/score/preview")
def score_preview(app_in: Application):
    """Same as /score but nothing is logged: for 'what would change this decision?' on the application page."""
    try:
        out = scoring.score(app_in.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["percentile"] = round(scoring.percentile(out["score"]), 3)
    return out


@app.post("/watch")
def watch(req: WatchRequest):
    if not req.consent:
        raise HTTPException(400, "borrower consent is required before statement data is used")
    rows = [r.model_dump() for r in req.rows]
    try:
        out = watching.watch(rows, req.emi, req.due_day, req.other_emis, req.loan_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out["model_version"] = "watch-rules-" + str(watching.RULES_FILE.stat().st_mtime_ns)[-8:]
    out["explain_id"] = decision_log.log("watch", req.model_dump(), out, synthetic=out["synthetic"])
    return out


@app.get("/samples/{profile}")
def sample(profile: Literal["steady", "slipping", "stressed"]):
    """A SYNTHETIC 12-week statement for the demo; every row is labelled SYNTHETIC."""
    df = watching.make_synthetic_statement(profile)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return {"synthetic": True, "emi": 3300, "due_day": 5, "rows": df.to_dict(orient="records")}


@app.get("/samples/{profile}/check")
def sample_check(profile: Literal["steady", "slipping", "stressed"]):
    """Watch result for a SYNTHETIC sample, for demos. Not a decision, so it is not written to the audit log."""
    df = watching.make_synthetic_statement(profile)
    return watching.watch(df, 3300, 5, loan_id=f"SYNTHETIC-{profile}")


@app.get("/decisions")
def decisions(limit: int = 50, kind: Literal["score", "watch"] | None = None):
    return decision_log.recent(min(limit, 500), kind)


@app.get("/decisions/{did}")
def decision(did: str):
    d = decision_log.get(did)
    if not d:
        raise HTTPException(404, "no decision with that id")
    return d


@app.post("/decisions/{did}/final")
def final_decision(did: str, f: FinalDecision):
    """The credit officer's final decision on a scored application. GigScore advises; the officer decides."""
    d = decision_log.get(did)
    if not d or d["kind"] != "score":
        raise HTTPException(404, "no scored application with that id")
    a = decision_log.add_action(did, "final", f.decision, f.officer, f.reason)
    return a | {"gigscore": d["outcome"], "override": d["outcome"] != "refer" and f.decision != d["outcome"]}


@app.post("/decisions/{did}/followup")
def followup(did: str, f: FollowUp):
    """A collections follow-up on a watched loan (call made, promise to pay, EMI date moved ...)."""
    d = decision_log.get(did)
    if not d or d["kind"] != "watch":
        raise HTTPException(404, "no early-warning check with that id")
    return decision_log.add_action(did, "followup", f.status, f.officer, f.note)


def _scored(did: str) -> dict:
    d = decision_log.get(did)
    if not d or d["kind"] != "score":
        raise HTTPException(404, "no scored application with that id")
    return d


@app.post("/decisions/{did}/assign")
def assign(did: str, o: Officer):
    """Take an application from the inbox ("assign to me")."""
    _scored(did)
    return decision_log.add_action(did, "assign", "assigned", o.officer)


@app.post("/decisions/{did}/disburse")
def disburse(did: str, o: Officer):
    """Mark an approved application as disbursed; note = the loan account number used for early warning."""
    d = _scored(did)
    final = next((a for a in d["actions"] if a["kind"] == "final"), None)
    if not final or final["status"] != "approve":
        raise HTTPException(409, "only an application the officer approved can be disbursed")
    loan_no = (o.note or "").strip() or "TW-" + (d["inputs"].get("application_id") or did[:8]).removeprefix("APP-")
    return decision_log.add_action(did, "disbursed", "disbursed", o.officer, loan_no)


@app.get("/inbox")
def inbox(branch: str | None = None):
    """Applications waiting for a credit officer's final decision (workspace inbox)."""
    return decision_log.inbox(branch)


@app.get("/cases")
def cases(limit: int = 200, branch: str | None = None):
    return decision_log.cases(min(limit, 1000), branch)


@app.get("/cases/{aid}")
def case(aid: str):
    c = decision_log.case(aid)
    if not c:
        raise HTTPException(404, "no application with that number")
    return c


@app.get("/search")
def search(q: str):
    return decision_log.search(q)


@app.get("/collections/queue")
def collections_queue(limit: int = 100):
    """Call list for the collections team: loans whose latest check is red or amber, red first."""
    return decision_log.queue(min(limit, 500))


@app.get("/portfolio/summary")
def portfolio():
    ref = scoring.app_reference()
    return {"live": decision_log.summary(),
            "historical_test_book": {"data": ref["data"], "n": ref["n_test"], "default_rate": ref["test_default_rate"],
                                     "band_table": ref["band_table"]},
            "par": "PAR 30 / PAR 90 need repayment history beyond the first EMI, which L&T does not contain; "
                   "shown once a lender's book is connected."}


@app.get("/model")
def model_info():
    cfg = scoring.load_config()
    return {**scoring.app_reference(), "config": {k: cfg[k] for k in ("model", "fallback_model", "bands",
                                                                       "calibration_offset", "kill_switch")
                                                   if k in cfg} | {"force_fallback": bool(cfg.get("force_fallback"))},
            "watch_rules": watching.load_rules()}


@app.get("/model/analytics")
def analytics():
    """Chart-ready aggregates from the October 2018 out-of-time test (src/analytics.py)."""
    p = Path("models/analytics.json")
    if not p.exists():
        raise HTTPException(404, "analytics not generated: run python -m src.analytics")
    import json
    return json.loads(p.read_text(encoding="utf-8"))


DOCS = {"model_card": "reports/model_card.md", "validation_report": "reports/validation_report.md",
        "features": "reports/features.md", "profile": "reports/profile.md"}


@app.get("/about/{doc}")
def about(doc: Literal["model_card", "validation_report", "features", "profile"]):
    p = Path(DOCS[doc])
    if not p.exists():
        raise HTTPException(404, f"{p} has not been generated")
    return {"doc": doc, "markdown": p.read_text(encoding="utf-8")}


# --- admin (runtime updates, audited) -----------------------------------------------------------

@app.get("/admin/changes", dependencies=[Depends(require_admin)])
def admin_changes(limit: int = 50):
    return decision_log.recent(min(limit, 500), "config")


@app.put("/admin/thresholds", dependencies=[Depends(require_admin)])
def update_thresholds(u: ThresholdUpdate):
    cfg = yaml.safe_load(scoring.CONFIG.read_text(encoding="utf-8"))
    before = {k: cfg.get(k) for k in ("model", "bands", "calibration_offset", "force_fallback")}
    if u.model:
        cfg["model"] = u.model
        cfg["fallback_model"] = "scorecard_" + u.model.split("_", 1)[1]
    m = cfg["model"]
    if u.approve is not None or u.refer is not None:
        b = cfg["bands"][m]
        b["approve"] = u.approve if u.approve is not None else b["approve"]
        b["refer"] = u.refer if u.refer is not None else b["refer"]
        if b["refer"] > b["approve"]:
            raise HTTPException(422, "refer cut-off must not be above the approve cut-off")
    if u.calibration_offset is not None:
        cfg["calibration_offset"] = u.calibration_offset
    if u.force_fallback is not None:
        cfg["force_fallback"] = u.force_fallback
    if cfg["model"] not in cfg["bands"] or cfg["fallback_model"] not in cfg["bands"]:
        raise HTTPException(422, f"no cut-offs configured for {cfg['model']} / {cfg['fallback_model']}")
    scoring.CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _reload()
    after = {k: cfg.get(k) for k in ("model", "bands", "calibration_offset", "force_fallback")}
    decision_log.log("config", {"change": "thresholds", "reason": u.reason, "before": before},
                     {"model_version": "config", "flag": "thresholds_updated", "after": after})
    return {"updated": after}


@app.put("/admin/watch-rules", dependencies=[Depends(require_admin)])
def update_rules(u: RulesUpdate):
    before = watching.load_rules()
    unknown = set(u.rules) - set(before)
    if unknown:
        raise HTTPException(422, f"unknown rule sections: {sorted(unknown)}")
    after = _merge(before, u.rules)
    watching.RULES_FILE.write_text(yaml.safe_dump(after, sort_keys=False), encoding="utf-8")
    _reload()
    decision_log.log("config", {"change": "watch_rules", "reason": u.reason, "before": before},
                     {"model_version": "config", "flag": "watch_rules_updated", "after": after})
    return {"updated": after}


@app.post("/admin/reload", dependencies=[Depends(require_admin)])
def reload_models():
    """Pick up retrained model files, encoders and reference data without restarting the server."""
    _reload()
    cfg = scoring.load_config()
    versions = {m: scoring._bundle(m)["version"] for m in (cfg["model"], cfg["fallback_model"])}
    decision_log.log("config", {"change": "reload"}, {"model_version": "config", "flag": "models_reloaded",
                                                      "versions": versions})
    return {"reloaded": versions}


# The built Astro site (frontend/dist), served here too so one local server runs everything:
# http://localhost:8000/. Mounted last so the API routes above take precedence.
if Path("frontend/dist").exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="site")
