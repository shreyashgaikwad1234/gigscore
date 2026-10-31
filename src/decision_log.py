"""Audit log: every score and watch decision with its inputs, model version and output (guide 9.4).

SQLite file at $DECISION_DB (default logs/decisions.sqlite). A decision can be retrieved by id to
answer a customer complaint.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,            -- 'score' or 'watch'
    model_version TEXT,
    outcome TEXT,                  -- band (score) or flag (watch)
    score INTEGER,
    pd REAL,
    inputs TEXT NOT NULL,          -- JSON, as received
    output TEXT NOT NULL,          -- JSON, as returned
    synthetic INTEGER DEFAULT 0    -- 1 when the inputs were SYNTHETIC demo data
);
CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    decision_id TEXT NOT NULL,     -- the score or watch decision acted on
    kind TEXT NOT NULL,            -- 'final' (officer's final decision) or 'followup' (collections call)
    status TEXT NOT NULL,          -- final: approve / decline; followup: called, promised, ...
    officer TEXT,
    note TEXT
)
"""


def db_path() -> Path:
    return Path(os.environ.get("DECISION_DB", "logs/decisions.sqlite"))


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(kind: str, inputs: dict, output: dict, synthetic: bool = False, ts: str | None = None) -> str:
    """ts is only passed by the demo seed (src/demo_seed.py) to back-date SYNTHETIC cases."""
    did = uuid.uuid4().hex
    with connect() as con:
        con.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (did, ts or _now(), kind, output.get("model_version"),
             output.get("band") or output.get("flag"), output.get("score"), output.get("pd"),
             json.dumps(inputs, default=str), json.dumps(output, default=str), int(synthetic)))
    return did


def add_action(decision_id: str, kind: str, status: str, officer: str | None = None, note: str | None = None,
               ts: str | None = None) -> dict:
    """Record what a person did with a decision: final decision, assignment, disbursal or a collections call."""
    a = {"id": uuid.uuid4().hex, "ts": ts or _now(),
         "decision_id": decision_id, "kind": kind, "status": status, "officer": officer, "note": note}
    with connect() as con:
        con.execute("INSERT INTO actions VALUES (:id,:ts,:decision_id,:kind,:status,:officer,:note)", a)
    return a


def actions(decision_id: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM actions WHERE decision_id = ? ORDER BY ts DESC, rowid DESC",
                           (decision_id,)).fetchall()
    return [dict(r) for r in rows]


def _latest(con: sqlite3.Connection, ids: list[str], kind: str) -> dict:
    if not ids:
        return {}
    rows = con.execute(f"SELECT * FROM actions WHERE kind = ? AND decision_id IN ({','.join('?' * len(ids))}) "
                       "ORDER BY ts, rowid", (kind, *ids)).fetchall()
    return {r["decision_id"]: dict(r) for r in rows}  # later rows overwrite earlier ones


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["inputs"], d["output"] = json.loads(d["inputs"]), json.loads(d["output"])
    d["synthetic"] = bool(d["synthetic"])
    return d


def get(did: str) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM decisions WHERE id = ?", (did,)).fetchone()
    if not r:
        return None
    return _row(r) | {"actions": actions(did)}


def recent(limit: int = 50, kind: str | None = None) -> list[dict]:
    """Newest decisions, each with the latest officer final decision / follow-up attached (or None)."""
    q, args = "SELECT * FROM decisions", []
    if kind:
        q, args = q + " WHERE kind = ?", [kind]
    with connect() as con:
        rows = [_row(r) for r in con.execute(q + " ORDER BY ts DESC, rowid DESC LIMIT ?", (*args, limit)).fetchall()]
        ids = [d["id"] for d in rows]
        final, follow = _latest(con, ids, "final"), _latest(con, ids, "followup")
    for d in rows:
        d["final"], d["followup"] = final.get(d["id"]), follow.get(d["id"])
    return rows


def queue(limit: int = 100) -> list[dict]:
    """Collections call list: watched loans flagged red or amber, latest check per loan, red first."""
    with connect() as con:
        rows = [_row(r) for r in con.execute(
            "SELECT * FROM decisions WHERE kind = 'watch' AND outcome IN ('red', 'amber') "
            "ORDER BY ts DESC, rowid DESC LIMIT ?",
            (limit * 3,)).fetchall()]
        follow = _latest(con, [d["id"] for d in rows], "followup")
        loan_app = {}  # loan account number -> application number, from disbursals
        for a in con.execute("SELECT a.note, d.inputs FROM actions a JOIN decisions d ON d.id = a.decision_id "
                             "WHERE a.kind = 'disbursed'").fetchall():
            loan_app[a["note"]] = json.loads(a["inputs"]).get("application_id")
        apps = {json.loads(r["inputs"]).get("application_id") for r in
                con.execute("SELECT inputs FROM decisions WHERE kind = 'score'").fetchall()} - {None}
    seen, out = set(), []
    for d in rows:
        loan = d["inputs"].get("loan_id") or d["id"]
        if loan in seen:
            continue
        seen.add(loan)
        o = d["output"]
        out.append({"id": d["id"], "ts": d["ts"], "loan_id": loan, "flag": d["outcome"], "synthetic": d["synthetic"],
                    "suggested_action": o.get("suggested_action"), "triggers": o.get("triggers", []),
                    "days_to_next_emi": o.get("days_to_next_emi"), "emi": d["inputs"].get("emi"),
                    "followup": follow.get(d["id"]),
                    "application_id": loan_app.get(loan) or (loan if loan in apps else None)})
    out.sort(key=lambda r: (r["flag"] != "red", r["days_to_next_emi"] if r["days_to_next_emi"] is not None else 99))
    return out[:limit]


# --- workspace: applications grouped into cases by application_id ---------------------------------

def _applications(con: sqlite3.Connection) -> dict[str, list[dict]]:
    """Scored applications that carry an application_id, grouped by it (re-scores share one id), newest last."""
    out: dict[str, list[dict]] = {}
    for r in con.execute("SELECT * FROM decisions WHERE kind = 'score' ORDER BY ts, rowid").fetchall():
        d = _row(r)
        aid = d["inputs"].get("application_id")
        if aid:
            out.setdefault(aid, []).append(d)
    return out


def _acts(con: sqlite3.Connection, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    return [dict(r) for r in con.execute(f"SELECT * FROM actions WHERE decision_id IN ({','.join('?' * len(ids))}) "
                                         "ORDER BY ts, rowid", ids).fetchall()]


def _last(acts: list[dict], kind: str) -> dict | None:
    return next((a for a in reversed(acts) if a["kind"] == kind), None)


def _brief(aid: str, decs: list[dict], acts: list[dict]) -> dict:
    d = decs[-1]
    return {"application_id": aid, "decision_id": d["id"], "ts": decs[0]["ts"], "last_scored": d["ts"],
            "applicant_name": d["inputs"].get("applicant_name"), "branch": d["inputs"].get("branch"),
            "loan_amount": d["inputs"].get("disbursed_amount"), "score": d["score"], "band": d["outcome"], "pd": d["pd"],
            "synthetic": d["synthetic"], "assigned": _last(acts, "assign"), "final": _last(acts, "final"),
            "disbursed": _last(acts, "disbursed")}


def inbox(branch: str | None = None) -> list[dict]:
    """Applications still waiting for a credit officer's final decision: referrals first, then oldest first."""
    with connect() as con:
        apps = _applications(con)
        rows = []
        for aid, decs in apps.items():
            acts = _acts(con, [d["id"] for d in decs])
            b = _brief(aid, decs, acts)
            if not b["final"] and (not branch or b["branch"] == branch):
                rows.append(b)
    return sorted(rows, key=lambda r: (r["band"] != "refer", r["ts"]))


def cases(limit: int = 200, branch: str | None = None) -> list[dict]:
    with connect() as con:
        apps = _applications(con)
        rows = [_brief(aid, decs, _acts(con, [d["id"] for d in decs])) for aid, decs in apps.items()]
    rows = [r for r in rows if not branch or r["branch"] == branch]
    return sorted(rows, key=lambda r: r["ts"], reverse=True)[:limit]


def case(aid: str) -> dict | None:
    """Everything about one application: the scores, what officers did, and the loan's early-warning checks."""
    with connect() as con:
        decs = _applications(con).get(aid)
        if not decs:
            return None
        acts = _acts(con, [d["id"] for d in decs])
        loan_nos = {aid} | {a["note"] for a in acts if a["kind"] == "disbursed" and a["note"]}
        watches = [_row(r) for r in con.execute("SELECT * FROM decisions WHERE kind = 'watch' ORDER BY ts, rowid")
                   if json.loads(r["inputs"]).get("loan_id") in loan_nos]
        wacts = _acts(con, [w["id"] for w in watches])
    events = [{"ts": d["ts"], "type": "scored", "decision_id": d["id"], "band": d["outcome"], "score": d["score"]} for d in decs]
    events += [{"ts": a["ts"], "type": a["kind"], "status": a["status"], "officer": a["officer"], "note": a["note"]} for a in acts]
    events += [{"ts": w["ts"], "type": "watch", "decision_id": w["id"], "flag": w["outcome"],
                "action": w["output"].get("suggested_action")} for w in watches]
    events += [{"ts": a["ts"], "type": "followup", "status": a["status"], "officer": a["officer"], "note": a["note"]} for a in wacts]
    return {**_brief(aid, decs, acts), "application": decs[-1], "rescored": len(decs) - 1,
            "loan_numbers": sorted(loan_nos - {aid}), "watch": watches, "timeline": sorted(events, key=lambda e: e["ts"])}


def search(q: str, limit: int = 12) -> list[dict]:
    """Find cases by application number, applicant name or loan number (case-insensitive substring)."""
    q = q.strip().lower()
    if len(q) < 2:
        return []
    hits = []
    for c in cases(1000):
        hay = " ".join(str(x or "") for x in (c["application_id"], c["applicant_name"], c["branch"],
                                               c["disbursed"]["note"] if c["disbursed"] else "")).lower()
        if q in hay:
            hits.append(c)
    return hits[:limit]


def delete_cases(prefix: str) -> dict:
    """Remove every case whose application number starts with prefix — its scores, officer actions, the
    early-warning checks of its loans and their calls. Used only to reset the SYNTHETIC demo cases."""
    if len(prefix) < 5:
        raise ValueError("prefix too short; refusing to delete broadly")
    with connect() as con:
        apps = {aid: decs for aid, decs in _applications(con).items() if aid.startswith(prefix)}
        score_ids = [d["id"] for decs in apps.values() for d in decs]
        loans = set(apps) | {a["note"] for a in _acts(con, score_ids) if a["kind"] == "disbursed" and a["note"]}
        watch_ids = [r["id"] for r in con.execute("SELECT id, inputs FROM decisions WHERE kind = 'watch'")
                     if json.loads(r["inputs"]).get("loan_id") in loans]
        ids = score_ids + watch_ids
        q = ",".join("?" * len(ids))
        n_act = con.execute(f"DELETE FROM actions WHERE decision_id IN ({q})", ids).rowcount if ids else 0
        if ids:
            con.execute(f"DELETE FROM decisions WHERE id IN ({q})", ids)
    return {"cases": len(apps), "decisions": len(ids), "actions": n_act}


def summary() -> dict:
    """Portfolio view of everything logged: bands for scored applications, flags for watched loans."""
    with connect() as con:
        s = con.execute("SELECT outcome, COUNT(*) n, AVG(pd) mean_pd FROM decisions WHERE kind='score' "
                        "GROUP BY outcome").fetchall()
        w = con.execute("SELECT outcome, COUNT(*) n FROM decisions WHERE kind='watch' GROUP BY outcome").fetchall()
        syn = con.execute("SELECT COUNT(*) FROM decisions WHERE synthetic = 1").fetchone()[0]
        scored = [r["id"] for r in con.execute("SELECT id FROM decisions WHERE kind='score'").fetchall()]
        final = _latest(con, scored, "final")
        outcome = dict(con.execute("SELECT id, outcome FROM decisions WHERE kind='score'").fetchall())
    bands = {r["outcome"]: {"n": r["n"], "mean_pd": r["mean_pd"]} for r in s}
    n = sum(b["n"] for b in bands.values())
    appr = bands.get("approve", {"n": 0, "mean_pd": None})
    overrides = sum(1 for did, a in final.items() if outcome[did] != "refer" and a["status"] != outcome[did])
    return {"scored": n, "bands": bands, "approval_rate": appr["n"] / n if n else None,
            "expected_bad_rate_approved": appr["mean_pd"],
            "watched": {r["outcome"]: r["n"] for r in w}, "synthetic_decisions": syn,
            "officer_decided": len(final), "overrides": overrides}
