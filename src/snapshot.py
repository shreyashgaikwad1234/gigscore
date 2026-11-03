"""Write the API's read-only responses into the website as static files:  python -m src.snapshot

The public pages read these first, so charts, governance documents and the hero demo appear at once even
while the free API is asleep (about a minute to wake). Re-run after retraining, after src.analytics, or
after editing a report; tests/test_snapshot.py fails if the files fall out of date.
"""
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

OUT = Path("frontend/public/snapshot")
FIGURES = Path("frontend/public/figures")
DOCS = ["model_card", "validation_report", "features", "profile"]
# The hero application on the Overview page (frontend/src/pages/index.astro, SAMPLE); keep the two identical.
HERO = dict(disbursed_amount=48000, asset_cost=72000, age=29, employment_type="Self employed",
            docs={"aadhar": 1, "pan": 1, "voterid": 1}, bureau_score=None, bureau_status="no_history", synthetic=True)


def responses(c: TestClient) -> dict[str, dict]:
    """Static responses: identical on every call until the model or reports change."""
    out = {"model": c.get("/model").json(), "analytics": c.get("/model/analytics").json()}
    out |= {f"about_{d}": c.get(f"/about/{d}").json() for d in DOCS}
    out["sample_slipping_check"] = c.get("/samples/slipping/check").json()
    return out


def main() -> None:
    from api.main import app
    c = TestClient(app)
    OUT.mkdir(parents=True, exist_ok=True)
    files = responses(c)
    # The score depends on the disbursal day, so the hero result is a recorded example, not a checked constant.
    files["hero_preview"] = c.post("/score/preview", json=HERO).json()
    for name, body in files.items():
        (OUT / f"{name}.json").write_text(json.dumps(body, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shutil.copytree("reports/figures", FIGURES, dirs_exist_ok=True)
    print(f"wrote {len(files)} files to {OUT} and copied figures to {FIGURES}")


if __name__ == "__main__":
    main()
