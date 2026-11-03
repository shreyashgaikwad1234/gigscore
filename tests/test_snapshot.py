import json
from pathlib import Path

from fastapi.testclient import TestClient

from src import snapshot


def test_website_snapshot_matches_api():
    """The static copies the website shows while the API wakes must equal what the API returns."""
    from api.main import app
    live = snapshot.responses(TestClient(app))
    for name, body in live.items():
        f = snapshot.OUT / f"{name}.json"
        assert f.exists(), f"{f} missing: run python -m src.snapshot"
        assert json.loads(f.read_text(encoding="utf-8")) == json.loads(json.dumps(body)), \
            f"{f} is out of date: run python -m src.snapshot"
    assert (snapshot.OUT / "hero_preview.json").exists()
    assert {p.name for p in Path("reports/figures").iterdir()} <= {p.name for p in snapshot.FIGURES.iterdir()}
