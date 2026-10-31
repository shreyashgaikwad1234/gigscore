"""Crop full-page screenshots of the web app (docs/screens/full/*.png, captured with headless Chrome at
1440 px wide) into page-sized sections for the explainer PDF (docs/screens/*.jpg).

Capture again after UI changes, e.g. for each page:
    chrome --headless=new --window-size=1440,<height> --virtual-time-budget=12000 --screenshot=<file> <url>
Home: capture http://localhost:8000/?still=1 so the hero animation shows its finished state.
Open /apply without ?example=1 for screenshots — the live preview is shown and nothing is written to the decision log.
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
FULL, OUT = ROOT / "screens" / "full", ROOT / "screens"

# name -> (source page, top px, bottom px); x is trimmed to the content column
CROPS = {
    # public website (Home captured with ?still=1 so the hero shows its finished state)
    "overview_hero": ("overview", 0, 955, True),
    "overview_challenge": ("overview", 969, 1660, True),
    "overview_platform": ("overview", 1660, 2393, True),
    "overview_evidence": ("overview", 2393, 3212, True),
    "overview_ladder": ("overview", 3212, 3765, True),
    "overview_estimate": ("overview", 3765, 4438, True),
    "overview_segments": ("overview", 4438, 5068, True),
    "overview_teams": ("overview", 5068, 5618, True),
    "overview_governance": ("overview", 5618, 6279, True),
    "overview_fairness": ("overview", 6279, 6943, True),
    "overview_pilot": ("overview", 6943, 7546, True),
    "overview_close": ("overview", 7546, 8743, True),
    "analytics_top": ("analytics", 0, 808),
    "analytics_playground": ("analytics", 818, 1379),
    "analytics_roc_gains": ("analytics", 1379, 2084),
    "analytics_distribution": ("analytics", 2084, 2694),
    "analytics_ladder_flow": ("analytics", 2694, 3327),
    "analytics_tradeoff_calibration": ("analytics", 3327, 3980),
    "analytics_segments_trend": ("analytics", 3980, 4510),
    "analytics_drivers": ("analytics", 4510, 5121),
    "analytics_states": ("analytics", 5121, 5852),
    "analytics_next": ("analytics", 5880, 6090),
    "about": ("about", 0, 1394),
    "pilot_top": ("pilot", 0, 817),
    "pilot_data": ("pilot", 851, 1779),
    "pilot_governance": ("pilot", 1813, 2560),
    # workspace (full width, so the sidebar shows)
    "queue": ("queue", 0, 1440, True),
    "case": ("case", 0, 1080, True),
    "timeline": ("timeline", 0, 1300, True),
    "letter": ("letter", 0, 1600, True),
    "apply": ("apply", 0, 1520, True),
    "watch": ("watch", 0, 1640, True),
    "portfolio": ("portfolio", 0, 1340, True),
    "admin": ("admin", 0, 1346, True),
}
X0, X1 = 80, 1360  # the content column; crops marked True keep the full width (the full-bleed hero)


def main():
    for name, (src, top, bottom, *full) in CROPS.items():
        im = Image.open(FULL / f"{src}.png").convert("RGB")
        bottom = min(bottom, im.height)
        x0, x1 = (0, im.width) if full else (X0, min(X1, im.width))
        im.crop((x0, top, x1, bottom)).save(OUT / f"{name}.jpg", quality=88, optimize=True)
        print(f"{name}.jpg {x1 - x0}x{bottom - top}")


if __name__ == "__main__":
    main()
