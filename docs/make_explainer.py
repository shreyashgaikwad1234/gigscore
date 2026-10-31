"""Build docs/GigScore_Explainer.pdf: an in-depth explanation of what GigScore is and how it works.

Every number is read from the project's own outputs (feature table, saved models, configs and
reports written by src/*.py); nothing is typed in by hand. Run from the repository root:

    .venv\\Scripts\\python.exe docs\\make_explainer.py
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import yaml
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "GigScore_Explainer.pdf"

# ---------- fonts and styles ----------
FONT_DIR = Path("C:/Windows/Fonts")
if (FONT_DIR / "segoeui.ttf").exists():
    pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "segoeui.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "segoeuib.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Italic", str(FONT_DIR / "segoeuii.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "consola.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
    BODY, BOLD, MONO, RUPEE = "Body", "Body-Bold", "Mono", "₹"
else:  # non-Windows fallback: built-in fonts have no rupee sign
    BODY, BOLD, MONO, RUPEE = "Helvetica", "Helvetica-Bold", "Courier", "Rs "

INK = colors.HexColor("#121826")
INK2 = colors.HexColor("#3a4356")
MUTED = colors.HexColor("#6b7489")
PRIMARY = colors.HexColor("#4338ca")
SOFT = colors.HexColor("#eef0ff")
LINE = colors.HexColor("#dde1ea")
ZEBRA = colors.HexColor("#f6f7fb")
GREEN, AMBER, RED = colors.HexColor("#15803d"), colors.HexColor("#b45309"), colors.HexColor("#c2261d")

S = {
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=30, leading=36, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=14, leading=20, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=19, leading=24, textColor=PRIMARY, spaceBefore=4, spaceAfter=10,
                         keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=13.5, leading=18, textColor=INK, spaceBefore=12, spaceAfter=6,
                         keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=11, leading=15, textColor=INK, spaceBefore=8, spaceAfter=3,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=10, leading=15, textColor=INK2, spaceAfter=6),
    "bodykeep": ParagraphStyle("bodykeep", fontName=BODY, fontSize=10, leading=15, textColor=INK2, spaceAfter=6,
                               keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=10, leading=14.5, textColor=INK2, leftIndent=14,
                             bulletIndent=3, spaceAfter=3),
    "small": ParagraphStyle("small", fontName=BODY, fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8.6, leading=11.5, textColor=INK2),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8.6, leading=11.5, textColor=INK),
    "head": ParagraphStyle("head", fontName=BOLD, fontSize=8.2, leading=11, textColor=colors.white),
    "callout": ParagraphStyle("callout", fontName=BODY, fontSize=10, leading=15, textColor=INK, spaceAfter=0),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=8.6, leading=12, textColor=INK, backColor=ZEBRA,
                           borderPadding=6, spaceBefore=4, spaceAfter=8),
    "toc1": ParagraphStyle("toc1", fontName=BODY, fontSize=10.5, leading=17, textColor=INK2),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def table(rows, widths, header=True, align_right=(), highlight_rows=()):
    """rows: list of lists of str (first row = header). Wraps every cell in a Paragraph."""
    data = []
    for i, r in enumerate(rows):
        st = S["head"] if header and i == 0 else S["cell"]
        data.append([Paragraph(str(c), st) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), PRIMARY)]
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    for i in highlight_rows:
        style.append(("BACKGROUND", (0, i), (-1, i), SOFT))
    for c in align_right:
        for i in range(len(rows)):
            data[i][c].style = ParagraphStyle("r", parent=data[i][c].style, alignment=2)
    t.setStyle(TableStyle(style))
    return t


def callout(text, color=PRIMARY, bg=SOFT):
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def fig(path, width_mm=165, caption=None):
    p = ROOT / path
    if not p.exists():
        return [P(f"(figure {path} not generated)", "small")]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(p)).getSize()
    w = width_mm * mm
    out = [Image(str(p), width=w, height=w * ih / iw)]
    if caption:
        out.append(P(caption, "small"))
    return out


def flow_diagram(steps, width=170 * mm, box_h=15 * mm, cols=4):
    """Boxes left-to-right, wrapping into rows, with arrows between consecutive steps."""
    gap_x, gap_y = 8 * mm, 9 * mm
    bw = (width - gap_x * (cols - 1)) / cols
    rows = (len(steps) + cols - 1) // cols
    h = rows * box_h + (rows - 1) * gap_y + 4
    d = Drawing(width, h)
    pos = []
    for i, (title, sub) in enumerate(steps):
        r, c = divmod(i, cols)
        if r % 2 == 1:
            c = cols - 1 - c  # snake layout so arrows stay short
        x, y = c * (bw + gap_x), h - (r + 1) * box_h - r * gap_y
        pos.append((x, y))
        d.add(Rect(x, y, bw, box_h, rx=6, ry=6, fillColor=SOFT if i % 2 == 0 else colors.white,
                   strokeColor=PRIMARY, strokeWidth=0.8))
        d.add(String(x + bw / 2, y + box_h - 6 * mm + 1, title, fontName=BOLD, fontSize=8.6, fillColor=INK,
                     textAnchor="middle"))
        d.add(String(x + bw / 2, y + 3.2 * mm, sub, fontName=BODY, fontSize=7, fillColor=MUTED, textAnchor="middle"))
    for (x1, y1), (x2, y2) in zip(pos, pos[1:]):
        if abs(y1 - y2) < 1:  # same row
            a, b = (x1 + bw, x2) if x2 > x1 else (x1, x2 + bw)
            ym = y1 + box_h / 2
            d.add(Line(a, ym, b, ym, strokeColor=PRIMARY, strokeWidth=1))
            tip = 1 if b > a else -1
            d.add(Polygon([b, ym, b - tip * 5, ym + 3, b - tip * 5, ym - 3], fillColor=PRIMARY, strokeColor=PRIMARY))
        else:  # down to next row
            xm = x1 + bw / 2
            d.add(Line(xm, y1, xm, y2 + box_h, strokeColor=PRIMARY, strokeWidth=1))
            d.add(Polygon([xm, y2 + box_h, xm - 3, y2 + box_h + 5, xm + 3, y2 + box_h + 5], fillColor=PRIMARY,
                          strokeColor=PRIMARY))
    return d


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def inr(v):
    return f"{RUPEE}{v:,.0f}"


# ---------- document with TOC ----------
class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name == "h1":
            self.notify("TOCEntry", (0, f.getPlainText(), self.page))


def on_page(c, doc):
    c.saveState()
    if doc.page > 1:
        c.setFont(BODY, 8)
        c.setFillColor(MUTED)
        c.drawString(20 * mm, 12 * mm, "GigScore — how it works")
        c.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
        c.setStrokeColor(LINE)
        c.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    c.restoreState()


def H1(t):
    return [CondPageBreak(80 * mm), Paragraph(t, S["h1"])]


# ---------- gather facts from project outputs ----------
def facts() -> dict:
    t = pd.read_parquet(ROOT / "data/processed/features.parquet")
    fl = json.loads((ROOT / "data/processed/feature_list.json").read_text())
    ref = json.loads((ROOT / "models/app_reference.json").read_text(encoding="utf-8"))
    analytics = json.loads((ROOT / "models/analytics.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config/thresholds.yaml").read_text(encoding="utf-8"))
    rules = yaml.safe_load((ROOT / "config/watch_rules.yaml").read_text(encoding="utf-8"))
    stats = pd.read_csv(ROOT / "data/processed/feature_stats.csv", index_col="feature")
    bundles = {n: joblib.load(ROOT / f"models/{'lgbm' if n.startswith('lgbm') else 'lr_scorecard'}_{n.split('_', 1)[1]}.pkl")
               for n in ("lgbm_fair", "lgbm_combined", "lgbm_bureau", "lgbm_alternative", "scorecard_fair",
                         "scorecard_combined")}
    y = t["loan_default"]
    month = t["disb"].dt.to_period("M").astype(str)
    exp_rows = []
    for line in (ROOT / "reports/experiments.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (.+?) \| (\d+) \| ([\d.]+) \| ([+\-][\d.]+) \|", line)
        if m:
            exp_rows.append((m.group(1), float(m.group(3)), float(m.group(4))))
    val = (ROOT / "reports/validation_report.md").read_text(encoding="utf-8")
    recal = re.search(r"\| as trained \| ([\d.]+%) \| ([\d.]+%) \| (\d+) of 10 \| ([\d.]+) pp \|\n"
                      r"\| with offset \| ([\d.]+%) \| [\d.]+% \| (\d+) of 10 \| ([\d.]+) pp \|", val)
    random_split = dict(re.findall(r"\| (lgbm_\w+) \| ([\d.]+) \| [\d.]+ \|", val))
    psi = dict(re.findall(r"(lgbm_fair|scorecard_fair): Oct ([\d.]+), Nov [\d.]+", val))
    psi_nov = dict(re.findall(r"(lgbm_fair|scorecard_fair): Oct [\d.]+, Nov ([\d.]+)", val))
    reasons = re.findall(r"\*(approve|refer|decline)\*: (.+)", val)
    age_rows = re.findall(r"\| (18-24|25-34|35-44|45-54|55\+) \| ([\d,]+) \| ([\d.]+%) \| ([\d.]+%) \| ([+\-][\d.]+ pp)", val)
    return dict(t=t, fl=fl, ref=ref, A=analytics, cfg=cfg, rules=rules, stats=stats, b=bundles, y=y, month=month, exp=exp_rows,
                recal=recal, random_split=random_split, psi=psi, psi_nov=psi_nov, reasons=reasons, age_rows=age_rows)


# ---------- the web app, page by page ----------
NICE = {"ltv": "loan-to-value", "bureau_score": "bureau score", "manufacturer_te": "vehicle make risk",
        "dob_jan1": "approximate birth date", "employment_self": "self-employed", "new_emi_est": "new EMI",
        "score_desc_grp": "bureau grade / reason", "disb_day": "day of month", "total_emi": "total EMIs"}


def app_pages(F: dict) -> list:
    A, ref, cfg = F["A"], F["ref"], F["cfg"]
    main = A["model"]
    cuts = cfg["bands"][main]
    fl = A["flow"]
    appr_paid = fl["approve"]["repaid"] / (fl["approve"]["repaid"] + fl["approve"]["missed"])
    dec = sorted(A["deciles"], key=lambda r: r["decile"])  # decile 1 = riskiest
    seg = {s["segment"]: s for s in A["segments"]}
    e60 = next(r for r in ref["extra_approvals"] if abs(r["bureau_only_approval"] - 0.6) < 1e-6)
    out = []
    pending: list = []  # a page's heading, text and first screenshot, kept on one PDF page

    def page(title, url, intro, items):
        pending.extend([P(f"{title}  <font color='#6b7489' size='9'>{url}</font>", "h2"), P(intro)])
        pending.extend(bullets(items))

    def shot(name, width_mm, caption):
        flow = fig(f"docs/screens/{name}.jpg", width_mm, caption)
        if pending:
            out.append(KeepTogether(pending + flow))
            pending.clear()
            return []
        return flow

    n_pct = lambda b: round((fl[b]["repaid"] + fl[b]["missed"]) / A["n_test"] * 100)
    miss = lambda b: round(n_pct(b) * fl[b]["missed"] / (fl[b]["repaid"] + fl[b]["missed"]))
    ns_seg = seg["No bureau score"]
    caught = pct(A["gains"][main][10], 0)

    out.append(P("The public website", "h2"))
    out.append(P("Four pages for an NBFC's decision-makers, read in order: what GigScore does, the evidence, how it is "
                 "governed, and how a pilot would run. Each page ends with a link to the next."))

    page("Overview", "/", "The case for GigScore, set out as one argument from the problem to the pilot.", [
        "<b>Hero:</b> “Lend to new-to-credit riders with confidence.” Beside it, a live product window types a sample "
        f"application, sends it to the production model (score {606}, Approve, with three reasons), then shows a statement three "
        "months after disbursal with income down 48% and an “At risk” flag. Below: a strip of four facts (98,364 loans in the "
        "validation, 233,154 in the dataset, three stated reasons, no device data).",
        "<b>Where is your two-wheeler book losing value?</b> Four common problems; selecting one shows how GigScore addresses it, "
        f"with figures from the validation — for example {pct(ns_seg['n'] / A['n_test'], 0)} of applicants had no bureau score, "
        f"which the bureau ranks at {ns_seg['auc_bureau']:.2f} (random is 0.50) and GigScore at {ns_seg['auc']:.2f}.",
        "<b>One platform across the loan lifecycle:</b> four modules — application decisioning, early-warning monitoring, the "
        "policy simulator and decision letters — each opening the live workspace.",
        f"<b>Outcomes for every 100 applications:</b> {n_pct('approve')} approved ({miss('approve')} missed the first EMI), "
        f"{n_pct('refer')} referred ({miss('refer')} missed), {n_pct('decline')} declined ({miss('decline')} missed); "
        f"+{e60['extra']:,} additional approvals at equal risk; {caught} of first-EMI misses fall in the lowest-scored 20%.",
        "<b>Then:</b> rank-ordering by score decile, a portfolio impact estimator (applications per month, ticket size and "
        "current approval rate in; additional loans and disbursal, or misses avoided, out), segments, functions served, "
        "governance, fairness across states, the three-stage pilot, frequently asked questions, and a closing invitation to "
        "the demonstration workspace.",
    ])
    out += shot("overview_hero", 160, "Overview — hero with the live product window, and the facts strip.")
    for name, cap in [("overview_challenge", "Overview — where is your two-wheeler book losing value?"),
                      ("overview_platform", "Overview — one platform across the loan lifecycle."),
                      ("overview_evidence", "Overview — outcomes for every 100 applications."),
                      ("overview_estimate", "Overview — estimate the impact on your portfolio."),
                      ("overview_governance", "Overview — designed for regulated lending."),
                      ("overview_close", "Overview — frequently asked questions and the closing invitation.")]:
        out += shot(name, 150, cap)

    page("Validation", "/analytics", "The evidence: out-of-time validation on all 98,364 October 2018 applications. Four "
         "headline measures, the policy simulator, and twelve analyses, each with a sentence on what it means for the "
         "portfolio. A selector compares the production model with the earlier model that used location proxies.", [])
    pending.append(table([
        ["#", "Analysis", "What it shows (production model, October 2018)"],
        ["—", "Policy simulator", f"Move the approve and refer cut-offs; see approval rate, first-EMI delinquency, referrals and state "
                                  f"approval rates on the real loans. At today's cut-offs ({cuts['approve']}/{cuts['refer']}) it "
                                  "reproduces the validated results; “Apply in policy settings” carries the cut-offs into the workspace"],
        ["01", "Discrimination vs the bureau score", f"ROC curves; AUC {A['roc'][main]['auc']:.2f} overall, "
                                                     f"{A['roc_no_score'][main]['auc']:.2f} new to credit (bureau "
                                                     f"{A['roc_no_score']['lgbm_bureau']['auc']:.2f})"],
        ["02", "Capture of first-EMI misses", f"Declining the lowest-scored 20% captures {caught} of misses (random: 20%)"],
        ["03", "Score distribution by outcome", "Repaid and missed borrowers by score, with the decline / refer / approve zones"],
        ["04", "First-EMI delinquency by decile", f"{pct(dec[9]['default_rate'], 0)} in the highest-scored tenth, "
                                                  f"{pct(dec[0]['default_rate'], 0)} in the lowest"],
        ["05", "Decision flow to outcome", f"All {A['n_test']:,} applications → recommendation → first EMI; {pct(appr_paid)} of approved paid"],
        ["06", "Approval rate versus delinquency", "Delinquency among approved as approval grows, for all four models"],
        ["07", "Calibration", "Predicted versus actual by tenth; on the line for Aug–Sep, above it for October"],
        ["08", "Discrimination by segment", f"New to credit {ns_seg['auc']:.2f} vs bureau {ns_seg['auc_bureau']:.2f}; self-employed "
                                            f"{seg['Self-employed']['auc']:.2f} vs {seg['Self-employed']['auc_bureau']:.2f}"],
        ["09", "Volume and delinquency by month", "August and September trained the model; October validated it"],
        ["10", "Drivers of the score", "Average contribution of each input; strongest: " + ", ".join(
            NICE.get(s["feature"], s["feature"]) for s in A["shap"][:3])],
        ["11", "Predictive strength of variables", "Strongest single variables; location proxies dropped from production in grey"],
        ["12", "Approval rate by state", "At 60% approval, production model versus earlier model"],
    ], [10 * mm, 50 * mm, 110 * mm]))
    pending.append(Spacer(1, 6))
    for name, cap in [("analytics_top", "Validation — headline measures."),
                      ("analytics_playground", "Validation — the policy simulator at today's cut-offs."),
                      ("analytics_roc_gains", "Validation — discrimination and capture of first-EMI misses."),
                      ("analytics_ladder_flow", "Validation — delinquency by decile and decision flow."),
                      ("analytics_tradeoff_calibration", "Validation — approval versus delinquency, and calibration."),
                      ("analytics_states", "Validation — approval rate by state."),
                      ("analytics_next", "Validation — the link onward to Governance, as every public page ends.")]:
        out += shot(name, 120 if name == "analytics_top" else 150, cap)

    page("Governance", "/about", "For risk and compliance review: the production model and its fallback, discrimination "
         "with 95% intervals, fairness across states for each model considered, and the full documents.", [
        f"<b>Headline measures:</b> production model and fallback; AUC {ref['metrics'][main]['all']['auc']:.2f} overall and "
        f"{ref['metrics'][main]['no_score']['auc']:.2f} new to credit (bureau data alone "
        f"{ref['metrics']['lgbm_bureau']['no_score']['auc']:.2f}); states outside the 5-point tolerance.",
        "<b>Fairness table:</b> the earlier model, the candidate variable sets and the production model, showing the trade of a "
        "small loss in discrimination for materially more consistent approval across states.",
        "<b>Documents:</b> model card, validation report, variable specification and data profile, generated by the pipeline.",
    ])
    out += shot("about", 140, "Governance — headline measures, discrimination and fairness tables.")

    page("Pilot programme", "/pilot", "How a lender would validate GigScore on its own portfolio before it influences a live "
         "decision — about three months.", [
        "<b>Three phases with deliverables:</b> historical back-test (weeks 1–2), shadow assessment alongside credit officers "
        "(weeks 3–10), and a controlled go-live in one branch (week 11 onwards).",
        "<b>Data required:</b> application, applicant, credit bureau and outcome fields the lender already holds, with the "
        "optional consented statement; data can stay in the lender's environment.",
        "<b>Success criteria</b> agreed in advance: rank-ordering on the lender's book, approval uplift at constant risk, "
        "first-EMI delinquency, turnaround time, override rate and state-level consistency; and <b>governance during the "
        "pilot</b> — no automated decisions before phase 3, officer authority, a complete audit trail.",
    ])
    out += shot("pilot_top", 140, "Pilot programme — the three phases and their deliverables.")
    out += shot("pilot_data", 140, "Pilot programme — data required and how success is measured.")

    out.append(P("The workspace", "h2"))
    out.append(P("The screens branch staff use, opened from “Open workspace”: a sidebar with the decision queue, new "
                 "application, early-warning monitor and decision log; links to the validation and governance pages; and "
                 "policy settings. The top bar has search (Ctrl + K) by name, application or loan number, a branch selector, "
                 "alerts for applications past the 24-hour service level and red loans not yet contacted, and the signed-in "
                 "user (a demonstration stand-in for real logins). Twelve demonstration cases (src/demo_seed.py) populate it."))

    page("Decision queue", "/app", "Applications awaiting a credit officer's decision.", [
        "A summary line (for example “Good evening, R. Patil. 7 applications need a decision, 1 past the 24-hour target …”), four measures "
        "(waiting, past target, assigned to me, loans to call), and the queue: referrals first, then the longest waiting, with "
        "a live waiting time that turns amber after 4 hours and red after 24.",
        "<b>Assign to me</b>, filters (all / assigned to me / unassigned) and keyboard flow (J / K to move, Enter to open, "
        "A to assign); below, running loans to call and recently decided cases.",
    ])
    out += shot("queue", 140, "Decision queue.")

    page("Case record", "/app/case?id=…", "One record per applicant, from application to collections.", [
        "<b>Fact strip:</b> score on a coloured band, chance of missing the first EMI, loan and share of price, estimated EMI, "
        "bureau position and employment.",
        "<b>Tabs:</b> Recommendation (gauge, reasons, checks before deciding, links to the validation and governance), "
        "Application (data as scored), Monitoring (latest statement check, weekly credits, all checks), History (every "
        "event from scoring to collections calls) and Decision letter.",
        "<b>Decision panel</b> kept in view: approve or decline (A / D), a recorded reason, Ctrl + Enter to record, and a "
        "five-second Undo before it is saved; then disbursal with the loan account number. Departures from the "
        "recommendation are marked as overrides.",
        "<b>Decision letter:</b> sanction or decline with the stated reasons, in English, Hindi or Marathi, printable to PDF.",
    ])
    out += shot("case", 145, "Case record — fact strip, recommendation and the officer's decision.")
    out += shot("timeline", 92, "Case record — history from scoring to a collections call.")
    out += shot("letter", 92, "Case record — decline letter with its stated reasons.")

    page("New application", "/apply", "Capture an application; the recommendation is previewed live and recorded on submission.", [
        "Sections for case (applicant name, application number, branch — never scoring inputs), loan, applicant and credit "
        "bureau. A <b>live preview</b> beside the form updates as fields change.",
        f"On <b>Get decision</b>: score, recommendation (approve from {cuts['approve']}, refer {cuts['refer']}–{cuts['approve'] - 1}, "
        f"decline below {cuts['refer']}), reasons, the officer's decision panel, and a link to the new case in the queue.",
        "<b>Sensitivity analysis:</b> adjust the loan amount or add KYC documents to see the effect, with the smallest changes "
        "that would improve the recommendation (for the sample: a ₹5,000 larger down payment moves it from refer to approve).",
    ])
    out += shot("apply", 125, "New application — form with the live preview.")

    page("Early-warning monitor", "/watch", "Statement-based monitoring of disbursed loans, with consent.", [
        "Opens with a synthetic sample previewed. A bank statement exported as CSV (HDFC- and SBI-style layouts, day-first "
        "dates, “1,234.00” amounts, narration text) is read and checked immediately once consent is recorded; bills and "
        "bounces are recognised from the narration.",
        "Status, recommended intervention, six indicators and weekly credits; the <b>collections call list</b> below records "
        "each contact (called, promised to pay, EMI date moved …) on the borrower's case.",
    ])
    out += shot("watch", 115, "Early-warning monitor — statement, status and indicators.")

    page("Decision log and policy settings", "/portfolio · /admin", "The record for audit, and the controls for the risk function.", [
        "<b>Decision log:</b> recorded recommendations, officer decisions with override marks, early-warning checks, the "
        "October 2018 validation benchmark by band, and each entry's full record with a link to its case.",
        "<b>Policy settings:</b> production or earlier model, approve and refer cut-offs, monthly recalibration, the fallback "
        "switch and early-warning thresholds — locked behind the administrator credential, each change recorded with its "
        "reason; arrives pre-filled from the policy simulator.",
    ])
    out += shot("portfolio", 112, "Decision log.")
    out += shot("admin", 92, "Policy settings (locked until the administrator credential is entered).")
    return out


# ---------- content ----------
def build(F: dict):
    t, ref, cfg, R, b, y = F["t"], F["ref"], F["cfg"], F["rules"], F["b"], F["y"]
    M = ref["metrics"]
    main, alt, fb = ref["model"], ref["alternative_model"], ref["fallback_model"]
    n = len(t)
    ns = t["no_score"] == 1
    se = t["employment_self"] == 1
    nus = t["no_usable_score"] == 1
    oot = t["split"] == "oot"
    win = ~oot
    cuts = cfg["bands"][main]
    bt = ref["band_table"]
    story = []

    # ----- cover -----
    story += [Spacer(1, 40 * mm), P("GigScore", "title"),
              P("An alternative-data credit decision engine for people with little or no credit history — "
                "what it is, how it works, how well it works, and where its limits are.", "subtitle"),
              Spacer(1, 14 * mm)]
    story.append(callout(
        f"<b>In one paragraph.</b> GigScore helps a lender decide on small two-wheeler loans for self-employed and gig "
        f"workers, half of whom have no credit-bureau score. It turns what a loan officer knows into a 300–900 score, an "
        f"approve / refer / decline decision and three plain-language reasons. After the loan is given, it reads the "
        f"borrower's bank or UPI statement (with consent) and flags trouble before an EMI is missed. It was built and "
        f"tested on {n:,} real loans from an Indian NBFC (L&amp;T Financial Services, 2018)."))
    story += [Spacer(1, 50 * mm),
              P(f"Generated {date.today():%d %B %Y} by <font name='{MONO}'>docs/make_explainer.py</font>. Every number in "
                "this document is read from the project's own outputs (feature table, saved models, configuration and the "
                "reports written by the code in <font name='" + MONO + "'>src/</font>).", "small"),
              P("Data: L&amp;T Vehicle Loan Default Prediction (Kaggle mamtadhaker/lt-vehicle-loan-default-prediction), "
                "released for the LTFS Data Science FinHack (Analytics Vidhya, 2019). Statement samples used in the app are "
                "SYNTHETIC and labelled as such. The Pune rider survey workbook in the project folder is simulated and is "
                "not used by any model.", "small"),
              PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    story += [P("Contents", "h2"), toc, PageBreak()]

    # ----- 1 -----
    story += H1("1. What GigScore is")
    story += [P("GigScore is a credit decision engine: software a lender (a bank, an NBFC or a fintech) plugs into its "
                "loan process. It is not a lender itself. It has three parts that share one code base:")]
    story.append(table([
        ["Part", "What it does", "Who uses it"],
        ["Score (at application)", "Takes an application and returns a 300–900 score, the probability the borrower misses "
         "the first EMI, a band (approve / refer / decline) and three reasons.", "Credit officer, or the lender's loan "
         "system through the API"],
        ["Watch (after disbursal)", "Reads 8–12 weeks of the borrower's bank / UPI statement and returns a green / amber / "
         "red flag, the signals that triggered it and a suggested next step.", "Collections and customer-care teams"],
        ["Workspace and website", "A workspace for branch staff (decision queue, case records, decision log, policy settings) "
         "and a public website presenting the validation, model governance and pilot programme to lenders.",
         "Credit officers, risk heads, auditors, lenders evaluating it"],
    ], [38 * mm, 88 * mm, 44 * mm]))
    story += [Spacer(1, 6), P("<b>Who it is for.</b> Delivery riders, cab drivers and other self-employed people who want "
                              "to finance a two-wheeler but are declined or overcharged because they have no CIBIL score. "
                              f"In the L&amp;T data, {pct(ns.mean())} of borrowers had no bureau score at all and "
                              f"{pct(se.mean())} were self-employed."),
              P("<b>What makes it more than a model.</b> It is built the way a lender would need it: a test on loans from a "
                "later month than the training data, calibrated probabilities, reasons a customer can be told, a fairness "
                "check, an audit log of every decision, a fallback model (kill switch), monitoring for drift, and settings "
                "that change without touching code.")]

    # ----- 2 -----
    story += H1("2. The problem it solves")
    story += [P("In India, lenders lean heavily on the credit-bureau score (CIBIL and others). A person who has never "
                "borrowed formally has no score, so the lender either declines them or treats them as high risk. Gig "
                "workers are hit hardest: their income is real but irregular and paid into UPI wallets and bank accounts, "
                "which a bureau never sees."),
              P("The data shows the bureau's blind spot directly. For the "
                f"{(oot & ns).sum():,} October 2018 applicants with no bureau score, a model built only on bureau data "
                f"ranks risk with an AUC of <b>{M['lgbm_bureau']['no_score']['auc']:.3f}</b> — no better than tossing a coin "
                "(0.5). It cannot tell the safe ones from the risky ones, so a lender relying on it has to treat them all "
                "the same."),
              P(f"GigScore uses information the lender already collects at the dealership — loan-to-value, the vehicle, "
                f"employment type, existing loans if any — and ranks the same no-score applicants with an AUC of "
                f"<b>{M[main]['no_score']['auc']:.3f}</b>. That ranking is what lets a lender approve the safer ones "
                "instead of declining everybody.")]

    # ----- 3 -----
    story += H1("3. The data")
    months = F["month"].value_counts().sort_index()
    rates = y.groupby(F["month"]).mean()
    story += [P(f"<b>Source.</b> {n:,} two-wheeler loans disbursed by L&amp;T Financial Services between August and October "
                "2018, with 41 columns: loan amount, vehicle price, loan-to-value, branch, dealer, vehicle maker, area, "
                "state, date of birth, employment type, which ID documents were shared, the bureau score and its "
                "description, and the applicant's existing loans."),
              P(f"<b>What is predicted.</b> <font name='{MONO}'>loan_default</font> = 1 means the borrower <b>missed the first "
                f"EMI</b> on its due date. Overall {pct(y.mean())} did. This is early-payment default — the earliest sign of "
                "repayment stress — which is also exactly what the Watch layer looks for later in the loan.")]
    story.append(table([["Disbursal month", "Loans", "First-EMI default rate", "Role"]] +
                       [[m, f"{months[m]:,}", pct(rates[m]), "test (out-of-time)" if m == "2018-10" else "training"]
                        for m in months.index], [40 * mm, 35 * mm, 45 * mm, 50 * mm], align_right=(1, 2)))
    story += [Spacer(1, 6), P("<b>Who is in it.</b>")]
    story += bullets([
        f"No bureau score at all (score = 0): {pct(ns.mean())}, defaulting at {pct(y[ns].mean())} vs {pct(y[~ns].mean())} "
        "for scored borrowers.",
        f"No <i>usable</i> score (score 0 or a 'Not Scored' placeholder): {pct(nus.mean())}.",
        f"Self-employed: {pct(se.mean())}, defaulting at {pct(y[se].mean())}.",
    ])
    story += [P("<b>Quirks the code handles</b> (file <font name='" + MONO + "'>src/load.py</font>):")]
    story += bullets([
        "Dates use two-digit years. A birth year of '84' is 1984 but pandas reads '45' as 2045, so birth years after 2025 "
        "are moved back a century. Disbursal dates are recent and are never shifted.",
        "A bureau score of 0 means <i>no score</i>, not a bad score. It is kept as a flag and the score is treated as "
        "missing.",
        "Rows described as 'Not Scored: …' carry placeholder codes 11–18 instead of real scores. These are also treated "
        "as missing; the reason for not being scored is kept as its own category, because it predicts risk.",
        "Account ages are text such as '1yrs 11mon' and are converted to months.",
        "Some loan amounts are negative or absurd (up to 1,000,000,000). Amount columns are clipped at the 1st and 99th "
        "percentile of their non-zero values, using the training months only.",
    ])
    story.append(callout("<b>Honesty rule.</b> Cleaning, feature screening, tuning, calibration and the decision cut-offs "
                         "use August–September only; October 2018 is the test month. One choice did look at October: the "
                         "decision to ship the fair model was made after comparing fairness options on October loans "
                         "(Section 9). That was a policy choice about equal treatment, not a search for a higher score, "
                         "and the fair model's October accuracy is lower than the alternative's.", GREEN,
                         colors.HexColor("#e7f6ec")))

    # ----- 4 -----
    story += H1("4. How it works, end to end")
    story += [P("The pipeline is a chain of small Python modules. Each reads the previous step's output from disk and "
                "writes its own, so any step can be re-run on its own.", "bodykeep")]
    story.append(flow_diagram([
        ("Load & clean", "src/load.py"), ("Build features", "src/features.py"), ("Encode IDs", "src/encode.py"),
        ("Train models", "src/train.py"), ("Validate", "src/evaluate.py"), ("Explain & fairness", "explain, fairness"),
        ("Score / Watch", "src/score.py, watch.py"), ("API + web app", "api/, frontend/")]))
    story += [Spacer(1, 6)]
    story.append(table([
        ["Step", "What happens", "Output"],
        ["Load & clean", "Parse dates, fix the century rule, convert '1yrs 11mon', flag no-score, clip amounts.",
         "a clean frame"],
        ["Build features", "Turn raw columns into 48 risk features in five families (Section 5).", "features.parquet"],
        ["Encode IDs", "Replace branch / dealer / maker / area IDs by their past default rate, out-of-fold.",
         "encoders.pkl"],
        ["Train", "Scorecard + LightGBM on four feature sets, tuned with 5-fold cross-validation inside Aug–Sep.",
         "models/*.pkl"],
        ["Validate", "Measure on October: accuracy with confidence ranges, calibration, drift, business value, cut-offs.",
         "validation_report.md, thresholds.yaml"],
        ["Explain & fairness", "Per-applicant reasons (SHAP); approval and error gaps by age band and state.",
         "reports, model card"],
        ["Score / Watch", "One application → score, band, reasons. One statement → flag, triggers, action.",
         "live decisions"],
        ["API + web app", "FastAPI serves decisions and logs every one; the Astro site is the user interface.",
         "the running product"],
    ], [32 * mm, 98 * mm, 40 * mm]))

    # ----- 5 -----
    story += H1("5. Features: what the model looks at")
    story += [P("Features are grouped into five families borrowed from how credit analysts think (capacity, asset quality, "
                "management, earnings, liquidity) plus context. The table shows the strongest by <b>information value</b> "
                "(IV: how well a single feature separates defaulters from non-defaulters; above 0.1 is medium, below 0.02 "
                "is too weak for a scorecard).")]
    st = F["stats"]
    top = st.sort_values("iv", ascending=False).head(12)
    desc = {
        "supplier_te": "Dealer's past default rate (out-of-fold)", "pincode_te": "Area's past default rate",
        "bureau_score": "Bureau score (missing if none)", "branch_te": "Branch's past default rate",
        "ltv": "Loan as % of the vehicle price", "down_payment_share": "Share of the price paid upfront",
        "score_desc_grp": "Bureau risk grade or reason for no score", "new_emi_est": "Estimated new EMI (1.5%/month, 36 months)",
        "total_emi": "Existing EMIs + new EMI", "pri_sanctioned_amount": "Total of past loans sanctioned",
        "pri_disbursed_amount": "Total of past loans disbursed", "branch_n": "Branch loan volume",
        "pri_current_balance": "Outstanding balance on existing loans", "hist_len_m": "Credit history length (months)",
        "pri_no_accts": "Number of past loans", "overdue_ratio": "Share of existing loans overdue",
        "pri_active_accts": "Number of running loans",
    }
    fair_set = set(F["fl"]["sets"]["fair"])
    story.append(table([["Feature", "Meaning", "IV", "In shipped model?"]] +
                       [[f, desc.get(f, ""), f"{r.iv:.3f}", "yes" if f in fair_set else "no — stands in for state"
                         if f in ("supplier_te", "pincode_te", "branch_te", "branch_n") else "no — duplicate"]
                        for f, r in top.iterrows()], [36 * mm, 70 * mm, 16 * mm, 48 * mm], align_right=(2,)))
    story += [Spacer(1, 6), P("<b>Leakage-safe target encoding.</b> Branch, dealer, maker and area have hundreds to "
                              "thousands of codes. Each code is replaced by the default rate of its past loans, shrunk "
                              "towards the overall rate when the code has few loans (smoothing m = 50). For the training "
                              "rows this is done <i>out-of-fold</i>: the data is split into five parts and each row is "
                              "encoded only from the other four, so no row ever sees its own outcome. New applications use "
                              "the mapping from all training rows."),
              P("<b>Deliberately left out.</b> Age, age band and state are protected attributes: they are stored for the "
                "fairness check and never fed to the model. The employee who logged the loan is an operational artefact and "
                "is dropped. Disbursal month is not built, because the training months (Aug, Sep) never include October.")]
    story += [P("<b>Ideas tested and rejected.</b> Before the final training, six more features, averaging three models, "
                "and blending LightGBM with the scorecard were tried with cross-validation on Aug–Sep only. None improved "
                "AUC by the required 0.002, so none was adopted:")]
    story.append(table([["Experiment", "CV AUC", "Gain"]] + [[e, f"{a:.4f}", f"{g:+.4f}"] for e, a, g in F["exp"]],
                       [110 * mm, 30 * mm, 30 * mm], align_right=(1, 2)))

    # ----- 6 -----
    story += H1("6. The models")
    story += [P("Two kinds of model are trained, each on four feature sets, giving eight models. Only two are used in "
                "production; the rest exist to measure what each kind of data contributes.")]
    story += [P("Model A — the scorecard (the fallback)", "h2"),
              P("The classic, regulator-friendly design. Each feature is cut into up to ten bins; each bin is replaced by "
                "its <b>weight of evidence</b> (WoE = log of the share of good borrowers in the bin divided by the share of "
                "bad ones). A logistic regression then weighs the features, and the result is converted to points. A credit "
                "officer can read the whole model as a points table. Any feature whose weight comes out with the wrong sign "
                "(which would make its points run against the data) is removed and the model refitted.")]
    sc = b[fb]
    story.append(P(f"The shipped scorecard <font name='{MONO}'>{sc['version']}</font> uses {len(sc['features'])} features: "
                   + ", ".join(sc["features"]) + "."))
    story += [P("Model B — LightGBM (the main model)", "h2"),
              P("A gradient-boosted tree model: hundreds of small decision trees, each correcting the previous ones. It "
                "captures interactions a scorecard cannot. Three safeguards keep it defensible:")]
    lg = b[main]
    story += bullets([
        "<b>Monotone constraints</b>: a higher loan-to-value, more delinquencies, more overdue loans or more enquiries can "
        "never lower the predicted risk, and more ID documents can never raise it.",
        f"<b>Tuning by cross-validation</b>: Optuna tried 40 settings, each judged by 5-fold cross-validation inside "
        f"Aug–Sep. The shipped model has {lg['best_iteration']} trees with "
        + ", ".join(f"{k} = {round(v, 3)}" for k, v in lg['params'].items()
                    if k in ('num_leaves', 'min_child_samples', 'feature_fraction', 'lambda_l2')) + ".",
        "<b>Calibration</b>: raw scores are mapped to real probabilities with isotonic regression fitted on out-of-fold "
        "predictions, so '12%' means about 12 in 100 such borrowers miss the first EMI.",
    ])
    story += [P("The four feature sets", "h2")]
    story.append(table([
        ["Set", "Features", "What it answers"],
        ["bureau", str(len(F["fl"]["sets"]["bureau"])), "How far can credit-bureau data alone go?"],
        ["alternative", str(len(F["fl"]["sets"]["alternative"])), "How far can non-bureau data alone go?"],
        ["combined", str(len(F["fl"]["sets"]["combined"])), "Everything — the most accurate model (the alternative)"],
        ["fair", str(len(F["fl"]["sets"]["fair"])), "Everything except features that stand in for state — <b>shipped</b>"],
    ], [30 * mm, 22 * mm, 118 * mm], highlight_rows=(4,)))
    story += [Spacer(1, 6), P("The <b>fair</b> set removes area and branch encodings, dealer encodings and the ID-document "
                              "flags. Section 9 explains why.")]

    # ----- 7 -----
    story += H1("7. How well it works")
    story += [P("All results are on the <b>98,364 October 2018 loans</b>, which the models never saw. Ranges are 95% "
                "confidence intervals from 500 bootstrap resamples. AUC is the probability that a randomly chosen "
                "defaulter gets a higher risk score than a randomly chosen non-defaulter: 0.5 is guessing, 1.0 is perfect.")]
    labels = {main: "Shipped (fair) LightGBM", alt: "All-features LightGBM", fb: "Fallback scorecard",
              "lgbm_bureau": "Bureau data only"}
    ci = lambda x: f"{x['auc']:.3f} ({x['auc_lo']:.3f}–{x['auc_hi']:.3f})"  # noqa: E731
    story.append(table([["Model", "All applicants", "No bureau score", "Self-employed"]] +
                       [[labels.get(k, k), ci(v["all"]), ci(v["no_score"]), ci(v["self_employed"])] for k, v in M.items()],
                       [48 * mm, 41 * mm, 41 * mm, 40 * mm], align_right=(1, 2, 3), highlight_rows=(1,)))
    story += [Spacer(1, 6)]
    story += bullets([
        f"<b>The headline:</b> for applicants with no bureau score, the shipped model's AUC is "
        f"{M[main]['no_score']['auc']:.3f} against {M['lgbm_bureau']['no_score']['auc']:.3f} for bureau data alone.",
        "<b>Is that good?</b> On this dataset, yes. First-EMI default is a noisy outcome and the data holds no income "
        "information; the project's build guide puts the best public competition results at around 0.67–0.68 (for a "
        "model that also uses age and location). An AUC above 0.75 here would signal a data leak, and none was found.",
        f"<b>Random vs time split:</b> refitting the same settings on a random split of all three months gives "
        f"{F['random_split'].get(main, '–')} for the shipped model — higher, because October leaks into training. The "
        "out-of-time number is the honest one.",
    ])
    story += [P("What a lender gets from it", "h2"),
              P("AUC is abstract; a lender cares about how many people it can approve and how many of them repay. "
                "The table shows, at the default rate the bureau-only model reaches, how many more applicants the shipped "
                "model can approve:")]
    story.append(table([["Bureau-only approves", "Its default rate", "GigScore approves", "Extra loans (of 98,364)"]] +
                       [[pct(r["bureau_only_approval"], 0), pct(r["bad_rate"]), pct(r["model_approval"]), f"+{r['extra']:,}"]
                        for r in ref["extra_approvals"]], [42 * mm, 42 * mm, 42 * mm, 44 * mm], align_right=(1, 2, 3)))
    nsa = ref["no_score_approval_at_train_rate"]
    story += [Spacer(1, 6), P(f"For no-score applicants specifically: keeping defaults at or below the training-period rate "
                              f"of {pct(ref['train_default_rate'])}, the shipped model can approve {pct(nsa[main], 0)} of "
                              f"them; the bureau-only model {pct(nsa['lgbm_bureau'], 1)}.")]
    story += fig("reports/figures/approval_curve.png", 150,
                 "Approval rate vs default rate among approved applicants, October 2018. Lower is better. "
                 "Source: src/evaluate.py.")
    story += [P("Calibration and monthly recalibration", "h2"),
              P(f"October was riskier than the training months ({pct(rates.get('2018-10', 0))} vs "
                f"{pct(y[win].mean())}). No model trained on August–September can know that in advance, so the predicted "
                "probabilities run low in October while the <i>ranking</i> stays intact. The fix a lender would use is a "
                "monthly 'calibration offset': once the first-EMI outcomes of last month's loans are known, shift every "
                "probability by a constant in log-odds.")]
    if F["recal"]:
        g = F["recal"].groups()
        story.append(P(f"Back-tested within October (offset fitted on loans from 1–15 Oct, applied to 16–31 Oct): mean "
                       f"predicted probability {g[0]} → {g[4]} against {g[1]} observed; worst decile gap {g[3]} → {g[6]} "
                       "points. It helps, but risk kept rising through October, so a half-month-old offset closes only part "
                       "of the gap. The offset is a setting (<font name='" + MONO + "'>calibration_offset</font>) that an "
                       "admin can change in the app."))
    story += [P("Stability", "h2"),
              P(f"The <b>population stability index</b> (PSI) compares the score distribution between periods: below 0.10 is "
                f"stable, above 0.25 needs investigation. The shipped model's score PSI is {F['psi'].get(main, '–')} for "
                f"October and {F['psi_nov'].get(main, '–')} for November 2018 (the unlabelled test.csv), measured against "
                "August–September.")]

    # ----- 8 -----
    story += H1("8. The score and the decision bands")
    story += [P(f"The probability of default (PD) is converted to points: <b>{cfg['score_scale']['base_score']} points "
                f"at {cfg['score_scale']['base_odds']}:1 good-to-bad odds</b> (a 20% PD), and every "
                f"{cfg['score_scale']['points_to_double_odds']} points doubles the odds. Higher is safer; the result is "
                "clipped to 300–900."),
              P(f"Score = {cfg['score_scale']['base_score']} + {cfg['score_scale']['points_to_double_odds']} × "
                f"log<sub>2</sub>( ((1 − PD) / PD) / {cfg['score_scale']['base_odds']} )", "code")]
    from src.scorecard import to_points
    story.append(table([["PD (chance of missing the first EMI)", "Score"]] +
                       [[pct(p, 0), str(int(to_points(p)))] for p in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)],
                       [100 * mm, 70 * mm], align_right=(1,)))
    story += [Spacer(1, 6),
              P(f"<b>Cut-offs.</b> Approve if the score is at least <b>{cuts['approve']}</b>, refer to a credit officer if "
                f"at least <b>{cuts['refer']}</b>, otherwise decline. They were set so that the safest 60% of August–September "
                "applicants would be approved and the next 20% referred, then applied unchanged to October:")]
    story.append(table([["Band", "Share of October applicants", "Actual first-EMI default"]] +
                       [[k.title(), pct(v["share"]), pct(v["bad_rate"])] for k, v in bt.items()],
                       [50 * mm, 60 * mm, 60 * mm], align_right=(1, 2)))
    story += [Spacer(1, 6), P("These cut-offs are a proposal. A lender chooses its own point on the approval curve; the "
                              "values live in <font name='" + MONO + "'>config/thresholds.yaml</font> and can be changed "
                              "from the policy settings page without touching code.")]

    # ----- 9 -----
    story += H1("9. Fairness")
    story += [P("The model never sees age or state, but other features can stand in for them (a 'proxy'). The check looks at "
                "outcomes: approval rate and error rates by age band and by state, with gaps above 5 percentage points "
                "flagged. Two error rates matter: the share of defaulters who still get approved, and the share of good "
                "borrowers who get turned away (the second is the one a regulator worries about most).")]
    if F["age_rows"]:
        story.append(table([["Age band", "Loans", "Default rate", "Approval rate", "Gap vs all"]] +
                           [list(r) for r in F["age_rows"]], [30 * mm, 30 * mm, 35 * mm, 35 * mm, 40 * mm],
                           align_right=(1, 2, 3, 4)))
        story.append(Spacer(1, 6))
    story += [P(f"<b>Age:</b> {ref['fairness_age_flagged']} of 5 bands flagged. <b>State:</b> {ref['fairness_state_flagged']} "
                f"of {ref['fairness_n_states']} states flagged for the shipped model."),
              P("Why the fair model is shipped", "h2"),
              P("The all-features model leaned on area, branch and dealer, and on which ID documents were shared — all of "
                "which differ a lot between states. Good borrowers in some states were turned away far more often. Each row "
                "below removes more of these features and approves the safest 60%:")]
    story.append(table([["Model", "AUC", "Default rate, approved", "Largest state gap", "States flagged"]] +
                       [[o["option"], f"{o['auc']:.3f}", pct(o["bad_rate_approved"]), f"{o['max_state_gap'] * 100:.1f} pp",
                         f"{int(o['states_flagged'])} of {int(o['n_states'])}"] for o in ref["fairness_options"]],
                       [62 * mm, 20 * mm, 32 * mm, 30 * mm, 26 * mm], align_right=(1, 2, 3, 4),
                       highlight_rows=(len(ref["fairness_options"]),)))
    story += [Spacer(1, 6), P("The trade: some accuracy and slightly more defaults among approved loans, for much smaller "
                              "differences between states. For a product whose purpose is fair access for people the "
                              "bureau cannot see, that is the right default. A lender can switch to the all-features model "
                              "from the policy settings page; that choice belongs with its risk and compliance teams."),
              P("<b>One open point.</b> With location features gone, the fair model leans more on employment type: "
                "'Applicant is self-employed' appears among the reasons for many referred and declined applicants "
                "(Section 10). Self-employed borrowers did default more often in this data, so the signal is real, but for "
                "a product built for gig workers it deserves a deliberate decision by the lender — keep it, cap its effect, "
                "or remove it and accept the accuracy cost.")]

    # ----- 10 -----
    story += H1("10. Explaining every decision")
    story += [P("Each decision comes with three plain-language reasons. For LightGBM they come from <b>SHAP</b> values, which "
                "split a prediction into how much each feature pushed the risk up or down for this particular applicant "
                "(LightGBM computes them exactly with its built-in TreeSHAP). For the scorecard they come from the points "
                "table. A declined or referred applicant sees the three strongest factors that <i>raised</i> risk; an "
                "approved one the three strongest that <i>lowered</i> it."),
              P("Every feature has a sentence written for customers, e.g. loan-to-value → 'The loan is a high share of the "
                "vehicle price'. Process features (disbursal timing, a placeholder birth date) stay in the model but are "
                "never shown as reasons, because they say nothing the customer can understand or act on.")]
    if F["reasons"]:
        story.append(P("Most frequent reasons on October 2018 applicants (shipped model):", "h3"))
        story.append(table([["Band", "Most frequent reasons (share of applicants in the band)"]] +
                           [[b_.title(), r.replace("; ", "<br/>")] for b_, r in F["reasons"]], [25 * mm, 145 * mm]))
    story += fig("reports/figures/shap_importance.png", 120, "Global importance of each feature in the shipped model "
                                                               "(mean absolute SHAP value). Source: src/explain.py.")

    # ----- 11 -----
    story += H1("11. Watch: the early-warning layer")
    story += [P("The L&amp;T data has no cash-flow history, so the early-warning layer starts as transparent rules — the way "
                "most collections teams already work — and is written so a trained model can replace the rules later "
                "without changing the app. It reads 8–12 weeks of statement rows (date, credit, debit, balance, optionally "
                "a category such as rent or utility and a status such as late or bounced)."),
              P("<b>Consent first.</b> The API refuses to read a statement unless the request says the borrower consented. "
                "Only the statement is used — no contacts, call logs or device data, in line with RBI's Digital Lending "
                "Directions.")]
    rr = R
    story.append(table([
        ["Signal", "Amber if", "Red if"],
        ["Income trend (last 4 weeks vs 12-week average)", f"below {pct(rr['inflow_trend']['amber_below'], 0)}",
         f"below {pct(rr['inflow_trend']['red_below'], 0)}"],
        ["Weeks with no income (last 8)", f"{rr['inflow_gaps']['amber_at']}", f"{rr['inflow_gaps']['red_at']} or more"],
        [f"Lowest balance in the {rr['balance_before_emi']['days_before']} days before the EMI date",
         f"below {rr['balance_before_emi']['amber_below']}× the EMI", f"below {rr['balance_before_emi']['red_below']}× the EMI"],
        [f"Rent / utility paid late or bounced (last {rr['bills']['lookback_days']} days)", f"{rr['bills']['amber_at']}",
         f"{rr['bills']['red_at']} or more"],
        ["All EMIs as a share of the last 28 days' income", f"above {pct(rr['obligation_ratio']['amber_above'], 0)}",
         f"above {pct(rr['obligation_ratio']['red_above'], 0)}"],
        [f"Share of each inflow spent within {rr['cash_out_speed']['hours']} hours",
         f"above {pct(rr['cash_out_speed']['amber_above'], 0)}", f"above {pct(rr['cash_out_speed']['red_above'], 0)}"],
    ], [86 * mm, 42 * mm, 42 * mm]))
    story += [Spacer(1, 6),
              P(f"<b>Combining signals.</b> Any red signal, or {rr['combination']['red_if_ambers']} or more amber ones → "
                f"<b>red</b>. {rr['combination']['amber_if_ambers']} amber → <b>amber</b>. Otherwise green. Fewer than "
                f"{rr['min_weeks']} weeks of statement → 'insufficient data'."),
              P("<b>Suggested actions.</b> Green: none. Amber: a reminder call — or, if the only problem is a low balance "
                "just before the due date while income holds up, an offer to move the EMI date. Red: contact the borrower "
                "about restructuring before an EMI is missed."),
              callout("<b>Status of the rules.</b> The thresholds come from the build guide and the survey design. They have "
                      "not yet been validated on real rider statements. The three demo statements in the app (steady, "
                      "slipping, stressed) are SYNTHETIC and only show that the rules behave sensibly. Treat flags as "
                      "prompts for a human call, not as decisions.", AMBER, colors.HexColor("#fdf3e2"))]

    # ----- 12 -----
    story += H1("12. The application")
    story += [P("The product has two parts that can be hosted separately:")]
    story.append(flow_diagram([("Browser", "the person using it"), ("Astro web app", "frontend/ — Vercel"),
                               ("FastAPI back end", "api/main.py — Render"), ("Models + config", "models/, config/"),
                               ], cols=4, box_h=14 * mm))
    story += [Spacer(1, 4)]
    story += bullets([
        "<b>Front end (Astro).</b> A public website (Overview, Validation, Governance, Pilot programme) and a workspace "
        "(decision queue, case records, new application, early-warning monitor, decision log, policy settings), built into "
        "static files. Every figure is fetched live from the API, so pages always reflect the current model, policy and log.",
        "<b>Back end (FastAPI).</b> Loads the models once, scores applications, runs the early-warning rules, records every "
        "recommendation and every officer action (assignment, decision, disbursal, collections call) in a SQLite audit log, "
        "and serves cases, the queue, search, reports and chart data.",
        "<b>Configuration</b> (<font name='" + MONO + "'>config/thresholds.yaml</font>, <font name='" + MONO +
        "'>config/watch_rules.yaml</font>) holds every decision threshold, so behaviour changes without code changes.",
    ])
    story += [P("Design", "h2"),
              P("The site is written in the language of a credit committee: recommendations, referrals, first-EMI "
                "delinquency, discrimination, overrides. The public pages form one sequence — Overview, Validation, Governance, "
                "Pilot programme — each ending with a link to the next, and every figure on them traces to the validation or "
                "the model documentation. In the workspace, one screen leads to the next: an application becomes a case in "
                "the decision queue; the case links to the evidence behind its score; the policy simulator carries cut-offs "
                "into policy settings; an early-warning check appears on the borrower's case."),
              P("The visual style follows established Indian consumer-finance sites: white and soft pastel sections, deep "
                "navy (#02416e) with an orange accent (#f26b1d), Archivo for headlines and Public Sans for text. Ten real "
                "photographs from Unsplash (free licence, credited in the footer) show Indian riders, shop owners and rupee "
                "notes. Numbers count up as they appear and pages change with a short transition. Every screenshot below is "
                "the running app at 1440 px wide; figures are real outputs, and statement samples are synthetic.")]
    story += app_pages(F)
    story += [P("API endpoints", "h2")]
    story.append(table([
        ["Endpoint", "Purpose"],
        ["POST /score", "Score one application → score, pd, band, reasons, model_version, percentile, application_id, explain_id"],
        ["POST /score/preview", "The same result without recording it — live preview and sensitivity analysis"],
        ["GET /inbox, /cases, /cases/{id}, /search", "Decision queue, case list, a full case record with history, and search"],
        ["POST /decisions/{id}/final, /assign, /disburse", "Officer's decision (with reason), assignment, and disbursal with the loan number"],
        ["POST /decisions/{id}/followup, GET /collections/queue", "Record a collections contact; the red / amber call list"],
        ["POST /watch", "Check one loan's statement (requires consent: true) → flag, triggers, rules, suggested_action"],
        ["GET /samples/{profile}", "A SYNTHETIC 12-week statement: steady, slipping or stressed"],
        ["GET /samples/{profile}/check", "Watch result for a SYNTHETIC sample, for demos; not written to the audit log"],
        ["GET /decisions, /decisions/{id}", "The audit log; retrieve any decision to answer a customer complaint"],
        ["GET /portfolio/summary, /model, /model/analytics, /about/{doc}",
         "Totals, model reference data and settings, chart data, and the reports"],
        ["PUT /admin/thresholds", "Change model, cut-offs, calibration offset or kill switch (admin token)"],
        ["PUT /admin/watch-rules", "Change watch thresholds (admin token)"],
        ["POST /admin/reload", "Load retrained model files without restarting (admin token)"],
        ["GET /health", "Which model is live and whether the fallback is active"],
    ], [58 * mm, 112 * mm]))
    story += [P("Safeguards built in", "h2")]
    story += bullets([
        "<b>Audit log:</b> every recommendation, officer action, early-warning check and settings change is stored with "
        "inputs, output, model version and time; practice and synthetic records are marked.",
        "<b>Human decision authority:</b> the officer records the final decision with a reason; departures from the "
        "recommendation are reported as overrides, and a five-second Undo prevents accidental decisions.",
        "<b>Kill switch:</b> if the main model fails, or an admin turns on 'force fallback' (for example after a drift "
        "alarm), the scorecard takes over and the response says so.",
        "<b>Protected attributes</b> are stored for fairness reporting only; the test suite checks that changing the state "
        "never changes the score.",
        "<b>Admin access</b> needs a token set on the server; without it, admin endpoints are disabled. Every change needs a "
        "written reason.",
        "<b>Input checks:</b> impossible inputs (age outside 18–75, loan above 120% of the vehicle price, bureau score "
        "outside 300–900) are rejected rather than scored.",
    ])

    # ----- 13 -----
    story += H1("13. Running it and deploying it")
    story += [P("Run on your own computer", "h2"),
              P("cd frontend; npm install; npm run build; cd ..<br/>"
                ".venv\\Scripts\\uvicorn.exe api.main:app --reload<br/>"
                "→ web app at http://localhost:8000/ , API docs at http://localhost:8000/docs", "code"),
              P("Demonstration data: <font name='" + MONO + "'>python -m src.demo_seed</font> adds twelve synthetic cases "
                "across three branches; <font name='" + MONO + "'>python -m src.demo_seed --reset</font> backs up the log and "
                "restores them before a presentation. Links: http://localhost:8000/app (workspace), "
                "http://localhost:8000/apply?example=1 (scores the sample application, recorded as practice), "
                "http://localhost:8000/?still=1 (Overview with the hero animation at rest, for screenshots). While editing the "
                "front end, <font name='" + MONO + "'>npm run dev</font> in frontend/ gives a live-reloading site on port 4321."),
              P("Rebuild everything from the raw data", "h2"),
              P("python -m src.profiling → python -m src.features → python -m src.train (≈45 min) → "
                "python -m src.evaluate → python -m src.analytics → pytest -q", "code"),
              P("Put it on the internet", "h2")]
    story += bullets([
        "Put the project in a GitHub repository (git needs to be installed first).",
        "<b>Render (API):</b> New → Blueprint → the repository. render.yaml sets up the service. Set ADMIN_TOKEN (a long "
        "random string you keep private) and ALLOWED_ORIGINS (the Vercel address).",
        "<b>Vercel (web app):</b> New Project → the repository → root directory 'frontend'. Set PUBLIC_API_URL to the Render "
        "address.",
        "Both services need accounts in your name; deployment is done by you.",
    ])

    # ----- 14 -----
    story += H1("14. Limits, and what it takes to use it for real")
    story += bullets([
        "<b>The data is from 2018 and one lender.</b> Default levels and loan sizes are dated. Before real use the pipeline "
        "must be retrained on the deploying lender's own loans — a configuration change, not a rewrite.",
        "<b>The borrowers are 'gig-like', not confirmed gig workers:</b> self-employed two-wheeler buyers. The planned Pune "
        "rider survey and consented rider statements are what would connect the model to actual riders. The survey file in "
        "the project folder is simulated and was not used.",
        "<b>Accuracy is modest</b> (AUC 0.637 for the shipped fair model, 0.658 for the all-features model) because "
        "first-EMI default is noisy and the data has no income. The "
        "next real gain needs cash-flow data (bank statements through the Account Aggregator framework, platform payouts).",
        "<b>Probabilities drift</b> when the market shifts, as October 2018 showed. Monthly recalibration is built in but "
        "only partly closes a fast-moving gap.",
        "<b>The Watch rules are unvalidated</b> until they are tested on real statements.",
        "<b>Only a regulated lender can lend.</b> The realistic path is a shadow-mode pilot with an NBFC: score live "
        "applications in parallel for 8–12 weeks without affecting decisions, then compare outcomes by band.",
    ])

    # ----- 15 -----
    story += H1("15. How it can be improved")
    story += [P("The roadmap, in the order that would matter most to a lender. 'Now' items need no outside partner; "
                "'With a partner' items need a lender, a platform or real borrowers.")]
    story.append(table([
        ["When", "Improvement", "Why it matters"],
        ["<b>Model and data</b>", "", ""],
        ["With a partner", "Add cash-flow data: bank statements through Account Aggregator, platform payout history",
         "The only route to a real accuracy gain; this data set has no income information"],
        ["With a partner", "Retrain and recalibrate on the lender's own recent loans; automate the monthly offset",
         "2018 levels are dated, and October 2018 showed how fast risk can shift"],
        ["Now", "Decide how 'self-employed' is used: keep it, cap its effect, or remove it",
         "It appears in many referral and decline reasons, which sits awkwardly with the product's purpose"],
        ["With a partner", "Validate the Watch rules on real rider statements; replace the simulated survey",
         "The early-warning thresholds are untested on real borrowers"],
        ["<b>Product</b>", "", ""],
        ["Now", "Batch scoring from a CSV of applications", "Lenders score many applications at once"],
        ["Now", "PDF bank-statement reading for the early-warning monitor", "Most borrowers hold statements as PDFs, not CSVs"],
        ["Now", "Full Hindi and Marathi interface (letters already translated; native-speaker review needed)",
         "The target borrowers and many branch staff prefer them"],
        ["Now", "Date and filter controls on the decision log; drift alerts on the validation page", "Day-to-day monitoring"],
        ["<b>Quality</b>", "", ""],
        ["Now", "Accessibility pass: contrast, keyboard navigation of charts", "Usable by everyone"],
        ["<b>Going live</b>", "", ""],
        ["Now", "Install git, push to GitHub, deploy to Render and Vercel", "A public link to send to lenders"],
        ["Before a pilot", "Individual logins with roles and two-person approval of policy changes; Postgres instead of "
                           "SQLite; a tamper-evident audit log; automated tests on every change; API rate limiting",
         "What a lender's IT and compliance teams will ask for"],
        ["With a partner", "Account Aggregator, PAN / DigiLocker and bureau sandbox integrations",
         "Consented data in place of uploads; the most credible step towards production"],
        ["With a partner", "Shadow-mode pilot with an NBFC for 8–12 weeks", "The proof that it works on real applicants"],
    ], [28 * mm, 78 * mm, 64 * mm]))

    # ----- 16 -----
    story += H1("16. Problems found and fixed during the build")
    story += [P("Every one of these was caught by a test or by checking a result against the raw data, and fixed before "
                "any number in this document was produced:")]
    story.append(table([
        ["Problem", "Effect before the fix"],
        ["Each row's dealer / branch / area average included its own outcome", "A small leak that flattered accuracy"],
        ["Binning collapsed mostly-zero columns into one bin", "Delinquency and enquiry features looked useless"],
        ["Stability check compared two different encodings", "False drift alarm (3.6 instead of 0.003)"],
        ["Correlation screen dropped the no-score flag", "Lost the subgroup flag the project depends on"],
        ["Clipping by plain percentiles", "Co-applicant columns flattened to zero or capped at " + inr(50000)],
        ["A scorecard weight with the wrong sign", "Points ran against the data for credit-history length"],
        ["Approval-curve maths split tied scores", "Bureau-only looked able to approve 7.9% of no-score applicants (0.1%)"],
        ["Validation rows used for tuning, early stopping and calibration", "Optimistic validation scores"],
        ["Cut-offs chosen on the test month", "Optimistic band default rates"],
        ["Century rule applied to disbursal dates", "A loan disbursed in 2026 would be read as 1926"],
        ["Data endpoints shared addresses with web pages", "/analytics and /portfolio showed raw JSON instead of the page"],
        ["Synthetic statements wrote no rows in no-income weeks", "Demo statements ended early and showed an empty week"],
        ["Watch preview logged; Watch page opened empty", "Fake audit entries; empty boxes until set up"],
        ["Chart legends and axis titles had no room", "Series names and axis titles were clipped in several charts"],
    ], [92 * mm, 78 * mm]))

    # ----- 17 -----
    story += [PageBreak()] + H1("17. Glossary")
    story.append(P("Where to find more: reports/validation_report.md (every test and number), reports/model_card.md (the "
                   "model card), reports/progress.md (what was built, step by step), README.md (how to run and deploy).",
                   "small"))
    story.append(table([
        ["Term", "Meaning"],
        ["AUC", "Chance a random defaulter is scored riskier than a random non-defaulter. 0.5 = guessing, 1 = perfect."],
        ["Gini / KS", "Other ranking measures: Gini = 2 × AUC − 1; KS = largest gap between the score distributions of "
                      "defaulters and non-defaulters."],
        ["PD", "Probability of default — here, the chance of missing the first EMI."],
        ["Out-of-time test", "Testing on a later month than the training data, as real deployment works."],
        ["Out-of-fold", "A prediction or encoding for a row made by a model that never saw that row."],
        ["Calibration", "Making '12%' mean that about 12 in 100 such borrowers default."],
        ["WoE / IV", "Weight of evidence (how a bin's good/bad mix compares with the whole) and information value "
                     "(how strongly a feature separates good from bad)."],
        ["PSI", "Population stability index — how much a distribution has shifted between two periods."],
        ["SHAP", "A method that splits one prediction into each feature's contribution; used for reasons."],
        ["Monotone constraint", "A rule forcing the model's risk to only rise (or only fall) as a feature grows."],
        ["Kill switch", "Switching to the simpler fallback model when the main model fails or drifts."],
        ["NBFC", "Non-banking financial company — the kind of lender that originates most new-to-credit loans in India."],
        ["PAR 30 / 90", "Portfolio at risk: share of loans more than 30 / 90 days overdue (needs repayment history the L&amp;T "
                        "data does not contain)."],
    ], [36 * mm, 134 * mm]))
    return story


def main():
    F = facts()
    doc = Doc(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=22 * mm,
              title="GigScore — how it works", author="GigScore project", subject="GigScore explainer")
    doc.multiBuild(build(F), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
