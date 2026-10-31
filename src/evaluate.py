"""Validation of the step 3 models on the October 2018 out-of-time test (build guide Section 7).

Discrimination with bootstrap CIs, random-split comparison, calibration, score and feature
stability, the approval-rate vs bad-rate curve, proposed cut-offs, plus the explainability
(src/explain.py) and fairness (src/fairness.py) sections.

Run:  python -m src.evaluate [--boot 500]
Writes reports/validation_report.md, config/thresholds.yaml, reports/figures/*.png and the
evaluation section of reports/model_card.md.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import explain, fairness  # noqa: E402
from src.encode import apply_encoders, fit_encoders, fold_ids, psi  # noqa: E402
from src.features import build_features  # noqa: E402
from src.load import TARGET, load  # noqa: E402
from src.metrics import bootstrap, diff_ci  # noqa: E402
from src.predict import apply_offset, featurize_raw_file, load_bundle, predict_pd  # noqa: E402
from src.scorecard import BASE_ODDS, BASE_SCORE, PDO, SCORE_MAX, SCORE_MIN, to_points  # noqa: E402
from src.train import FEATURE_LIST, FEATURES, GROUPS, MONOTONE, SETS  # noqa: E402

FIG = Path("reports/figures")
THRESHOLDS = Path("config/thresholds.yaml")
OOF = "data/processed/oof_predictions.parquet"
# Proposed policy (guide Section 6.4): approve the safest 60%, refer the next 20%, decline the rest.
APPROVE_SHARE, REFER_SHARE = 0.60, 0.20
# Shipped model and fallback (src/train.py PRODUCTION_SET), the higher-accuracy alternative, and the
# bureau-only baseline the gain is measured against.
MAIN, FALLBACK, ALT, BUREAU = "lgbm_fair", "scorecard_fair", "lgbm_combined", "lgbm_bureau"
ALT_FALLBACK = "scorecard_combined"  # fallback when a lender switches to ALT (cut-offs needed for both)
CAL_TOL = 0.02          # guide: predicted within ±2 pp of observed per decile
PSI_STABLE, PSI_INVESTIGATE = 0.10, 0.25


# --- business view --------------------------------------------------------------------------

def _tie_blocks(y: np.ndarray, p: np.ndarray):
    """Sort safest first and group equal PDs: a lender cannot choose between tied applicants."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    start = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    size = np.diff(np.r_[start, len(ps)])
    bads = np.add.reduceat(ys, start)
    return start, size, bads


def expected_bads(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Expected bads among the k safest (k = 1..n), drawing at random within a tie block."""
    start, size, bads = _tie_blocks(y, p)
    before = np.r_[0, np.cumsum(bads)[:-1]]
    block = np.repeat(np.arange(len(start)), size)
    k = np.arange(1, len(y) + 1)
    return before[block] + (k - start[block]) * bads[block] / size[block]


def approval_curve(y: np.ndarray, p: np.ndarray, rates=None) -> pd.DataFrame:
    """Approve the safest share of applicants (lowest PD first); bad rate among those approved."""
    rates = np.linspace(0.05, 1.0, 96) if rates is None else np.asarray(rates)
    cum = expected_bads(y, p)
    k = np.maximum((rates * len(y)).round().astype(int), 1)
    return pd.DataFrame({"approval_rate": k / len(y), "bad_rate": cum[k - 1] / k, "approved": k,
                         "bads": cum[k - 1]})


def approval_at_bad_rate(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Largest approval rate whose bad rate among the approved stays at or below `target`.
    Cut-offs fall only between distinct PD values."""
    start, size, bads = _tie_blocks(y, p)
    k = start + size
    ok = np.flatnonzero(np.cumsum(bads) / k <= target)
    return float(k[ok.max()] / len(y)) if len(ok) else 0.0


def cutoff_for_share(scores: np.ndarray, share: float) -> int:
    """Score cut-off s (approve if score >= s) whose approved share is closest to `share`."""
    cand = np.unique(scores)
    share_ge = np.array([(scores >= s).mean() for s in cand])
    return int(cand[np.argmin(np.abs(share_ge - share))])


def assign_bands(scores: np.ndarray, approve: int, refer: int) -> np.ndarray:
    return np.where(scores >= approve, "approve", np.where(scores >= refer, "refer", "decline"))


def band_table(y: np.ndarray, bands: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame({"band": bands, "y": y}).groupby("band")["y"].agg(rows="size", bad_rate="mean")
    t["share"] = t["rows"] / len(y)
    return t.reindex(["approve", "refer", "decline"])[["rows", "share", "bad_rate"]]


# --- calibration and stability --------------------------------------------------------------

def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Deciles of predicted PD: mean predicted vs observed default rate."""
    q = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    t = pd.DataFrame({"bin": q, "p": p, "y": y}).groupby("bin").agg(rows=("y", "size"), mean_pd=("p", "mean"),
                                                                    observed=("y", "mean"))
    t["diff_pp"] = (t["mean_pd"] - t["observed"]) * 100
    return t


def fit_offset(p: np.ndarray, y: np.ndarray) -> float:
    """Log-odds shift that makes the mean predicted PD equal the observed default rate."""
    return float(brentq(lambda d: apply_offset(p, d).mean() - y.mean(), -5, 5))


def recalibration_backtest(oot: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    """The monthly procedure a lender would run: first-EMI outcomes for loans disbursed 1-15 Oct are known
    by mid-November; fit the offset on them and apply it to loans disbursed 16-31 Oct."""
    first = (oot["disb"].dt.day <= 15).to_numpy()
    d = fit_offset(p[first], y[first])
    p2, y2 = p[~first], y[~first]
    before, after = reliability(y2, p2), reliability(y2, apply_offset(p2, d))
    return {"offset": d, "n_fit": int(first.sum()), "n_test": int((~first).sum()), "observed": float(y2.mean()),
            "mean_before": float(p2.mean()), "mean_after": float(apply_offset(p2, d).mean()),
            "deciles_off_before": int((before["diff_pp"].abs() > CAL_TOL * 100).sum()),
            "deciles_off_after": int((after["diff_pp"].abs() > CAL_TOL * 100).sum()),
            "max_dev_before": float(before["diff_pp"].abs().max()), "max_dev_after": float(after["diff_pp"].abs().max()),
            "auc_unchanged": bool(abs(roc_auc_score(y2, p2) - roc_auc_score(y2, apply_offset(p2, d))) < 1e-12)}


def score_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1))[1:-1])
    lab = lambda s: pd.Series(np.searchsorted(edges, s, side="right"))  # noqa: E731
    return psi(lab(expected), lab(actual))


# --- random split (guide Section 4.4: report it, but the headline is out-of-time) -------------

def random_split_auc(bundles: dict, seed: int = 42) -> pd.DataFrame:
    df = load()
    f = build_features(df)
    tr, te = train_test_split(df.index, test_size=0.2, random_state=seed, stratify=df[TARGET])
    te_tr, enc = fit_encoders(df.loc[tr], df.loc[tr, TARGET], fold_ids(len(tr)))
    X = f.join(pd.concat([te_tr, apply_encoders(df.loc[te], enc)]))
    rows = []
    for name in (BUREAU, ALT, MAIN):
        b = bundles[name]
        feats = b["features"]
        params = {**b["params"], "monotone_constraints": [MONOTONE.get(c, 0) for c in feats]}
        m = lgb.train(params, lgb.Dataset(X.loc[tr, feats], df.loc[tr, TARGET]), b["best_iteration"])
        rows.append({"model": name, "random_split_auc": roc_auc_score(df.loc[te, TARGET], m.predict(X.loc[te, feats]))})
    return pd.DataFrame(rows).set_index("model")


# --- figures --------------------------------------------------------------------------------

def plot_curves(curves: dict, base_rate: float, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        ax.plot(c["approval_rate"] * 100, c["bad_rate"] * 100, label=name)
    ax.axhline(base_rate * 100, color="grey", ls="--", lw=1, label=f"approve everyone ({base_rate:.1%})")
    ax.axvline(APPROVE_SHARE * 100, color="black", ls=":", lw=1)
    ax.set_xlabel("Approval rate (%) — safest applicants first")
    ax.set_ylabel("First-EMI default rate among approved (%)")
    ax.set_title("Approval rate vs bad rate, October 2018 (out-of-time)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(tables: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hi = max(max(t["mean_pd"].max(), t["observed"].max()) for t in tables.values()) * 100 + 2
    ax.plot([0, hi], [0, hi], color="grey", ls="--", lw=1)
    for name, t in tables.items():
        ax.plot(t["mean_pd"] * 100, t["observed"] * 100, marker="o", label=name)
    ax.set_xlabel("Mean predicted PD (%)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Reliability by decile, October 2018")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- main -----------------------------------------------------------------------------------

def main(n_boot: int = 500) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(FEATURES)
    fl = json.loads(Path(FEATURE_LIST).read_text())
    oot = t[t["split"] == "oot"]
    y = oot[TARGET].to_numpy()

    bundles = {f"{k}_{s}": load_bundle(f"models/{'lr_scorecard' if k == 'scorecard' else 'lgbm'}_{s}.pkl")
               for s in SETS for k in ("scorecard", "lgbm")}
    preds = {m: predict_pd(b, oot) for m, b in bundles.items()}
    raw_main = predict_pd(bundles[MAIN], oot, raw=True)
    # Aug-Sep out-of-fold predictions (src/train.py): the reference for cut-offs, PSI and calibration
    oof = pd.read_parquet(OOF)
    va_preds = {m: oof[f"pd_{m}"].to_numpy() for m in (MAIN, FALLBACK, ALT, ALT_FALLBACK)}
    y_va = oof[TARGET].to_numpy()

    groups = {"all": np.ones(len(oot), bool), "no_score": (oot["no_score"] == 1).to_numpy(),
              "no_usable_score": (oot["no_usable_score"] == 1).to_numpy(),
              "self_employed": (oot["employment_self"] == 1).to_numpy()}
    res, draws = bootstrap(y, preds, groups, n_boot=n_boot)
    lift_ns = (res.set_index(["model", "group"]).loc[(MAIN, "no_score"), "auc"]
               - res.set_index(["model", "group"]).loc[(BUREAU, "no_score"), "auc"],
               *diff_ci(draws, (MAIN, "no_score"), (BUREAU, "no_score")))
    print("discrimination done")

    rnd = random_split_auc(bundles)
    rnd["oot_auc"] = [res.set_index(["model", "group"]).loc[(m, "all"), "auc"] for m in rnd.index]
    print("random split done")

    # calibration
    rel = {f"{MAIN} (isotonic)": reliability(y, preds[MAIN]),
           f"{MAIN} (raw, before isotonic)": reliability(y, raw_main),
           FALLBACK: reliability(y, preds[FALLBACK])}
    rel_va = reliability(y_va, va_preds[MAIN])
    brier = {f"{MAIN} raw": brier_score_loss(y, raw_main),
             f"{MAIN} isotonic": brier_score_loss(y, preds[MAIN]),
             FALLBACK: brier_score_loss(y, preds[FALLBACK]),
             f"{ALT} isotonic": brier_score_loss(y, preds[ALT]),
             "constant (Aug–Sep default rate)": brier_score_loss(y, np.full(len(y), t.loc[t.split != "oot", TARGET].mean()))}
    plot_calibration({k: v for k, v in rel.items() if "raw" not in k}, FIG / "calibration.png")
    cal_shift = fit_offset(preds[MAIN], y)
    recal = recalibration_backtest(oot, y, preds[MAIN])

    # stability: score PSI (Aug–Sep validation rows as reference) and feature PSI from step 2
    nov = featurize_raw_file()
    shipped = (MAIN, FALLBACK, ALT, ALT_FALLBACK)
    nov_preds = {m: predict_pd(bundles[m], nov) for m in shipped}
    s_va = {m: to_points(va_preds[m]) for m in shipped}
    s_oot = {m: to_points(preds[m]) for m in shipped}
    s_nov = {m: to_points(nov_preds[m]) for m in shipped}
    psi_scores = pd.DataFrame({m: {"Oct vs Aug–Sep": score_psi(s_va[m], s_oot[m]),
                                   "Nov vs Aug–Sep": score_psi(s_va[m], s_nov[m])} for m in shipped}).T
    fstats = pd.read_csv("data/processed/feature_stats.csv", index_col="feature")
    fstats = fstats.loc[[f for f in fl["sets"]["combined"] if f in fstats.index]]
    psi_watch = fstats[(fstats["psi_oct"] > PSI_STABLE) | (fstats["psi_nov"] > PSI_STABLE)][["psi_oct", "psi_nov"]]
    print("stability done")

    # business: approval vs bad rate, extra approvals at the same bad rate
    curves = {m: approval_curve(y, preds[m]) for m in (MAIN, ALT, FALLBACK, BUREAU)}
    plot_curves(curves, y.mean(), FIG / "approval_curve.png")
    ns = groups["no_score"]
    curves_ns = {m: approval_curve(y[ns], preds[m][ns]) for m in (MAIN, ALT, BUREAU)}
    plot_curves(curves_ns, y[ns].mean(), FIG / "approval_curve_no_score.png")
    curve_rows = []
    for a in (0.5, 0.6, 0.7, 0.8, 0.9):
        row = {"approval_rate": a}
        for m in (MAIN, ALT, FALLBACK, BUREAU):
            row[m] = float(approval_curve(y, preds[m], [a])["bad_rate"].iloc[0])
        curve_rows.append(row)
    curve_tab = pd.DataFrame(curve_rows)
    extra = []
    for a in (0.5, 0.6, 0.7):
        target = float(approval_curve(y, preds[BUREAU], [a])["bad_rate"].iloc[0])
        row = {"bureau_only_approval": a, "bad_rate": target}
        for m in (MAIN, ALT):
            row[m] = approval_at_bad_rate(y, preds[m], target)
            row[f"extra_{m}"] = int(round((row[m] - a) * len(y)))
        extra.append(row)
    extra = pd.DataFrame(extra)
    train_rate = float(t.loc[t.split != "oot", TARGET].mean())
    ns_at = {m: approval_at_bad_rate(y[ns], preds[m][ns], train_rate) for m in (MAIN, ALT, BUREAU)}

    # cut-offs: approve the safest 60%, refer the next 20%, chosen on Aug-Sep out-of-fold scores so that
    # October (where they are evaluated below) plays no part in choosing them
    cuts = {}
    for m in shipped:
        sc = s_va[m]
        cuts[m] = {"approve": cutoff_for_share(sc, APPROVE_SHARE),
                   "refer": cutoff_for_share(sc, APPROVE_SHARE + REFER_SHARE)}
    bands = assign_bands(s_oot[MAIN], **cuts[MAIN])
    bands_fb = assign_bands(s_oot[FALLBACK], **cuts[FALLBACK])
    bt = {"all": band_table(y, bands), "no_score": band_table(y[ns], bands[ns]),
          "self_employed": band_table(y[groups["self_employed"]], bands[groups["self_employed"]]),
          "fallback (scorecard)": band_table(y, bands_fb),
          f"alternative ({ALT})": band_table(y, assign_bands(s_oot[ALT], **cuts[ALT]))}
    bands_va = assign_bands(s_va[MAIN], **cuts[MAIN])
    bt_va = band_table(y_va, bands_va)
    write_thresholds(cuts, bt["all"], bt_va)
    print("business + cut-offs done")

    ex = explain.run(bundles[MAIN], oot, bands)
    print("explainability done")
    fa = fairness.run(oot, y, preds[MAIN], bands, bundles[MAIN], t, cuts[MAIN],
                      base_bundle=bundles[ALT], base_p=preds[ALT])
    print("fairness done")

    out = dict(res=res, lift_ns=lift_ns, rnd=rnd, rel=rel, rel_va=rel_va, brier=brier, cal_shift=cal_shift,
               recal=recal,
               psi_scores=psi_scores, psi_watch=psi_watch, curve_tab=curve_tab, extra=extra, ns_at=ns_at,
               train_rate=train_rate, cuts=cuts, bt=bt, bt_va=bt_va, ex=ex, fa=fa, n_oot=len(oot),
               n_nov=len(nov), base_rate=float(y.mean()), n_boot=n_boot)
    Path("reports/validation_report.md").write_text(validation_md(out), encoding="utf-8")
    update_model_card(out)
    write_app_reference(out, s_oot[MAIN], y, bundles)
    return out


APP_REFERENCE = Path("models/app_reference.json")


def write_app_reference(o: dict, scores: np.ndarray, y: np.ndarray, bundles: dict):
    """Small, real (October out-of-time) reference the web app and API show; no raw data needed."""
    ri = o["res"].set_index(["model", "group"])
    order = np.argsort(-scores, kind="stable")  # highest score (safest) first
    cum_bad = np.cumsum(y[order])
    curve = []
    for a in np.round(np.linspace(0.05, 1.0, 20), 2):
        k = int(round(a * len(y)))
        curve.append({"approval_rate": float(a), "bad_rate": float(cum_bad[k - 1] / k),
                      "score_cutoff": int(scores[order][k - 1])})
    ref = {
        "generated_by": f"src/evaluate.py on {date.today()}",
        "data": "L&T Vehicle Loan Default, October 2018 out-of-time test (real data)",
        "model": MAIN, "fallback_model": FALLBACK, "alternative_model": ALT,
        "versions": {m: bundles[m]["version"] for m in (MAIN, FALLBACK, ALT)},
        "n_test": int(len(y)), "test_default_rate": float(y.mean()),
        "score_quantiles": [int(v) for v in np.quantile(scores, np.linspace(0, 1, 201))],
        "curve": curve,
        "bands": o["cuts"],
        "band_table": {b: {"share": float(r["share"]), "bad_rate": float(r["bad_rate"])}
                       for b, r in o["bt"]["all"].iterrows()},
        "metrics": {m: {g: {"auc": float(ri.loc[(m, g), "auc"]), "auc_lo": float(ri.loc[(m, g), "auc_lo"]),
                            "auc_hi": float(ri.loc[(m, g), "auc_hi"])} for g in ("all", "no_score", "self_employed")}
                    for m in (MAIN, ALT, FALLBACK, BUREAU)},
        "fairness_options": [{"option": k.strip("* "), **{c: float(v) for c, v in r.items()}}
                             for k, r in o["fa"]["options"].iterrows()],
        "extra_approvals": [{"bureau_only_approval": float(r["bureau_only_approval"]), "bad_rate": float(r["bad_rate"]),
                             "model_approval": float(r[MAIN]), "extra": int(r[f"extra_{MAIN}"])}
                            for _, r in o["extra"].iterrows()],
        "no_score_approval_at_train_rate": {m: float(v) for m, v in o["ns_at"].items()},
        "train_default_rate": float(o["train_rate"]),
        "fairness_age_flagged": int(o["fa"]["tables"]["age_band"]["flag"].sum()),
        "fairness_state_flagged": int(o["fa"]["tables"]["State_ID"]["flag"].sum()),
        "fairness_n_states": int(len(o["fa"]["tables"]["State_ID"])),
    }
    APP_REFERENCE.write_text(json.dumps(ref, indent=1), encoding="utf-8")


def write_thresholds(cuts: dict, bt: pd.DataFrame, bt_chosen: pd.DataFrame | None = None):
    THRESHOLDS.parent.mkdir(exist_ok=True)
    cfg = {
        "model": MAIN,
        "fallback_model": FALLBACK,
        "alternative_model": ALT,
        "score_scale": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "points_to_double_odds": PDO,
                        "min": SCORE_MIN, "max": SCORE_MAX},
        "bands": {m: {"approve": c["approve"], "refer": c["refer"]} for m, c in cuts.items()},
        "policy": {"approve_share": APPROVE_SHARE, "refer_share": REFER_SHARE,
                   "chosen_on": "Aug-Sep 2018 out-of-fold scores (src/evaluate.py)",
                   "october_bad_rate": {b: round(float(r), 4) for b, r in bt["bad_rate"].items()},
                   "october_share": {b: round(float(r), 4) for b, r in bt["share"].items()},
                   **({"aug_sep_bad_rate": {b: round(float(r), 4) for b, r in bt_chosen["bad_rate"].items()}}
                      if bt_chosen is not None else {})},
        # Log-odds shift added to every PD (recalibration-in-the-large). Re-fit monthly on the latest
        # matured month of first-EMI outcomes; 0 = probabilities exactly as trained.
        "calibration_offset": 0.0,
        "kill_switch": {"psi_max": PSI_INVESTIGATE,
                        "action": "fall back to fallback_model, or route every application to manual review"},
    }
    header = (f"# Decision thresholds. PROPOSED by src/evaluate.py on {date.today()}; the lender chooses.\n"
              "# Score >= approve -> approve; >= refer -> refer; else decline. Edit here, never in code.\n")
    THRESHOLDS.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


# --- report ---------------------------------------------------------------------------------

def _pct(v):
    return f"{v:.1%}"


def _ci(r, k):
    return f"{r[k]:.3f} ({r[k + '_lo']:.3f}–{r[k + '_hi']:.3f})"


def validation_md(o: dict) -> str:
    res = o["res"]
    L = [
        "# Validation report — GigScore",
        "",
        f"Produced by `src/evaluate.py` (`python -m src.evaluate --boot {o['n_boot']}`), which calls "
        f"`src/explain.py` and `src/fairness.py`, on {date.today()}. Real L&T data. Models from `src/train.py` "
        "(see `reports/model_results.md`). Test set: **October 2018, out-of-time** "
        f"({o['n_oot']:,} loans, default rate {_pct(o['base_rate'])}); models were trained on Aug–Sep 2018.",
        "",
        "## 1. Tests run",
        "",
        f"| test | how | result |",
        "|:--|:--|:--|",
        f"| Discrimination | AUC, Gini, KS; {o['n_boot']} paired bootstrap resamples | Section 2 |",
        "| Random vs out-of-time split | same LightGBM settings refit on a random 80/20 split of all months | Section 2 |",
        "| Calibration | reliability by decile, Brier score, before / after isotonic | Section 3 |",
        "| Stability | PSI of the score (Oct, Nov vs Aug–Sep) and of every feature | Section 4 |",
        "| Business | approval-rate vs bad-rate curve; extra approvals at equal bad rate | Section 5 |",
        "| Cut-offs | approve safest 60%, refer next 20% of Aug–Sep; applied to October → `config/thresholds.yaml` | Section 6 |",
        "| Recalibration | monthly log-odds offset fitted on 1–15 Oct, tested on 16–31 Oct | Section 3 |",
        "| Explainability | TreeSHAP (LightGBM `pred_contrib`), reason codes for every applicant | Section 7 |",
        "| Fairness | approval and error rates by age band and state; proxy check; mitigation test | Section 8 |",
        "",
        "## 2. Discrimination (October, out-of-time)",
        "",
        "| model | all | no score | no usable score | self-employed |",
        "|:--|--:|--:|--:|--:|",
    ]
    ri = res.set_index(["model", "group"])
    for m in res["model"].unique():
        L.append(f"| {m} | " + " | ".join(_ci(ri.loc[(m, g)], "auc") for g in GROUPS) + " |")
    d, lo, hi = o["lift_ns"]
    L += ["", f"AUC with 95% CI. **No-score subgroup: {MAIN} vs {BUREAU} {d:+.3f} ({lo:+.3f} to {hi:+.3f}).** "
              "Gini and KS for every model and subgroup are in `reports/model_results.md`.", "",
          "| model | random-split AUC | out-of-time AUC |", "|:--|--:|--:|"]
    for m, r in o["rnd"].iterrows():
        L.append(f"| {m} | {r['random_split_auc']:.3f} | {r['oot_auc']:.3f} |")
    L += ["", "The random split mixes October into training, so it flatters the model; the out-of-time number is "
              "the one to quote.", "",
          "## 3. Calibration", "",
          f"Brier score on October (lower is better): " + "; ".join(f"{k} {v:.4f}" for k, v in o["brier"].items()) + ".",
          ""]
    for name, tab in o["rel"].items():
        bad = int((tab["diff_pp"].abs() > CAL_TOL * 100).sum())
        L += [f"**{name}** — {bad} of 10 deciles outside ±2 pp", "",
              "| decile | mean predicted | observed | diff (pp) |", "|--:|--:|--:|--:|"]
        L += [f"| {i + 1} | {_pct(r['mean_pd'])} | {_pct(r['observed'])} | {r['diff_pp']:+.1f} |"
              for i, r in tab.iterrows()]
        L.append("")
    bad_va = int((o["rel_va"]["diff_pp"].abs() > CAL_TOL * 100).sum())
    rc = o["recal"]
    L += [f"On Aug–Sep out-of-fold predictions (where isotonic was fitted) {bad_va} of 10 deciles are outside ±2 pp. "
          f"On October the model under-predicts because October's default rate ({_pct(o['base_rate'])}) is above "
          f"the training window's ({_pct(o['train_rate'])}); no model trained on Aug–Sep can know that in advance. "
          f"Hindsight offset for the whole of October: {o['cal_shift']:+.3f} log-odds.", "",
          "**Monthly recalibration (the deployment procedure, backtested).** First-EMI outcomes mature about a "
          f"month after disbursal, so by mid-November a lender knows them for loans disbursed 1–15 October "
          f"({rc['n_fit']:,} loans). Fitting the log-odds offset on those ({rc['offset']:+.3f}) and applying it to "
          f"loans disbursed 16–31 October ({rc['n_test']:,}):", "",
          "| 16–31 Oct | mean predicted PD | observed | deciles outside ±2 pp | worst decile gap |",
          "|:--|--:|--:|--:|--:|",
          f"| as trained | {_pct(rc['mean_before'])} | {_pct(rc['observed'])} | {rc['deciles_off_before']} of 10 | "
          f"{rc['max_dev_before']:.1f} pp |",
          f"| with offset | {_pct(rc['mean_after'])} | {_pct(rc['observed'])} | {rc['deciles_off_after']} of 10 | "
          f"{rc['max_dev_after']:.1f} pp |", "",
          f"Ranking is unchanged by the offset (AUC identical: {rc['auc_unchanged']}). The offset lives in "
          "`config/thresholds.yaml` (`calibration_offset`, 0 as shipped) so a lender updates it without code changes. "
          "Score cut-offs are on the score, which also moves with the offset; re-check band shares after updating it.",
          "", "![calibration](figures/calibration.png)", "",
          "## 4. Stability", "",
          "Score PSI (reference: Aug–Sep out-of-fold scores): " + "; ".join(
              f"{m}: Oct {r['Oct vs Aug–Sep']:.3f}, Nov {r['Nov vs Aug–Sep']:.3f}" for m, r in o["psi_scores"].iterrows())
          + f" (< {PSI_STABLE} stable, > {PSI_INVESTIGATE} investigate). November = test.csv "
          f"({o['n_nov']:,} unlabelled loans).", "",
          f"Model features with PSI above {PSI_STABLE} (from `data/processed/feature_stats.csv`, step 2):", "",
          "| feature | PSI Oct | PSI Nov |", "|:--|--:|--:|"]
    L += [f"| `{f}` | {r['psi_oct']:.3f} | {r['psi_nov']:.3f} |" for f, r in o["psi_watch"].iterrows()]
    L += ["", "## 5. Business view", "",
          "![approval curve](figures/approval_curve.png)", "",
          "Bad rate among approved applicants when approving the safest share (October):", "",
          f"| approval rate | {MAIN} (shipped) | {ALT} | {FALLBACK} | {BUREAU} |", "|--:|--:|--:|--:|--:|"]
    L += [f"| {_pct(r['approval_rate'])} | {_pct(r[MAIN])} | {_pct(r[ALT])} | {_pct(r[FALLBACK])} | "
          f"{_pct(r[BUREAU])} |" for _, r in o["curve_tab"].iterrows()]
    L += ["", f"Approving everyone gives {_pct(o['base_rate'])}.", "",
          f"Extra approvals (of {o['n_oot']:,}) at the same bad rate as the bureau-only model:", "",
          f"| bureau-only approves | its bad rate | {MAIN} approves | extra | {ALT} approves | extra |",
          "|--:|--:|--:|--:|--:|--:|"]
    L += [f"| {_pct(r['bureau_only_approval'])} | {_pct(r['bad_rate'])} | {_pct(r[MAIN])} | "
          f"{int(r[f'extra_{MAIN}']):+,} | {_pct(r[ALT])} | {int(r[f'extra_{ALT}']):+,} |"
          for _, r in o["extra"].iterrows()]
    L += ["", f"**No-score applicants:** holding the bad rate at or below the training window's "
              f"{_pct(o['train_rate'])}, {MAIN} can approve {_pct(o['ns_at'][MAIN])} of them, {ALT} "
              f"{_pct(o['ns_at'][ALT])}, the bureau-only model {_pct(o['ns_at'][BUREAU])}.", "",
          "![approval curve, no score](figures/approval_curve_no_score.png)", "",
          "## 6. Proposed cut-offs (`config/thresholds.yaml`)", "",
          f"Policy: approve the safest {APPROVE_SHARE:.0%}, refer the next {REFER_SHARE:.0%}, decline the rest. "
          "Cut-offs are chosen on Aug–Sep out-of-fold scores and then applied unchanged to October, so the October "
          "band rates below are a genuine out-of-time result (the guide's suggestion of choosing them on the test "
          "set would make them optimistic). The lender should pick its own point on the curve. "
          f"Score scale: {BASE_SCORE} points at {BASE_ODDS}:1 good:bad odds (PD 20%), {PDO} points to double the odds.",
          ""]
    for m, c in o["cuts"].items():
        L.append(f"- **{m}**: approve if score ≥ {c['approve']}, refer if ≥ {c['refer']}, else decline.")
    for name, tab in o["bt"].items():
        L += ["", f"**{name}** (October)", "", "| band | loans | share | default rate |", "|:--|--:|--:|--:|"]
        L += [f"| {b} | {int(r['rows']):,} | {_pct(r['share'])} | {_pct(r['bad_rate'])} |" for b, r in tab.iterrows()]
    L += ["", "Main-model bands on Aug–Sep out-of-fold scores, where the cut-offs were chosen:", "",
          "| band | loans | share | default rate |", "|:--|--:|--:|--:|"]
    L += [f"| {b} | {int(r['rows']):,} | {_pct(r['share'])} | {_pct(r['bad_rate'])} |" for b, r in o["bt_va"].iterrows()]
    L += ["", "## 7. Explainability", "", *o["ex"]["md"], "", "## 8. Fairness", "", *o["fa"]["md"], "",
          "## 9. Sign-off", "",
          "To be completed by an independent validator; this report is generated, not signed.", "",
          "| role | name | date | decision |", "|:--|:--|:--|:--|",
          "| Model owner | | | |", "| Independent validator | | | |", "| Risk committee | | | |", ""]
    return "\n".join(L)


CARD_START, CARD_END = "<!-- evaluation:start -->", "<!-- evaluation:end -->"


def update_model_card(o: dict, path: str = "reports/model_card.md"):
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    if CARD_START in text:
        text = text[:text.index(CARD_START)].rstrip() + "\n"
    c = o["cuts"][MAIN]
    d, lo, hi = o["lift_ns"]
    sec = [
        "", CARD_START,
        "## Evaluation, explainability and fairness (step 4)",
        "",
        f"From `src/evaluate.py` on {date.today()}; details in `reports/validation_report.md`.",
        "",
        f"- **Shipped model: {MAIN}** (fallback {FALLBACK}). It leaves out area, branch, dealer and ID-document "
        "features because they stand in for state and produced large state approval gaps in the all-features "
        f"model; the cost is some accuracy (Section 2 and 8 of the validation report). {ALT} is the documented "
        "higher-accuracy alternative; switching is a config change (`model` in `config/thresholds.yaml`) and a "
        "lender and compliance decision. Chosen 2026-10-29 for the project's purpose: fair access for "
        "credit-invisible borrowers.",
        f"- **Headline:** for October applicants with no bureau score, {MAIN} AUC exceeds bureau-only by "
        f"{d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}).",
        f"- **Calibration:** isotonic on Aug–Sep out-of-fold predictions; as trained, October is under-predicted "
        f"(observed {_pct(o['base_rate'])}). Monthly recalibration backtest (offset fitted on 1–15 Oct, applied to "
        f"16–31 Oct): deciles outside ±2 pp {o['recal']['deciles_off_before']} → {o['recal']['deciles_off_after']} of 10. "
        "Offset is `calibration_offset` in `config/thresholds.yaml`.",
        f"- **Stability:** score PSI Oct {o['psi_scores'].loc[MAIN, 'Oct vs Aug–Sep']:.3f}, "
        f"Nov {o['psi_scores'].loc[MAIN, 'Nov vs Aug–Sep']:.3f}.",
        f"- **Proposed bands** ({MAIN}): approve ≥ {c['approve']}, refer ≥ {c['refer']}; October default rate "
        + ", ".join(f"{b} {_pct(r)}" for b, r in o["bt"]["all"]["bad_rate"].items()) + ".",
        *[f"- **Explainability:** {s}" for s in o["ex"]["card"]],
        *[f"- **Fairness:** {s}" for s in o["fa"]["card"]],
        CARD_END, "",
    ]
    p.write_text(text + "\n".join(sec), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    out = main(ap.parse_args().boot)
    print(out["rnd"].round(3))
    print(pd.Series(out["brier"]).round(4))
    print(out["psi_scores"].round(3))
    print(out["curve_tab"].round(3))
    print(out["extra"])
    print("no-score approvals at train bad rate:", out["ns_at"])
    print(out["cuts"])
    for k, v in out["bt"].items():
        print(k, "\n", v.round(3))
