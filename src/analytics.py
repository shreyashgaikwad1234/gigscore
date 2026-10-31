"""Chart-ready analytics for the web app, computed from real predictions on the October 2018
out-of-time test (and Aug-Sep out-of-fold predictions for calibration). No raw data is shipped:
the output is aggregates only.

Run:  python -m src.analytics      (after src.train and src.evaluate; writes models/analytics.json)
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from src.encode import psi
from src.explain import shap_values
from src.load import TARGET
from src.predict import load_bundle, predict_pd
from src.scorecard import to_points
from src.train import FEATURE_LIST, FEATURES

OUT = Path("models/analytics.json")
MODELS = {"lgbm_fair": "models/lgbm_fair.pkl", "lgbm_combined": "models/lgbm_combined.pkl",
          "scorecard_fair": "models/lr_scorecard_fair.pkl", "lgbm_bureau": "models/lgbm_bureau.pkl"}
LABELS = {"lgbm_fair": "GigScore (shipped)", "lgbm_combined": "All features", "scorecard_fair": "Scorecard",
          "lgbm_bureau": "Bureau only"}


def _r(v, d=4):
    return round(float(v), d)


def roc_points(y, p, n=120):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return {"fpr": [_r(v) for v in fpr[idx]], "tpr": [_r(v) for v in tpr[idx]], "auc": _r(roc_auc_score(y, p))}


def gains(y, p, steps=np.linspace(0, 1, 51)):
    """Share of all defaulters caught when declining the riskiest x% of applicants."""
    ys = y[np.argsort(-p, kind="stable")]
    cum = np.r_[0, np.cumsum(ys)] / ys.sum()
    return [_r(cum[int(round(s * len(ys)))]) for s in steps]


def approval_curve(y, p, steps=np.linspace(0.05, 1, 40)):
    ys = y[np.argsort(p, kind="stable")]
    cum = np.cumsum(ys)
    out = []
    for a in steps:
        k = max(1, int(round(a * len(ys))))
        out.append(_r(cum[k - 1] / k))
    return out


def main() -> dict:
    t = pd.read_parquet(FEATURES)
    fl = json.loads(Path(FEATURE_LIST).read_text())
    cfg = yaml.safe_load(Path("config/thresholds.yaml").read_text(encoding="utf-8"))
    main_name = cfg["model"]
    oot = t[t["split"] == "oot"]
    y = oot[TARGET].to_numpy()
    bundles = {m: load_bundle(p) for m, p in MODELS.items()}
    preds = {m: predict_pd(b, oot) for m, b in bundles.items()}
    p = preds[main_name]
    score = to_points(p)
    cuts = cfg["bands"][main_name]
    band = np.where(score >= cuts["approve"], "approve", np.where(score >= cuts["refer"], "refer", "decline"))
    ns = (oot["no_score"] == 1).to_numpy()

    out = {"generated_by": f"src/analytics.py on {date.today()}", "data": "L&T 2018, October out-of-time test (real)",
           "model": main_name, "labels": LABELS, "n_test": int(len(y)), "base_rate": _r(y.mean()), "cuts": cuts}

    # ROC (all / no-score), gains, approval curves
    out["roc"] = {m: roc_points(y, q) for m, q in preds.items()}
    out["roc_no_score"] = {m: roc_points(y[ns], q[ns]) for m, q in preds.items()}
    steps = np.linspace(0, 1, 51)
    out["gains"] = {"x": [_r(s) for s in steps], **{m: gains(y, q) for m, q in preds.items()}}
    a_steps = np.linspace(0.05, 1, 40)
    out["approval"] = {"x": [_r(s) for s in a_steps], **{m: approval_curve(y, q, a_steps) for m, q in preds.items()}}

    # score distribution of repaid vs missed-first-EMI borrowers
    edges = np.arange(420, 781, 10)
    sc = np.clip(score, edges[0], edges[-1] - 1)
    good, _ = np.histogram(sc[y == 0], edges)
    bad, _ = np.histogram(sc[y == 1], edges)
    out["score_hist"] = {"bins": [int(e) for e in edges[:-1]], "repaid": good.tolist(), "missed": bad.tolist()}

    # risk by score decile (decile 1 = riskiest)
    d = pd.DataFrame({"score": score, "pd": p, "y": y})
    d["decile"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1
    dec = d.groupby("decile").agg(n=("y", "size"), default_rate=("y", "mean"), mean_pd=("pd", "mean"),
                                  lo=("score", "min"), hi=("score", "max"))
    out["deciles"] = [{"decile": int(i), "n": int(r.n), "default_rate": _r(r.default_rate), "mean_pd": _r(r.mean_pd),
                       "score_lo": int(r.lo), "score_hi": int(r.hi)} for i, r in dec.iterrows()]

    # calibration: October as trained vs Aug-Sep out-of-fold
    oof = pd.read_parquet("data/processed/oof_predictions.parquet")

    def rel(yy, pp):
        q = pd.qcut(pd.Series(pp).rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"q": q, "p": pp, "y": yy}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"))
        return {"pred": [_r(v) for v in g["pred"]], "obs": [_r(v) for v in g["obs"]]}
    out["calibration"] = {"october": rel(y, p), "aug_sep_oof": rel(oof[TARGET].to_numpy(), oof[f"pd_{main_name}"].to_numpy())}

    # flow: applications -> band -> outcome
    flow = pd.crosstab(band, np.where(y == 1, "missed", "repaid"))
    out["flow"] = {b: {"repaid": int(flow.loc[b, "repaid"]), "missed": int(flow.loc[b, "missed"])}
                   for b in ("approve", "refer", "decline")}

    # segments
    seg = {"No bureau score": ns, "Has bureau score": ~ns, "Self-employed": (oot["employment_self"] == 1).to_numpy(),
           "Salaried": ((oot["employment_self"] == 0) & (oot["employment_unknown"] == 0)).to_numpy()}
    out["segments"] = [{"segment": k, "n": int(m.sum()), "default_rate": _r(y[m].mean()),
                        "approval": _r((band[m] == "approve").mean()),
                        "approved_default": _r(y[m & (band == "approve")].mean()),
                        "auc": _r(roc_auc_score(y[m], p[m])), "auc_bureau": _r(roc_auc_score(y[m], preds["lgbm_bureau"][m]))}
                       for k, m in seg.items()]

    # months and stability
    allw = t.assign(month=t["disb"].dt.to_period("M").astype(str))
    out["months"] = [{"month": mth, "n": int(len(g)), "default_rate": _r(g[TARGET].mean())}
                     for mth, g in allw.groupby("month")]
    ref_scores = to_points(oof[f"pd_{main_name}"].to_numpy())
    q = np.unique(np.quantile(ref_scores, np.linspace(0, 1, 11))[1:-1])
    lab = lambda s: pd.Series(np.searchsorted(q, s, side="right"))  # noqa: E731
    out["score_psi_oct"] = _r(psi(lab(ref_scores), lab(score)))

    # fairness: approval by state and age band at the same 60% approval rate, shipped vs all-features
    def approve_top(pp, share=0.6):
        a = np.zeros(len(pp), bool)
        a[np.argsort(pp, kind="stable")[:int(round(share * len(pp)))]] = True
        return a
    a_main, a_alt = approve_top(p), approve_top(preds["lgbm_combined"])
    st = oot["State_ID"].to_numpy()
    rows = []
    for s_ in np.unique(st):
        m = st == s_
        if m.sum() >= 500:
            rows.append({"state": int(s_), "n": int(m.sum()), "default_rate": _r(y[m].mean()),
                         "approval_shipped": _r(a_main[m].mean()), "approval_all_features": _r(a_alt[m].mean())})
    out["states"] = sorted(rows, key=lambda r: r["default_rate"])
    out["overall_approval"] = {"shipped": _r(a_main.mean()), "all_features": _r(a_alt.mean())}

    # per-score counts for the cut-off playground: any cut-off -> approvals, missed EMIs and state approval rates
    big = [r["state"] for r in out["states"]]
    sd = pd.DataFrame({"score": score, "y": y, "state": st})
    u = np.sort(sd["score"].unique())
    g = sd.groupby("score")["y"].agg(["size", "sum"]).reindex(u)
    stn = (sd[sd["state"].isin(big)].groupby(["state", "score"]).size().unstack(fill_value=0)
           .reindex(columns=u, fill_value=0))
    out["by_score"] = {"score": [int(s) for s in u], "n": g["size"].astype(int).tolist(),
                       "missed": g["sum"].astype(int).tolist(),
                       "states": {str(s): stn.loc[s].astype(int).tolist() for s in big}}
    ages = oot["age_band"].to_numpy()
    out["age_bands"] = [{"band": b, "n": int((ages == b).sum()), "default_rate": _r(y[ages == b].mean()),
                         "approval": _r((band[ages == b] == "approve").mean())} for b in sorted(set(ages))]

    # explanations: global SHAP importance on a sample; information value of features
    rng = np.random.default_rng(0)
    pick = rng.choice(len(oot), 20000, replace=False)
    contrib, _ = shap_values(bundles[main_name], oot.iloc[pick])
    imp = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=bundles[main_name]["features"]).sort_values(ascending=False)
    out["shap"] = [{"feature": f, "value": _r(v)} for f, v in imp.head(14).items()]
    stats = pd.read_csv("data/processed/feature_stats.csv", index_col="feature")
    fair = set(fl["sets"]["fair"])
    out["iv"] = [{"feature": f, "iv": _r(r.iv), "in_shipped": f in fair} for f, r in stats.sort_values("iv", ascending=False).head(14).iterrows()]

    OUT.write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = main()
    print(f"wrote {OUT}: {len(json.dumps(o)) / 1024:.0f} KB; AUC {o['roc'][o['model']]['auc']}; flow {o['flow']}")
