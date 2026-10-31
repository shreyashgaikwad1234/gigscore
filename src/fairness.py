"""Subgroup fairness for the main model (build guide Section 7; CLAUDE.md rule 3).

Age band and State_ID are never model inputs; they are used only here. For each group: approval
rate, TPR (defaulters not approved) and FPR (non-defaulters not approved) at the proposed cut-offs,
with gaps over 5 pp flagged. Also: how strongly each model feature tracks age or state (proxy
check), and a mitigation test that retrains without location features.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features import DEALER_FEATURES, FAIR_EXCLUDE, LOCATION_FEATURES
from src.load import TARGET

GAP_FLAG = 0.05
MIN_ROWS = 500
# Mitigation options: feature groups that most directly stand in for state (proxy check, step 4).
# The last option is the shipped 'fair' feature set, here refit with the full model's settings.
OPTIONS = {
    "without area & branch": LOCATION_FEATURES,
    "without area, branch & dealer": LOCATION_FEATURES + DEALER_FEATURES,
    "without area, branch, dealer & ID documents": FAIR_EXCLUDE,
}
ATTRS = {"age_band": "age band", "State_ID": "state"}


def group_table(attr: pd.Series, y: np.ndarray, p: np.ndarray, bands: np.ndarray, min_rows: int = MIN_ROWS) -> pd.DataFrame:
    approved = bands == "approve"
    rows = []
    overall = {"approval": approved.mean(), "tpr": (~approved[y == 1]).mean(), "fpr": (~approved[y == 0]).mean()}
    for g, idx in attr.reset_index(drop=True).groupby(attr.to_numpy()).groups.items():
        idx = np.asarray(idx)
        if len(idx) < min_rows:
            continue
        yg, ag = y[idx], approved[idx]
        r = {"group": g, "rows": len(idx), "default_rate": yg.mean(), "approval": ag.mean(),
             "approve_or_refer": (bands[idx] != "decline").mean(),
             "tpr": (~ag[yg == 1]).mean(), "fpr": (~ag[yg == 0]).mean(),
             "auc": roc_auc_score(yg, p[idx]) if 0 < yg.mean() < 1 else np.nan}
        for k in ("approval", "tpr", "fpr"):
            r[f"{k}_gap"] = r[k] - overall[k]
        r["flag"] = bool(abs(r["approval_gap"]) > GAP_FLAG or abs(r["fpr_gap"]) > GAP_FLAG)
        rows.append(r)
    return pd.DataFrame(rows).set_index("group").sort_index()


def proxy_check(X: pd.DataFrame, features: list[str], age: pd.Series, state: pd.Series) -> pd.DataFrame:
    """|Spearman| with age and correlation ratio (eta) with state for every numeric model feature."""
    rows = []
    for f in features:
        if f in X and pd.api.types.is_numeric_dtype(X[f]):
            v = X[f].astype(float)
            ok = v.notna()
            means = v[ok].groupby(state[ok]).transform("mean")
            ss_tot = ((v[ok] - v[ok].mean()) ** 2).sum()
            eta = np.sqrt(((means - v[ok].mean()) ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
            rows.append({"feature": f, "age_spearman": abs(v.corr(age, method="spearman")), "state_eta": eta})
    return pd.DataFrame(rows).set_index("feature")


def _approve_top(p: np.ndarray, share: float) -> np.ndarray:
    """Approve the `share` of applicants with the lowest PD (rank-based, ties broken by order)."""
    k = int(round(share * len(p)))
    a = np.zeros(len(p), bool)
    a[np.argsort(p, kind="stable")[:k]] = True
    return a


def option_metrics(oot: pd.DataFrame, y: np.ndarray, p: np.ndarray, share: float) -> dict:
    """Approve the safest `share` and measure the state gaps (approval and good-applicant decline rate)."""
    a = _approve_top(p, share)
    st = oot["State_ID"].to_numpy()
    df = pd.DataFrame({"state": st, "a": a, "good": y == 0})
    n = df.groupby("state").size()
    keep = n[n >= MIN_ROWS].index
    appr = df.groupby("state")["a"].mean()[keep]
    fpr = 1 - df[df["good"]].groupby("state")["a"].mean()[keep]
    fpr_all = 1 - a[y == 0].mean()
    gap, fgap = (appr - a.mean()).abs(), (fpr - fpr_all).abs()
    ns = (oot["no_score"] == 1).to_numpy()
    return {"auc": roc_auc_score(y, p), "auc_no_score": roc_auc_score(y[ns], p[ns]),
            "bad_rate_approved": float(y[a].mean()),
            "max_state_gap": float(gap.max()), "max_state_fpr_gap": float(fgap.max()),
            "states_flagged": int(((gap > GAP_FLAG) | (fgap > GAP_FLAG)).sum()), "n_states": int(len(keep))}


def mitigation_options(t: pd.DataFrame, oot: pd.DataFrame, y: np.ndarray, p_main: np.ndarray, main_name: str,
                       base_bundle: dict, base_p: np.ndarray, share: float) -> pd.DataFrame:
    """Refit the all-features LightGBM (same parameters and tree count, all of Aug-Sep) without groups of
    features that stand in for state, and compare with it and with the shipped, separately tuned model
    at the same approval rate."""
    import lightgbm as lgb

    from src.train import MONOTONE  # local import: train imports nothing from here

    win = t[t["split"] != "oot"]
    rows = {f"{base_bundle['version']} (all features)": option_metrics(oot, y, base_p, share)}
    for name, drop in OPTIONS.items():
        feats = [f for f in base_bundle["features"] if f not in drop]
        params = {**base_bundle["params"], "monotone_constraints": [MONOTONE.get(f, 0) for f in feats]}
        m = lgb.train(params, lgb.Dataset(win[feats], win[TARGET]), base_bundle["best_iteration"])
        rows[f"  {name}"] = option_metrics(oot, y, m.predict(oot[feats]), share)
    rows[f"**{main_name} (shipped; own tuning)**"] = option_metrics(oot, y, p_main, share)
    return pd.DataFrame(rows).T


def _pct(v):
    return f"{v:.1%}"


def run(oot: pd.DataFrame, y: np.ndarray, p: np.ndarray, bands: np.ndarray, bundle: dict, t: pd.DataFrame,
        cuts: dict, base_bundle: dict, base_p: np.ndarray, approve_share: float = 0.60) -> dict:
    tables = {a: group_table(oot[a], y, p, bands) for a in ATTRS}
    proxy = proxy_check(oot, base_bundle["features"], oot["age"], oot["State_ID"])
    opts = mitigation_options(t, oot, y, p, bundle["version"], base_bundle, base_p, approve_share)

    md = [f"Protected attributes (age band, state) are **not** model inputs. Results below use {bundle['version']} "
          f"with the proposed cut-offs (approve ≥ {cuts['approve']}, refer ≥ {cuts['refer']}) on October. "
          "TPR = share of defaulters not approved; FPR = share of non-defaulters not approved (good applicants "
          f"turned away). Gaps are vs all applicants; flagged when approval or FPR gap exceeds "
          f"{GAP_FLAG * 100:.0f} pp. Groups with fewer than {MIN_ROWS} October loans are not shown.", ""]
    card = []
    for a, label in ATTRS.items():
        tab = tables[a]
        md += [f"### By {label}", "",
               "| group | loans | default rate | approval | gap | FPR | FPR gap | TPR | AUC | flag |",
               "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
        md += [f"| {g} | {int(r['rows']):,} | {_pct(r['default_rate'])} | {_pct(r['approval'])} | "
               f"{r['approval_gap'] * 100:+.1f} pp | {_pct(r['fpr'])} | {r['fpr_gap'] * 100:+.1f} pp | {_pct(r['tpr'])} | "
               f"{r['auc']:.3f} | {'**yes**' if r['flag'] else ''} |" for g, r in tab.iterrows()]
        flagged = tab[tab["flag"]]
        corr = tab["default_rate"].corr(tab["approval"]) if len(tab) > 2 else np.nan
        md += ["", f"{len(flagged)} of {len(tab)} {label} groups flagged. Correlation between a group's actual "
                   f"default rate and its approval rate: {corr:+.2f} (strongly negative = gaps follow real risk "
                   "differences rather than arbitrary ones).", ""]
        card.append(f"{len(flagged)} of {len(tab)} {label} groups have an approval or FPR gap > 5 pp "
                    f"(max approval gap {tab['approval_gap'].abs().max() * 100:.1f} pp).")

    top_age = proxy.sort_values("age_spearman", ascending=False).head(6)
    top_state = proxy.sort_values("state_eta", ascending=False).head(6)
    md += ["### Proxy check", "",
           "Features most associated with age (|Spearman|): " + ", ".join(
               f"`{f}` {v:.2f}" for f, v in top_age["age_spearman"].items()) + ".", "",
           "Features most associated with state (correlation ratio η): " + ", ".join(
               f"`{f}` {v:.2f}" for f, v in top_state["state_eta"].items()) + ".", "",
           "### Mitigation options: retrain without the features that stand in for state", "",
           "Indented rows refit the all-features LightGBM on all of Aug–Sep with its parameters and tree count, "
           "dropping more proxy features each time; the last row is the shipped model, tuned on its own. Each "
           f"approves the safest {approve_share:.0%} of October. State gaps are vs all applicants. The proxy "
           "check above is on the all-features model, which is why the shipped model leaves those features out.", "",
           "| option | AUC | AUC no-score | bad rate approved | largest state approval gap | largest state FPR gap | "
           "states flagged |", "|:--|--:|--:|--:|--:|--:|--:|"]
    md += [f"| {name} | {r['auc']:.3f} | {r['auc_no_score']:.3f} | {_pct(r['bad_rate_approved'])} | "
           f"{r['max_state_gap'] * 100:.1f} pp | {r['max_state_fpr_gap'] * 100:.1f} pp | "
           f"{int(r['states_flagged'])} of {int(r['n_states'])} |" for name, r in opts.iterrows()]
    md += ["", "Shipped choice: the model without area, branch, dealer and ID-document features, trading some "
               "accuracy for far smaller state gaps (rationale in `reports/model_card.md`). A lender can switch "
               "`model` in `config/thresholds.yaml` to the all-features model."]
    cur, last = opts.iloc[0], opts.iloc[-1]
    card.append(f"Shipped model vs all-features model at 60% approval: October AUC {cur['auc']:.3f} → {last['auc']:.3f}, "
                f"largest state approval gap {cur['max_state_gap'] * 100:.1f} → {last['max_state_gap'] * 100:.1f} pp, "
                f"states flagged {int(cur['states_flagged'])} → {int(last['states_flagged'])} of {int(last['n_states'])}.")
    return {"md": md, "card": card, "tables": tables, "proxy": proxy, "options": opts}
