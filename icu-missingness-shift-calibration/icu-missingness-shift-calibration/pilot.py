#!/usr/bin/env python3
"""
Pilot v2: does calibration degrade faster for models that use observation counts
when the missingness pattern shifts?  (PhysioNet/CinC Challenge 2012, in-hospital mortality)

Hypothesis (H1)
    Models that use observation counts are better calibrated in-distribution but
    degrade faster in ECE under a shift in observation rate than values-only models.
Result (see README): the set-a -> set-b effect did NOT replicate when train and test were swapped,
    so no general claim is made.

What v2 adds
    1. Recalibration arm: refit an intercept (or Platt slope+intercept) on half of the shifted
       test patients, evaluate on the other half.
       NOTE: this uses labels from the shifted data, so it is an upper bound on what is fixable.
    2. Bootstrap CIs over test patients for ECE and for the key contrast
       "extra ECE increase caused by using counts" = dECE(counts) - dECE(values).
    3. Replication: train a->test b, train b->test a, and train a->test c (if set-c is available).

Setup per pair
    Test measurements are perturbed:  mcar: each dropped w.p. p;  mnar: |z|<1 values dropped w.p. p.
    Models: {LR, GBM} x {values only, values + log1p counts}. Metrics: AUROC, Brier, ECE (10 equal-mass bins),
    calibration-in-the-large (CIL = mean prediction - observed rate).
    "seeds" vary only the random dropping (training is deterministic), so the seed spread is NOT
    test-set uncertainty. The bootstrap CIs are the honest uncertainty estimate.

Usage (Colab):  !python pilot.py --seeds 5 --boot 200       # auto-downloads data
Pipeline test:  python pilot.py --synthetic --data_dir synth_raw   # results are meaningless, never report
"""
import argparse, glob, io, os, tarfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = "https://physionet.org/files/challenge-2012/1.0.0/"
STATIC = ["Age", "Gender", "Height", "ICUType"]
SKIP = set(STATIC) | {"RecordID"}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------------------- data
def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    required = ["set-a.tar.gz", "set-b.tar.gz", "Outcomes-a.txt", "Outcomes-b.txt"]
    optional = ["set-c.tar.gz", "Outcomes-c.txt"]            # best effort
    for f in required + optional:
        dst = os.path.join(data_dir, f)
        try:
            if not os.path.exists(dst):
                print("downloading", f)
                urllib.request.urlretrieve(BASE + f, dst)
        except Exception as e:
            if f in required:
                raise
            print(f"(optional file {f} unavailable: {e})")
            continue
        if f.endswith(".tar.gz") and not os.path.isdir(os.path.join(data_dir, f[:5])):
            with tarfile.open(dst) as t:
                try:
                    t.extractall(data_dir, filter="data")
                except TypeError:
                    t.extractall(data_dir)


def load_set(data_dir, tag):
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"set-{tag}", "*.txt"))):
        d = pd.read_csv(path)
        d.columns = ["Time", "param", "value"]
        d["rid"] = int(os.path.basename(path)[:-4])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    hm = df["Time"].str.split(":", expand=True).astype(int)
    df["t"] = hm[0] * 60 + hm[1]
    df = df.drop(columns="Time")
    df.loc[df["value"] == -1, "value"] = np.nan
    static = df[df["param"].isin(STATIC)].groupby(["rid", "param"])["value"].first().unstack()
    ts = df[~df["param"].isin(SKIP)].dropna(subset=["value"]).sort_values(["rid", "t"])
    out = pd.read_csv(os.path.join(data_dir, f"Outcomes-{tag}.txt")).set_index("RecordID")
    y = out["In-hospital_death"].reindex(static.index)
    return ts[["rid", "t", "param", "value"]], static, y


def make_synthetic(out_dir, n=2000, seed=0):
    """Fake data in the SAME raw format with informative missingness. Pipeline test only."""
    rng = np.random.default_rng(seed)
    names = ["HR", "Temp", "GCS", "Lactate", "Creatinine", "BUN", "WBC", "Na", "K", "Glucose", "pH", "Weight"]
    mu = rng.normal(0, 1, len(names)) * 5 + 50
    for tag, off in [("a", 0), ("b", 100000), ("c", 200000)]:
        os.makedirs(os.path.join(out_dir, f"set-{tag}"), exist_ok=True)
        rows_out = []
        for i in range(n):
            rid = off + i
            s = rng.normal()
            die = int(rng.random() < 1 / (1 + np.exp(2.2 - 1.2 * s)))
            lines = ["Time,Parameter,Value", f"00:00,RecordID,{rid}",
                     f"00:00,Age,{int(np.clip(rng.normal(65, 15), 18, 95))}",
                     f"00:00,Gender,{rng.integers(0, 2)}", f"00:00,Height,{-1 if rng.random() < .5 else 170}",
                     f"00:00,ICUType,{rng.integers(1, 5)}"]
            ev = []
            for j, nm in enumerate(names):
                for _ in range(rng.poisson(3 * np.exp(0.35 * s))):
                    t = int(rng.integers(0, 2880))
                    ev.append((t, nm, round(mu[j] + 3 * (0.6 * s * (-1) ** j + rng.normal()), 2)))
            for t, nm, v in sorted(ev):
                lines.append(f"{t // 60:02d}:{t % 60:02d},{nm},{v}")
            with open(os.path.join(out_dir, f"set-{tag}", f"{rid}.txt"), "w") as f:
                f.write("\n".join(lines))
            rows_out.append((rid, 0, 0, 0, 0, die))
        pd.DataFrame(rows_out, columns=["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival",
                                        "In-hospital_death"]).to_csv(
            os.path.join(out_dir, f"Outcomes-{tag}.txt"), index=False)


# ----------------------------------------------------------------------------- features / shift
def build_features(ts, static, ids, cols=None):
    g = ts.groupby(["rid", "param"])["value"].agg(["last", "mean", "min", "max", "count"])
    w = g.unstack("param")
    w.columns = [f"{p}__{s}" for s, p in w.columns]
    w = w.reindex(ids)
    st = static.reindex(ids).copy()
    for k in (1, 2, 3, 4):
        st[f"ICUType_{k}"] = (st["ICUType"] == k).astype(float)
    st = st.drop(columns="ICUType")
    st.columns = [f"{c}__static" for c in st.columns]
    X = pd.concat([w, st], axis=1)
    cnt = [c for c in X.columns if c.endswith("__count")]
    X[cnt] = np.log1p(X[cnt].fillna(0))
    if cols is not None:
        X = X.reindex(columns=cols)
        cnt = [c for c in cols if c.endswith("__count")]
        X[cnt] = X[cnt].fillna(0)
    return X


def perturb(ts, kind, p, rng, stats):
    if p == 0:
        return ts
    r = rng.random(len(ts))
    if kind == "mcar":
        keep = r >= p
    elif kind == "mnar":
        z = ((ts["value"] - ts["param"].map(stats["mu"])) / ts["param"].map(stats["sd"])).abs()
        keep = ~((z < 1.0) & (r < p))
    else:
        raise ValueError(kind)
    return ts[np.asarray(keep)]


# ----------------------------------------------------------------------------- metrics / recalibration
def ece(y, p, bins=10):
    o = np.argsort(p)
    y, p = y[o], p[o]
    return sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean())
               for c in np.array_split(np.arange(len(p)), bins))


def metrics(y, p):
    return dict(auroc=roc_auc_score(y, p), brier=brier_score_loss(y, p), ece=ece(y, p), cil=p.mean() - y.mean())


def reliability(y, p, bins=10):
    idx = np.array_split(np.argsort(p), bins)
    return np.array([p[i].mean() for i in idx]), np.array([y[i].mean() for i in idx])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_intercept(y, p):
    z = logit(p)
    f = lambda b: -np.mean(y * np.log(sigmoid(z + b) + 1e-12) + (1 - y) * np.log(1 - sigmoid(z + b) + 1e-12))
    b = minimize_scalar(f, bounds=(-6, 6), method="bounded").x
    return lambda q: sigmoid(logit(q) + b)


def fit_platt(y, p):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return lambda q: m.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def make_models(seed):
    return {
        "LR": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=0.05, max_iter=3000)),
        "GBM": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                     HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                    max_iter=200, random_state=seed)),
    }


# ----------------------------------------------------------------------------- one train/test pair
def run_pair(tr, te, D, a, ps, outdir):
    os.makedirs(outdir, exist_ok=True)
    ts_tr, st_tr, y_tr = D[tr]
    ts_te, st_te, y_te = D[te]
    log(f"\n{'=' * 78}\nTRAIN set-{tr} (n={len(y_tr)}, mortality {y_tr.mean():.3f})  ->  "
        f"TEST set-{te} (n={len(y_te)}, mortality {y_te.mean():.3f})\n{'=' * 78}")
    stats = {"mu": ts_tr.groupby("param")["value"].mean(),
             "sd": ts_tr.groupby("param")["value"].std().replace(0, 1).fillna(1)}
    Xtr = build_features(ts_tr, st_tr, y_tr.index)
    cols = list(Xtr.columns)
    variants = {"values": [c for c in cols if not c.endswith("__count")], "values+counts": cols}
    ytr, yte = y_tr.values.astype(int), y_te.values.astype(int)
    n = len(yte)
    seeds = list(range(a.seeds))
    conds = [("clean", 0.0)] + [(k, p) for k in ("mcar", "mnar") for p in ps]

    # fixed patient split for the recalibration arm
    perm = np.random.default_rng(7).permutation(n)
    cal_i, ev_i = perm[: n // 2], perm[n // 2:]

    preds, rows, rrows = {}, [], []
    for seed in seeds:
        models = {}
        for mname, mk in make_models(seed).items():
            for vname, vcols in variants.items():
                m = mk(); m.fit(Xtr[vcols], ytr)
                models[f"{mname} {vname}"] = (m, vcols)
        for kind, p in conds:
            rng = np.random.default_rng(1000 + seed)
            Xte = build_features(perturb(ts_te, kind, p, rng, stats), st_te, y_te.index, cols)
            for name, (m, vcols) in models.items():
                pr = m.predict_proba(Xte[vcols])[:, 1]
                preds[(name, kind, p, seed)] = pr
                rows.append(dict(model=name, shift=kind, p=p, seed=seed, **metrics(yte, pr)))
                # recalibration arm: fit on cal half, evaluate on held-out half
                fits = {"none": lambda q: q,
                        "intercept": fit_intercept(yte[cal_i], pr[cal_i]),
                        "platt": fit_platt(yte[cal_i], pr[cal_i])}
                for meth, fn in fits.items():
                    rrows.append(dict(model=name, shift=kind, p=p, seed=seed, method=meth,
                                      **metrics(yte[ev_i], fn(pr[ev_i]))))
        log(f"  seed {seed} done")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "results_raw.csv"), index=False)
    clean = res[res["shift"] == "clean"]
    for k in ("mcar", "mnar"):
        c = clean.copy(); c["shift"] = k
        res = pd.concat([res, c], ignore_index=True)
    res = res[res["shift"] != "clean"].sort_values(["shift", "model", "p", "seed"])
    res.to_csv(os.path.join(outdir, "results.csv"), index=False)
    rec = pd.DataFrame(rrows)
    rec.to_csv(os.path.join(outdir, "recalibration.csv"), index=False)
    pm = max(ps)

    # ---------------- bootstrap over test patients
    rng = np.random.default_rng(2026)
    names = sorted({k[0] for k in preds})
    E = {}
    for b in range(a.boot):
        idx = rng.integers(0, n, n); yb = yte[idx]
        for nm in names:
            E.setdefault((nm, "clean", 0.0), []).append(ece(yb, preds[(nm, "clean", 0.0, 0)][idx]))
            for kind, p in conds[1:]:
                E.setdefault((nm, kind, p), []).append(
                    np.mean([ece(yb, preds[(nm, kind, p, s)][idx]) for s in seeds]))
    E = {k: np.array(v) for k, v in E.items()}
    crow = []
    for kind in ("mcar", "mnar"):
        for p in ps:
            for fam in ("LR", "GBM"):
                d = ((E[(f"{fam} values+counts", kind, p)] - E[(f"{fam} values+counts", "clean", 0.0)])
                     - (E[(f"{fam} values", kind, p)] - E[(f"{fam} values", "clean", 0.0)]))
                crow.append(dict(shift=kind, p=p, family=fam, extra_dECE_from_counts=d.mean(),
                                 lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                 excludes_zero=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)))
    con = pd.DataFrame(crow)
    con.to_csv(os.path.join(outdir, "contrast_bootstrap.csv"), index=False)
    erow = [dict(model=k[0], shift=k[1], p=k[2], ece=v.mean(), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5))
            for k, v in E.items()]
    pd.DataFrame(erow).to_csv(os.path.join(outdir, "ece_bootstrap.csv"), index=False)

    # ---------------- printed summaries
    log(f"\n[A] Clean-data comparison (mean over seeds; ECE bootstrap 95% CI):")
    t = res[(res["shift"] == "mcar") & (res["p"] == 0)].groupby("model")[["auroc", "brier", "ece", "cil"]].mean()
    t["ece_CI"] = [f"[{np.percentile(E[(m, 'clean', 0.0)], 2.5):.4f}, {np.percentile(E[(m, 'clean', 0.0)], 97.5):.4f}]"
                   for m in t.index]
    log(t.round(4).to_string())
    log("\n[B] Extra ECE increase caused by using counts, dECE(counts)-dECE(values), 95% bootstrap CI:")
    log(con.round(4).to_string(index=False))
    log(f"\n[C] Recalibration on held-out half (p={pm}); lower ECE/|CIL| = better. 'none' = no recalibration:")
    r = rec[(rec["p"] == pm)].groupby(["shift", "model", "method"])[["ece", "brier", "cil"]].mean().round(4)
    log(r.unstack("method").to_string())

    # ---------------- figures
    sty = {"values": "--", "values+counts": "-"}; colr = {"LR": "tab:blue", "GBM": "tab:red"}
    fig, ax = plt.subplots(3, 2, figsize=(11, 10), sharex="col")
    for j, k in enumerate(("mcar", "mnar")):
        for i, met in enumerate(("ece", "brier", "auroc")):
            for name, g in res[res["shift"] == k].groupby("model"):
                mn, vn = name.split(" ", 1)
                s = g.groupby("p")[met].agg(["mean", "min", "max"])
                ax[i, j].plot(s.index, s["mean"], sty[vn], color=colr[mn], marker="o", label=name)
                ax[i, j].fill_between(s.index, s["min"], s["max"], color=colr[mn], alpha=0.12)
            ax[i, j].set_ylabel(met.upper() if met != "brier" else "Brier")
            if i == 0: ax[i, j].set_title(f"{k.upper()} shift")
            if i == 2: ax[i, j].set_xlabel("fraction of measurements dropped (p)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"train set-{tr} -> test set-{te}  (band = min-max over perturbation seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_shift_curves.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for i, fam in enumerate(("LR", "GBM")):
        for j, k in enumerate(("mcar", "mnar")):
            g = con[(con["shift"] == k) & (con["family"] == fam)]
            ax[i, j].axhline(0, color="k", lw=0.8)
            ax[i, j].errorbar(g["p"], g["extra_dECE_from_counts"],
                              yerr=[g["extra_dECE_from_counts"] - g["lo"], g["hi"] - g["extra_dECE_from_counts"]],
                              fmt="o-", capsize=3, color=colr[fam])
            ax[i, j].set_title(f"{fam}, {k.upper()}", fontsize=10)
            if j == 0: ax[i, j].set_ylabel("extra ECE rise from counts")
            if i == 1: ax[i, j].set_xlabel("fraction dropped (p)")
    fig.suptitle(f"set-{tr}->set-{te}: does using counts worsen calibration under shift? (95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_contrast_ci.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a_, k in zip(ax, ("mcar", "mnar")):
        g = rec[(rec["shift"] == k) & (rec["p"] == pm)].groupby(["model", "method"])["ece"].mean().unstack("method")
        g = g[["none", "intercept", "platt"]]
        g.plot.bar(ax=a_, rot=20); a_.set_title(f"{k.upper()} p={pm}: ECE on held-out half"); a_.set_xlabel("")
    ax[0].set_ylabel("ECE")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_recalibration.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    for a_, name in zip(ax, names):
        a_.plot([0, 1], [0, 1], "k:", lw=1)
        for kind, c in (("clean", "tab:green"), ("mnar", "tab:red")):
            pr = preds[(name, kind, 0.0 if kind == "clean" else pm, 0)]
            px, py = reliability(yte, pr)
            a_.plot(px, py, "o-", color=c, label="clean" if kind == "clean" else f"MNAR p={pm}")
        a_.set_title(name, fontsize=10); a_.set_xlabel("predicted risk")
    ax[0].set_ylabel("observed mortality"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_reliability.png"), dpi=160); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="physionet2012")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--pairs", default="ab,ba,ac", help="train->test pairs, e.g. ab,ba,ac")
    ap.add_argument("--n_synth", type=int, default=2000)
    a, _ = ap.parse_known_args()      # tolerate Jupyter/Colab's extra '-f kernel.json'
    os.makedirs(a.out, exist_ok=True)
    ps = [float(x) for x in a.ps.split(",")]

    if a.synthetic:
        log("!! SYNTHETIC DATA: pipeline test only, results are meaningless !!")
        make_synthetic(a.data_dir, n=a.n_synth)
    elif a.download or not os.path.isdir(os.path.join(a.data_dir, "set-a")):
        download(a.data_dir)

    D = {}
    for pair in a.pairs.split(","):
        tr, te = pair[0], pair[1]
        for tag in (tr, te):
            if tag not in D:
                try:
                    D[tag] = load_set(a.data_dir, tag)
                except Exception as e:
                    log(f"(set-{tag} unavailable, skipping pairs that need it: {type(e).__name__}: {e})")
                    D[tag] = None
        if D[tr] is None or D[te] is None:
            continue
        run_pair(tr, te, D, a, ps, os.path.join(a.out, f"{tr}_to_{te}"))

    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write("\n".join(LOG))
    log(f"\nAll outputs in {a.out}/  (summary.txt + one folder per train->test pair)")


if __name__ == "__main__":
    main()
