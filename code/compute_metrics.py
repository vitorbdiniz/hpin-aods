"""Analysis tables (plan Etapa 4).

Reads the rolling-window estimates (data/pins/sinteticos) and produces all
tables consumed by the figure script and the paper, in results/tables:

  mean.csv, std.csv, MIV.csv          (model, param, window, value)
  pacf.csv                            (model, param, window, lag, PACF)
  half_life.csv                       (model, param, window, half-life)
  PLS_PIN.csv, PLS_HPIN.csv           wide: index i x windows
  pls_stats.csv                       descriptive stats of PLS per window
  transition_metrics_{MODEL}_{w}.csv  per-window transition-derived metrics
  wilcoxon_{criterion}.csv            paired block-bootstrap Wilcoxon tests
  stats_description.csv               appendix descriptive statistics

Metric definitions (documented in the paper after reviewer requests):
- MIV: mean absolute difference of estimates between consecutive windows (levels).
- PACF: statsmodels pacf (Yule-Walker adjusted), lags 1..5, on each rolling
  parameter series.
- Half-life: -ln 2 / ln(rho1), defined only for rho1 in (0, 1); otherwise NaN.
- PLS: out-of-sample one-step-ahead log predictive density of the first
  observation after each estimation window (both models, same aggregation).
- Spectral gap: 1 - |lambda_2| of the row-normalized Gamma (right eigenvalues);
  u_infty from the left eigenvector of the eigenvalue closest to 1.
- Wilcoxon: paired moving-block bootstrap (block length 50, resample size 500,
  5000 replicates, seed 7); the per-replicate statistic is compared across
  models with a two-sided Wilcoxon signed-rank test.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.tsa.stattools import pacf as sm_pacf

from hpin import (build_transition_matrix, stationary_distribution,
                  predictive_logscore_pin)

HERE = os.path.dirname(__file__)
PINS_DIR = os.path.join(HERE, "..", "data", "pins", "sinteticos")
FLUX_DIR = os.path.join(HERE, "..", "data", "fluxos", "sinteticos")
OUT_DIR = os.path.join(HERE, "..", "results", "tables")

WINDOWS = [60, 90, 120, 150, 180]
DATASET = "fluxos_sinteticos_personalizado"
PARAMS = ["pin", "alpha", "delta", "epsilon_b", "epsilon_s", "mu"]
N_BOOT = 5000
BOOT_SIZE = 500
BLOCK_LEN = 50
SEED = 7


def load_pins():
    pins = {"PIN": {}, "HPIN": {}}
    for w in WINDOWS:
        pin = pd.read_csv(os.path.join(PINS_DIR, f"pin_ekop_{w}_{DATASET}.csv"), index_col=0)
        hpin = pd.read_csv(os.path.join(PINS_DIR, f"pin_hmm_{w}_{DATASET}.csv"), index_col=0)
        pins["PIN"][w] = pin
        pins["HPIN"][w] = hpin
    return pins


# ---------------------------------------------------------------------------
# scalar metrics
# ---------------------------------------------------------------------------

def mean_incremental_variation(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(np.abs(np.diff(x))))


def half_life(rho1):
    if not np.isfinite(rho1) or rho1 <= 0.0 or rho1 >= 1.0:
        return np.nan
    return float(np.log(0.5) / np.log(rho1))


def spectral_gap(G):
    G = np.asarray(G, dtype=float)
    G = G / G.sum(axis=1, keepdims=True)
    ev = np.sort(np.abs(np.linalg.eigvals(G)))[::-1]
    return float(np.real(1.0 - ev[1]))


def switch_rate(G, u=None):
    u = stationary_distribution(G) if u is None else u
    return float(np.dot(u, 1.0 - np.diag(G)))


def transition_entropy(G, u=None, normalize=True):
    G = np.asarray(G, dtype=float)
    G = G / G.sum(axis=1, keepdims=True)
    u = stationary_distribution(G) if u is None else u
    logG = np.where(G > 0, np.log(G), 0.0)
    H = -float(np.sum(u[:, None] * G * logG))
    return H / np.log(G.shape[0]) if normalize else H


def persistence_asymmetry(phi_plus, phi_minus):
    return float(abs(phi_plus - phi_minus))


def gamma_from_row(row, model):
    if model == "HPIN":
        return build_transition_matrix(row["eta"], row["kappa"], row["phi_plus"],
                                       row["phi_minus"], row["omega_plus"], row["omega_minus"])
    a, d = row["alpha"], row["delta"]
    r = np.array([1.0 - a, a * (1.0 - d), a * d])
    r = np.clip(r, 0.0, None)
    r = r / r.sum() if r.sum() > 0 else np.array([1.0, 0.0, 0.0])
    return np.tile(r, (3, 1))


# ---------------------------------------------------------------------------
# block bootstrap + Wilcoxon
# ---------------------------------------------------------------------------

def block_bootstrap_stat(x, y, stat_fn, rng):
    """Paired moving-block bootstrap replicates of stat_fn for two aligned series."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    n_blocks = int(np.ceil(BOOT_SIZE / BLOCK_LEN))
    starts = rng.integers(0, n - BLOCK_LEN, size=(N_BOOT, n_blocks))
    sx = np.empty(N_BOOT)
    sy = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK_LEN) for s in starts[b]])[:BOOT_SIZE]
        sx[b] = stat_fn(x[idx])
        sy[b] = stat_fn(y[idx])
    return sx, sy


def paired_wilcoxon(sx, sy):
    d = sx - sy
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return np.nan, np.nan
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pins = load_pins()
    rng = np.random.default_rng(SEED)

    mean_rows, std_rows, miv_rows, pacf_rows, hl_rows = [], [], [], [], []
    wtests = {"mean": [], "std": [], "MIV": [], "PACF": []}

    for w in WINDOWS:
        common = pins["PIN"][w].index.intersection(pins["HPIN"][w].index)
        dfs = {m: pins[m][w].loc[common] for m in ("PIN", "HPIN")}
        for p in PARAMS:
            series = {m: dfs[m][p].astype(float).values for m in ("PIN", "HPIN")}
            for m in ("PIN", "HPIN"):
                x = series[m]
                mean_rows.append((m, p, w, float(np.nanmean(x))))
                std_rows.append((m, p, w, float(np.nanstd(x, ddof=1))))
                miv_rows.append((m, p, w, mean_incremental_variation(x)))
                pv = sm_pacf(x[~np.isnan(x)], nlags=5)
                for lag in range(1, 6):
                    pacf_rows.append((m, p, w, lag, float(pv[lag])))
                hl_rows.append((m, p, w, half_life(float(pv[1]))))

            # Wilcoxon: mean / std / MIV via paired block bootstrap
            for crit, fn in (("mean", np.nanmean),
                             ("std", lambda v: np.nanstd(v, ddof=1)),
                             ("MIV", mean_incremental_variation)):
                sx, sy = block_bootstrap_stat(series["HPIN"], series["PIN"], fn, rng)
                stat, pval = paired_wilcoxon(sx, sy)
                wtests[crit].append((w, p, stat, pval))
            # Wilcoxon on PACF lags 1..5
            for lag in range(1, 6):
                fn = lambda v, lag=lag: sm_pacf(v, nlags=lag)[lag] if len(v) > 3 * lag else np.nan
                sx, sy = block_bootstrap_stat(series["HPIN"], series["PIN"], fn, rng)
                stat, pval = paired_wilcoxon(sx, sy)
                wtests["PACF"].append((w, p, lag, stat, pval))
        print(f"w={w}: scalar metrics + wilcoxon done", flush=True)

    pd.DataFrame(mean_rows, columns=["model", "param", "window", "mean"]).to_csv(
        os.path.join(OUT_DIR, "mean.csv"), index=False)
    pd.DataFrame(std_rows, columns=["model", "param", "window", "std"]).to_csv(
        os.path.join(OUT_DIR, "std.csv"), index=False)
    pd.DataFrame(miv_rows, columns=["model", "param", "window", "MIV"]).to_csv(
        os.path.join(OUT_DIR, "MIV.csv"), index=False)
    pd.DataFrame(pacf_rows, columns=["model", "param", "window", "lag", "PACF"]).to_csv(
        os.path.join(OUT_DIR, "pacf.csv"), index=False)
    pd.DataFrame(hl_rows, columns=["model", "param", "window", "half-life"]).to_csv(
        os.path.join(OUT_DIR, "half_life.csv"), index=False)
    for crit in ("mean", "std", "MIV"):
        pd.DataFrame(wtests[crit], columns=["window", "param", "statistics", "p_value"]).to_csv(
            os.path.join(OUT_DIR, f"wilcoxon_{crit}.csv"), index=False)
    pd.DataFrame(wtests["PACF"], columns=["window", "param", "lag", "statistics", "p_value"]).to_csv(
        os.path.join(OUT_DIR, "wilcoxon_PACF.csv"), index=False)

    # ------------------------------------------------------------------
    # PLS (out-of-sample one-step-ahead)
    # ------------------------------------------------------------------
    flux = pd.read_csv(os.path.join(FLUX_DIR, DATASET + ".csv"), index_col=0)
    B, S = flux["buyer"].values.astype(float), flux["seller"].values.astype(float)

    pls_pin = pd.DataFrame(index=pd.RangeIndex(len(flux), name="i"), columns=WINDOWS, dtype=float)
    pls_hpin = pd.DataFrame(index=pd.RangeIndex(len(flux), name="i"), columns=WINDOWS, dtype=float)
    for w in WINDOWS:
        hp = pins["HPIN"][w]
        pls_hpin.loc[hp.index, w] = hp["pls_next"].values
        pn = pins["PIN"][w].dropna(subset=["alpha"])
        vals = [predictive_logscore_pin(r["alpha"], r["delta"], r["epsilon_b"],
                                        r["epsilon_s"], r["mu"], B[i], S[i])
                for i, r in pn.iterrows()]
        pls_pin.loc[pn.index, w] = vals

    pls_pin.dropna(how="all").to_csv(os.path.join(OUT_DIR, "PLS_PIN.csv"))
    pls_hpin.dropna(how="all").to_csv(os.path.join(OUT_DIR, "PLS_HPIN.csv"))

    stats, wrows, diff_rows = [], [], []
    for w in WINDOWS:
        both = pd.concat([pls_pin[w].rename("PIN"), pls_hpin[w].rename("HPIN")], axis=1).dropna()
        for m in ("HPIN", "PIN"):
            d = both[m].describe()
            stats.append([m, w] + [d[k] for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]])
        diff = both["HPIN"] - both["PIN"]
        res = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        wrows.append((w, float(res.statistic), float(res.pvalue)))
        # paired per-observation comparison: the raw PLS levels are dominated by
        # the common overdispersion shock Z_t, so the paired difference is the
        # informative object
        diff_rows.append(dict(window=w, n=len(diff),
                              median_diff=float(diff.median()),
                              q25_diff=float(diff.quantile(0.25)),
                              q75_diff=float(diff.quantile(0.75)),
                              frac_hpin_better=float((diff > 0).mean()),
                              mean_diff=float(diff.mean()),
                              median_hpin=float(both["HPIN"].median()),
                              median_pin=float(both["PIN"].median())))
    pd.DataFrame(stats, columns=["model", "window", "mean", "std", "min",
                                 "q25", "q50", "q75", "max"]).to_csv(
        os.path.join(OUT_DIR, "pls_stats.csv"), index=False)
    pd.DataFrame(wrows, columns=["window", "statistics", "p_value"]).to_csv(
        os.path.join(OUT_DIR, "wilcoxon_PLS.csv"), index=False)
    pd.DataFrame(diff_rows).to_csv(os.path.join(OUT_DIR, "pls_diff_stats.csv"), index=False)
    print("PLS done", flush=True)

    # ------------------------------------------------------------------
    # transition-derived metrics
    # ------------------------------------------------------------------
    tm_wrows = []
    tm_all = {}
    for m in ("PIN", "HPIN"):
        for w in WINDOWS:
            rows = []
            for _, r in pins[m][w].dropna(subset=["alpha"]).iterrows():
                G = gamma_from_row(r, m)
                u = stationary_distribution(G)
                if m == "HPIN":
                    pa = persistence_asymmetry(r["phi_plus"], r["phi_minus"])
                else:
                    pa = persistence_asymmetry(G[1, 1], G[2, 2])
                rows.append(dict(persistence_asymmetry=pa,
                                 switch_rate=switch_rate(G, u),
                                 spectral_gap=spectral_gap(G),
                                 transition_entropy=transition_entropy(G, u)))
            tm = pd.DataFrame(rows)
            tm.to_csv(os.path.join(OUT_DIR, f"transition_metrics_{m}_{w}.csv"))
            tm_all[(m, w)] = tm
    for w in WINDOWS:
        for metric in ["persistence_asymmetry", "switch_rate", "spectral_gap", "transition_entropy"]:
            a = tm_all[("HPIN", w)][metric].values
            b = tm_all[("PIN", w)][metric].values
            n = min(len(a), len(b))
            stat, pval = paired_wilcoxon(a[:n], b[:n])
            tm_wrows.append((w, metric, stat, pval))
    pd.DataFrame(tm_wrows, columns=["window", "metric", "statistics", "p_value"]).to_csv(
        os.path.join(OUT_DIR, "wilcoxon_transition_metrics.csv"), index=False)
    print("transition metrics done", flush=True)

    # ------------------------------------------------------------------
    # appendix descriptive statistics
    # ------------------------------------------------------------------
    desc_rows = []
    for w in WINDOWS:
        for stat_name in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
            row = {"statistics": stat_name, "window": w}
            for p in PARAMS:
                for m in ("PIN", "HPIN"):
                    d = pins[m][w][p].astype(float).describe()
                    row[f"{p}_{m}"] = d[stat_name]
            desc_rows.append(row)
    pd.DataFrame(desc_rows).to_csv(os.path.join(OUT_DIR, "stats_description.csv"), index=False)
    print("done ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
