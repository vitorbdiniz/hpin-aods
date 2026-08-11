"""Persistence grid: how much true temporal dependence must exist before
modelling it changes the measurement of informed trading?

The grid varies one parameter and holds everything else fixed:

    Gamma(rho) = (1 - rho) * 1 u  +  rho * I

Properties (exact, not approximate):
  - stationary distribution is u for every rho, so the unconditional alpha,
    delta and PIN are IDENTICAL across the grid — the classical PIN's target
    does not move, only the temporal structure does;
  - eigenvalues are {1, rho, rho}, so the spectral gap is exactly 1 - rho;
  - switch rate is exactly (1 - rho) * (1 - sum u_i^2).

rho = 0 is the i.i.d. (degenerate) case — the classical PIN's own DGP.
rho = 0.45 reproduces the baseline dynamic DGP of the paper (true gap 0.55).

Usage:
    python persistence_grid.py generate    # simulate the six series
    python persistence_grid.py hpin        # rolling HPIN on all six
    python persistence_grid.py aggregate   # summary table (needs the R run)
"""

import os
import sys
import numpy as np
import pandas as pd

from hpin import emission_rates, estimate_pin, stationary_distribution
from generate_series import simulate, T, EPS_B, EPS_S, MU, IG_SCALE
import rolling_hpin

HERE = os.path.dirname(__file__)
FLUX_DIR = os.path.join(HERE, "..", "data", "fluxos", "sinteticos")
PINS_DIR = os.path.join(HERE, "..", "data", "pins", "sinteticos")
OUT_DIR = os.path.join(HERE, "..", "results", "tables")

# stationary distribution held fixed across the grid: the one implied by the
# paper's baseline Gamma_s
U = np.array([0.47368421, 0.26315789, 0.26315789])
RHOS = [0.0, 0.2, 0.45, 0.6, 0.8, 0.9]
WINDOWS = [60, 90, 120, 150, 180]
SEED0 = 42


def gamma_rho(rho, u=U):
    """Gamma(rho) = (1-rho) 1u + rho I."""
    return (1.0 - rho) * np.tile(u, (3, 1)) + rho * np.eye(3)


def dataset_name(rho, clean=False):
    suffix = "_clean" if clean else ""
    return f"fluxos_sinteticos_rho{int(round(rho * 100)):03d}{suffix}"


def truth(rho, u=U):
    """Analytic truth for this grid point."""
    G = gamma_rho(rho, u)
    alpha = float(u[1] + u[2])
    return dict(
        rho=rho,
        spectral_gap_true=1.0 - rho,
        switch_rate_true=float((1.0 - rho) * (1.0 - np.sum(u ** 2))),
        entropy_true=float(-np.sum(u[:, None] * G * np.where(G > 0, np.log(G), 0.0))
                           / np.log(3)),
        alpha_true=alpha,
        delta_true=float(u[2] / alpha),
        pin_true=estimate_pin(alpha, EPS_B, EPS_S, MU),
    )


# ---------------------------------------------------------------------------

def cmd_generate():
    os.makedirs(FLUX_DIR, exist_ok=True)
    rows = []
    for k, rho in enumerate(RHOS):
        G = gamma_rho(rho)
        df = simulate(G, T, EPS_B, EPS_S, MU, seed=SEED0 + k, ig_scale=IG_SCALE)
        name = dataset_name(rho)
        df.to_csv(os.path.join(FLUX_DIR, name + ".csv"))
        t = truth(rho)
        rows.append(dict(dataset=name, seed=SEED0 + k, **t,
                         state_freq=np.round(np.bincount(df.state, minlength=3) / len(df), 4).tolist()))
        print(f"{name}: gap_true={t['spectral_gap_true']:.2f} "
              f"switch_true={t['switch_rate_true']:.4f} "
              f"pin_true={t['pin_true']:.4f} | B mean={df.buyer.mean():.0f} "
              f"states={np.bincount(df.state, minlength=3) / len(df)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "persistence_grid_truth.csv"), index=False)
    print("\nDatasets for the R baseline run:")
    print(" ".join(dataset_name(r) for r in RHOS))


def cmd_generate_clean():
    """Same grid without the Inverse Gaussian factor: isolates how much of the
    structural estimation error is due to the overdispersion misspecification
    rather than to the estimator itself."""
    os.makedirs(FLUX_DIR, exist_ok=True)
    for k, rho in enumerate(RHOS):
        df = simulate(gamma_rho(rho), T, EPS_B, EPS_S, MU,
                      seed=SEED0 + k, ig_noise=False)
        name = dataset_name(rho, clean=True)
        df.to_csv(os.path.join(FLUX_DIR, name + ".csv"))
        print(f"{name}: gap_true={1 - rho:.2f} | B mean={df.buyer.mean():.0f} "
              f"std={df.buyer.std():.0f}")


def cmd_hpin():
    for rho in RHOS:
        rolling_hpin.run(dataset_name(rho))


def cmd_hpin_clean():
    for rho in RHOS:
        rolling_hpin.run(dataset_name(rho, clean=True))


def cmd_aggregate():
    """Join HPIN and classical PIN rolling estimates against the analytic truth,
    for both the overdispersed (IG) and the correctly specified (clean) grids."""
    from compute_metrics import (spectral_gap, switch_rate, transition_entropy,
                                 gamma_from_row, mean_incremental_variation)
    rows = []
    for clean in (False, True):
      for rho in RHOS:
        ds = dataset_name(rho, clean=clean)
        t = truth(rho)
        for w in WINDOWS:
            f_h = os.path.join(PINS_DIR, f"pin_hmm_{w}_{ds}.csv")
            f_p = os.path.join(PINS_DIR, f"pin_ekop_{w}_{ds}.csv")
            if not os.path.exists(f_h):
                continue
            h = pd.read_csv(f_h, index_col=0)
            rec = dict(dgp="clean" if clean else "overdispersed", window=w, **t)

            # HPIN structural estimates
            gh = [gamma_from_row(r, "HPIN") for _, r in h.iterrows()]
            rec["gap_hpin"] = float(np.mean([spectral_gap(G) for G in gh]))
            rec["switch_hpin"] = float(np.mean([switch_rate(G) for G in gh]))
            rec["entropy_hpin"] = float(np.mean([transition_entropy(G) for G in gh]))
            rec["pin_hpin"] = float(h["pin"].median())
            rec["miv_hpin"] = mean_incremental_variation(h["pin"].values)
            rec["bias_pin_hpin"] = rec["pin_hpin"] - t["pin_true"]
            rec["err_gap_hpin"] = rec["gap_hpin"] - t["spectral_gap_true"]
            rec["err_switch_hpin"] = rec["switch_hpin"] - t["switch_rate_true"]

            if os.path.exists(f_p):
                p = pd.read_csv(f_p, index_col=0).dropna(subset=["pin"])
                gp = [gamma_from_row(r, "PIN") for _, r in p.iterrows()]
                rec["gap_pin"] = float(np.mean([spectral_gap(G) for G in gp]))
                rec["switch_pin"] = float(np.mean([switch_rate(G) for G in gp]))
                rec["entropy_pin"] = float(np.mean([transition_entropy(G) for G in gp]))
                rec["pin_pin"] = float(p["pin"].median())
                rec["miv_pin"] = mean_incremental_variation(p["pin"].values)
                rec["bias_pin_pin"] = rec["pin_pin"] - t["pin_true"]
                rec["err_gap_pin"] = rec["gap_pin"] - t["spectral_gap_true"]
                rec["err_switch_pin"] = rec["switch_pin"] - t["switch_rate_true"]
                common = h.index.intersection(p.index)
                if "pls_next" in h.columns:
                    rec["n_paired"] = len(common)
            rows.append(rec)

    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "persistence_grid.csv")
    out.to_csv(path, index=False)
    cols = [c for c in ["dgp", "rho", "window", "spectral_gap_true", "gap_hpin", "gap_pin",
                        "switch_rate_true", "switch_hpin", "switch_pin",
                        "pin_true", "pin_hpin", "pin_pin"] if c in out.columns]
    print(out[cols].round(4).to_string(index=False))
    print("\nsaved ->", path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    {"generate": cmd_generate, "hpin": cmd_hpin, "aggregate": cmd_aggregate,
     "generate-clean": cmd_generate_clean, "hpin-clean": cmd_hpin_clean}[cmd]()
