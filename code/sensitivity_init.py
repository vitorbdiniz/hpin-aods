"""Initialization-sensitivity experiment (reviewer #2 item 1).

N randomized initializations of the tied Baum-Welch:
  - pi ~ Dirichlet(1,1,1)
  - each row of Gamma ~ Dirichlet(1,1,1)
  - eps_b0, eps_s0 = sample means x U(0.7, 1.3)
  - mu0 = 0.5 (eps_b0 + eps_s0) / 2 x U(0.2, 2.0)

Cases:
  (a) full dynamic (IG) series, T=5000 — the main, misspecified DGP;
  (b) a representative 60-period window of (a) — the rolling-analysis regime;
  (c) full clean series (well-specified Markov-Poisson) — isolates what the
      model structure contributes vs. what the overdispersion misspecification
      contributes to multimodality.

Under the tied parameterization the only structurally bad basin is the known
identifiability boundary mu = 0 (all three emission laws collapse); its
log-likelihood is drastically worse, so a best-of-restarts rule removes it
trivially. The script also counts degenerate (mu ~ 0) solutions among the
production rolling fits, where the conservative initialization is used.
"""

import glob
import os
import numpy as np
import pandas as pd

from hpin import fit_hpin

HERE = os.path.dirname(__file__)
FLUX_DIR = os.path.join(HERE, "..", "data", "fluxos", "sinteticos")
PINS_DIR = os.path.join(HERE, "..", "data", "pins", "sinteticos")
DATA = os.path.join(FLUX_DIR, "fluxos_sinteticos_personalizado.csv")
DATA_CLEAN = os.path.join(FLUX_DIR, "fluxos_sinteticos_poisson_puro.csv")
OUT_DIR = os.path.join(HERE, "..", "results", "tables")

N_INIT = 50
SEED = 123
WINDOW_START = 2000          # representative mid-sample 60-period window


def random_init(rng, B, S):
    eps_b0 = float(np.mean(B)) * rng.uniform(0.7, 1.3)
    eps_s0 = float(np.mean(S)) * rng.uniform(0.7, 1.3)
    mu0 = 0.5 * (eps_b0 + eps_s0) / 2.0 * rng.uniform(0.2, 2.0)
    G = rng.dirichlet(np.ones(3), size=3)
    return dict(pi=rng.dirichlet(np.ones(3)), Gamma=G, eps_b=eps_b0, eps_s=eps_s0, mu=mu0)


def run_case(label, B, S):
    rng = np.random.default_rng(SEED)
    rows = []
    for j in range(N_INIT):
        fit = fit_hpin(B, S, init=random_init(rng, B, S))
        rows.append(dict(run=j,
                         eps_b=fit["eps_b"], eps_s=fit["eps_s"], mu=fit["mu"],
                         alpha=fit["steady_alpha"], delta=fit["steady_delta"],
                         pin=fit["steady_pin"],
                         spectral_diag=float(np.diag(fit["Gamma"]).mean()),
                         loglik=fit["loglik"], n_iter=fit["n_iter"]))
    df = pd.DataFrame(rows)
    best = df["loglik"].max()
    df["at_best"] = np.abs(df["loglik"] - best) < 1e-4 * abs(best)
    df["degenerate"] = df["mu"] < 1e-6
    print(f"\n{label}: {N_INIT} random inits | best loglik={best:.2f} | "
          f"at best: {df['at_best'].mean():.0%} | degenerate (mu=0): "
          f"{df['degenerate'].sum()} | rel. loglik spread: "
          f"{(best - df['loglik'].min()) / abs(best):.2e}")
    print(df[["eps_b", "eps_s", "mu", "alpha", "pin", "loglik"]].describe()
          .loc[["mean", "std", "min", "max"]].round(4))
    nd = df[~df["degenerate"]]
    print("range among non-degenerate runs:",
          {k: (round(float(nd[k].min()), 4), round(float(nd[k].max()), 4))
           for k in ["mu", "pin"]})
    return df.assign(case=label)


def production_degenerate_count():
    """Degenerate (mu ~ 0) solutions among the production rolling fits."""
    total, degen = 0, 0
    for f in glob.glob(os.path.join(PINS_DIR, "pin_hmm_*.csv")):
        d = pd.read_csv(f, index_col=0)
        total += len(d)
        degen += int((d["mu"] < 1e-6).sum())
    print(f"\nproduction rolling fits: {total} | degenerate (mu=0): {degen}")
    return total, degen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ig = pd.read_csv(DATA, index_col=0)
    B, S = ig["buyer"].values.astype(float), ig["seller"].values.astype(float)
    clean = pd.read_csv(DATA_CLEAN, index_col=0)

    cases = [
        run_case("ig_full_series", B, S),
        run_case("ig_window_60", B[WINDOW_START:WINDOW_START + 60],
                 S[WINDOW_START:WINDOW_START + 60]),
        run_case("clean_full_series", clean["buyer"].values.astype(float),
                 clean["seller"].values.astype(float)),
    ]
    out = pd.concat(cases, ignore_index=True)
    out.to_csv(os.path.join(OUT_DIR, "sensitivity_init.csv"), index=False)

    total, degen = production_degenerate_count()
    summary = (out.groupby("case")
               .agg(n=("run", "size"), at_best=("at_best", "mean"),
                    degenerate=("degenerate", "sum"),
                    pin_min=("pin", "min"), pin_max=("pin", "max"),
                    mu_min=("mu", "min"), mu_max=("mu", "max"))
               .reset_index())
    summary.loc[len(summary)] = ["production_rolling", total, np.nan, degen,
                                 np.nan, np.nan, np.nan, np.nan]
    summary.to_csv(os.path.join(OUT_DIR, "sensitivity_summary.csv"), index=False)
    print("\nsaved ->", os.path.join(OUT_DIR, "sensitivity_init.csv"),
          "and sensitivity_summary.csv")


if __name__ == "__main__":
    main()
