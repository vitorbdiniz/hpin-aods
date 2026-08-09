"""Rolling-window HPIN estimation (plan Etapa 2b).

Window convention: for each i in [w, T), estimate on the half-open window
[i-w, i) (0-based) and record the row under index i — the index of the first
out-of-window observation, for which the one-step-ahead predictive log score
(PLS) is also computed. Cold start at every window (no warm-start), with the
published conservative initialization.

alpha / delta / pin follow Eq. 12 of the paper implemented through the
stationary distribution of the estimated Gamma (u_infty); the prior-based
variants are stored alongside for completeness.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from hpin import fit_hpin, predictive_logscore_hpin

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data", "fluxos", "sinteticos")
OUT_DIR = os.path.join(HERE, "..", "data", "pins", "sinteticos")

WINDOWS = [60, 90, 120, 150, 180]
DATASETS = ["fluxos_sinteticos_personalizado", "fluxos_sinteticos_iid"]

COLS = ["eta", "kappa", "phi_plus", "phi_minus", "omega_plus", "omega_minus",
        "epsilon_b", "epsilon_s", "mu",
        "alpha", "delta", "pin",
        "alpha_prior", "delta_prior", "pin_prior",
        "log_likelihood", "n_iter", "pls_next"]


def run(dataset):
    df = pd.read_csv(os.path.join(DATA_DIR, dataset + ".csv"), index_col=0)
    B = df["buyer"].values.astype(float)
    S = df["seller"].values.astype(float)
    T = len(df)
    os.makedirs(OUT_DIR, exist_ok=True)

    for w in WINDOWS:
        t0 = time.time()
        rows = {}
        for i in range(w, T):
            fit = fit_hpin(B[i - w:i], S[i - w:i])
            pls = predictive_logscore_hpin(fit, B[i], S[i]) if i < T else np.nan
            rows[i] = [fit["eta"], fit["kappa"], fit["phi_plus"], fit["phi_minus"],
                       fit["omega_plus"], fit["omega_minus"],
                       fit["eps_b"], fit["eps_s"], fit["mu"],
                       fit["steady_alpha"], fit["steady_delta"], fit["steady_pin"],
                       fit["alpha"], fit["delta"], fit["pin"],
                       fit["loglik"], fit["n_iter"], pls]
        out = pd.DataFrame.from_dict(rows, orient="index", columns=COLS)
        out.index.name = "i"
        path = os.path.join(OUT_DIR, f"pin_hmm_{w}_{dataset}.csv")
        out.to_csv(path)
        print(f"{dataset} w={w}: {len(out)} fits in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] or DATASETS
    for ds in targets:
        run(ds)
