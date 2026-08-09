"""Validation gate (plan Etapa 2).

Hard gate  — CLEAN DGP (pure Markov-Poisson): the tied Baum-Welch must recover
             the true parameters (eps within 5%, mu within 10%, Gamma diagonal
             within 0.10) with a monotone log-likelihood.
Diagnostics — IG DGP (main experiment): estimates are reported but not gated;
             under the intentional overdispersion misspecification the Poisson
             HMM MLE is biased, which both PIN and HPIN face equally.
"""

import os
import time
import numpy as np
import pandas as pd

from hpin import fit_hpin, build_transition_matrix

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data", "fluxos", "sinteticos")

TRUE = dict(eps_b=10_000, eps_s=10_000, mu=5_000)
G_TRUE = build_transition_matrix(0.5, 0.5, 0.5, 0.5, 0.1, 0.1)
TRUE_PIN = 0.5263 * 5000 / (0.5263 * 5000 + 20000)


def report(label, fit, states, gate):
    ok = True
    for k, tol_rel in (("eps_b", 0.05), ("eps_s", 0.05), ("mu", 0.10)):
        rel = abs(fit[k] - TRUE[k]) / TRUE[k]
        flag = "OK" if rel < tol_rel else "FAIL"
        print(f"  {k:6s}: {fit[k]:10.1f} (true {TRUE[k]}, rel err {rel:.2%}) {flag if gate else ''}")
        ok &= rel < tol_rel
    diag_err = np.abs(np.diag(fit["Gamma"]) - np.diag(G_TRUE)).max()
    print(f"  Gamma diag max err: {diag_err:.3f} {'OK' if diag_err < 0.10 else 'FAIL' if gate else ''}")
    ok &= diag_err < 0.10
    acc = (fit["gamma_t"].argmax(axis=1) == states).mean()
    print(f"  steady_alpha={fit['steady_alpha']:.3f} (true 0.526) | "
          f"steady_pin={fit['steady_pin']:.4f} (true {TRUE_PIN:.4f}) | decode acc={acc:.3f}")
    dl = np.diff(fit["loglik_hist"])
    mono = (dl >= -1e-9 * np.abs(fit["loglik_hist"][:-1])).all()
    print(f"  loglik={fit['loglik']:.2f} in {fit['n_iter']} iters | monotone={mono}")
    ok &= bool(mono)
    if gate:
        print(f"=> {label}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ok = True
    for fname, label, gate in [
        ("fluxos_sinteticos_poisson_puro.csv", "CLEAN (hard gate)", True),
        ("fluxos_sinteticos_personalizado.csv", "IG (diagnostics)", False),
    ]:
        df = pd.read_csv(os.path.join(DATA_DIR, fname), index_col=0)
        t0 = time.time()
        fit = fit_hpin(df["buyer"].values, df["seller"].values)
        print(f"\n{label}: T={len(df)} fit in {time.time() - t0:.1f}s")
        res = report(label, fit, df["state"].values, gate)
        if gate:
            ok &= res

    # timing on short windows (rolling-window feasibility)
    df = pd.read_csv(os.path.join(DATA_DIR, "fluxos_sinteticos_personalizado.csv"), index_col=0)
    B, S = df["buyer"].values, df["seller"].values
    t0 = time.time()
    fits = [fit_hpin(B[i:i + 60], S[i:i + 60]) for i in range(0, 1200, 60)]
    dt = (time.time() - t0) / len(fits)
    iters = [f["n_iter"] for f in fits]
    print(f"\n20 window-60 fits: {dt * 1000:.0f} ms/fit | iters min/med/max = "
          f"{min(iters)}/{int(np.median(iters))}/{max(iters)}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
