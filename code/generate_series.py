"""Synthetic order-flow generation for the HPIN paper (revised).

Differences from the original `gerador de séries.ipynb`, as documented in the
revision:
- The Inverse Gaussian factor Z_t ~ IG(mean=1, scale=9) now scales the
  *intensities* (lambda * Z_t) before Poisson sampling, one common Z_t per
  period for both sides. This preserves the conditional Poisson structure
  (mixed-Poisson marginals), matching the description in the paper and the
  PIG construction of Griffin et al. (2021). The original code multiplied the
  sampled counts and truncated to int.
- True hidden states are saved alongside (B, S) for recovery diagnostics.
- All randomness flows through one numpy Generator seeded explicitly.

DGPs produced:
1. dynamic  — Markov states, eta=0.5, kappa=0.5, phi=0.5, omega=0.1 (paper's Gamma_s)
2. iid      — degenerate Gamma whose identical rows equal the stationary
              distribution of the dynamic Gamma (fairness check: PIN's own DGP)
3. clean    — same Markov states as (1) but no IG factor (pure Markov-Poisson):
              used for the estimator-consistency / parameter-recovery check
"""

import os
import numpy as np
import pandas as pd

from hpin import build_transition_matrix, stationary_distribution, emission_rates

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fluxos", "sinteticos")

T = 5000
EPS_B = 10_000
EPS_S = 10_000
MU = 5_000
SEED = 42
IG_SCALE = 9.0


def simulate(Gamma, T, eps_b, eps_s, mu, seed, ig_scale=IG_SCALE, ig_noise=True):
    rng = np.random.default_rng(seed)
    lam_b, lam_s = emission_rates(eps_b, eps_s, mu)

    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(3, p=Gamma[states[t - 1]])

    if ig_noise:
        Z = rng.wald(1.0, ig_scale, size=T)      # one factor per period, common to B and S
    else:
        Z = np.ones(T)
    B = rng.poisson(lam_b[states] * Z)
    S = rng.poisson(lam_s[states] * Z)
    return pd.DataFrame({"buyer": B, "seller": S, "state": states, "Z": Z})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- dynamic DGP (paper's Gamma_s) ---
    G_dyn = build_transition_matrix(eta=0.5, kappa=0.5, phi_plus=0.5, phi_minus=0.5,
                                    omega_plus=0.1, omega_minus=0.1)
    df_dyn = simulate(G_dyn, T, EPS_B, EPS_S, MU, seed=SEED)
    df_dyn.to_csv(os.path.join(OUT_DIR, "fluxos_sinteticos_personalizado.csv"))

    # --- iid DGP (degenerate Gamma = stationary of the dynamic one) ---
    u = stationary_distribution(G_dyn)
    G_iid = np.tile(u, (3, 1))
    df_iid = simulate(G_iid, T, EPS_B, EPS_S, MU, seed=SEED + 1)
    df_iid.to_csv(os.path.join(OUT_DIR, "fluxos_sinteticos_iid.csv"))

    # --- clean DGP (no IG factor; estimator-recovery check) ---
    df_clean = simulate(G_dyn, T, EPS_B, EPS_S, MU, seed=SEED, ig_noise=False)
    df_clean.to_csv(os.path.join(OUT_DIR, "fluxos_sinteticos_poisson_puro.csv"))

    print("dynamic Gamma:\n", np.round(G_dyn, 4))
    print("stationary u:", np.round(u, 4), "| alpha_true =", round(u[1] + u[2], 4))
    for name, df in [("dynamic", df_dyn), ("iid", df_iid), ("clean", df_clean)]:
        print(f"{name}: B mean={df.buyer.mean():.1f} std={df.buyer.std():.1f} | "
              f"S mean={df.seller.mean():.1f} std={df.seller.std():.1f} | "
              f"state freq={np.bincount(df.state, minlength=3) / len(df)}")


if __name__ == "__main__":
    main()
