"""HPIN — Hidden Probability of Informed Trading.

Clean tied Baum-Welch implementation matching the paper's Appendix A
(ap:baum-welch) equation by equation:

- 3-state HMM (N, +, -) with conditionally independent Poisson emissions,
  intensities tied through the PIN structure (Eq. lambda-relations):
      lambda_B = [eps_b, eps_b + mu, eps_b]
      lambda_S = [eps_s, eps_s, eps_s + mu]
- E-step: scaled forward-backward (log-sum-exp per period; the per-period
  normalization cancels the common Poisson factors, playing the role of the
  Ersan (2016) likelihood factorization in the dynamic setting).
- M-step: Dirichlet-type pseudocount c on pi and Gamma (Rabiner 1989);
  tied emissions via the closed-form conditional updates (positive roots of
  scalar quadratics, eqs. mstep-eb / mstep-es / mstep-mu), cycled once per
  iteration (ECM); mu >= 0 enforced by truncation.
- Convergence: relative log-likelihood change below `tol`, capped at
  `max_iter` iterations.

No temperature annealing, no sticky prior, no damping, no free-emission
warm-up: state identity is fixed a priori by the parameter binding, so label
switching cannot occur (see paper, sec:identifiability).
"""

import numpy as np
from scipy.special import gammaln

STATES = ("N", "+", "-")
K = 3


# ---------------------------------------------------------------------------
# Parametrization helpers (identical conventions to the paper / old notebook)
# ---------------------------------------------------------------------------

def build_transition_matrix(eta, kappa, phi_plus, phi_minus, omega_plus, omega_minus):
    """Gamma in the economically-parametrized form (paper Eq. hmm-transition-matrix)."""
    G = np.array([
        [1.0 - eta, eta * (1.0 - kappa), eta * kappa],
        [(1.0 - phi_plus) * (1.0 - omega_plus), phi_plus, (1.0 - phi_plus) * omega_plus],
        [(1.0 - phi_minus) * (1.0 - omega_minus), (1.0 - phi_minus) * omega_minus, phi_minus],
    ])
    return G / G.sum(axis=1, keepdims=True)


def extract_transition_params(G):
    """Inverse of build_transition_matrix."""
    return dict(
        eta=float(1.0 - G[0, 0]),
        kappa=float(G[0, 2] / (1.0 - G[0, 0])) if G[0, 0] < 1.0 else 0.0,
        phi_plus=float(G[1, 1]),
        phi_minus=float(G[2, 2]),
        omega_plus=float(G[1, 2] / (1.0 - G[1, 1])) if G[1, 1] < 1.0 else 0.0,
        omega_minus=float(G[2, 1] / (1.0 - G[2, 2])) if G[2, 2] < 1.0 else 0.0,
    )


def emission_rates(eps_b, eps_s, mu):
    lam_b = np.array([eps_b, eps_b + mu, eps_b], dtype=float)
    lam_s = np.array([eps_s, eps_s, eps_s + mu], dtype=float)
    return lam_b, lam_s


def stationary_distribution(G):
    """Left eigenvector of Gamma for eigenvalue 1, sum-normalized."""
    eigvals, eigvecs = np.linalg.eig(G.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    v = np.real(eigvecs[:, idx])
    v = np.abs(v)
    return v / v.sum()


def estimate_pin(alpha, eps_b, eps_s, mu):
    denom = alpha * mu + eps_b + eps_s
    return float(alpha * mu / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# E-step: scaled forward-backward
# ---------------------------------------------------------------------------

def log_emissions(B, S, lam_b, lam_s):
    """(T, K) matrix of log b_h(y_t), full Poisson pmf including gammaln terms."""
    B = np.asarray(B, dtype=float)[:, None]
    S = np.asarray(S, dtype=float)[:, None]
    lb = np.log(lam_b)[None, :]
    ls = np.log(lam_s)[None, :]
    return (B * lb - lam_b[None, :] - gammaln(B + 1.0)
            + S * ls - lam_s[None, :] - gammaln(S + 1.0))


def forward_backward(logE, pi, G):
    """Scaled forward-backward.

    Returns (gamma_t, xi_sum, loglik, alpha_filt_last):
      gamma_t        (T, K) posterior state probabilities
      xi_sum         (K, K) sum over t of pairwise posteriors xi_t(i, j)
      loglik         observed-data log-likelihood
      alpha_filt_last (K,)  filtered state distribution at T (for prediction)
    """
    T = logE.shape[0]
    # per-period max: cancels factors common to the three states (Ersan-style)
    M = logE.max(axis=1, keepdims=True)
    Eb = np.exp(logE - M)          # bounded in (0, 1]

    alpha = np.zeros((T, K))
    cnorm = np.zeros(T)
    a = pi * Eb[0]
    cnorm[0] = a.sum()
    alpha[0] = a / cnorm[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ G) * Eb[t]
        cnorm[t] = a.sum()
        alpha[t] = a / cnorm[t]

    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (G @ (Eb[t + 1] * beta[t + 1])) / cnorm[t + 1]

    gamma_t = alpha * beta
    gamma_t /= gamma_t.sum(axis=1, keepdims=True)

    # xi summed over t (vectorized): xi_t = alpha_t (x) [G * Eb_{t+1} beta_{t+1}] / c_{t+1}
    w = (Eb[1:] * beta[1:]) / cnorm[1:, None]          # (T-1, K)
    xi_sum = G * (alpha[:-1].T @ w)                    # (K, K)

    loglik = float(np.log(cnorm).sum() + M.sum())
    return gamma_t, xi_sum, loglik, alpha[-1]


# ---------------------------------------------------------------------------
# M-step: tied emission updates (closed-form conditional maximizers)
# ---------------------------------------------------------------------------

def _positive_root(a, b, c):
    """Larger root of a x^2 + b x + c = 0, truncated at 0."""
    disc = max(b * b - 4.0 * a * c, 0.0)
    return max((-b + np.sqrt(disc)) / (2.0 * a), 0.0)


def update_emissions_tied(gamma_t, B, S, eps_b, eps_s, mu):
    """One ECM cycle on (eps_b, eps_s, mu) — paper eqs. mstep-eb/es/mu."""
    T = len(B)
    n = gamma_t.sum(axis=0)                    # (K,)
    Bw = gamma_t.T @ B                         # (K,)
    Sw = gamma_t.T @ S
    B_tot = float(B.sum())
    S_tot = float(S.sum())

    # eps_b given mu:  T e^2 + (T mu - B_tot) e - (B_N + B_-) mu = 0
    eps_b = _positive_root(T, T * mu - B_tot, -(Bw[0] + Bw[2]) * mu)
    # eps_s given mu
    eps_s = _positive_root(T, T * mu - S_tot, -(Sw[0] + Sw[1]) * mu)
    # mu given eps_b, eps_s:  m mu^2 + [m(eb+es) - B_+ - S_-] mu + [m eb es - B_+ es - S_- eb] = 0
    m = n[1] + n[2]
    if m > 0:
        mu = _positive_root(m,
                            m * (eps_b + eps_s) - Bw[1] - Sw[2],
                            m * eps_b * eps_s - Bw[1] * eps_s - Sw[2] * eps_b)
    else:
        mu = 0.0
    return float(eps_b), float(eps_s), float(mu)


# ---------------------------------------------------------------------------
# Initialization (as published in the paper, sec:experiment)
# ---------------------------------------------------------------------------

def default_init(B, S, eta=0.2, kappa=0.5, phi_plus=0.5, phi_minus=0.5,
                 omega_plus=0.05, omega_minus=0.05, priors=(0.8, 0.1, 0.1)):
    eps_b0 = float(np.mean(B))
    eps_s0 = float(np.mean(S))
    mu0 = 0.5 * (eps_b0 + eps_s0) / 2.0
    return dict(
        pi=np.array(priors, dtype=float),
        Gamma=build_transition_matrix(eta, kappa, phi_plus, phi_minus, omega_plus, omega_minus),
        eps_b=eps_b0, eps_s=eps_s0, mu=mu0,
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_hpin(B, S, init=None, max_iter=1000, tol=1e-8, pseudocount=1e-3):
    """Tied Baum-Welch (paper alg:hpin). Returns dict of estimates + diagnostics."""
    B = np.asarray(B, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float).reshape(-1)
    if init is None:
        init = default_init(B, S)
    pi = init["pi"].copy()
    G = init["Gamma"].copy()
    eps_b, eps_s, mu = init["eps_b"], init["eps_s"], init["mu"]
    c = pseudocount

    loglik_hist = []
    loglik_old = None
    for it in range(max_iter):
        lam_b, lam_s = emission_rates(eps_b, eps_s, mu)
        logE = log_emissions(B, S, lam_b, lam_s)
        gamma_t, xi_sum, loglik, alpha_last = forward_backward(logE, pi, G)
        loglik_hist.append(loglik)
        if loglik_old is not None and abs(loglik - loglik_old) < tol * abs(loglik_old):
            break
        loglik_old = loglik

        # M-step: dynamics (with pseudocounts) + tied emissions (ECM cycle)
        pi = (gamma_t[0] + c) / (1.0 + K * c)
        G = (xi_sum + c) / (gamma_t[:-1].sum(axis=0)[:, None] + K * c)
        G /= G.sum(axis=1, keepdims=True)
        eps_b, eps_s, mu = update_emissions_tied(gamma_t, B, S, eps_b, eps_s, mu)

    # sanity: with tied emissions the labels are fixed by construction
    lam_b, lam_s = emission_rates(eps_b, eps_s, mu)
    assert lam_b[1] >= lam_b[0] and lam_s[2] >= lam_s[0], "label binding violated"

    alpha = float(pi[1] + pi[2])
    delta = float(pi[2] / alpha) if alpha > 0 else 0.0
    u_inf = stationary_distribution(G)
    steady_alpha = float(u_inf[1] + u_inf[2])
    steady_delta = float(u_inf[2] / steady_alpha) if steady_alpha > 0 else np.nan

    out = dict(
        eps_b=eps_b, eps_s=eps_s, mu=mu,
        pi=pi, Gamma=G,
        alpha=alpha, delta=delta,
        pin=estimate_pin(alpha, eps_b, eps_s, mu),
        steady_alpha=steady_alpha, steady_delta=steady_delta,
        steady_pin=estimate_pin(steady_alpha, eps_b, eps_s, mu),
        loglik=loglik_hist[-1], n_iter=len(loglik_hist),
        loglik_hist=np.array(loglik_hist),
        alpha_filt_last=alpha_last,
        gamma_t=gamma_t,
    )
    out.update(extract_transition_params(G))
    return out


# ---------------------------------------------------------------------------
# One-step-ahead predictive log score (out of sample)
# ---------------------------------------------------------------------------

def predictive_logscore_hpin(fit, B_next, S_next):
    """log p(y_{T+1} | y_{1:T}) under the fitted HPIN."""
    lam_b, lam_s = emission_rates(fit["eps_b"], fit["eps_s"], fit["mu"])
    pred = fit["alpha_filt_last"] @ fit["Gamma"]          # P(X_{T+1} | y_{1:T})
    logE = log_emissions(np.array([B_next]), np.array([S_next]), lam_b, lam_s)[0]
    m = logE.max()
    return float(m + np.log(np.dot(pred, np.exp(logE - m))))


def predictive_logscore_pin(alpha, delta, eps_b, eps_s, mu, B_next, S_next):
    """log p(y) under the static PIN mixture (i.i.d. => predictive = marginal)."""
    w = np.array([1.0 - alpha, alpha * (1.0 - delta), alpha * delta])
    w = np.clip(w, 1e-300, None)
    w /= w.sum()
    lam_b, lam_s = emission_rates(eps_b, eps_s, mu)
    logE = log_emissions(np.array([B_next]), np.array([S_next]), lam_b, lam_s)[0]
    x = np.log(w) + logE
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))
