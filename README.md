# HPIN — Replication package

Replication code and data for *"Unveiling Latent Market Dynamics: A Hidden
Markov Framework for Informed Trading"* (under review, Annals of Data Science).

The HPIN generalizes the classical PIN model of Easley et al. (1996) by letting
the informational state follow a first-order hidden Markov chain with Poisson
emissions tied through the PIN structure (ε_b, ε_s, μ). Estimation is a tied
Baum–Welch (EM) routine with closed-form conditional M-step updates, derived in
Appendix A of the paper and implemented in [`code/hpin.py`](code/hpin.py).

## Environment

- **Python** ≥ 3.12 — `pip install -r code/requirements.txt`
  (numpy, pandas, scipy, statsmodels)
- **R** ≥ 4.4 — packages `PINstimation` (classical PIN baseline, Ersan–Alıcı
  initial sets), `ggplot2`, `dplyr`, `tidyr`, `stringr` (figures)

All experiments are seeded (data generation: 42; bootstrap: 7; sensitivity: 123).

## Execution order (from `code/`)

| Step | File | What it does |
|------|------|--------------|
| 1 | `01_generate_series.ipynb` | Simulates the three DGPs (dynamic / iid / clean) into `data/fluxos/sinteticos/` — outputs are also committed, so this step can be skipped |
| 2 | `02_hpin_validation.ipynb` | Parameter-recovery gate on the clean DGP |
| 3 | `03_hpin_rolling.ipynb` | Rolling-window HPIN (60–180, cold start, out-of-sample PLS) |
| 4 | `Rscript run_pin_classic.R` | Rolling-window classical PIN via PINstimation `pin_ea` (parallel; takes several hours — outputs are committed under `data/pins/` so this step can be skipped) |
| 5 | `04_analysis.ipynb` | All tables: metrics, Wilcoxon tests, initialization sensitivity, fairness check |
| 6 | `Rscript make_figures.R` | All paper figures into `results/figs/` |

## Layout

- `code/hpin.py` — model, tied Baum–Welch, predictive scores (paper Alg. 1 + App. A)
- `code/*.py`, `code/*.ipynb` — pipeline modules and notebook drivers
- `code/run_pin_classic.R`, `code/make_figures.R` — R side (baseline + figures)
- `data/fluxos/` — simulated order-flow series (buyer, seller, true state, Z factor)
- `data/pins/` — rolling-window estimates for both models
- `results/` — tables and figures consumed by the paper

## Conventions

- Rolling windows are half-open `[i−w, i)` (0-based); each row is indexed by
  `i`, the first out-of-window observation, which is also the observation
  scored by the one-step-ahead predictive log score (PLS).
- Every window is estimated from a cold start with the conservative
  initialization described in the paper (no warm starts).
- The half-life metric is reported as undefined when the lag-1 PACF ∉ (0, 1).
