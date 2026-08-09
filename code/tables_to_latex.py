"""Renders the paper's appendix tables from results/tables CSVs into LaTeX
snippets (results/tables/latex/*.tex), matching the manuscript's structures.
"""

import os
import pandas as pd

HERE = os.path.dirname(__file__)
TAB = os.path.join(HERE, "..", "results", "tables")
OUT = os.path.join(TAB, "latex")

PARAMS = ["pin", "alpha", "delta", "epsilon_b", "epsilon_s", "mu"]
PARAM_TEX = {"pin": "pin", "alpha": r"$\alpha$", "delta": r"$\delta$",
             "epsilon_b": r"$\epsilon_b$", "epsilon_s": r"$\epsilon_s$", "mu": r"$\mu$"}
WINDOWS = [60, 90, 120, 150, 180]


def fnum(x, dec=4):
    if pd.isna(x):
        return "--"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.{dec}f}"


def fpval(p):
    if pd.isna(p):
        return "--"
    return "$<10^{-4}$" if p < 1e-4 else f"{p:.2f}"


def stats_description():
    df = pd.read_csv(os.path.join(TAB, "stats_description.csv"))
    lines = []
    for w in WINDOWS:
        sub = df[df["window"] == w]
        for _, r in sub.iterrows():
            stat = r["statistics"].replace("%", r"\%")
            cells = []
            for p in PARAMS_ORDER:
                for m in ("PIN", "HPIN"):
                    cells.append(fnum(r[f"{p}_{m}"], 4 if p in ("pin", "alpha", "delta") else 2))
            lines.append(f"{stat} & {w} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if w != WINDOWS[-1] else "")
    with open(os.path.join(OUT, "stats_description.tex"), "w") as f:
        f.write("\n".join(lines))


PARAMS_ORDER = ["alpha", "delta", "epsilon_b", "epsilon_s", "mu", "pin"]


def wilcox_simple(crit, fname_out):
    df = pd.read_csv(os.path.join(TAB, f"wilcoxon_{crit}.csv"))
    lines = []
    for wi, w in enumerate(WINDOWS):
        sub = df[df["window"] == w]
        for p in ["pin", "alpha", "delta", "epsilon_b", "epsilon_s", "mu"]:
            r = sub[sub["param"] == p].iloc[0]
            lines.append(f"{w} & {PARAM_TEX[p]} & {fnum(r['statistics'], 1)} & "
                         f"{fpval(r['p_value'])} \\\\")
        if wi < len(WINDOWS) - 1:
            lines.append(r"\midrule")
    with open(os.path.join(OUT, fname_out), "w") as f:
        f.write("\n".join(lines))


def wilcox_pacf():
    df = pd.read_csv(os.path.join(TAB, "wilcoxon_PACF.csv"))
    lines = []
    for wi, w in enumerate(WINDOWS):
        for p in ["pin", "alpha", "delta", "epsilon_b", "epsilon_s", "mu"]:
            cells = []
            for lag in range(1, 6):
                r = df[(df["window"] == w) & (df["param"] == p) & (df["lag"] == lag)].iloc[0]
                cells.append(f"{lag} & {fnum(r['statistics'], 1)} & {fpval(r['p_value'])}")
            lines.append(f"{w} & {PARAM_TEX[p]} & " + " & ".join(cells) + r" \\")
        if wi < len(WINDOWS) - 1:
            lines.append(r"\midrule")
    with open(os.path.join(OUT, "wilcoxon_PACF.tex"), "w") as f:
        f.write("\n".join(lines))


def pls_tables():
    df = pd.read_csv(os.path.join(TAB, "pls_stats.csv"))
    lines = []
    for mi, m in enumerate(("HPIN", "PIN")):
        for stat in ["mean", "std", "min", "q25", "q50", "q75", "max"]:
            label = {"q25": r"25\%", "q50": r"50\%", "q75": r"75\%"}.get(stat, stat)
            cells = [fnum(df[(df["model"] == m) & (df["window"] == w)][stat].iloc[0], 2)
                     for w in WINDOWS]
            lines.append(f"\\gls{{{m.lower()}}} & {label} & " + " & ".join(cells) + r" \\")
        if mi == 0:
            lines.append(r"\midrule")
    with open(os.path.join(OUT, "pls_stats.tex"), "w") as f:
        f.write("\n".join(lines))

    dfw = pd.read_csv(os.path.join(TAB, "wilcoxon_PLS.csv"))
    lines = [f"{int(r['window'])} & {fnum(r['statistics'], 1)} & {fpval(r['p_value'])} \\\\"
             for _, r in dfw.iterrows()]
    with open(os.path.join(OUT, "wilcoxon_PLS.tex"), "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    stats_description()
    for crit, out in [("mean", "wilcoxon_mean.tex"), ("std", "wilcoxon_std.tex"),
                      ("MIV", "wilcoxon_MIV.tex")]:
        wilcox_simple(crit, out)
    wilcox_pacf()
    pls_tables()
    print("latex snippets ->", OUT)


if __name__ == "__main__":
    main()
