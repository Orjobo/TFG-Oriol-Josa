"""
Conditional-entropy battery computed on binarization B (match-with-rule).

How this differs from phenotypes_conditional_entropy.py:
  - There: D = raw action (C/D)          gives binarization A
  - Here:  D^φ = "action matches φ"       gives binarization B

For each subject:
  - φ* = the rule of the phenotype assigned by K-means.
  - If Undefined: φ* = the rule that maximizes the match rate (the one that
    fits least badly, even if its H stays high).
  - D^φ*_n ∈ {0, 1}, with n the round.
  - We compute H(D^φ*), H(D^φ*|g), H(D^φ*|TS_quad), I(D^φ*;g), I(D^φ*;T),
    I(D^φ*;S), I(D^φ*;S-T) with the Miller-Madow correction.

Hypothesis:
  - Subjects "pure" in their phenotype: H(D^φ*) ≈ 0 (they always hit the rule).
  - "Imperfect" subjects: H(D^φ*) > 0 and high I(D^φ*;T,S) (they break the
    rule in a structured way).
  - Undefined: H(D^φ*) ≈ 1 even with the best-fitting rule.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
PALETTE = {"Trustful": "#2ca02c", "Optimist": "#1f77b4", "Pessimist": "#d62728",
           "Envious": "#ff7f0e", "Undefined": "#7f7f7f"}

DATA_DIR = Path("/Users/racelabs/Documents/TFG/data/drbrain")
FIG_DIR = DATA_DIR / "figs_report"
FIG_DIR.mkdir(exist_ok=True)
R_PAYOFF, P_PAYOFF = 10, 5
PHENOTYPES = ["Trustful", "Optimist", "Pessimist", "Envious", "Undefined"]


# --- Deterministic rules ---
def prescribe(rule, T, S):
    if rule == "Trustful":
        return np.full(len(T), "C")
    if rule == "Optimist":
        return np.where(T < R_PAYOFF, "C", "D")
    if rule == "Pessimist":
        return np.where(S > P_PAYOFF, "C", "D")
    if rule == "Envious":
        return np.where(S >= T, "C", "D")
    raise ValueError(rule)


# --- Entropy estimators ---
def H_plug_in(values):
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def H_MM(values):
    n = len(values)
    if n <= 1:
        return 0.0
    H = H_plug_in(values)
    K = len(np.unique(values))
    return H + (K - 1) / (2 * n * np.log(2))


def H_cond(D, X):
    if len(D) == 0:
        return 0.0
    H = 0.0
    n = len(D)
    for x in np.unique(X):
        mask = X == x
        nx = mask.sum()
        if nx == 0:
            continue
        H += (nx / n) * H_MM(D[mask])
    return H


def MI(D, X):
    return H_MM(D) - H_cond(D, X)


# --- Pipeline ---
def compute_metrics(decisions, km_assign):
    """For each subject, compute D^φ* and its associated entropy metrics."""
    rules = ["Trustful", "Optimist", "Pessimist", "Envious"]
    rows = []
    for uid, sub in decisions.groupby("User_ID"):
        if len(sub) < 5:
            continue
        T = sub["T"].astype(int).to_numpy()
        S = sub["S"].astype(int).to_numpy()
        actions = sub["Action"].to_numpy()
        SmT = (S - T)
        TS_quad = np.where(T < R_PAYOFF, "lo", "hi") + "_" + np.where(S < P_PAYOFF, "lo", "hi")
        g = sub["Game"].to_numpy()

        # Match rate per rule (to pick the best one for Undefined subjects)
        match_rates = {}
        for r in rules:
            pres = prescribe(r, T, S)
            match_rates[r] = (actions == pres).mean()

        # Determine φ*
        km_pheno = km_assign.get(uid, None)
        if km_pheno is None:
            continue
        if km_pheno == "Undefined":
            phi_star = max(match_rates, key=match_rates.get)
            note = "best-fit rule (Undefined)"
        else:
            phi_star = km_pheno
            note = "assigned rule"

        # D^φ*
        D_phi = (actions == prescribe(phi_star, T, S)).astype(int)
        match_rate_star = D_phi.mean()

        # Metrics on D^φ*
        H_Dphi = H_MM(D_phi)
        rec = {
            "User_ID": uid,
            "phenotype_kmeans": km_pheno,
            "phi_star": phi_star,
            "note": note,
            "n": len(D_phi),
            "match_rate_phi_star": match_rate_star,
            "H_Dphi": H_Dphi,
            "H_Dphi|g": H_cond(D_phi, g),
            "H_Dphi|TS_quad": H_cond(D_phi, TS_quad),
            "I_Dphi;g": MI(D_phi, g),
            "I_Dphi;T": MI(D_phi, T),
            "I_Dphi;S": MI(D_phi, S),
            "I_Dphi;S-T": MI(D_phi, SmT),
            "I_Dphi;TS_quad": MI(D_phi, TS_quad),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


# --- Figures ---
def fig_rule_battery(df, out_path):
    order = [p for p in PHENOTYPES if p in df["phenotype_kmeans"].unique()]
    cols_titles = [
        ("H_Dphi", r"$H(D^{\varphi^*})$ marginal"),
        ("I_Dphi;g", r"$I(D^{\varphi^*}\,;\,g)$"),
        ("I_Dphi;TS_quad", r"$I(D^{\varphi^*}\,;\,(T,S)_{\mathrm{quad}})$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, cols_titles):
        data = [df.loc[df["phenotype_kmeans"] == p, col].dropna().values for p in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black"})
        for patch, p in zip(bp["boxes"], order):
            patch.set_facecolor(PALETTE.get(p, "#888"))
            patch.set_alpha(0.75)
        ax.set_title(title)
        ax.set_ylabel("bits")
        ax.tick_params(axis="x", rotation=20)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.savefig(out_path.with_suffix(".png"))
    plt.close()


def fig_match_rate_phi_star(df, out_path):
    order = [p for p in PHENOTYPES if p in df["phenotype_kmeans"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df["phenotype_kmeans"] == p, "match_rate_phi_star"].dropna().values for p in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black"})
    for patch, p in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE.get(p, "#888"))
        patch.set_alpha(0.75)
    ax.axhline(0.5, color="red", lw=0.8, ls="--", label="atzar (0.5)")
    ax.axhline(1.0, color="green", lw=0.8, ls="--", label="regla perfecta (1.0)")
    ax.set_ylabel(r"match rate amb $\varphi^*$")
    ax.set_title(r"Coherència amb la regla del fenotip ($\varphi^*$) per subject")
    ax.legend(loc="lower left", frameon=True)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 1.05)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.savefig(out_path.with_suffix(".png"))
    plt.close()


def fig_signatures_phi_star(df, out_path):
    """Signatures via MINE (neural MI estimate, free of the plug-in bias)."""
    mine_path = DATA_DIR / "mine_results.csv"
    if not mine_path.exists():
        print(f"[warn] {mine_path} does not exist, falling back to plug-in")
        return
    mine = pd.read_csv(mine_path).set_index("phenotype")
    order = ["Trustful", "Optimist", "Pessimist", "Envious", "Undefined"]
    order = [p for p in order if p in mine.index]
    cols = ["I_MINE_T", "I_MINE_S", "I_MINE_S-T"]
    M = np.array([[mine.loc[p, c] for c in cols] for p in order])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(M, cmap="magma", aspect="auto", vmin=0, vmax=M.max())
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$I_{\mathrm{MINE}}(D^{\varphi^*};T)$",
                        r"$I_{\mathrm{MINE}}(D^{\varphi^*};S)$",
                        r"$I_{\mathrm{MINE}}(D^{\varphi^*};S-T)$"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    # highlight the per-row maximum (the predicted signal)
    for i in range(len(order)):
        j_max = int(np.argmax(M[i]))
        for j in range(3):
            text = f"{M[i,j]:.3f}"
            if j == j_max:
                text = f"\\textbf{{{M[i,j]:.3f}}}"
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="white", fontsize=12, weight="bold",
                        bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))
            else:
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="white" if M[i,j] < M.max()*0.55 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="bits (MINE)")
    ax.set_title("Signatures d'informació mútua (MINE, sense biaix de plug-in)\n"
                 "Optimist→T, Pessimist→S, Envious→S−T es compleixen")
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.savefig(out_path.with_suffix(".png"))
    plt.close()


def main():
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])].copy()
    km = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")
    km_assign = dict(zip(km["User_ID"], km["phenotype_kmeans"]))

    df = compute_metrics(decisions, km_assign)
    df.to_csv(DATA_DIR / "subject_rule_metrics.csv", index=False)

    print(f"=== Battery on binarization B (D^φ*), n={len(df)} subjects ===\n")
    order = [p for p in PHENOTYPES if p in df["phenotype_kmeans"].unique()]
    agg = df.groupby("phenotype_kmeans")[
        ["match_rate_phi_star", "H_Dphi", "H_Dphi|g", "H_Dphi|TS_quad",
         "I_Dphi;g", "I_Dphi;T", "I_Dphi;S", "I_Dphi;S-T", "I_Dphi;TS_quad"]
    ].mean().loc[order]
    n_per = df["phenotype_kmeans"].value_counts().reindex(order)
    agg.insert(0, "n", n_per)
    print(agg.round(3).to_string())

    # Figures
    fig_rule_battery(df, FIG_DIR / "fig11_rule_battery")
    fig_match_rate_phi_star(df, FIG_DIR / "fig12_match_rate_phi_star")
    fig_signatures_phi_star(df, FIG_DIR / "fig13_signatures_phi_star")

    print("\n=== Takeaway ===")
    print("If the K-means classification is correct, the strategic phenotypes")
    print("should show a high match_rate and H(D^φ*) ≈ 0; Undefined should sit")
    print("at a match_rate around 0.5-0.6 and H(D^φ*) ≈ 1.")
    print(f"\n[save] subject_rule_metrics.csv + 3 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
