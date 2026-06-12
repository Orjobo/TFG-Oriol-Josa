"""
Generates the final report figures with a unified aesthetic.
Output goes to FIG_DIR (see below): the thesis figures folder.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

# ---------- aesthetics ----------
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
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
PALETTE = {"Trustful": "#009E73", "Optimist": "#56B4E9", "Pessimist": "#D55E00",
           "Envious": "#CC79A7", "Undefined": "#000000", "R": "#999999"}

DATA_DIR = Path("/Users/racelabs/Library/Mobile Documents/com~apple~CloudDocs/Documents/TFG/data/drbrain")
FIG_DIR = Path("/Users/racelabs/Desktop/TFG_memoria/figures")
FIG_DIR.mkdir(exist_ok=True)
PURE_GAMES = ["Harmony", "Snow Drift", "Stag-Hunt", "Prisoner's Dilemma"]
PURE_LABELS = ["HG", "SG", "SH", "PD"]
PHENOTYPES = ["Trustful", "Optimist", "Pessimist", "Envious", "Undefined"]
R_PAYOFF, P_PAYOFF = 10, 5


def color_list(order):
    return [PALETTE.get(p, "#888") for p in order]


# ---------- Fig 1: (T,S) plane and the 4 games ----------
def fig_TS_plane():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: schematic of the 4 games
    ax = axes[0]
    # Quadrants coloured from the same viridis map as the right panel,
    # ordered by cooperation level (HG high -> PD low) so the schematic
    # matches the empirical heatmap.
    _vir = plt.cm.viridis
    ax.add_patch(plt.Rectangle((5, 5), 5, 5, fc=_vir(0.85), ec="black", alpha=0.6))
    ax.add_patch(plt.Rectangle((10, 5), 5, 5, fc=_vir(0.55), ec="black", alpha=0.6))
    ax.add_patch(plt.Rectangle((5, 0), 5, 5, fc=_vir(0.35), ec="black", alpha=0.6))
    ax.add_patch(plt.Rectangle((10, 0), 5, 5, fc=_vir(0.12), ec="black", alpha=0.6))
    ax.text(7.5, 7.5, "HG\nHarmony", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(12.5, 7.5, "SG\nSnowdrift", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(7.5, 2.5, "SH\nStag Hunt", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(12.5, 2.5, "PD\nPris. Dilemma", ha="center", va="center", fontsize=12, weight="bold")
    ax.axhline(P_PAYOFF, color="black", lw=1, ls="--")
    ax.axvline(R_PAYOFF, color="black", lw=1, ls="--")
    ax.set_xlim(5, 15)
    ax.set_ylim(0, 10)
    ax.set_xlabel("T (Temptation)")
    ax.set_ylabel("S (loser's payoff)")
    # ax.set_title("Els 4 jocs al pla (T, S) — R=10, P=5 fixos")
    ax.set_aspect("equal")

    # Right: empirical cooperation across the lattice
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])]
    decisions["coop"] = (decisions["Action"] == "C").astype(int)
    grid = decisions.groupby(["T", "S"])["coop"].mean().reset_index()
    pivot = grid.pivot(index="S", columns="T", values="coop")
    ax2 = axes[1]
    im = ax2.imshow(pivot.values, origin="lower", cmap="viridis", vmin=0, vmax=1,
                    extent=[4.5, 15.5, -0.5, 10.5], aspect="auto")
    ax2.set_xlabel("T")
    ax2.set_ylabel("S")
    # ax2.set_title("Cooperació mitjana empírica per (T, S)")
    ax2.axhline(4.5, color="black", lw=1, ls="--")
    ax2.axvline(9.5, color="black", lw=1, ls="--")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("⟨C⟩")
    plt.savefig(FIG_DIR / "fig01_TS_plane.pdf")
    plt.savefig(FIG_DIR / "fig01_TS_plane.png")
    plt.close()


# ---------- Fig 2: replicate the paper's proportions vs our 3 methods ----------
def fig_proportions_comparison():
    paper = {"Envious": 29.8, "Pessimist": 20.9, "Optimist": 20.3, "Trustful": 16.6, "Undefined": 12.2}
    km = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")
    ent = pd.read_csv(DATA_DIR / "phenotype_entropy_assignments.csv")
    comp = pd.read_csv(DATA_DIR / "phenotype_composition_weights.csv")

    def pct(df, col):
        c = df[col].value_counts()
        n = len(df)
        return {p: 100 * c.get(p, 0) / n for p in paper.keys()}

    km_p = pct(km, "phenotype_kmeans")
    ent_p = pct(ent, "phenotype_entropy")
    comp_p = pct(comp, "phenotype_composition")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = ["Envious", "Pessimist", "Optimist", "Trustful", "Undefined"]
    x = np.arange(len(order))
    w = 0.2
    ax.bar(x - 1.5 * w, [paper[p] for p in order], w, label="Paper", color="#444", edgecolor="black")
    ax.bar(x - 0.5 * w, [km_p[p] for p in order], w, label="K-means replicat", color="#0072B2", edgecolor="black")
    ax.bar(x + 0.5 * w, [ent_p[p] for p in order], w, label="Entròpic (Bin. B)", color="#009E73", edgecolor="black")
    ax.bar(x + 1.5 * w, [comp_p[p] for p in order], w, label="Composicional MLE", color="#E69F00", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("% subjects")
    ax.set_title("Proporcions per fenotip — paper vs els tres mètodes propis")
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(0, 40)
    plt.savefig(FIG_DIR / "fig02_proportions.pdf")
    plt.savefig(FIG_DIR / "fig02_proportions.png")
    plt.close()


# ---------- Fig 3: cooperation heatmaps per phenotype (replicates Fig 2 of the paper) ----------
def fig_phenotype_heatmaps():
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])].copy()
    decisions["coop"] = (decisions["Action"] == "C").astype(int)
    km = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")
    df = decisions.merge(km, on="User_ID")

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, p in zip(axes, ["Trustful", "Optimist", "Pessimist", "Envious", "Undefined"]):
        sub = df[df["phenotype_kmeans"] == p]
        grid = sub.groupby(["T", "S"])["coop"].mean().reset_index()
        pivot = grid.pivot(index="S", columns="T", values="coop")
        # Re-index to cover the full lattice (some cells may be empty per phenotype)
        pivot = pivot.reindex(index=range(0, 11), columns=range(5, 16))
        im = ax.imshow(pivot.values, origin="lower", cmap="viridis", vmin=0, vmax=1,
                       extent=[4.5, 15.5, -0.5, 10.5], aspect="auto")
        ax.set_title(f"{p} (n={len(sub['User_ID'].unique())})")
        ax.set_xlabel("T")
        if ax is axes[0]:
            ax.set_ylabel("S")
        ax.axhline(4.5, color="black", lw=0.6, ls="--")
        ax.axvline(9.5, color="black", lw=0.6, ls="--")
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.85, pad=0.02)
    cbar.set_label("⟨C⟩")
    fig.suptitle(r"Empirical mean cooperation per phenotype ($K$-means, $k=5$)", y=1.03)
    plt.savefig(FIG_DIR / "fig03_phenotype_heatmaps.pdf")
    plt.savefig(FIG_DIR / "fig03_phenotype_heatmaps.png")
    plt.close()


# ---------- Fig 4: battery of conditional entropies per phenotype ----------
def fig_entropy_battery():
    metrics = pd.read_csv(DATA_DIR / "subject_entropy_metrics.csv")
    metrics = metrics.dropna(subset=["phenotype_kmeans"])
    order = [p for p in PHENOTYPES if p in metrics["phenotype_kmeans"].unique()]
    cols = [("H_D", "H(D) marginal"), ("I_D;g", "I(D ; g)"), ("I_D;TS", "I(D ; (T,S))")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, cols):
        data = [metrics[metrics["phenotype_kmeans"] == p][col].dropna() for p in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"})
        for patch, c in zip(bp["boxes"], color_list(order)):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel("bits")
        ax.tick_params(axis="x", rotation=20)
    plt.savefig(FIG_DIR / "fig04_entropy_battery.pdf")
    plt.savefig(FIG_DIR / "fig04_entropy_battery.png")
    plt.close()


# ---------- Fig 5: normalised signatures (phenotype x signal heatmap) ----------
def fig_signatures_heatmap():
    sig = pd.read_csv(DATA_DIR / "subject_signatures.csv").dropna(subset=["phenotype_kmeans"])
    order = [p for p in PHENOTYPES if p in sig["phenotype_kmeans"].unique()]
    signals = ["I_D;T_norm", "I_D;S_norm", "I_D;S-T_norm"]
    sig_lbl = ["I(D;T)/H(D)", "I(D;S)/H(D)", "I(D;S−T)/H(D)"]
    M = np.array([[sig.loc[sig["phenotype_kmeans"] == p, c].mean() for c in signals] for p in order])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.3, vmax=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(sig_lbl)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(3):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.55 else "black", fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax, label="Normalised MI")
    # ax.set_title("Signatures d'informació mútua per fenotip\n(màxim per fila marca la senyal predita)")
    plt.savefig(FIG_DIR / "fig05_signatures.pdf")
    plt.savefig(FIG_DIR / "fig05_signatures.png")
    plt.close()


# ---------- Fig 6: weight distributions from the compositional MLE ----------
def fig_weight_distributions():
    w = pd.read_csv(DATA_DIR / "phenotype_composition_weights.csv")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    cols = ["w_R", "w_Trustful", "w_Optimist", "w_Pessimist", "w_Envious"]
    titles = ["w_Random", "w_Trustful", "w_Optimist", "w_Pessimist", "w_Envious"]
    cols_color = ["R", "Trustful", "Optimist", "Pessimist", "Envious"]
    for ax, c, t, color_key in zip(axes, cols, titles, cols_color):
        ax.hist(w[c], bins=25, color=PALETTE[color_key], edgecolor="black", alpha=0.85)
        ax.set_title(t)
        ax.set_xlim(0, 1)
        ax.set_xlabel("pes")
        if ax is axes[0]:
            ax.set_ylabel("# subjects")
    plt.suptitle("Distribució dels pesos del model composicional MLE (5 estratègies)", y=1.03)
    plt.savefig(FIG_DIR / "fig06_weight_distributions.pdf")
    plt.savefig(FIG_DIR / "fig06_weight_distributions.png")
    plt.close()


# ---------- Fig 7: real vs shuffled match rate over D^phi* (binarisation B) ----------
def fig_real_vs_shuffled():
    df = pd.read_csv(DATA_DIR / "shuffle_null_b.csv")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    # Colour by phenotype
    for p in PHENOTYPES:
        mask = df["phenotype_kmeans"] == p
        if mask.sum() == 0:
            continue
        ax.scatter(df.loc[mask, "match_rate_shuffled"],
                   df.loc[mask, "match_rate_real"],
                   alpha=0.65, s=24, color=PALETTE.get(p, "#888"),
                   edgecolor="white", linewidth=0.4, label=p)
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1)
    ax.fill_between([0.0, 1.0], [0.0, 1.0], 1.0, alpha=0.06, color="green")
    # Fraction above the diagonal (real > shuffled) is reported in the
    # figure caption and body, not annotated on the plot, to avoid
    # duplicating the number inside the figure.
    ax.set_xlabel(r"Match rate with $\varphi^*$ (shuffled actions)")
    ax.set_ylabel(r"Match rate with $\varphi^*$ (real actions)")
    # ax.set_title(r"Shuffle null sobre $D^{\varphi^*}$ — gap = adherència real a la regla")
    ax.set_xlim(0.27, 1.02)
    ax.set_ylim(0.27, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    plt.savefig(FIG_DIR / "fig07_real_vs_shuffled.pdf")
    plt.savefig(FIG_DIR / "fig07_real_vs_shuffled.png")
    plt.close()


# ---------- Fig 8: structural / observational collinearity ----------
def fig_collinearity():
    Cs = pd.read_csv(DATA_DIR / "collinearity_structural.csv", index_col=0)
    Co = pd.read_csv(DATA_DIR / "collinearity_observational.csv", index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, C, t in zip(axes, [Cs, Co],
                        ["Estructural (Pearson sobre reticle)",
                         "Observacional (Spearman entre pesos MLE)"]):
        # Replace NaN with 0 just for display (the cell is still marked with a dash)
        Cv = C.fillna(0).values
        im = ax.imshow(Cv, cmap="PuOr_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(C)))
        ax.set_yticks(range(len(C)))
        ax.set_xticklabels(C.columns, rotation=20)
        ax.set_yticklabels(C.index)
        for i in range(len(C)):
            for j in range(len(C)):
                v = Cv[i, j]
                if pd.isna(C.values[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", color="gray")
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if abs(v) > 0.5 else "black", fontsize=9)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Test de degeneració observacional — cap parell amb |ρ|>0.9 (a diferència del MI≡WSLS de Mr. Banks)",
                 y=1.02, fontsize=10)
    plt.savefig(FIG_DIR / "fig08_collinearity.pdf")
    plt.savefig(FIG_DIR / "fig08_collinearity.png")
    plt.close()


# ---------- Fig 9: PCA of the weights (spectrum and PC1-PC2 scatter) ----------
def fig_pca_weights():
    w = pd.read_csv(DATA_DIR / "phenotype_composition_weights.csv")
    cols = ["w_R", "w_Trustful", "w_Optimist", "w_Pessimist", "w_Envious"]
    X = w[cols].to_numpy()
    Xc = X - X.mean(axis=0)
    U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    PC = Xc @ Vt.T
    H_pc = -np.sum(var[var > 0] * np.log2(var[var > 0]))
    n_eff = 2 ** H_pc

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(range(1, 6), var * 100, color="#0072B2", edgecolor="black")
    axes[0].plot(range(1, 6), np.cumsum(var) * 100, "ko-", linewidth=1.5)
    axes[0].set_xlabel("Component principal")
    axes[0].set_ylabel("% variància explicada")
    axes[0].set_title(f"Espectre PCA — {n_eff:.2f} estratègies efectives")
    axes[0].set_xticks(range(1, 6))

    # Scatter PC1 vs PC2 coloured by phenotype
    if "phenotype_composition" in w.columns:
        for p in PHENOTYPES:
            mask = w["phenotype_composition"] == p
            axes[1].scatter(PC[mask, 0], PC[mask, 1], s=22, alpha=0.7,
                            color=PALETTE.get(p, "#888"), label=p, edgecolor="white", linewidth=0.4)
        axes[1].legend(loc="best", frameon=True, fontsize=9)
    axes[1].set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    axes[1].set_title("Pesos projectats — fenotip dominant")
    plt.savefig(FIG_DIR / "fig09_pca.pdf")
    plt.savefig(FIG_DIR / "fig09_pca.png")
    plt.close()


# ---------- Fig 10: demographics, weights vs age ----------
def fig_demographics():
    w = pd.read_csv(DATA_DIR / "phenotype_composition_weights.csv")
    u = pd.read_csv(DATA_DIR / "drbrain_users.csv")
    df = w.merge(u, on="User_ID")
    df["age_bin"] = pd.cut(df["Age"], bins=[0, 15, 30, 45, 100],
                            labels=["≤15", "16-30", "31-45", "46+"])

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
    cols = ["w_R", "w_Trustful", "w_Optimist", "w_Pessimist", "w_Envious"]
    color_keys = ["R", "Trustful", "Optimist", "Pessimist", "Envious"]
    titles = ["w_R", "w_Trustful", "w_Optimist*", "w_Pessimist", "w_Envious*"]
    for ax, c, color_key, t in zip(axes, cols, color_keys, titles):
        data = [df[df["age_bin"] == b][c].values for b in df["age_bin"].cat.categories]
        bp = ax.boxplot(data, labels=df["age_bin"].cat.categories, patch_artist=True,
                        showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE[color_key])
            patch.set_alpha(0.7)
        ax.set_title(t)
        ax.set_xlabel("edat")
        if ax is axes[0]:
            ax.set_ylabel("pes")
    fig.suptitle("Pesos composicionals per franja d'edat (* = KW p < 0.05)", y=1.03)
    plt.savefig(FIG_DIR / "fig10_demographics.pdf")
    plt.savefig(FIG_DIR / "fig10_demographics.png")
    plt.close()


# ---------- Main ----------
if __name__ == "__main__":
    print("Generating figures...")
    fig_TS_plane();             print("  fig01 done")
    fig_proportions_comparison(); print("  fig02 done")
    fig_phenotype_heatmaps();   print("  fig03 done")
    fig_entropy_battery();      print("  fig04 done")
    fig_signatures_heatmap();   print("  fig05 done")
    fig_weight_distributions(); print("  fig06 done")
    fig_real_vs_shuffled();     print("  fig07 done")
    fig_collinearity();         print("  fig08 done")
    fig_pca_weights();          print("  fig09 done")
    fig_demographics();         print("  fig10 done")
    print(f"\nFigures in {FIG_DIR}")
