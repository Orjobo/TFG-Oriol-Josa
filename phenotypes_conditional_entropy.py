"""
Battery of conditional entropies per phenotype (Poncela-Casasnovas 2016 paper).

Analogue of Mr. Banks' I(D;M); here we measure:
  - H(D)            marginal entropy of the action
  - H(D | g)        conditioned on game type (4 categories)
  - H(D | (T,S))    conditioned on the fine payoff (cell of the 11x11 lattice)
  - I(D; g)         mutual information with the game type
  - I(D; (T,S))     mutual information with the fine payoff

Hypothesis (parallel to Mr. Banks' 15x):
  - Trustful + Undefined: I(D; .) ~ 0 (not context-sensitive)
  - Optimist, Pessimist, Envious: I(D; .) > 0, context-sensitive

Requires: phenotype_entropy_assignments.csv (output of the first pipeline)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path("/Users/racelabs/Documents/TFG/data/drbrain")
PURE_GAMES = ["Harmony", "Snow Drift", "Stag-Hunt", "Prisoner's Dilemma"]
PHENOTYPES = ["Trustful", "Optimist", "Pessimist", "Envious", "Undefined"]


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def H_marginal(actions: np.ndarray) -> float:
    """H(D) for a binary sequence D in {C,D}."""
    p = (actions == "C").mean() if len(actions) > 0 else 0.5
    return binary_entropy(p)


def H_conditional(actions: np.ndarray, context: np.ndarray) -> float:
    """H(D | context) = sum_x p(x) * H(D | context=x)."""
    if len(actions) == 0:
        return 0.0
    H = 0.0
    for x in np.unique(context):
        mask = context == x
        if mask.sum() == 0:
            continue
        H += (mask.sum() / len(actions)) * H_marginal(actions[mask])
    return H


def per_subject_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, sub in decisions.groupby("User_ID"):
        if len(sub) < 5:
            continue
        actions = sub["Action"].to_numpy()
        g = sub["Game"].to_numpy()
        TS = (sub["T"].astype(int).astype(str) + "_" + sub["S"].astype(int).astype(str)).to_numpy()
        H_D = H_marginal(actions)
        H_D_g = H_conditional(actions, g)
        H_D_TS = H_conditional(actions, TS)
        rows.append({
            "User_ID": uid,
            "n": len(actions),
            "coop_rate": (actions == "C").mean(),
            "H_D": H_D,
            "H_D|g": H_D_g,
            "H_D|TS": H_D_TS,
            "I_D;g": H_D - H_D_g,
            "I_D;TS": H_D - H_D_TS,
        })
    return pd.DataFrame(rows)


def aggregate_by_phenotype(metrics: pd.DataFrame, label_col: str) -> pd.DataFrame:
    g = metrics.groupby(label_col)
    agg = g[["H_D", "H_D|g", "H_D|TS", "I_D;g", "I_D;TS", "coop_rate", "n"]].agg(
        ["mean", "std", "count"]
    )
    return agg


def plot_summary(metrics: pd.DataFrame, label_col: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    order = [p for p in PHENOTYPES if p in metrics[label_col].unique()]

    for ax, col, title in zip(
        axes,
        ["H_D", "I_D;g", "I_D;TS"],
        ["H(D) marginal", "I(D; game)", "I(D; (T,S))"],
    ):
        data = [metrics[metrics[label_col] == p][col].dropna() for p in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(order)))):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.set_ylabel("bits")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Bateria entròpica per fenotip ({label_col})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])].copy()

    ent_assign = pd.read_csv(DATA_DIR / "phenotype_entropy_assignments.csv")[
        ["User_ID", "phenotype_entropy"]
    ]
    km_assign = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")[
        ["User_ID", "phenotype_kmeans"]
    ]

    metrics = per_subject_metrics(decisions)
    metrics = metrics.merge(ent_assign, on="User_ID", how="left").merge(
        km_assign, on="User_ID", how="left"
    )
    metrics.to_csv(DATA_DIR / "subject_entropy_metrics.csv", index=False)

    print(f"=== Entropy battery, n={len(metrics)} subjects ===\n")

    for label_col in ["phenotype_kmeans", "phenotype_entropy"]:
        print(f"\n--- Grouping by {label_col} ---")
        sub = metrics.dropna(subset=[label_col])
        agg = sub.groupby(label_col)[
            ["H_D", "H_D|g", "H_D|TS", "I_D;g", "I_D;TS", "coop_rate"]
        ].mean()
        n_per = sub[label_col].value_counts()
        agg["n"] = n_per
        order = [p for p in PHENOTYPES if p in agg.index]
        print(agg.loc[order].round(3).to_string())

        plot_summary(sub, label_col, DATA_DIR / f"entropy_battery_{label_col}.png")

    # --- Key hypothesis: I(D;g) per phenotype
    print("\n" + "=" * 70)
    print("Hypothesis: I(D;.) ~ 0 for Trustful/Undefined; > 0 for Optimist/Pessimist/Envious")
    print("=" * 70)
    sub = metrics.dropna(subset=["phenotype_kmeans"])
    means = sub.groupby("phenotype_kmeans")[["I_D;g", "I_D;TS"]].mean()
    print(means.round(4))

    # Ratio relative to Undefined (like Mr. Banks' 15x)
    if "Undefined" in means.index:
        ratio = means.divide(means.loc["Undefined"]).round(2)
        print("\nRatios relative to Undefined (like Mr. Banks' 15x):")
        print(ratio.to_string())

    print(f"\n[save] CSV and plots at {DATA_DIR}")


if __name__ == "__main__":
    main()
