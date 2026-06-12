"""
Compositional phenotype model, a direct analogue of the Mr. Banks one
(w_R + w_MI + w_WSLS, later reduced to w_R + w_I).

Here each subject is a mixture of 5 deterministic strategies (except R, which is random):
  - R (Random)        p(C) = 0.5
  - T (Trustful)      p(C) = 1
  - O (Optimist)      p(C) = 1 if T_n < 10, else 0
  - P (Pessimist)     p(C) = 1 if S_n > 5, else 0
  - E (Envious)       p(C) = 1 if S_n >= T_n, else 0

For each subject we fit w = (w_R, w_T, w_O, w_P, w_E) with w_i >= 0, sum(w_i) = 1
by maximizing the log-likelihood:
  L(w) = sum_n log[ p(C_n; w) ]   where  p(C_n; w) = sum_phi w_phi * p_phi(C | T_n, S_n)

Also includes:
  - Shuffle null model (to check the weights are not an artefact)
  - KW p-values for gender and age on the weights
  - Comparison against the K-means clustering from the paper
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import kruskal
import matplotlib.pyplot as plt

DATA_DIR = Path("/Users/racelabs/Documents/TFG/data/drbrain")
R_PAYOFF, P_PAYOFF = 10, 5
STRATEGIES = ["R", "Trustful", "Optimist", "Pessimist", "Envious"]


def predict_per_strategy(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Return an (n_decisions x 5) matrix of p_phi(C | T_n, S_n) for each strategy."""
    n = len(T)
    M = np.zeros((n, 5))
    M[:, 0] = 0.5                                  # R: random
    M[:, 1] = 1.0                                  # Trustful: always C
    M[:, 2] = (T < R_PAYOFF).astype(float)         # Optimist: T<R
    M[:, 3] = (S > P_PAYOFF).astype(float)         # Pessimist: S>P
    M[:, 4] = (S >= T).astype(float)               # Envious: S>=T
    return M


def neg_log_lik(w: np.ndarray, predictions: np.ndarray, action_C: np.ndarray) -> float:
    """Negative log-likelihood of the mixture, given w (sums to 1) and binary action C=1/D=0."""
    p_C = predictions @ w                          # p(C_n; w) for each decision
    p_C = np.clip(p_C, 1e-9, 1 - 1e-9)
    L = action_C * np.log(p_C) + (1 - action_C) * np.log(1 - p_C)
    return -L.sum()


def fit_subject(T: np.ndarray, S: np.ndarray, actions: np.ndarray) -> dict:
    """MLE of the 5 weights w for a single subject."""
    pred = predict_per_strategy(T, S)
    action_C = (actions == "C").astype(float)
    n = len(actions)
    if n < 5:
        return None

    # Start from uniform weights
    w0 = np.ones(5) / 5
    bounds = [(0.0, 1.0)] * 5
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    res = minimize(neg_log_lik, w0, args=(pred, action_C),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    w = res.x / res.x.sum()  # re-normalize, the constraint may not hold exactly
    p_pred = (pred @ w).clip(1e-9, 1 - 1e-9)

    # Metrics
    log_lik = -res.fun
    cross_ent = -log_lik / n / np.log(2)           # bits/decision
    accuracy = ((p_pred > 0.5) == action_C.astype(bool)).mean()

    return dict(
        w_R=w[0], w_Trustful=w[1], w_Optimist=w[2],
        w_Pessimist=w[3], w_Envious=w[4],
        log_lik=log_lik, cross_ent=cross_ent, accuracy=accuracy, n=n,
    )


def fit_all(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, sub in decisions.groupby("User_ID"):
        T = sub["T"].astype(int).to_numpy()
        S = sub["S"].astype(int).to_numpy()
        act = sub["Action"].to_numpy()
        out = fit_subject(T, S, act)
        if out is None:
            continue
        out["User_ID"] = uid
        rows.append(out)
    return pd.DataFrame(rows)


def assign_dominant_strategy(weights: pd.DataFrame, threshold: float = 0.4) -> pd.Series:
    """Dominant strategy = argmax_phi w_phi; if max < threshold it becomes Mixed/Undefined."""
    cols = [f"w_{s}" for s in STRATEGIES]
    M = weights[cols].to_numpy()
    best_idx = M.argmax(axis=1)
    best_w = M.max(axis=1)
    labels = np.array([STRATEGIES[i] for i in best_idx], dtype=object)
    labels[best_w < threshold] = "Mixed"
    # Rename R+Mixed to Undefined (to match the paper's convention)
    labels[labels == "R"] = "Undefined"
    labels[labels == "Mixed"] = "Undefined"
    return pd.Series(labels, index=weights.index, name="phenotype_composition")


# --------------------------------------------------------------------------
# Shuffle null model: shuffle each subject's actions and refit
# --------------------------------------------------------------------------
def fit_shuffled(decisions: pd.DataFrame, n_iter: int = 1, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for uid, sub in decisions.groupby("User_ID"):
        T = sub["T"].astype(int).to_numpy()
        S = sub["S"].astype(int).to_numpy()
        act = sub["Action"].to_numpy()
        accs = []
        for _ in range(n_iter):
            shuffled = rng.permutation(act)
            out = fit_subject(T, S, shuffled)
            if out is not None:
                accs.append(out["accuracy"])
        if accs:
            rows.append({"User_ID": uid, "shuffled_accuracy": np.mean(accs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Demographics: KW test for gender and age on the weights
# --------------------------------------------------------------------------
def kw_demographics(weights: pd.DataFrame, users: pd.DataFrame):
    df = weights.merge(users, on="User_ID")
    df["age_bin"] = pd.cut(
        df["Age"], bins=[0, 15, 30, 45, 100],
        labels=["≤15", "16-30", "31-45", "46+"]
    )
    print("\n=== Kruskal-Wallis: weights vs gender ===")
    for s in STRATEGIES:
        col = f"w_{s}"
        groups = [df.loc[df["Gender"] == g, col].values for g in df["Gender"].unique()]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        means = df.groupby("Gender")[col].mean().round(3).to_dict()
        sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        print(f"  {s:10s}  KW p={p:.4f}{sig}  | {means}")

    print("\n=== Kruskal-Wallis: weights vs age (4 bins) ===")
    for s in STRATEGIES:
        col = f"w_{s}"
        groups = [df.loc[df["age_bin"] == b, col].values for b in df["age_bin"].cat.categories]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        sig = " ***" if p < 0.001 else (" **" if p < 0.01 else (" *" if p < 0.05 else ""))
        print(f"  {s:10s}  KW p={p:.4f}{sig}")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_weight_distribution(weights: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
    for ax, s in zip(axes, STRATEGIES):
        ax.hist(weights[f"w_{s}"], bins=20, color="steelblue", edgecolor="black")
        ax.set_title(f"w_{s}")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("# subjects")
    fig.suptitle("Distribució dels pesos del model composicional", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_real_vs_shuffled(weights: pd.DataFrame, shuffled: pd.DataFrame, out_path: Path):
    m = weights.merge(shuffled, on="User_ID")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(m["shuffled_accuracy"], m["accuracy"], alpha=0.4, s=20)
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", linewidth=1)
    ax.set_xlabel("Accuracy on shuffled actions (null model)")
    ax.set_ylabel("Accuracy on real actions")
    ax.set_title("Real vs shuffled — el gap és la 'estructura real'")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_with_kmeans(merged: pd.DataFrame, out_path: Path):
    crosstab = pd.crosstab(merged["phenotype_composition"], merged["phenotype_kmeans"])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(crosstab.values, cmap="Blues")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns, rotation=20)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index)
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            v = crosstab.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > crosstab.values.max() / 2 else "black")
    ax.set_xlabel("K-means (paper)")
    ax.set_ylabel("Composicional MLE")
    ax.set_title("Concordança composicional vs K-means")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])].copy()
    users = pd.read_csv(DATA_DIR / "drbrain_users.csv")

    print(f"[load] {decisions['User_ID'].nunique()} subjects, {len(decisions)} decisions")

    print("\n[fit] Fitting 5 weights per subject (compositional MLE)...")
    weights = fit_all(decisions)
    print(f"  {len(weights)} subjects fitted. Mean accuracy: {weights['accuracy'].mean():.3f}")
    print(f"  Mean cross-entropy: {weights['cross_ent'].mean():.3f} bits/decision")

    weights["phenotype_composition"] = assign_dominant_strategy(weights)
    weights.to_csv(DATA_DIR / "phenotype_composition_weights.csv", index=False)

    print("\n=== Weight distribution (mean ± std over all subjects) ===")
    for s in STRATEGIES:
        col = f"w_{s}"
        m, sd = weights[col].mean(), weights[col].std()
        print(f"  w_{s:10s}  μ={m:.3f}  σ={sd:.3f}")

    print("\n=== Dominant phenotype (compositional) ===")
    counts = weights["phenotype_composition"].value_counts()
    PAPER_PCT = {"Envious": 29.8, "Pessimist": 20.9, "Optimist": 20.3,
                 "Trustful": 16.6, "Undefined": 12.2}
    for phi in ["Envious", "Pessimist", "Optimist", "Trustful", "Undefined"]:
        n = counts.get(phi, 0)
        pct = 100 * n / len(weights)
        print(f"  {phi:10s}  n={n:4d} ({pct:4.1f}%)  | paper: {PAPER_PCT[phi]:.1f}%")

    print("\n[null] Shuffle null model, refitting with shuffled actions...")
    shuffled = fit_shuffled(decisions, n_iter=1)
    merged_acc = weights.merge(shuffled, on="User_ID")
    real_minus_shuffled = (merged_acc["accuracy"] - merged_acc["shuffled_accuracy"]).mean()
    pct_above_null = (merged_acc["accuracy"] > merged_acc["shuffled_accuracy"]).mean()
    print(f"  Accuracy real:     μ={merged_acc['accuracy'].mean():.3f}")
    print(f"  Accuracy shuffled: μ={merged_acc['shuffled_accuracy'].mean():.3f}")
    print(f"  Δ (real − null):   {real_minus_shuffled:.3f}")
    print(f"  Subjects with accuracy > null: {pct_above_null:.1%}")
    shuffled.to_csv(DATA_DIR / "shuffled_null_accuracy.csv", index=False)

    # Demographics
    kw_demographics(weights, users)

    # Comparison against K-means
    km = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")[
        ["User_ID", "phenotype_kmeans"]
    ]
    merged = weights.merge(km, on="User_ID", how="inner")
    agree = (merged["phenotype_composition"] == merged["phenotype_kmeans"]).mean()
    print(f"\n=== Agreement compositional vs K-means (n={len(merged)}) ===")
    print(f"  Agreement: {agree:.3f}")
    print(pd.crosstab(merged["phenotype_composition"], merged["phenotype_kmeans"]).to_string())

    # Figures
    plot_weight_distribution(weights, DATA_DIR / "composition_weights_dist.png")
    plot_accuracy_real_vs_shuffled(weights, shuffled, DATA_DIR / "real_vs_shuffled.png")
    plot_confusion_with_kmeans(merged, DATA_DIR / "composition_vs_kmeans_confusion.png")

    print(f"\n[save] results written to {DATA_DIR}")


if __name__ == "__main__":
    main()
