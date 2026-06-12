"""
MINE (Mutual Information Neural Estimation) on phenotypes.

Belghazi et al. 2018 — doi:10.48550/arXiv.1801.04062

Adapted from mine_analysis.py (Mr. Banks) to the Poncela-Casasnovas dataset.

Estimates I(D^φ*; X) per phenotype, with X in {T, S, (T,S)}.
For each phenotype we pool all (T_n, S_n, D^φ*_n) from its subjects.
Samples per phenotype: 950-2300, enough for MINE to converge.

Goal: compare against the plug-in estimator (Miller-Madow) to:
  (1) Check whether the plug-in is significantly biased with 121 cells x 15 samples
  (2) Confirm the signatures: Optimist for T, Pessimist for S, Envious for S-T
  (3) Provide a statistically robust figure for the thesis
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

torch.manual_seed(42)
np.random.seed(42)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

DATA_DIR = Path("/Users/racelabs/Documents/TFG/data/drbrain")
FIG_DIR = DATA_DIR / "figs_report"
FIG_DIR.mkdir(exist_ok=True)
R_PAYOFF, P_PAYOFF = 10, 5
PHENOTYPES = ["Trustful", "Optimist", "Pessimist", "Envious", "Undefined"]
PALETTE = {"Trustful": "#2ca02c", "Optimist": "#1f77b4", "Pessimist": "#d62728",
           "Envious": "#ff7f0e", "Undefined": "#7f7f7f"}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# --------------- Rules ---------------
def prescribe(rule, T, S):
    if rule == "Trustful":  return np.full(len(T), "C")
    if rule == "Optimist":  return np.where(T < R_PAYOFF, "C", "D")
    if rule == "Pessimist": return np.where(S > P_PAYOFF, "C", "D")
    if rule == "Envious":   return np.where(S >= T, "C", "D")


def best_fit_rule(actions, T, S):
    rates = {r: (actions == prescribe(r, T, S)).mean()
             for r in ["Trustful", "Optimist", "Pessimist", "Envious"]}
    return max(rates, key=rates.get)


# --------------- MINE ---------------
class TNet(nn.Module):
    def __init__(self, x_dim, hidden=64):
        super().__init__()
        # x = [feature(s), D^φ*]
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y.view(-1, 1)], dim=1)).squeeze(-1)


def mine_estimate(X, Y, n_epochs=1500, batch=256, lr=1e-3, hidden=64,
                  warmup=200, device=DEVICE, seed=0):
    """X (n, d), Y (n,) binary. Returns I_MINE in bits (mean over the last 100 epochs)."""
    rng = np.random.default_rng(seed)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    net = TNet(X.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    n = X.shape[0]
    for ep in range(n_epochs):
        idx = rng.choice(n, batch, replace=True)
        idx_m = rng.choice(n, batch, replace=True)
        x = X_t[idx]; y = Y_t[idx]; y_m = Y_t[idx_m]
        t_joint = net(x, y)
        t_marg = net(x, y_m)
        # Donsker-Varadhan
        dv = t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)
        loss = -dv
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(dv.item() / np.log(2))  # convert nats to bits
    return float(np.mean(history[-100:])), history


# --------------- Per-phenotype pipeline ---------------
def build_per_phenotype(decisions, km_assign):
    """For each phenotype returns (T, S, D^φ*) as arrays."""
    out = {}
    for uid, sub in decisions.groupby("User_ID"):
        if uid not in km_assign:
            continue
        T = sub["T"].astype(int).to_numpy()
        S = sub["S"].astype(int).to_numpy()
        actions = sub["Action"].to_numpy()
        km_pheno = km_assign[uid]
        phi_star = best_fit_rule(actions, T, S) if km_pheno == "Undefined" else km_pheno
        D_phi = (actions == prescribe(phi_star, T, S)).astype(np.float32)
        out.setdefault(km_pheno, {"T": [], "S": [], "D": []})
        out[km_pheno]["T"].append(T.astype(np.float32))
        out[km_pheno]["S"].append(S.astype(np.float32))
        out[km_pheno]["D"].append(D_phi)
    for p in out:
        out[p] = {k: np.concatenate(v) for k, v in out[p].items()}
    return out


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])].copy()
    km = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")
    km_assign = dict(zip(km["User_ID"], km["phenotype_kmeans"]))

    per = build_per_phenotype(decisions, km_assign)
    print(f"Samples per phenotype:")
    for p, d in per.items():
        print(f"  {p}: {len(d['D'])} samples")

    rows = []
    print(f"\nMINE running on device: {DEVICE}\n")
    for p in PHENOTYPES:
        if p not in per:
            continue
        d = per[p]
        T = normalize(d["T"]).reshape(-1, 1)
        S = normalize(d["S"]).reshape(-1, 1)
        TS = np.column_stack([normalize(d["T"]), normalize(d["S"])])
        SmT = normalize(d["S"] - d["T"]).reshape(-1, 1)
        Y = d["D"]

        print(f"--- {p} (n={len(Y)}) ---")
        mi_T, _ = mine_estimate(T, Y, seed=1)
        mi_S, _ = mine_estimate(S, Y, seed=2)
        mi_TS, _ = mine_estimate(TS, Y, seed=3)
        mi_SmT, _ = mine_estimate(SmT, Y, seed=4)
        print(f"  I_MINE(D^φ*; T)     = {mi_T:.4f} bits")
        print(f"  I_MINE(D^φ*; S)     = {mi_S:.4f} bits")
        print(f"  I_MINE(D^φ*; S−T)   = {mi_SmT:.4f} bits")
        print(f"  I_MINE(D^φ*; (T,S)) = {mi_TS:.4f} bits")

        rows.append({
            "phenotype": p, "n": len(Y),
            "I_MINE_T": mi_T, "I_MINE_S": mi_S,
            "I_MINE_S-T": mi_SmT, "I_MINE_TS": mi_TS,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "mine_results.csv", index=False)
    print(f"\n[save] {DATA_DIR / 'mine_results.csv'}")

    # --- Comparison against the plug-in estimator (Miller-Madow) per phenotype ---
    plug = pd.read_csv(DATA_DIR / "subject_signatures.csv")
    plug = plug.dropna(subset=["phenotype_kmeans"])
    plug_agg = plug.groupby("phenotype_kmeans")[
        ["I_D;T", "I_D;S", "I_D;S-T", "I_D;TS_quad"]
    ].mean().rename(columns={
        "I_D;T": "plug_T", "I_D;S": "plug_S",
        "I_D;S-T": "plug_S-T", "I_D;TS_quad": "plug_TS"
    })
    comparison = df.set_index("phenotype").join(plug_agg)
    print("\n=== MINE vs plug-in (Miller-Madow) comparison ===")
    print(comparison.round(3).to_string())
    comparison.to_csv(DATA_DIR / "mine_vs_plugin.csv")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    order = [p for p in PHENOTYPES if p in df["phenotype"].values]
    x = np.arange(len(order))
    w = 0.18
    metrics = [("I_MINE_T", "I(D^φ*;T)"), ("I_MINE_S", "I(D^φ*;S)"),
               ("I_MINE_S-T", "I(D^φ*;S−T)"), ("I_MINE_TS", "I(D^φ*;(T,S))")]
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]
    for i, (col, label) in enumerate(metrics):
        vals = [df.loc[df["phenotype"] == p, col].values[0] for p in order]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=colors[i],
               edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("MI (bits, MINE)")
    ax.set_title(r"Estimació MINE de la informació mútua $I(D^{\varphi^*}; X)$ per fenotip")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.savefig(FIG_DIR / "fig14_mine.pdf")
    plt.savefig(FIG_DIR / "fig14_mine.png")
    plt.close()
    print(f"[save] fig14_mine.pdf/png")


if __name__ == "__main__":
    main()
