"""
TFG - MINE (Mutual Information Neural Estimation) on Mr. Banks
Belghazi et al. 2018, doi:10.48550/arXiv.1801.04062

Estimates I(X; Y) with a neural network via the Donsker-Varadhan representation:
    I(X;Y) >= sup_T  E_{P(X,Y)}[T(x,y)] - log E_{P(X)P(Y)}[exp(T(x,y))]

On Mr. Banks we use this to compare against the analytic MI from Perello's paper:
  I(M_{n-1}; D_n)    ~ 0.044 bits (paper)    captures MI (market imitation)
  I(R_{n-1}; C_n)    ~ 0.050 bits (paper)    captures WSLS

MINE lets us estimate the MI between:
  (A) the full context (the last k decisions+results) and D_n
If I_MINE > I_analytic, there is higher-order structure that the first-order
analytic MI misses, which is what justifies the LSTM.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])


def build_pairs(df, k):
    """
    For each round t >= k, X = concat(decisions, results, markets of the last k),
    Y = the decision at t. Returns arrays X (n, 3*k), Y (n,).
    """
    Xs, Ys = [], []
    for _, g in df.groupby("game"):
        d = g["dec_bin"].values
        r = g["res_bin"].values
        m = g["market_bin"].values
        n = len(d)
        for t in range(k, n):
            feat = np.concatenate([d[t - k:t], r[t - k:t], m[t - k:t]]).astype(np.float32)
            Xs.append(feat)
            Ys.append(d[t])
    return np.array(Xs), np.array(Ys, dtype=np.float32)


class TNet(nn.Module):
    """T(x,y) network for the Donsker-Varadhan bound."""
    def __init__(self, x_dim, y_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y.view(-1, 1)], dim=1)).squeeze(-1)


def mine_estimate(X, Y, n_epochs=600, batch=512, lr=1e-3, hidden=64, verbose=False):
    """Estimate I(X;Y) in bits via MINE with the DV bound."""
    X_t = torch.tensor(X)
    Y_t = torch.tensor(Y)
    net = TNet(X.shape[1], 1, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    ma_et = 1.0  # moving average to stabilize the gradient of the exp term
    ma_rate = 0.01
    history = []
    n = X.shape[0]
    for epoch in range(n_epochs):
        idx = np.random.choice(n, batch, replace=True)
        idx_m = np.random.choice(n, batch, replace=True)  # marginal Y (shuffled)
        x = X_t[idx]
        y = Y_t[idx]
        y_marg = Y_t[idx_m]

        t_joint = net(x, y)
        t_marg = net(x, y_marg)
        et = torch.exp(t_marg)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * et.mean().item()
        # bias-corrected gradient as in the MINE paper
        loss = -(t_joint.mean() - torch.log(et.mean() + 1e-8) * et.mean().detach() / (ma_et + 1e-8))
        # simplification: use DV directly (bias is small with many samples)
        dv = t_joint.mean() - torch.log(et.mean() + 1e-8)
        loss = -dv
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(dv.item() / np.log(2))  # convert nats to bits
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  I~{history[-1]:.4f} bits")

    return np.mean(history[-50:]), history


# --- Experiments over different horizons k ---
ks = [1, 2, 3, 5, 8]
results = []
print("MINE over different horizons k (past decisions+results+markets):")
for k in ks:
    X, Y = build_pairs(df, k)
    print(f"\nk={k}: X shape {X.shape}, Y shape {Y.shape}")
    mi, hist = mine_estimate(X, Y, n_epochs=800, batch=512, hidden=64, verbose=False)
    print(f"  I_MINE(past_{k}; D_n) ~ {mi:.4f} bits")
    results.append({"k": k, "I_MINE_bits": mi, "n_samples": len(X)})

# --- Plot results ---
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ks_arr = [r["k"] for r in results]
mis = [r["I_MINE_bits"] for r in results]
ax.plot(ks_arr, mis, "o-", color="darkblue", lw=2, markersize=10,
        label="MINE I(passat_k; D_n)")
ax.axhline(0.044, ls="--", color="steelblue",
           label="I(M_{n-1}; D_n) analítica (0.044)")
ax.axhline(0.050, ls="--", color="coral",
           label="I(R_{n-1}; C_n) analítica (0.050)")
ax.set_xlabel("k (longitud de l'historial)")
ax.set_ylabel("Informació mútua (bits)")
ax.set_title("MINE: MI entre historial i decisió\n"
             "Si creix amb k → hi ha estructura d'ordre alt (justifica la LSTM)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/mine_analysis.png", dpi=120)
print("\nSaved: mine_analysis.png")

pd.DataFrame(results).to_csv("/Users/racelabs/Desktop/TFG/data/mine_results.csv", index=False)
print("Saved: data/mine_results.csv")
