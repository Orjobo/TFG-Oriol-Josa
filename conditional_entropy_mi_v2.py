"""
TFG — redesigned "Mr. Banks replication" figure (1x3).

Replaces the earlier 6-panel version (two of those panels were just the H(p)
curve, a mathematical identity that added nothing). The story now builds up
panel by panel:

  Panel 1 (left)    "Apparent individual randomness"
                    Per-subject H(result) and H(change) are both close to 1 bit,
                    so looked at in isolation the players look random.

  Panel 2 (center)  "Hidden structure via MI"
                    I(M_{n-1};D_n)=0.044, I(R_{n-1};D_n)=0.050,
                    I(R_{n-1};change_n)=0.050, while the controls sit near 0.
                    There is a real round-to-round signal.

  Panel 3 (right)   "Numerical replication vs paper"
                    The four conditional probabilities match the original paper
                    to 3 decimals.

Output: /Users/racelabs/Downloads/presentacio 2/conditional_entropy_mi.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/Users/racelabs/Documents/TFG/data/rounds.csv")
OUT = Path("/Users/racelabs/Downloads/presentacio 2/conditional_entropy_mi.png")

df = pd.read_csv(DATA)
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].astype(str).str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)

df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])
df["prev_decision"] = df.groupby("game")["dec_bin"].shift(1)
df["prev_result"] = df.groupby("game")["res_bin"].shift(1)
df["prev_market"] = df.groupby("game")["market_bin"].shift(1)
df["changed"] = (df["dec_bin"] != df["prev_decision"]).astype(float)
dfp = df.dropna(subset=["prev_decision"]).copy()
for c in ["prev_decision", "prev_result", "prev_market", "changed"]:
    dfp[c] = dfp[c].astype(int)


def H(s):
    p = s.value_counts(normalize=True)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def Hcond(d, x, y):
    out = 0.0
    n = len(d)
    for _, g in d.groupby(x):
        out += (len(g) / n) * H(g[y])
    return out


def MI(d, x, y):
    return H(d[y]) - Hcond(d, x, y)


# ---------- per-subject H ----------
H_res = dfp.groupby("user")["res_bin"].apply(H).values
H_chg = dfp.groupby("user")["changed"].apply(H).values

# ---------- aggregate MI ----------
MI_market = MI(dfp, "prev_market", "dec_bin")
MI_result = MI(dfp, "prev_result", "dec_bin")
MI_change = MI(dfp, "prev_result", "changed")
MI_dec_dec = MI(dfp, "prev_decision", "dec_bin")
MI_mkt_mkt = MI(dfp, "prev_market", "market_bin")

# ---------- conditionals vs paper ----------
p_up_mu = dfp[dfp["prev_market"] == 1]["dec_bin"].mean()
p_dn_md = (1 - dfp[dfp["prev_market"] == 0]["dec_bin"]).mean()
p_rep_e = (1 - dfp[dfp["prev_result"] == 1]["changed"]).mean()
p_chg_f = dfp[dfp["prev_result"] == 0]["changed"].mean()
ours = [p_up_mu, p_dn_md, p_rep_e, p_chg_f]
paper = [0.714, 0.531, 0.682, 0.579]

# =======================================================
# 1x3 FIGURE
# =======================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# --- Panel 1: per-person H (overlaid) ---
ax = axes[0]
ax.hist(H_res, bins=25, color="#E8956E", edgecolor="white", alpha=0.75,
        label=f"H(resultat)  ⟨H⟩={H_res.mean():.3f}")
ax.hist(H_chg, bins=25, color="#8B7AB8", edgecolor="white", alpha=0.65,
        label=f"H(canvi)        ⟨H⟩={H_chg.mean():.3f}")
ax.axvline(1.0, color="black", ls="--", lw=1, alpha=0.6)
ax.text(1.0, ax.get_ylim()[1] * 0.95, " H_max",
        fontsize=8, va="top", ha="left", alpha=0.7)
ax.set_xlabel("Entropia per persona (bits)")
ax.set_ylabel("Nombre d'individus")
ax.set_title("(a) Aparent aleatorietat individual", fontweight="bold")
ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
ax.set_xlim(0.6, 1.02)

# --- Panel 2: MI bars ---
ax = axes[1]
labels = [r"$I(M_{n{-}1}; D_n)$" + "\nMarket Imit.",
          r"$I(R_{n{-}1}; D_n)$" + "\nWSLS",
          r"$I(R_{n{-}1}; \mathrm{canvi}_n)$" + "\nWSLS-canvi",
          r"$I(D_{n{-}1}; D_n)$" + "\nControl",
          r"$I(M_{n{-}1}; M_n)$" + "\nControl"]
vals = [MI_market, MI_result, MI_change, MI_dec_dec, MI_mkt_mkt]
colors = ["#2C5282", "#D9544D", "#8B7AB8", "#999999", "#999999"]
bars = ax.bar(range(5), vals, color=colors, edgecolor="white", linewidth=1.2)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Mutual information (bits)")
ax.set_title("(b) Estructura amagada via MI", fontweight="bold")
ax.axhline(0.045, color="#2C5282", ls=":", lw=1.2, alpha=0.7)
ax.axhline(0.050, color="#D9544D", ls=":", lw=1.2, alpha=0.7)
ax.text(4.45, 0.045, "paper 0.045", color="#2C5282", fontsize=7,
        va="center", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
ax.text(4.45, 0.054, "paper 0.050", color="#D9544D", fontsize=7,
        va="center", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
ax.set_ylim(0, max(vals) * 1.25)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f"{v:.4f}", ha="center", va="bottom", fontsize=8)

# --- Panel 3: conditionals vs paper ---
ax = axes[2]
cl = [r"p(↑|mkt↑)", r"p(↓|mkt↓)", "p(rep|encert)", "p(canvi|error)"]
x = np.arange(4)
w = 0.36
b1 = ax.bar(x - w / 2, ours, w, color="#2C5282", edgecolor="white",
            label="Replicació", linewidth=1)
b2 = ax.bar(x + w / 2, paper, w, color="#E8956E", edgecolor="white",
            label="Paper original", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(cl, fontsize=8)
ax.set_ylabel("Probabilitat condicional")
ax.set_title("(c) Coincidència numèrica amb el paper", fontweight="bold")
ax.set_ylim(0, 0.92)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.4)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
for i, (a, b) in enumerate(zip(ours, paper)):
    ax.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=7.5)
    ax.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close()
print(f"[save] {OUT}")
print(f"  H(result) mean = {H_res.mean():.4f}")
print(f"  H(change) mean = {H_chg.mean():.4f}")
print(f"  MI market   = {MI_market:.4f} (paper 0.045)")
print(f"  MI result   = {MI_result:.4f} (paper 0.050)")
print(f"  MI change   = {MI_change:.4f}")
print(f"  Control d-d = {MI_dec_dec:.4f}")
print(f"  Control m-m = {MI_mkt_mkt:.4f}")
print(f"  ours vs paper conditionals:")
for c, a, b in zip(cl, ours, paper):
    print(f"    {c}: {a:.3f}  vs  {b:.3f}")
