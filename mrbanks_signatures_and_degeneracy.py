"""
Mr. Banks: the per-decision fingerprint and the MI/WSLS degeneracy.

Produces two figures:

  mrbanks_signatures.png
    For each previous state (D_{n-1}, R_{n-1}) -- last decision (up/down)
    crossed with last outcome (hit/miss) -- the empirical P(D_n = up) against
    the three baseline strategies:
      Random  0.5 everywhere
      MI      copy the previous market direction M_{n-1}
      WSLS    stay after a hit, switch after a miss
    MI and WSLS agree in two of the four states and differ in the other two,
    but since M_{n-1} equals D_{n-1} after a hit and its complement after a
    miss, their aggregate predictions coincide. That is the degeneracy below.

  mrbanks_degeneracy.png
    Left: w_MI vs w_WSLS for the 280 players, with the Spearman rho.
    Right: Spearman correlation matrix of (w_R, w_MI, w_WSLS).
    The two weights are perfectly rank-correlated, so MI and WSLS cannot be
    told apart on this binary game; the fit only pins down w_MI + w_WSLS.

Figures are written to OUT (the TFG figures folder).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

DATA = Path("/Users/racelabs/Library/Mobile Documents/com~apple~CloudDocs/Documents/TFG/data")
OUT = Path("/Users/racelabs/Desktop/TFG_memoria/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# =====================================================================
# Load the rounds, drop non-responses and scenario 4, then build the lagged variables
# =====================================================================
df = pd.read_csv(DATA / "rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].astype(str).str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])
df["prev_decision"] = df.groupby("game")["dec_bin"].shift(1)
df["prev_result"] = df.groupby("game")["res_bin"].shift(1)
df["prev_market"] = df.groupby("game")["market_bin"].shift(1)
dfp = df.dropna(subset=["prev_decision"]).copy()
for c in ["prev_decision", "prev_result", "prev_market"]:
    dfp[c] = dfp[c].astype(int)


# =====================================================================
# Fig 1: per-state fingerprint
# =====================================================================
# The four previous states (D_{n-1}, R_{n-1}) and the empirical P(D_n = up)
states = [(1, 1), (1, 0), (0, 1), (0, 0)]
state_lbl = ["(↑, encert)", "(↑, error)", "(↓, encert)", "(↓, error)"]

emp = []
n_per = []
for d_prev, r_prev in states:
    sub = dfp[(dfp["prev_decision"] == d_prev) & (dfp["prev_result"] == r_prev)]
    emp.append(sub["dec_bin"].mean())
    n_per.append(len(sub))

# Model predictions for P(D_n = up | D_{n-1}, R_{n-1})
# Random: 0.5
pred_R = [0.5, 0.5, 0.5, 0.5]
# WSLS: stay if win, switch if lose → D_n = D_{n-1} if R=1 else 1-D_{n-1}
pred_WSLS = [1, 0, 0, 1]
# MI: copy market.  M_{n-1} = D_{n-1} if R=1, else 1-D_{n-1}
# Therefore D_n = M_{n-1} = D_{n-1} if R=1 else 1-D_{n-1}
# Same vector as WSLS: this is the degeneracy.
pred_MI = [1, 0, 0, 1]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
x = np.arange(4)
w = 0.21
b1 = ax.bar(x - 1.5 * w, pred_R, w, color="#999999", edgecolor="white",
            label="Random  (predicció 0.5)", linewidth=1.0)
b2 = ax.bar(x - 0.5 * w, pred_WSLS, w, color="#D9544D", edgecolor="white",
            label="WSLS  (mantenir si encert)", linewidth=1.0)
b3 = ax.bar(x + 0.5 * w, pred_MI, w, color="#2C5282", edgecolor="white",
            label="Market Imit.  (copia mercat)", linewidth=1.0,
            hatch="///", alpha=0.85)
b4 = ax.bar(x + 1.5 * w, emp, w, color="#3CA663", edgecolor="white",
            label="Empíric", linewidth=1.0)

for i, (v, n) in enumerate(zip(emp, n_per)):
    ax.text(i + 1.5 * w, v + 0.025, f"{v:.3f}", ha="center", fontsize=8.5,
            fontweight="bold", color="#1F6633")
    ax.text(i, -0.07, f"n={n}", ha="center", fontsize=7.5, color="gray")

ax.set_xticks(x)
ax.set_xticklabels(["$D_{n-1}, R_{n-1}$ = " + s for s in state_lbl], fontsize=9)
ax.set_ylabel(r"$P(D_n = \uparrow \, | \, D_{n-1}, R_{n-1})$")
ax.set_title("Empremta empírica vs prediccions de cada estratègia\n"
             r"(les barres MI i WSLS coincideixen sempre $\Rightarrow$ "
             "degeneració)",
             fontweight="bold", fontsize=10.5)
ax.set_ylim(-0.1, 1.10)
ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.4)
ax.legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.9,
          bbox_to_anchor=(0.5, -0.18))

plt.tight_layout()
out1 = OUT / "mrbanks_signatures.png"
plt.savefig(out1, dpi=180, bbox_inches="tight")
plt.close()
print(f"[save] {out1}")
print(f"  Empíric: {[f'{v:.3f}' for v in emp]}")
print(f"  N per estat: {n_per}")


# =====================================================================
# Fig 2: observational degeneracy (rho = 1)
# =====================================================================
W = pd.read_csv(DATA / "composition_weights.csv")
print(f"\n[load] {len(W)} jugadors amb pesos del model 3p")
rho_spear, p_spear = spearmanr(W["w_MI"], W["w_WSLS"])
print(f"  ρ_Spearman(w_MI, w_WSLS) = {rho_spear:.4f}  (p={p_spear:.2e})")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                          gridspec_kw={"width_ratios": [1.6, 1]})

# Left panel: scatter
ax = axes[0]
ax.scatter(W["w_MI"], W["w_WSLS"], s=24, alpha=0.55,
           color="#0072B2", edgecolor="white", linewidth=0.5)
xmax = max(W["w_MI"].max(), W["w_WSLS"].max()) * 1.05
lim = [-0.02, xmax]
ax.plot(lim, lim, ":", color="gray", lw=1.0, alpha=0.6,
        label="$w_{MI}=w_{WSLS}$")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$w_{\mathrm{MI}}$")
ax.set_ylabel(r"$w_{\mathrm{WSLS}}$")
# ax.set_title("(a) Relació monotònica perfecta entre $w_{MI}$ i $w_{WSLS}$\n"
#              rf"$\rho_{{\mathrm{{Spearman}}}}={rho_spear:.3f}$, $n={len(W)}$ jugadors",
#              fontweight="bold", fontsize=10.5)
ax.legend(loc="lower right")
ax.set_aspect("equal")

# Right panel: correlation matrix
ax = axes[1]
cols = ["w_R", "w_MI", "w_WSLS"]
labels = ["$w_R$", "$w_{MI}$", "$w_{WSLS}$"]
M = np.zeros((3, 3))
for i, a in enumerate(cols):
    for j, b in enumerate(cols):
        if i == j:
            M[i, j] = 1.0
        else:
            M[i, j], _ = spearmanr(W[a], W[b])
im = ax.imshow(M, cmap="viridis", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=14)
ax.set_yticklabels(labels, fontsize=14)
for i in range(3):
    for j in range(3):
        c = "white" if M[i, j] < -0.3 else "black"
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                color=c, fontsize=14, fontweight="bold")
# ax.set_title("(b) Matriu de correlació de Spearman\n"
#              "MI i WSLS perfectament redundants",
#              fontweight="bold")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out2 = OUT / "mrbanks_degeneracy.png"
plt.savefig(out2, dpi=180, bbox_inches="tight")
plt.close()
print(f"[save] {out2}")
print("\nLectura:")
print("  · w_MI ↔ w_WSLS amb ρ=1.000 → indistingibles per dades")
print("  · Causa: en aquest joc binari, MI i WSLS prediuen el mateix")
print("    en els 4 estats possibles (vegeu fingerprint a slide 5)")
print("  · Solució: col·lapsar a w_I = w_MI + w_WSLS (slide 7)")
