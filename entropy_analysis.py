"""
TFG - Entropy analysis of the Mr. Banks experiment
Step 2: per-individual entropy, its temporal evolution, and the collective entropy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- Load data ---
df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()  # keep only valid decisions

# Drop scenario 4 (excluded in the paper)
df = df[~df["scenario"].str.startswith("4")].copy()
print(f"Valid decisions (scenario 4 excluded): {len(df)}")
print(f"Unique users: {df['user'].nunique()}")

# Recode decision as binary: -1 (down) becomes 0, 1 (up) stays 1, so we can count easily
df["dec_bin"] = (df["decision"] == 1).astype(int)


# =====================================================================
# 1. PER-INDIVIDUAL ENTROPY (over each user's full decision sequence)
# =====================================================================

def shannon_entropy(decisions):
    """Shannon entropy of a binary sequence (in bits)."""
    n = len(decisions)
    if n == 0:
        return np.nan
    counts = Counter(decisions)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# Overall entropy of each user
user_entropy = df.groupby("user")["dec_bin"].apply(
    lambda x: shannon_entropy(x.values)
).reset_index()
user_entropy.columns = ["user", "H"]

# Number of decisions per user
user_n = df.groupby("user").size().reset_index(name="n_decisions")
user_entropy = user_entropy.merge(user_n, on="user")

# p(up) per user
user_pup = df.groupby("user")["dec_bin"].mean().reset_index()
user_pup.columns = ["user", "p_up"]
user_entropy = user_entropy.merge(user_pup, on="user")

print("\n" + "="*60)
print("1. PER-INDIVIDUAL ENTROPY")
print("="*60)
print(user_entropy["H"].describe())
print(f"\nMaximum possible entropy (binary): {1.0:.3f} bits")
print(f"Mean entropy: {user_entropy['H'].mean():.4f} bits")
print(f"Median: {user_entropy['H'].median():.4f} bits")


# =====================================================================
# 2. TEMPORAL ENTROPY — round-by-round evolution (cumulative window)
# =====================================================================

# Global round index: each user plays several 25-round games, so we chain
# their decisions into one running sequence (1, 2, 3, ...)
df = df.sort_values(["user", "game", "round"])
df["global_round"] = df.groupby("user").cumcount() + 1

def cumulative_entropy(decisions):
    """Cumulative entropy: H after the first t decisions, for t = 1..n."""
    n = len(decisions)
    entropies = []
    counts = {0: 0, 1: 0}
    for t in range(n):
        counts[decisions[t]] += 1
        total = t + 1
        probs = np.array([counts[k] / total for k in counts if counts[k] > 0])
        H = -np.sum(probs * np.log2(probs))
        entropies.append(H)
    return entropies

# Compute the cumulative entropy curve for each user
print("\nComputing cumulative entropy per user...")
all_cum_entropy = []
for user_id, group in df.groupby("user"):
    decs = group["dec_bin"].values
    Hs = cumulative_entropy(decs)
    for t, H in enumerate(Hs):
        all_cum_entropy.append({"user": user_id, "t": t + 1, "H_cum": H})

df_cum = pd.DataFrame(all_cum_entropy)

# Mean and std across users at each timestep
cum_stats = df_cum.groupby("t")["H_cum"].agg(["mean", "std", "count"]).reset_index()
cum_stats.columns = ["t", "H_mean", "H_std", "n_users"]


# =====================================================================
# 3. COLLECTIVE ENTROPY — group decision distribution per round
# =====================================================================

# For each round within each scenario, look at the collective decision
# distribution (what fraction of the group chose up vs down)

def collective_entropy_by_round(df):
    """Entropy of the group's decision distribution at each round (1-25) of every scenario."""
    results = []
    for (scenario, round_num), group in df.groupby(["scenario", "round"]):
        n = len(group)
        p_up = group["dec_bin"].mean()
        if p_up == 0 or p_up == 1:
            H = 0.0
        else:
            H = -p_up * np.log2(p_up) - (1 - p_up) * np.log2(1 - p_up)
        results.append({
            "scenario": scenario, "round": round_num,
            "H_collective": H, "p_up": p_up, "n": n
        })
    return pd.DataFrame(results)

df_collective = collective_entropy_by_round(df)

# Average each round across scenarios
coll_by_round = df_collective.groupby("round").agg(
    H_mean=("H_collective", "mean"),
    H_std=("H_collective", "std"),
    p_up_mean=("p_up", "mean")
).reset_index()

print("\n" + "="*60)
print("3. COLLECTIVE ENTROPY PER ROUND (averaged across scenarios)")
print("="*60)
print(coll_by_round.to_string(index=False))


# =====================================================================
# 4. SLIDING-WINDOW ENTROPY (per individual, 5-round window)
# =====================================================================

WINDOW = 5

def sliding_entropy(decisions, window=WINDOW):
    """Entropy over a sliding window of the given size."""
    n = len(decisions)
    if n < window:
        return []
    results = []
    for i in range(n - window + 1):
        chunk = decisions[i:i+window]
        H = shannon_entropy(chunk)
        results.append({"t_center": i + window // 2 + 1, "H_window": H})
    return results

print("\nComputing sliding-window entropy...")
all_sliding = []
for user_id, group in df.groupby("user"):
    decs = group["dec_bin"].values
    for entry in sliding_entropy(decs, WINDOW):
        entry["user"] = user_id
        all_sliding.append(entry)

df_sliding = pd.DataFrame(all_sliding)
slide_stats = df_sliding.groupby("t_center")["H_window"].agg(["mean", "std"]).reset_index()
slide_stats.columns = ["t_center", "H_mean", "H_std"]


# =====================================================================
# FIGURES
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis de Entropía — Mr. Banks Experiment", fontsize=14, fontweight="bold")

# --- Fig 1: histogram of per-individual entropy ---
ax = axes[0, 0]
ax.hist(user_entropy["H"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(1.0, color="red", ls="--", label="H_max = 1 bit")
ax.axvline(user_entropy["H"].mean(), color="orange", ls="--", label=f"Media = {user_entropy['H'].mean():.3f}")
ax.set_xlabel("Entropía H (bits)")
ax.set_ylabel("Número de individuos")
ax.set_title("Distribución de entropía individual")
ax.legend()

# --- Fig 2: mean cumulative entropy vs timestep ---
ax = axes[0, 1]
ax.plot(cum_stats["t"], cum_stats["H_mean"], color="steelblue", lw=1.5)
ax.fill_between(cum_stats["t"],
                cum_stats["H_mean"] - cum_stats["H_std"],
                cum_stats["H_mean"] + cum_stats["H_std"],
                alpha=0.2, color="steelblue")
ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="H_max")
ax.set_xlabel("Ronda global (1-75)")
ax.set_ylabel("H acumulativa (bits)")
ax.set_title("Entropía acumulativa media (± 1σ)")
ax.legend()
ax.set_xlim(1, 75)

# --- Fig 3: collective entropy per round (within each scenario) ---
ax = axes[1, 0]
for scenario, grp in df_collective.groupby("scenario"):
    ax.plot(grp["round"], grp["H_collective"], marker=".", ms=4, label=scenario, alpha=0.7)
ax.axhline(1.0, color="red", ls="--", alpha=0.3)
ax.set_xlabel("Ronda (dentro del scenario, 1-25)")
ax.set_ylabel("H colectiva (bits)")
ax.set_title("Entropía colectiva por ronda y scenario")
ax.legend(fontsize=7, ncol=2)

# --- Fig 4: sliding-window entropy ---
ax = axes[1, 1]
ax.plot(slide_stats["t_center"], slide_stats["H_mean"], color="darkgreen", lw=1.5)
ax.fill_between(slide_stats["t_center"],
                slide_stats["H_mean"] - slide_stats["H_std"],
                slide_stats["H_mean"] + slide_stats["H_std"],
                alpha=0.2, color="darkgreen")
ax.axhline(1.0, color="red", ls="--", alpha=0.5)
ax.set_xlabel("Ronda global")
ax.set_ylabel(f"H ventana ({WINDOW} rondas) (bits)")
ax.set_title(f"Entropía media ventana deslizante (w={WINDOW})")
ax.set_xlim(1, 75)

plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/entropy_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved to /Users/racelabs/Desktop/TFG/entropy_analysis.png")


# =====================================================================
# 5. PER-INDIVIDUAL ENTROPY vs p(up) — scatter
# =====================================================================

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(user_entropy["p_up"], user_entropy["H"],
                c=user_entropy["n_decisions"], cmap="viridis",
                alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
plt.colorbar(sc, label="Nº decisiones")

# Theoretical curve H = -p*log2(p) - (1-p)*log2(1-p)
p_range = np.linspace(0.01, 0.99, 200)
H_theory = -p_range * np.log2(p_range) - (1 - p_range) * np.log2(1 - p_range)
ax.plot(p_range, H_theory, "r--", lw=1.5, label="H_max teórica")

ax.set_xlabel("p(up) del individuo")
ax.set_ylabel("Entropía H (bits)")
ax.set_title("Entropía individual vs sesgo up/down")
ax.legend()
plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/entropy_vs_pup.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved to /Users/racelabs/Desktop/TFG/entropy_vs_pup.png")

print("\n" + "="*60)
print("DONE")
print("="*60)
