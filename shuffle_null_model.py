"""
TFG - Shuffle Null Model
Shuffle each player's decisions to build a random baseline.
Compare real vs shuffled entropy to spot players whose sequences carry structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# --- Load data ---
df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)

print(f"Decisions: {len(df)}, Users: {df['user'].nunique()}")

# --- Functions ---
def shannon_entropy(decisions):
    n = len(decisions)
    if n == 0:
        return np.nan
    counts = Counter(decisions)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def sliding_entropy_mean(decisions, window=5):
    n = len(decisions)
    if n < window:
        return np.nan
    entropies = []
    for i in range(n - window + 1):
        entropies.append(shannon_entropy(decisions[i:i+window]))
    return np.mean(entropies)

# --- 1. Marginal entropy, real vs shuffle, per player ---
print("\n1. Computing marginal entropy: real vs shuffle...")
N_SHUFFLES = 1000

results = []
for user_id, group in df.groupby("user"):
    decs = group["dec_bin"].values
    n = len(decs)
    if n < 10:
        continue

    # Real sequence
    H_real = shannon_entropy(decs)
    H_window_real = sliding_entropy_mean(decs, window=5)

    # Shuffled baseline
    H_shuffles = []
    H_window_shuffles = []
    for _ in range(N_SHUFFLES):
        shuffled = np.random.permutation(decs)
        H_shuffles.append(shannon_entropy(shuffled))
        H_window_shuffles.append(sliding_entropy_mean(shuffled, window=5))

    H_shuffle_mean = np.mean(H_shuffles)
    H_shuffle_std = np.std(H_shuffles)
    H_window_shuffle_mean = np.mean(H_window_shuffles)
    H_window_shuffle_std = np.std(H_window_shuffles)

    # Z-scores
    z_marginal = (H_real - H_shuffle_mean) / H_shuffle_std if H_shuffle_std > 0 else 0
    z_window = (H_window_real - H_window_shuffle_mean) / H_window_shuffle_std if H_window_shuffle_std > 0 else 0

    results.append({
        "user": user_id,
        "n_decisions": n,
        "H_real": H_real,
        "H_shuffle_mean": H_shuffle_mean,
        "H_shuffle_std": H_shuffle_std,
        "z_marginal": z_marginal,
        "H_window_real": H_window_real,
        "H_window_shuffle_mean": H_window_shuffle_mean,
        "H_window_shuffle_std": H_window_shuffle_std,
        "z_window": z_window
    })

df_results = pd.DataFrame(results)
print(f"Players analysed: {len(df_results)}")

# --- Results ---
print("\n" + "="*60)
print("SHUFFLE NULL MODEL RESULTS")
print("="*60)

print("\n--- Marginal entropy ---")
print(f"Marginal z-score: should not be significant")
print(f"  (shuffling leaves p(up) untouched, so marginal H is unchanged)")
print(f"  Mean z: {df_results['z_marginal'].mean():.4f}")
print(f"  Players with |z| > 1.96: {(df_results['z_marginal'].abs() > 1.96).sum()}")

print("\n--- Windowed entropy (w=5) ---")
print(f"  Mean real windowed H: {df_results['H_window_real'].mean():.4f}")
print(f"  Mean shuffle windowed H: {df_results['H_window_shuffle_mean'].mean():.4f}")
print(f"  Difference: {df_results['H_window_real'].mean() - df_results['H_window_shuffle_mean'].mean():.4f}")
print(f"  Mean z: {df_results['z_window'].mean():.4f}")
print(f"  Players with z < -1.96 (more structure than shuffle): {(df_results['z_window'] < -1.96).sum()}")
print(f"  Players with |z| < 1.96 (indistinguishable from random): {(df_results['z_window'].abs() < 1.96).sum()}")

# --- Figures ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Shuffle Null Model — Mr. Banks", fontsize=14, fontweight="bold")

# Fig 1: histogram of windowed z-scores
ax = axes[0, 0]
ax.hist(df_results["z_window"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(-1.96, color="red", ls="--", label="z = -1.96")
ax.axvline(1.96, color="red", ls="--")
ax.axvline(0, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Z-score (H finestra real vs shuffle)")
ax.set_ylabel("N jugadors")
ax.set_title("Z-scores: entropia finestra (w=5)")
ax.legend()

# Fig 2: real H vs shuffle H (windowed)
ax = axes[0, 1]
ax.scatter(df_results["H_window_shuffle_mean"], df_results["H_window_real"],
           alpha=0.5, s=20, color="steelblue", edgecolors="white", linewidth=0.3)
lim = [0.4, 1.05]
ax.plot(lim, lim, "r--", lw=1, label="H_real = H_shuffle")
ax.set_xlabel("H finestra shuffle (mitjana)")
ax.set_ylabel("H finestra real")
ax.set_title("H real vs H shuffle (finestra w=5)")
ax.legend()
ax.set_xlim(lim)
ax.set_ylim(lim)

# Fig 3: z-score vs number of decisions
ax = axes[1, 0]
ax.scatter(df_results["n_decisions"], df_results["z_window"],
           alpha=0.5, s=20, color="mediumpurple", edgecolors="white", linewidth=0.3)
ax.axhline(-1.96, color="red", ls="--", alpha=0.5)
ax.axhline(1.96, color="red", ls="--", alpha=0.5)
ax.axhline(0, color="gray", ls=":", alpha=0.3)
ax.set_xlabel("Nombre de decisions")
ax.set_ylabel("Z-score (finestra)")
ax.set_title("Z-score vs nombre de decisions")

# Fig 4: distribution of real vs shuffle H (aggregate)
ax = axes[1, 1]
ax.hist(df_results["H_window_real"], bins=30, alpha=0.6, color="steelblue",
        edgecolor="white", label="Real", density=True)
ax.hist(df_results["H_window_shuffle_mean"], bins=30, alpha=0.6, color="coral",
        edgecolor="white", label="Shuffle (mitjana)", density=True)
ax.set_xlabel("H finestra (w=5)")
ax.set_ylabel("Densitat")
ax.set_title("Distribucio H real vs shuffle")
ax.legend()

plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/shuffle_null_model.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: /Users/racelabs/Desktop/TFG/shuffle_null_model.png")

# Save results
df_results.to_csv("/Users/racelabs/Desktop/TFG/data/shuffle_results.csv", index=False)
print("Results: /Users/racelabs/Desktop/TFG/data/shuffle_results.csv")
print("\nDONE")
