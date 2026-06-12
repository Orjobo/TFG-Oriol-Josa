"""
TFG - Sliding-window analysis across several window sizes.
Looks for a critical w where the entropy changes behaviour.
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

def shannon_entropy(decisions):
    n = len(decisions)
    if n == 0:
        return np.nan
    counts = Counter(decisions)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def sliding_entropy_mean(decisions, window):
    n = len(decisions)
    if n < window:
        return np.nan
    entropies = []
    for i in range(n - window + 1):
        entropies.append(shannon_entropy(decisions[i:i+window]))
    return np.mean(entropies)

# --- Compute H over several window sizes ---
WINDOWS = [2, 3, 5, 7, 10, 15, 20, 25]
N_SHUFFLES = 200  # fewer than before to keep it fast

print("Computing H over several window sizes...")
print(f"Windows: {WINDOWS}")

results_by_w = {w: {"real": [], "shuffle": []} for w in WINDOWS}

for user_id, group in df.groupby("user"):
    decs = group["dec_bin"].values
    n = len(decs)

    for w in WINDOWS:
        if n < w:
            continue

        # Real
        H_real = sliding_entropy_mean(decs, w)
        results_by_w[w]["real"].append(H_real)

        # Shuffle (mean over N permutations)
        H_shuffles = []
        for _ in range(N_SHUFFLES):
            shuffled = np.random.permutation(decs)
            H_shuffles.append(sliding_entropy_mean(shuffled, w))
        results_by_w[w]["shuffle"].append(np.mean(H_shuffles))

# --- Aggregate ---
summary = []
for w in WINDOWS:
    reals = np.array(results_by_w[w]["real"])
    shuffles = np.array(results_by_w[w]["shuffle"])
    if len(reals) == 0:
        continue
    summary.append({
        "w": w,
        "H_real_mean": np.mean(reals),
        "H_real_std": np.std(reals),
        "H_shuffle_mean": np.mean(shuffles),
        "H_shuffle_std": np.std(shuffles),
        "delta": np.mean(reals) - np.mean(shuffles),
        "n_users": len(reals)
    })

df_summary = pd.DataFrame(summary)

print("\n" + "="*60)
print("RESULTS PER WINDOW")
print("="*60)
print(df_summary.to_string(index=False))

# --- Figures ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Entropia per finestra lliscant — Analisi multiescala", fontsize=14, fontweight="bold")

# Fig 1: H real vs H shuffle per w
ax = axes[0]
ax.errorbar(df_summary["w"], df_summary["H_real_mean"], yerr=df_summary["H_real_std"],
            marker="o", color="steelblue", label="Real", capsize=3)
ax.errorbar(df_summary["w"], df_summary["H_shuffle_mean"], yerr=df_summary["H_shuffle_std"],
            marker="s", color="coral", label="Shuffle", capsize=3)
ax.set_xlabel("Mida finestra (w)")
ax.set_ylabel("H mitjana (bits)")
ax.set_title("H real vs shuffle per finestra")
ax.legend()
ax.set_xticks(WINDOWS)

# Fig 2: Delta (real - shuffle) per w
ax = axes[1]
ax.plot(df_summary["w"], df_summary["delta"], marker="o", color="darkgreen", lw=2)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Mida finestra (w)")
ax.set_ylabel("Delta H (real - shuffle)")
ax.set_title("Diferencia real vs shuffle")
ax.set_xticks(WINDOWS)

# Fig 3: distribution of real H for each w (boxplot)
ax = axes[2]
data_for_box = [np.array(results_by_w[w]["real"]) for w in WINDOWS if len(results_by_w[w]["real"]) > 0]
labels_for_box = [str(w) for w in WINDOWS if len(results_by_w[w]["real"]) > 0]
bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("steelblue")
    patch.set_alpha(0.6)
ax.set_xlabel("Mida finestra (w)")
ax.set_ylabel("H per jugador (bits)")
ax.set_title("Distribucio H per finestra")

plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/window_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: /Users/racelabs/Desktop/TFG/window_analysis.png")
print("DONE")
