"""
TFG - Clustering players by strategy.
k-means + PCA to identify behavioural subpopulations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Load data ---
df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)

# Previous-round info, kept within each game (shift must not bleed across games)
df["prev_decision"] = df.groupby("game")["dec_bin"].shift(1)
df["prev_result"] = df.groupby("game")["res_bin"].shift(1)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])
df["prev_market"] = df.groupby("game")["market_bin"].shift(1)
df["changed"] = np.where(df["prev_decision"].isna(), np.nan,
                         (df["dec_bin"] != df["prev_decision"]).astype(float))

df_pairs = df.dropna(subset=["prev_decision"]).copy()
df_pairs["prev_decision"] = df_pairs["prev_decision"].astype(int)
df_pairs["prev_result"] = df_pairs["prev_result"].astype(int)
df_pairs["prev_market"] = df_pairs["prev_market"].astype(int)
df_pairs["changed"] = df_pairs["changed"].astype(int)

# --- Functions ---
def shannon_entropy(vals):
    n = len(vals)
    if n == 0:
        return np.nan
    counts = Counter(vals)
    probs = np.array([c / n for c in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# --- Build per-player features ---
print("Building per-player features...")
user_features = []

for user_id, group in df_pairs.groupby("user"):
    if len(group) < 15:
        continue

    decs = group["dec_bin"].values
    changes = group["changed"].values
    results = group["res_bin"].values

    # Conditional probabilities
    mkt_up = group[group["prev_market"] == 1]
    mkt_down = group[group["prev_market"] == 0]
    res_ok = group[group["prev_result"] == 1]
    res_fail = group[group["prev_result"] == 0]

    p_up_given_mkt_up = mkt_up["dec_bin"].mean() if len(mkt_up) > 0 else np.nan
    p_down_given_mkt_down = (1 - mkt_down["dec_bin"].mean()) if len(mkt_down) > 0 else np.nan
    p_repeat_given_success = (1 - res_ok["changed"].mean()) if len(res_ok) > 0 else np.nan
    p_change_given_fail = res_fail["changed"].mean() if len(res_fail) > 0 else np.nan

    user_features.append({
        "user": user_id,
        "H_decision": shannon_entropy(decs),
        "H_change": shannon_entropy(changes),
        "p_up": np.mean(decs),
        "p_change": np.mean(changes),
        "p_up_mkt_up": p_up_given_mkt_up,
        "p_down_mkt_down": p_down_given_mkt_down,
        "p_repeat_success": p_repeat_given_success,
        "p_change_fail": p_change_given_fail,
        "n_decisions": len(group)
    })

df_feat = pd.DataFrame(user_features).dropna()
print(f"Players with complete features: {len(df_feat)}")

# --- Features for clustering ---
feature_cols = ["H_decision", "H_change", "p_up", "p_change",
                "p_up_mkt_up", "p_down_mkt_down",
                "p_repeat_success", "p_change_fail"]

X = df_feat[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA ---
pca = PCA(n_components=len(feature_cols))
X_pca = pca.fit_transform(X_scaled)

print("\n--- PCA: explained variance ---")
for i, (var, cumvar) in enumerate(zip(pca.explained_variance_ratio_,
                                       np.cumsum(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {var:.3f} ({cumvar:.3f} cumulative)")

print("\n--- PCA: principal components (loadings) ---")
print(f"{'Feature':<20} {'PC1':>6} {'PC2':>6} {'PC3':>6}")
print("-" * 40)
for feat, loadings in zip(feature_cols, pca.components_.T):
    print(f"{feat:<20} {loadings[0]:>6.3f} {loadings[1]:>6.3f} {loadings[2]:>6.3f}")

# --- Silhouette for k=2..6 ---
print("\n--- Silhouette by number of clusters ---")
silhouettes = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouettes[k] = sil
    print(f"  k={k}: silhouette = {sil:.4f}")

best_k = max(silhouettes, key=silhouettes.get)
print(f"\n  Best k: {best_k}")

# --- Final clustering ---
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_feat["cluster"] = km_final.fit_predict(X_scaled)

print(f"\n--- Profile of the {best_k} clusters ---")
for c in range(best_k):
    mask = df_feat["cluster"] == c
    n = mask.sum()
    print(f"\nCluster {c} (n={n}):")
    for col in feature_cols:
        print(f"  {col:<20}: {df_feat.loc[mask, col].mean():.4f}")

# --- Figures ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Clustering de jugadors (k={best_k}) — Mr. Banks",
             fontsize=14, fontweight="bold")

colors_cluster = ["steelblue", "coral", "mediumpurple", "darkgreen", "orange", "brown"]

# Fig 1: PCA scatter PC1 vs PC2
ax = axes[0, 0]
for c in range(best_k):
    mask = df_feat["cluster"] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               alpha=0.6, s=30, color=colors_cluster[c],
               label=f"Cluster {c} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA: PC1 vs PC2")
ax.legend(fontsize=8)

# Fig 2: PCA scatter PC1 vs PC3
ax = axes[0, 1]
for c in range(best_k):
    mask = df_feat["cluster"] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 2],
               alpha=0.6, s=30, color=colors_cluster[c],
               label=f"Cluster {c}")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
ax.set_title("PCA: PC1 vs PC3")
ax.legend(fontsize=8)

# Fig 3: Silueta per k
ax = axes[0, 2]
ax.plot(list(silhouettes.keys()), list(silhouettes.values()),
        marker="o", color="darkgreen", lw=2)
ax.set_xlabel("Nombre de clusters (k)")
ax.set_ylabel("Silueta")
ax.set_title("Index de silueta vs k")
ax.set_xticks(list(silhouettes.keys()))

# Fig 4: cluster profile barplot - MI features
ax = axes[1, 0]
mi_features = ["p_up_mkt_up", "p_down_mkt_down"]
x_pos = np.arange(len(mi_features))
width = 0.8 / best_k
for c in range(best_k):
    mask = df_feat["cluster"] == c
    vals = [df_feat.loc[mask, f].mean() for f in mi_features]
    ax.bar(x_pos + c * width, vals, width, color=colors_cluster[c],
           label=f"Cluster {c}", alpha=0.8)
ax.set_xticks(x_pos + width * (best_k - 1) / 2)
ax.set_xticklabels(["p(up|mkt up)", "p(down|mkt down)"], fontsize=8)
ax.set_ylabel("Probabilitat")
ax.set_title("Market Imitation per cluster")
ax.legend(fontsize=7)
ax.axhline(0.5, color="gray", ls="--", alpha=0.3)

# Fig 5: cluster profile barplot - WSLS features
ax = axes[1, 1]
wsls_features = ["p_repeat_success", "p_change_fail"]
for c in range(best_k):
    mask = df_feat["cluster"] == c
    vals = [df_feat.loc[mask, f].mean() for f in wsls_features]
    ax.bar(x_pos + c * width, vals, width, color=colors_cluster[c],
           label=f"Cluster {c}", alpha=0.8)
ax.set_xticks(x_pos + width * (best_k - 1) / 2)
ax.set_xticklabels(["p(rep|encert)", "p(canvi|error)"], fontsize=8)
ax.set_ylabel("Probabilitat")
ax.set_title("WSLS per cluster")
ax.legend(fontsize=7)
ax.axhline(0.5, color="gray", ls="--", alpha=0.3)

# Fig 6: cluster profile barplot - entropies
ax = axes[1, 2]
h_features = ["H_decision", "H_change"]
for c in range(best_k):
    mask = df_feat["cluster"] == c
    vals = [df_feat.loc[mask, f].mean() for f in h_features]
    ax.bar(x_pos + c * width, vals, width, color=colors_cluster[c],
           label=f"Cluster {c}", alpha=0.8)
ax.set_xticks(x_pos + width * (best_k - 1) / 2)
ax.set_xticklabels(["H(decisio)", "H(canvi)"], fontsize=8)
ax.set_ylabel("H (bits)")
ax.set_title("Entropies per cluster")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("/Users/racelabs/Desktop/TFG/clustering_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: /Users/racelabs/Desktop/TFG/clustering_analysis.png")

# Save
df_feat.to_csv("/Users/racelabs/Desktop/TFG/data/clustering_results.csv", index=False)
print("Results: /Users/racelabs/Desktop/TFG/data/clustering_results.csv")
print("\nDONE")
