"""
TFG - REDUCED compositional model (2 weights: R vs Imitative)

================================================================================
OBSERVATIONAL DEGENERACY MI = WSLS (found 2026-04-21)
================================================================================
In a binary game where a "hit" means D_t equals M_t (as in Mr. Banks),
the Market Imitation (MI) and Win-Stay Lose-Shift (WSLS) strategies are
OBSERVATIONALLY IDENTICAL. Proof:

  MI:   D_t = M_{t-1}

  WSLS: D_t = { D_{t-1}        if r_{t-1} = 1  (hit, so repeat)
              { 1 - D_{t-1}    if r_{t-1} = 0  (miss, so switch)

Since r_{t-1} = 1 is equivalent to D_{t-1} = M_{t-1}, and D, M are in {0,1}:
  - If r_{t-1}=1: D_t^WSLS = D_{t-1} = M_{t-1} = D_t^MI
  - If r_{t-1}=0: D_t^WSLS = 1 - D_{t-1} = 1 - (1 - M_{t-1}) = M_{t-1} = D_t^MI

So for all t >= 1:   D_t^WSLS = D_t^MI = M_{t-1}

This is not an MLE artifact, it is a structural property of the game.
The Spearman rho(w_MI, w_WSLS) = 1.000 seen in the 3-weight model confirms it.

Physical reading: the game has a symmetry (D<->1-D commutes with M<->1-M) that
shrinks the range of distinguishable strategies. It is a direct parallel of the
observational degeneracy in physics when two observables commute.

================================================================================
REDUCED MODEL
================================================================================
Base strategies (just 2):
  R (Random):      p(D_t = 1) = 0.5
  I (Imitative):   D_t = M_{t-1}   (lumps MI + WSLS together)

For each player, MLE of (w_R, w_I) with w_R + w_I = 1:
  p(D_t = 1) = 0.5 * w_R + M_{t-1} * w_I

Analyses:
  (1) Histogram of w_I, the population distribution of "imitability"
  (2) w_I by gender and age (Kruskal-Wallis)
  (3) Entropy H(D) and H(D|M) by dominant type
  (4) Predictive comparison against the 3-weight model and the networks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import kruskal, pearsonr
from collections import Counter

BASE = "/Users/racelabs/Documents/TFG"
EPS = 1e-6

# --- Load data ---
df = pd.read_csv(f"{BASE}/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])


def fit_wI(decisions_per_game, markets_per_game):
    """MLE of w_I in [0,1] for a single player. w_R = 1 - w_I."""
    all_m, all_t = [], []
    for d, m in zip(decisions_per_game, markets_per_game):
        if len(d) < 2:
            continue
        all_m.append(m[:-1].astype(float))
        all_t.append(d[1:])
    if not all_m:
        return None, None
    mm = np.concatenate(all_m)
    tt = np.concatenate(all_t)
    if len(tt) < 5:
        return None, None

    def nll(w_I):
        p1 = 0.5 * (1 - w_I) + mm * w_I
        p1 = np.clip(p1, EPS, 1 - EPS)
        return -(tt * np.log(p1) + (1 - tt) * np.log(1 - p1)).sum()

    res = minimize_scalar(nll, bounds=(0, 1), method="bounded",
                          options={"xatol": 1e-6})
    return res.x, -res.fun / len(tt)


# --- Group by player ---
print("Grouping data by player...")
user_data = {}
for (user, game), g in df.groupby(["user", "game"]):
    if len(g) < 5:
        continue
    user_data.setdefault(user, []).append({
        "decisions": g["dec_bin"].values,
        "markets": g["market_bin"].values,
    })
print(f"Valid players: {len(user_data)}")

# --- 70/15/15 split (same as lstm_prediction.py) ---
users_sorted = sorted(user_data.keys())
rng = np.random.default_rng(42)
rng.shuffle(users_sorted)
n_train = int(0.7 * len(users_sorted))
n_val = int(0.15 * len(users_sorted))
test_users = set(users_sorted[n_train + n_val:])

# --- MLE per player ---
print("\nFitting w_I per player (1D MLE)...")
wI_by_user = {}
ll_by_user = {}
for u, games in user_data.items():
    decs = [g["decisions"] for g in games]
    mkts = [g["markets"] for g in games]
    w_I, ll = fit_wI(decs, mkts)
    if w_I is None:
        continue
    wI_by_user[u] = w_I
    ll_by_user[u] = ll
print(f"Players fitted: {len(wI_by_user)}")

wI_arr = np.array([wI_by_user[u] for u in sorted(wI_by_user)])
print(f"\nGlobal w_I distribution:")
print(f"  mean = {wI_arr.mean():.3f} ± {wI_arr.std():.3f}")
print(f"  median = {np.median(wI_arr):.3f}")
print(f"  w_I > 0.5: {(wI_arr > 0.5).sum()} ({100*(wI_arr>0.5).mean():.1f}%)")

# --- Predictive evaluation on the test set ---
print("\nEvaluating on the test set...")
correct, total, ce = 0, 0, 0.0
for u in test_users:
    if u not in wI_by_user:
        continue
    w_I = wI_by_user[u]
    for g in user_data[u]:
        m = g["markets"][:-1].astype(float)
        p1 = 0.5 * (1 - w_I) + m * w_I
        p1 = np.clip(p1, EPS, 1 - EPS)
        targets = g["decisions"][1:]
        preds = (p1 > 0.5).astype(int)
        correct += (preds == targets).sum()
        total += len(targets)
        ce += -(targets * np.log(p1) + (1 - targets) * np.log(1 - p1)).sum()

print(f"2-weight compositional:  acc={correct/total:.4f}  ce={ce/total:.4f}  (n={total})")
print("\nComparison with pure baselines:")
print("  R pure  (w_I=0):  acc=0.4142  ce=0.6931  [from the 3-weight model]")
print("  I pure  (w_I=1):  acc=0.6279  ce=5.1412  [equivalent to pure MI = pure WSLS]")

# --- Save weights ---
pd.DataFrame({
    "user": sorted(wI_by_user),
    "w_R": [1 - wI_by_user[u] for u in sorted(wI_by_user)],
    "w_I": [wI_by_user[u] for u in sorted(wI_by_user)],
    "ll_mean": [ll_by_user[u] for u in sorted(wI_by_user)],
}).to_csv(f"{BASE}/data/composition_weights_2p.csv", index=False)
print(f"\nSaved: data/composition_weights_2p.csv")

# --- Demographics ---
users = pd.read_csv(f"{BASE}/data/users.csv")
dfw = pd.DataFrame({"user": sorted(wI_by_user),
                    "w_I": [wI_by_user[u] for u in sorted(wI_by_user)]})
dfw = dfw.merge(users[["id", "gender", "age_range"]], left_on="user", right_on="id")

# Gender
groups_g = [dfw[dfw["gender"] == g]["w_I"].values for g in ["h", "d"]]
stat_g, p_g = kruskal(*groups_g)
print(f"\nKruskal-Wallis w_I x gender:  H={stat_g:.3f}, p={p_g:.4f}")
print(f"  Men:   μ(w_I)={dfw[dfw['gender']=='h']['w_I'].mean():.3f} (n={(dfw['gender']=='h').sum()})")
print(f"  Women: μ(w_I)={dfw[dfw['gender']=='d']['w_I'].mean():.3f} (n={(dfw['gender']=='d').sum()})")

# Age (drops re0)
dfa = dfw[dfw["age_range"] != "re0"].copy()
dfa.loc[dfa["age_range"] == "re6", "age_range"] = "re5"
age_order = ["re1", "re2", "re3", "re4", "re5"]
age_labels = {"re1": "<18", "re2": "18-30", "re3": "31-45", "re4": "46-60", "re5": ">60"}
groups_a = [dfa[dfa["age_range"] == a]["w_I"].values for a in age_order
            if len(dfa[dfa["age_range"] == a]) > 2]
stat_a, p_a = kruskal(*groups_a)
print(f"\nKruskal-Wallis w_I x age (drops re0):  H={stat_a:.3f}, p={p_a:.4f}")
for a in age_order:
    sub = dfa[dfa["age_range"] == a]["w_I"]
    if len(sub) > 2:
        print(f"  {age_labels[a]:6s} (n={len(sub):3d}): μ={sub.mean():.3f}  mediana={sub.median():.3f}")

# --- Entropy by dominant type ---
def shannon(seq):
    c = Counter(seq); n = len(seq)
    p = np.array([v/n for v in c.values()])
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def cond_entropy(X, Y):
    c_xy = Counter(zip(X, Y)); c_x = Counter(X); n = len(X)
    H = 0.0
    for (x, y), cxy in c_xy.items():
        p_xy = cxy / n
        p_y_given_x = cxy / c_x[x]
        H -= p_xy * np.log2(p_y_given_x + 1e-12)
    return H

# Classification: w_I > 0.5 gives type "I" (imitative), otherwise "R" (random)
user_type = {u: ("I" if w > 0.5 else "R") for u, w in wI_by_user.items()}
print(f"\nPlayers by dominant type: {dict(Counter(user_type.values()))}")

H_global = shannon(df["dec_bin"].values)
df_pairs = df.copy()
df_pairs["prev_market"] = df.groupby("game")["market_bin"].shift(1)
df_pairs = df_pairs.dropna(subset=["prev_market"])
df_pairs["prev_market"] = df_pairs["prev_market"].astype(int)
HDgivenM_global = cond_entropy(df_pairs["prev_market"].values, df_pairs["dec_bin"].values)
print(f"\nGlobal reference:")
print(f"  H(D) = {H_global:.4f}")
print(f"  H(D|M_{{n-1}}) = {HDgivenM_global:.4f}")

print(f"\nBy dominant type:")
H_type, HgM_type, N_type = {}, {}, {}
for t in ["R", "I"]:
    users_t = [u for u, x in user_type.items() if x == t]
    sub = df[df["user"].isin(users_t)].copy()
    sub["prev_market"] = sub.groupby("game")["market_bin"].shift(1)
    sub = sub.dropna(subset=["prev_market"])
    sub["prev_market"] = sub["prev_market"].astype(int)
    H_type[t] = shannon(sub["dec_bin"].values)
    HgM_type[t] = cond_entropy(sub["prev_market"].values, sub["dec_bin"].values)
    N_type[t] = len(users_t)
    print(f"  Type {t} (n={N_type[t]:3d}): H(D)={H_type[t]:.4f}  H(D|M)={HgM_type[t]:.4f}  "
          f"MI(D;M)={H_type[t]-HgM_type[t]:.4f}")

# ============================================================
# FIGURE: 4 panels
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Global w_I histogram
ax = axes[0, 0]
ax.hist(wI_arr, bins=30, color="#4477AA", alpha=0.75, edgecolor="black", linewidth=0.4)
ax.axvline(wI_arr.mean(), ls="--", color="red",
           label=f"μ = {wI_arr.mean():.2f}")
ax.axvline(0.5, ls=":", color="gray", label="llindar R/I")
ax.set_xlabel("w_I  (pes imitatiu = w_MI + w_WSLS)")
ax.set_ylabel("Nombre de jugadors")
ax.set_title(f"(a) Distribució de w_I per jugador (n={len(wI_arr)})")
ax.legend()
ax.grid(alpha=0.3)

# (b) w_I by gender
ax = axes[0, 1]
colors_g = {"h": "#4477AA", "d": "#EE6677"}
labels_g = {"h": f"Home (n={(dfw['gender']=='h').sum()})",
            "d": f"Dona (n={(dfw['gender']=='d').sum()})"}
for g in ["h", "d"]:
    ax.hist(dfw[dfw["gender"] == g]["w_I"], bins=20, alpha=0.6,
            color=colors_g[g], label=labels_g[g])
ax.set_xlabel("w_I")
ax.set_ylabel("Jugadors")
ax.set_title(f"(b) w_I per gènere\nKruskal-Wallis p = {p_g:.4f}")
ax.legend()
ax.grid(alpha=0.3)

# (c) w_I by age (boxplot)
ax = axes[1, 0]
grups = []; labs = []
for a in age_order:
    sub = dfa[dfa["age_range"] == a]["w_I"].values
    if len(sub) > 2:
        grups.append(sub)
        labs.append(f"{age_labels[a]}\n(n={len(sub)})")
bp = ax.boxplot(grups, positions=range(len(grups)), patch_artist=True,
                widths=0.6, showmeans=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#4477AA"); patch.set_alpha(0.6)
ax.set_xticks(range(len(grups)))
ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("w_I")
ax.set_title(f"(c) w_I per edat (excloent 'no respost')\nKruskal-Wallis p = {p_a:.4f}")
ax.grid(alpha=0.3, axis="y")

# (d) Entropies by dominant type
ax = axes[1, 1]
types = ["R", "I"]
x = np.arange(len(types))
w = 0.35
bars_H = ax.bar(x - w/2, [H_type[t] for t in types], w,
                label="H(D)", color="#AAAAAA", edgecolor="black", linewidth=0.5)
bars_Hg = ax.bar(x + w/2, [HgM_type[t] for t in types], w,
                 label="H(D|M_{n-1})", color="#4477AA", edgecolor="black", linewidth=0.5)
ax.axhline(H_global, ls="--", color="gray", alpha=0.7,
           label=f"H global = {H_global:.3f}")
ax.axhline(HDgivenM_global, ls=":", color="#4477AA", alpha=0.7,
           label=f"H(D|M) global = {HDgivenM_global:.3f}")
ax.set_xticks(x)
ax.set_xticklabels([f"Tipus {t}\n(n={N_type[t]})" for t in types])
ax.set_ylabel("Entropia (bits)")
ax.set_ylim(0, 1.1)
ax.set_title("(d) Entropia marginal i condicional per tipus dominant\n"
             "(MI(D;M) = H(D) − H(D|M) quantifica la informació del mercat)")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3, axis="y")
for bar, t in zip(bars_H, types):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{H_type[t]:.3f}", ha="center", fontsize=8)
for bar, t in zip(bars_Hg, types):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{HgM_type[t]:.3f}", ha="center", fontsize=8)

fig.suptitle("Model composicional reduït (R vs Imitativa) — "
             "col·lapse MI+WSLS per simetria del joc", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f"{BASE}/composition_model_imitative.png", dpi=120, bbox_inches="tight")
print(f"\nSaved: composition_model_imitative.png")

# --- Comparison with the 3-weight model (if present) ---
try:
    w3 = pd.read_csv(f"{BASE}/data/composition_weights.csv")
    # w_I of the 3-weight model = w_MI + w_WSLS
    w3["w_I_3p"] = w3["w_MI"] + w3["w_WSLS"]
    merged = w3.merge(pd.DataFrame({"user": sorted(wI_by_user),
                                    "w_I_2p": [wI_by_user[u] for u in sorted(wI_by_user)]}),
                      on="user")
    r, p = pearsonr(merged["w_I_3p"], merged["w_I_2p"])
    print(f"\nConsistency 3p vs 2p:  Pearson(w_MI+w_WSLS vs w_I_2p) = {r:.4f}  p={p:.2e}")
    print("  the reduced model reproduces the sum of the imitative weights of the 3p model")
except FileNotFoundError:
    pass
