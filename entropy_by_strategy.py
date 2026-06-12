"""
TFG - Entropy segmented by strategy.

Idea raised by Perello (2026-04-14): once each decision has been attributed
to one of the strategies, we can look at entropy again. The combined sequence
mixes every strategy, but if we say "these zeros and ones belong to this
strategy", each subset should reveal a pattern, i.e. lower entropy than the
pooled sequence, since attributing a decision to a strategy is what lets us
explain it.

Interpretation: if we assign each decision to the strategy that best explains
it (per-player compositional weights), the "strategy-conditioned" sequence
should carry less entropy than the global sequence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("/Users/racelabs/Documents/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(int)
df["res_bin"] = (df["result"] == 1).astype(int)
df["market_bin"] = np.where(df["result"] == 1, df["dec_bin"], 1 - df["dec_bin"])

weights = pd.read_csv("/Users/racelabs/Documents/TFG/data/composition_weights.csv")
weights_by_user = {r["user"]: (r["w_R"], r["w_MI"], r["w_WSLS"])
                    for _, r in weights.iterrows()}

EPS = 1e-12


def shannon(seq):
    if len(seq) == 0:
        return np.nan
    c = Counter(seq)
    n = len(seq)
    p = np.array([v / n for v in c.values()])
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def cond_entropy(X, Y):
    """H(Y|X) for discrete variables."""
    if len(X) == 0:
        return np.nan
    c_xy = Counter(zip(X, Y))
    c_x = Counter(X)
    n = len(X)
    H = 0.0
    for (x, y), cxy in c_xy.items():
        p_xy = cxy / n
        p_y_given_x = cxy / c_x[x]
        H -= p_xy * np.log2(p_y_given_x + EPS)
    return H


# --- For each decision t >= 1, compute the player's p_R, p_MI, p_WSLS ---
# and assign it to the strategy with the largest weight times correct prediction.
records = []
for (user, game), g in df.groupby(["user", "game"]):
    if len(g) < 3 or user not in weights_by_user:
        continue
    wR, wMI, wWSLS = weights_by_user[user]
    d = g["dec_bin"].values
    r = g["res_bin"].values
    m = g["market_bin"].values
    for t in range(1, len(d)):
        prev_r = r[t - 1]
        prev_d = d[t - 1]
        prev_m = m[t - 1]
        p_R = 0.5
        p_MI = 1.0 if prev_m == 1 else 0.0
        p_WSLS = float(prev_d) if prev_r == 1 else float(1 - prev_d)
        target = d[t]
        # Probability that each strategy produces the observed target:
        likes = {
            "R":    wR * (p_R if target == 1 else (1 - p_R)),
            "MI":   wMI * (p_MI if target == 1 else (1 - p_MI)),
            "WSLS": wWSLS * (p_WSLS if target == 1 else (1 - p_WSLS)),
        }
        # Assign to the strategy with the largest posterior likelihood
        total = sum(likes.values())
        if total < EPS:
            continue
        posteriors = {k: v / total for k, v in likes.items()}
        assigned = max(posteriors, key=posteriors.get)
        records.append({
            "user": user,
            "game": game,
            "t": t,
            "decision": target,
            "prev_market": prev_m,
            "prev_result": prev_r,
            "assigned_strategy": assigned,
            "post_R": posteriors["R"],
            "post_MI": posteriors["MI"],
            "post_WSLS": posteriors["WSLS"],
        })

events = pd.DataFrame(records)
print(f"Total segmented decisions: {len(events)}")
print("\nProportion assigned to each strategy:")
print(events["assigned_strategy"].value_counts(normalize=True).round(3))

# --- Global entropy vs per-strategy entropy ---
H_global = shannon(events["decision"].values)
print(f"\nGlobal entropy H(D) = {H_global:.4f} bits")

print("\nSEGMENTED entropy by assigned strategy (decision by decision):")
for strat in ["R", "MI", "WSLS"]:
    sub = events[events["assigned_strategy"] == strat]
    if len(sub) > 0:
        h = shannon(sub["decision"].values)
        print(f"  {strat:6s}: H(D) = {h:.4f}   n={len(sub)}")

# --- Entropy by PLAYER TYPE (cluster by dominant weight) ---
# Criterion: each player is assigned to their dominant weight (argmax of w_R, w_MI, w_WSLS).
# This differs from the per-decision assignment: it is more robust and makes
# sense as genuine "strategy clusters".
user_type = {}
for _, row in weights.iterrows():
    w = {"R": row["w_R"], "MI": row["w_MI"], "WSLS": row["w_WSLS"]}
    user_type[row["user"]] = max(w, key=w.get)

print("\nPlayers by dominant type:")
from collections import Counter as _C
print(dict(_C(user_type.values())))

print("\nEntropy by PLAYER TYPE (over all their decisions):")
H_by_type = {}
N_by_type = {}
for t in ["R", "MI", "WSLS"]:
    users_t = [u for u, x in user_type.items() if x == t]
    seqs = df[df["user"].isin(users_t)]["dec_bin"].values
    if len(seqs) > 0:
        H_by_type[t] = shannon(seqs)
        N_by_type[t] = len(seqs)
        print(f"  Players of type {t:6s}: H(D) = {H_by_type[t]:.4f}   decisions n={N_by_type[t]}   players={len(users_t)}")

# --- Global conditional entropy for comparison ---
df_pairs = df.copy()
df_pairs["prev_market"] = df.groupby("game")["market_bin"].shift(1)
df_pairs = df_pairs.dropna(subset=["prev_market"])
df_pairs["prev_market"] = df_pairs["prev_market"].astype(int)
H_D_given_M = cond_entropy(df_pairs["prev_market"].values, df_pairs["dec_bin"].values)
print(f"\nGlobal reference: H(D_n | M_{{n-1}}) = {H_D_given_M:.4f} bits (all players)")

# --- Plot: two panels ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
strats = ["R", "MI", "WSLS"]
colors = ["gray", "steelblue", "coral"]

# (a) By assigned decision (events)
ax = axes[0]
Hs = [shannon(events[events["assigned_strategy"] == s]["decision"].values)
      if len(events[events["assigned_strategy"] == s]) > 0 else 0 for s in strats]
Ns = [len(events[events["assigned_strategy"] == s]) for s in strats]
bars = ax.bar(strats, Hs, color=colors, alpha=0.8)
ax.axhline(H_global, ls="--", color="red", label=f"H global = {H_global:.3f}")
for bar, n in zip(bars, Ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"n={n}", ha="center", fontsize=10)
ax.set_ylabel("Entropia de Shannon (bits)")
ax.set_title("(a) H(D) per decisió assignada a cada estratègia\n"
             "(via posterior del model composicional)")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(alpha=0.3, axis="y")

# (b) By dominant player type
ax = axes[1]
Hs_u = [H_by_type.get(s, 0) for s in strats]
Ns_u = [N_by_type.get(s, 0) for s in strats]
bars = ax.bar(strats, Hs_u, color=colors, alpha=0.8)
ax.axhline(H_global, ls="--", color="red", label=f"H global = {H_global:.3f}")
for bar, n in zip(bars, Ns_u):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"n={n}", ha="center", fontsize=10)
ax.set_ylabel("Entropia de Shannon (bits)")
ax.set_title("(b) H(D) per jugadors segons tipus dominant\n"
             "(argmax dels pesos composicionals)")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/Users/racelabs/Documents/TFG/entropy_by_strategy.png", dpi=120)
print("\nSaved: entropy_by_strategy.png")
