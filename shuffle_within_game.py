"""
Shuffle null model, within-game variant (the conservative one).
Instead of shuffling all of a player's decisions together, we shuffle only
within each 25-round game. This keeps the boundaries between games intact
(different games can carry different market trends) and gives a stricter null.

Compared against the "all-decisions" shuffle at N = 25,000 shuffles.
"""

import pandas as pd
import numpy as np
import time

WINDOW = 5

df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(np.int8)

# Per-window entropy lookup table
H_LOOKUP = np.zeros(WINDOW + 1)
for c in range(WINDOW + 1):
    if 0 < c < WINDOW:
        p = c / WINDOW
        H_LOOKUP[c] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def sliding_entropy_fast(seq, window=WINDOW):
    n = len(seq)
    if n < window:
        return np.nan
    cs = np.concatenate(([0], np.cumsum(seq)))
    counts = cs[window:] - cs[:-window]
    return H_LOOKUP[counts].mean()


def shuffle_within_games(decs_per_game, rng):
    """Shuffle each game independently."""
    out = []
    for g in decs_per_game:
        g_shuffled = g.copy()
        rng.shuffle(g_shuffled)
        out.append(g_shuffled)
    return np.concatenate(out)


def run_within_game(N_SHUFFLES, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    results = []
    t0 = time.time()
    for user_id, group in df.groupby("user"):
        decs = group["dec_bin"].values
        n = len(decs)
        if n < 10:
            continue
        H_win_real = sliding_entropy_fast(decs)

        # Decisions split by game
        games = [g["dec_bin"].values.copy()
                 for _, g in group.groupby("game")]

        H_win_shuffles = np.empty(N_SHUFFLES)
        for i in range(N_SHUFFLES):
            buf = shuffle_within_games(games, rng)
            H_win_shuffles[i] = sliding_entropy_fast(buf)

        mu = H_win_shuffles.mean()
        sigma = H_win_shuffles.std()
        z = (H_win_real - mu) / sigma if sigma > 0 else 0.0
        p_emp = ((H_win_shuffles <= H_win_real).sum() + 1) / (N_SHUFFLES + 1)

        results.append({
            "user": user_id,
            "n_decisions": n,
            "H_window_real": H_win_real,
            "H_window_shuffle_mean": mu,
            "H_window_shuffle_std": sigma,
            "z_window": z,
            "p_emp_lower": p_emp,
        })
    return pd.DataFrame(results), time.time() - t0


N = 25_000
print(f"Running within-game shuffle with N = {N}...")
res, elapsed = run_within_game(N, rng_seed=42)
print(f"  (time: {elapsed:.1f} s)")

print(f"\n=== WITHIN-GAME SHUFFLE (N = {N}) ===")
print(f"  Players analysed:           {len(res)}")
print(f"  H window real (mean):       {res['H_window_real'].mean():.4f}")
print(f"  H window shuffle (mean):    {res['H_window_shuffle_mean'].mean():.4f}")
print(f"  Difference:                 {res['H_window_real'].mean() - res['H_window_shuffle_mean'].mean():+.4f}")
print(f"  Mean z:                     {res['z_window'].mean():+.4f}")
print(f"  z < -1.96:                  {(res['z_window'] < -1.96).sum()} "
      f"({100*(res['z_window'] < -1.96).sum()/len(res):.1f}%)")
print(f"  |z| < 1.96:                 {(res['z_window'].abs() < 1.96).sum()} "
      f"({100*(res['z_window'].abs() < 1.96).sum()/len(res):.1f}%)")
print(f"  p_emp < 0.05:               {(res['p_emp_lower'] < 0.05).sum()} "
      f"({100*(res['p_emp_lower'] < 0.05).sum()/len(res):.1f}%)")

res.to_csv("/Users/racelabs/Desktop/TFG/data/shuffle_within_game_25000.csv", index=False)
print("\nSaved: data/shuffle_within_game_25000.csv")
