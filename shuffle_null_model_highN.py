"""
TFG - Shuffle null model with high N (25k and 50k).
Reran with 25,000 and 50,000 shuffles to check that the original result (1,000
shuffles) is robust to the number of resamples.

Optimization: for binary sequences the windowed entropy with w=5 depends only on
the count of 1s inside each window (0-5), so we precompute the entropy table and
use a cumsum to get each shuffle in O(n).
"""

import pandas as pd
import numpy as np
import time
from collections import Counter

df = pd.read_csv("/Users/racelabs/Desktop/TFG/data/rounds.csv")
df = df[df["decision"] != 0].copy()
df = df[~df["scenario"].str.startswith("4")].copy()
df = df.sort_values(["user", "game", "round"]).reset_index(drop=True)
df["dec_bin"] = (df["decision"] == 1).astype(np.int8)

print(f"Decisions: {len(df)}, Users: {df['user'].nunique()}")

WINDOW = 5

# Entropy table indexed by the number of 1s inside a window of 5
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


def shannon_global(seq):
    n = len(seq)
    if n == 0:
        return np.nan
    c1 = int(seq.sum())
    c0 = n - c1
    probs = np.array([c0, c1]) / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def run_analysis(N_SHUFFLES, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    results = []
    t0 = time.time()
    for user_id, group in df.groupby("user"):
        decs = group["dec_bin"].values
        n = len(decs)
        if n < 10:
            continue
        H_real = shannon_global(decs)
        H_win_real = sliding_entropy_fast(decs)

        H_win_shuffles = np.empty(N_SHUFFLES)
        buf = decs.copy()
        for i in range(N_SHUFFLES):
            rng.shuffle(buf)
            H_win_shuffles[i] = sliding_entropy_fast(buf)

        mu = H_win_shuffles.mean()
        sigma = H_win_shuffles.std()
        z = (H_win_real - mu) / sigma if sigma > 0 else 0.0

        # Empirical lower-tail p-value (fraction of shuffles at or below the real value)
        n_ge = (H_win_shuffles <= H_win_real).sum()
        p_empirical = (n_ge + 1) / (N_SHUFFLES + 1)

        results.append({
            "user": user_id,
            "n_decisions": n,
            "H_real": H_real,
            "H_window_real": H_win_real,
            "H_window_shuffle_mean": mu,
            "H_window_shuffle_std": sigma,
            "z_window": z,
            "p_emp_lower": p_empirical,
        })
    elapsed = time.time() - t0
    return pd.DataFrame(results), elapsed


def summarize(name, res):
    n_total = len(res)
    mean_Hreal = res["H_window_real"].mean()
    mean_Hshuf = res["H_window_shuffle_mean"].mean()
    diff = mean_Hreal - mean_Hshuf
    n_struct = (res["z_window"] < -1.96).sum()
    n_random = (res["z_window"].abs() < 1.96).sum()
    n_p_sig = (res["p_emp_lower"] < 0.05).sum()
    n_p_001 = (res["p_emp_lower"] < 0.01).sum()
    print(f"\n=== {name} ===")
    print(f"  Players analyzed:              {n_total}")
    print(f"  Window H, real (mean):         {mean_Hreal:.4f}")
    print(f"  Window H, shuffle (mean):      {mean_Hshuf:.4f}")
    print(f"  Difference:                    {diff:+.4f} bits")
    print(f"  Mean z:                        {res['z_window'].mean():+.4f}")
    print(f"  Players z < -1.96:             {n_struct} ({100*n_struct/n_total:.1f}%)")
    print(f"  Players |z| < 1.96:            {n_random} ({100*n_random/n_total:.1f}%)")
    print(f"  Players p_emp < 0.05:          {n_p_sig} ({100*n_p_sig/n_total:.1f}%)")
    print(f"  Players p_emp < 0.01:          {n_p_001} ({100*n_p_001/n_total:.1f}%)")
    return {
        "n_total": n_total,
        "H_real_mean": mean_Hreal,
        "H_shuffle_mean": mean_Hshuf,
        "diff": diff,
        "z_mean": res["z_window"].mean(),
        "n_structural_z": int(n_struct),
        "n_random_z": int(n_random),
        "n_p05": int(n_p_sig),
        "n_p01": int(n_p_001),
    }


summary = {}
for N in (25_000, 50_000):
    print(f"\n--- Running N_SHUFFLES = {N} ---")
    res, elapsed = run_analysis(N, rng_seed=42)
    print(f"  (time: {elapsed:.1f} s)")
    summary[N] = summarize(f"N = {N}", res)
    res.to_csv(f"/Users/racelabs/Desktop/TFG/data/shuffle_results_{N}.csv", index=False)

print("\n\n========== COMPARISON ==========")
print(f"{'metric':30s}  {'25k':>12s}  {'50k':>12s}  {'Δ':>12s}")
for key in ["H_real_mean", "H_shuffle_mean", "diff", "z_mean",
            "n_structural_z", "n_random_z", "n_p05", "n_p01"]:
    a = summary[25_000][key]
    b = summary[50_000][key]
    delta = b - a
    if isinstance(a, float):
        print(f"{key:30s}  {a:12.4f}  {b:12.4f}  {delta:+12.4f}")
    else:
        print(f"{key:30s}  {a:12d}  {b:12d}  {delta:+12d}")
