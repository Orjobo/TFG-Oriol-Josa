"""
Shuffle null model for binarization B (D^phi* = match-with-rule).

The earlier shuffle operated on the raw action (binarization A), which was
the wrong target: shuffling the action directly destroys the rule match by
construction. Here we shuffle consistently on the "does it follow the
phenotype's rule or not" sequence instead.

Method:
  For each subject with a phenotype phi* assigned by K-means:
    - Real: D^phi*_n = 1[action_n == phi*(T_n, S_n)]
            gives match_rate, H(D^phi*), I(D^phi*; g), I(D^phi*; (T,S)_quad)
    - Shuffled: permute action_n while keeping (T_n, S_n) fixed,
                recompute D^phi*_n and the same metrics.
    - The real minus shuffled difference is the structure that is genuinely
      consistent with the rule, above what chance alignment produces.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/racelabs/Documents/TFG/data/drbrain")
R_PAYOFF, P_PAYOFF = 10, 5


def prescribe(rule, T, S):
    if rule == "Trustful":
        return np.full(len(T), "C")
    if rule == "Optimist":
        return np.where(T < R_PAYOFF, "C", "D")
    if rule == "Pessimist":
        return np.where(S > P_PAYOFF, "C", "D")
    if rule == "Envious":
        return np.where(S >= T, "C", "D")
    raise ValueError(rule)


def H_MM(values):
    n = len(values)
    if n <= 1:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    H = float(-np.sum(p * np.log2(p)))
    K = len(np.unique(values))
    return H + (K - 1) / (2 * n * np.log(2))


def H_cond(D, X):
    if len(D) == 0:
        return 0.0
    H = 0.0
    n = len(D)
    for x in np.unique(X):
        mask = X == x
        nx = mask.sum()
        if nx == 0:
            continue
        H += (nx / n) * H_MM(D[mask])
    return H


def MI(D, X):
    return H_MM(D) - H_cond(D, X)


def best_fit_rule(actions, T, S):
    rates = {}
    for r in ["Trustful", "Optimist", "Pessimist", "Envious"]:
        rates[r] = (actions == prescribe(r, T, S)).mean()
    return max(rates, key=rates.get)


def metrics_for(D_phi, g, TS_quad):
    return {
        "match_rate": float(np.mean(D_phi)),
        "H_Dphi": H_MM(D_phi),
        "I_Dphi;g": MI(D_phi, g),
        "I_Dphi;TS_quad": MI(D_phi, TS_quad),
    }


def main(n_iter=20, seed=42):
    decisions = pd.read_csv(DATA_DIR / "drbrain_decisions.csv")
    decisions = decisions[decisions["Action"].isin(["C", "D"])].copy()
    km = pd.read_csv(DATA_DIR / "phenotype_kmeans_assignments.csv")
    km_assign = dict(zip(km["User_ID"], km["phenotype_kmeans"]))

    rng = np.random.default_rng(seed)
    rows = []

    for uid, sub in decisions.groupby("User_ID"):
        if len(sub) < 5 or uid not in km_assign:
            continue
        T = sub["T"].astype(int).to_numpy()
        S = sub["S"].astype(int).to_numpy()
        actions = sub["Action"].to_numpy()
        g = sub["Game"].to_numpy()
        TS_quad = np.where(T < R_PAYOFF, "lo", "hi") + "_" + np.where(S < P_PAYOFF, "lo", "hi")

        # Pick phi*: if K-means left the subject Undefined, fall back to the
        # rule that best fits their actions so they still get a reference rule.
        km_pheno = km_assign[uid]
        phi_star = best_fit_rule(actions, T, S) if km_pheno == "Undefined" else km_pheno
        prescribed = prescribe(phi_star, T, S)

        # Real
        D_phi_real = (actions == prescribed).astype(int)
        m_real = metrics_for(D_phi_real, g, TS_quad)

        # Shuffled (average over n_iter permutations)
        m_shuffled_avg = {k: 0.0 for k in m_real}
        for _ in range(n_iter):
            actions_sh = rng.permutation(actions)
            D_phi_sh = (actions_sh == prescribed).astype(int)
            m_sh = metrics_for(D_phi_sh, g, TS_quad)
            for k in m_sh:
                m_shuffled_avg[k] += m_sh[k] / n_iter

        rec = {"User_ID": uid, "phenotype_kmeans": km_pheno, "phi_star": phi_star, "n": len(actions)}
        for k in m_real:
            rec[f"{k}_real"] = m_real[k]
            rec[f"{k}_shuffled"] = m_shuffled_avg[k]
            rec[f"{k}_delta"] = m_real[k] - m_shuffled_avg[k]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "shuffle_null_b.csv", index=False)

    # Summary
    print(f"=== Shuffle null model on D^phi* (n={len(df)} subjects, {n_iter} shuffle iters) ===\n")
    cols = ["match_rate_real", "match_rate_shuffled", "match_rate_delta",
            "H_Dphi_real", "H_Dphi_shuffled",
            "I_Dphi;g_real", "I_Dphi;g_shuffled",
            "I_Dphi;TS_quad_real", "I_Dphi;TS_quad_shuffled"]
    print(df[cols].mean().round(4).to_string())

    print("\n=== Fraction of subjects with real match_rate > shuffled ===")
    above = (df["match_rate_real"] > df["match_rate_shuffled"]).mean()
    print(f"  {above:.1%}")

    print("\n=== Per phenotype ===")
    g_cols = ["match_rate_real", "match_rate_shuffled", "match_rate_delta"]
    print(df.groupby("phenotype_kmeans")[g_cols].mean().round(3).to_string())

    print(f"\n[save] {DATA_DIR / 'shuffle_null_b.csv'}")


if __name__ == "__main__":
    main()
