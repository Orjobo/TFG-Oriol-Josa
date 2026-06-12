# TFG — Statistical-mechanical information measures in human decision-making

Analysis code for my undergraduate physics thesis (TFG, Universitat de Barcelona).
The project applies information-theoretic measures (Shannon entropy, mutual
information, a neural estimator, a within-subject shuffle null, and a
compositional decomposition over candidate strategies) to two laboratory
experiments on human decision-making, and reads each measure against its
counterpart in equilibrium statistical mechanics.

The two datasets:

- **Mr. Banks** — a sequential financial game where each subject guesses round
  by round whether a market index moves up or down (Gutiérrez-Roig et al., 2016).
- **Phenotypes** — a battery of four 2x2 dyadic games over a grid of payoffs
  (Poncela-Casasnovas et al., 2016).

## Layout

- `data/` — the processed datasets (raw rounds plus the intermediate tables the
  scripts produce: weights, shuffle nulls, MINE estimates, clustering results).
- `parse_mrbanks.py` — build the working Mr. Banks table from the raw rounds.

Mr. Banks analysis:
- `conditional_entropy_mi_v2.py` — plug-in mutual information of each strategy's cue.
- `shuffle_within_game.py`, `shuffle_null_model.py`, `shuffle_null_model_highN.py` — within-subject shuffle nulls.
- `window_analysis.py` — sliding-window entropy across round scales.
- `lstm_prediction.py`, `lstm_configs.py` — LSTM next-decision baseline.
- `composition_model_imitative.py` — compositional decomposition over {Random, MI, WSLS} and the MI/WSLS degeneracy.
- `entropy_analysis.py`, `entropy_by_strategy.py` — entropy summaries.

Phenotypes analysis:
- `clustering_analysis.py` — K-means assignment of behavioural phenotypes.
- `phenotypes_mine.py`, `mine_analysis.py` — neural estimation of mutual information (MINE).
- `phenotypes_shuffle_null_b.py` — shuffle null on the rule-match substrate.
- `phenotypes_composition_model.py`, `phenotypes_conditional_entropy.py`, `phenotypes_rule_conditional_entropy.py` — composition and conditional-entropy variants.

Figures:
- `phenotypes_report_figures.py` — the Phenotypes figures.
- `mrbanks_signatures_and_degeneracy.py` — the Mr. Banks degeneracy figure.

## Running

Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and
`torch` (for MINE and the LSTM).

Note: the input/output paths are currently hardcoded at the top of each script
(`DATA = ...`, `OUT = ...`). Point them at this `data/` folder and at a folder
of your choice for the figures before running.

## Author

Oriol Josa Bofarull — Facultat de Física, Universitat de Barcelona.
