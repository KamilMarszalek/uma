# Tournament-based Random Forest

A Python implementation of a **modified Random Forest** for **classification**, where the split (feature test) at each node is chosen using a **tournament selection** mechanism instead of always taking the globally best split.

In a 2-way tournament, we randomly sample **two candidate tests**, evaluate their quality (e.g., Information Gain / Gini Gain), and apply the better one.  
This injects *controlled randomness* into tree growth: more informed than purely random splits, but less greedy than classic decision trees.

---

## Key idea: tournament selection for node splits

Classic trees (and many RF variants) pick the **best** split among a candidate set.  
Here, we pick the split via a tournament:

1. Sample candidate tests (how we sample depends on the tree type).
2. Draw two candidates.
3. Evaluate both with a gain function.
4. Choose the better one.

Why do this?
- potentially **reduces overfitting** of individual trees (less greediness),
- can **increase diversity** in the forest (lower correlation between trees),
- may keep accuracy competitive with standard Random Forest.

> ⚠️ Empirically (see the report), this modification is **not universally better** — it can help on some datasets and hurt on others, so it’s best treated as a tunable variant.

---

## Implemented tree types

This project contains two tree families:

### 1) ID3 (categorical features)
- Designed for **discrete/categorical** attributes.
- Split quality options:
  - **Information Gain** (default baseline)
  - **Gain Ratio**
  - (optionally) **Gini Gain**

ID3 splits a node into multiple branches (one per attribute value).

### 2) CART (continuous + categorical)
- Supports **continuous** features using thresholds `x[f] <= θ`.
- Uses **Gini Gain** for split evaluation.
- Categorical attributes are handled via **one-hot encoding**.

---

## Random Forest construction

The forest follows the standard Random Forest recipe:
- **Bootstrap sampling** of training examples per tree (`sample_ratio`)
- Random subset of features per tree (classic RF behavior)
- Final prediction by **majority voting**

Where it differs:
- During node splitting, feature tests are selected using **tournament sampling**, where sampling may be **with replacement** (duplicates can occur), unlike scikit-learn’s typical “without replacement” feature subset in a node.

---

## Quality measures (metrics)

The experiment runner reports (per configuration):
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- Training time statistics

For stochastic stability, results in the report are aggregated across **25 independent runs** (with controlled seeding).

---

## Datasets used in experiments

Experiments were performed on several UCI datasets:

- **Mushroom** (UCI id 73): binary, categorical, easy (~100% expected).
- **Breast Cancer (recurrence)** (UCI id 14): small dataset, good for overfitting behavior.
- **Bank Marketing** (UCI id 222): imbalanced, harder minority detection.
- **Default of Credit Card Clients** (UCI id 350): many numeric attributes, good for CART thresholds.

Notes:
- ID3 was evaluated only on datasets with fully (or mostly) categorical attributes to avoid discretization artifacts.
- CART was evaluated on all datasets.

---

## Experimental protocol

- Train/test split: **70% / 30%**, **stratified**
- Each experiment repeated **25 times** with deterministic seeding
- One hyperparameter changed at a time (to keep plots readable)
- Inputs for experiments are defined via CSV files in `experiment_input/`
- Outputs (CSV logs) are stored in `experiment_output/`
- Plots are saved in `plots/`

---

## Main hyperparameters (as used in the report)

Default settings used as a baseline:

- `n_trees`: **100**
- `sample_ratio`: **1.0** (100%)
- `max_depth`: **None** (unlimited)
- `tournament_size`: **2** (later chosen **3** as a precision/recall trade-off in the report)
- CART-specific: `min_samples_split`: **2**
- Evaluation:
  - ID3: **Information Gain** (or Gain Ratio in the final comparison)
  - CART: **Gini Gain**

---

## Summary of findings (from the final report)

- The tournament approach **does not guarantee improvement**.
- It can help by changing the model’s operating point:
  - often **increases recall** for minority classes **at the cost of precision** when tournament size grows,
  - may be useful in domains where missing positives is expensive (e.g., medical screening).
- Compared to scikit-learn RandomForest (CART-based):
  - On **Breast Cancer (id 14)** the modified approach was competitive and sometimes slightly better in accuracy; F1 was best for sklearn / close for ID3 depending on metric trade-offs.
  - On **Bank Marketing (id 222)** sklearn performed clearly better overall (especially recall/F1).
  - On **Credit Default (id 350)** results were very close — no clear winner.

---

## Repository layout

```
.
├── src/                    # core implementation (trees, forest, data utilities)
│   ├── tree/               # ID3, CART, split evaluation, node definitions
│   ├── forest/             # Random Forest wrapper + config
│   └── data/               # UCI provider, encoding, split utilities
├── experiments/            # experiment runner, parsing configs, plotting, tables
├── experiment_input/       # CSV definitions of hyperparameter sweeps
├── experiment_output/      # CSV outputs from runs
├── plots/                  # generated plots for the report
├── tables/                 # LaTeX tables exported from results
├── tests/                  # pytest tests for core logic
├── dokumentacja.tex/pdf    # Polish report (source + compiled)
└── pyproject.toml / requirements.txt / uv.lock
```

---

## Getting started

### 1) Install dependencies

If you use `uv`:
```bash
uv sync
```

Or with pip:
```bash
pip install -r requirements.txt
```

### 2) Run tests
```bash
pytest -q
```

### 3) Define experiments in a CSV file

The experiment runner expects a CSV with a header like:

```csv
experiment,forest_type,eval_function,num_trees,sample_ratio,feature_ratio,tree_max_depth,tree_tournament_size,min_samples_split,set_id,train_size,base_random_seed,categorial_encoding,times_repeat
```

Example (model comparison on dataset `14`):

```csv
experiment,forest_type,eval_function,num_trees,sample_ratio,feature_ratio,tree_max_depth,tree_tournament_size,min_samples_split,set_id,train_size,base_random_seed,categorial_encoding,times_repeat
Compare_on_14,ID3,ID3_GAIN_RATIO,100,1.0,1.0,-1,3,2,14,0.7,42,CATEGORICAL,25
Compare_on_14,CART,CART_GINI_GAIN,100,1.0,1.0,-1,3,2,14,0.7,42,ONE_HOT,25
Compare_on_14,SKLEARN,SKLEARN_GINI,100,1.0,1.0,-1,3,2,14,0.7,42,ONE_HOT,25
```

In this repository, ready-to-run CSV definitions live in `experiment_input/`.

### 4) Run experiments (CSV → CSV results)

The main entrypoint takes the path to a CSV file and saves results to CSV outputs:

```bash
uv run python -m src.main experiment_input/compare_on_14.csv
```

To run a batch of CSVs from `experiment_input/`, use:

```bash
cd experiments
bash run_experiment.sh 0
```

`run_experiment.sh` iterates over files in `experiment_input/` and runs:
```bash
uv run python -m src.main "<csv-file>"
```

The script accepts `0` or `1` as an argument and then processes **every second file** (even/odd indices).  
This is handy if you want to split the workload across two terminals:

- Terminal A: `bash run_experiment.sh 0`
- Terminal B: `bash run_experiment.sh 1`

### 5) Generate plots from outputs

`plotter.py` reads experiment output CSVs (by default from `experiment_output/`) and generates charts.

Typical usage:

```bash
uv run python -m experiments.experiment.plotter --input-dir experiment_output --output-dir plots
```

- If the CSV filename starts with `Compare_on_...`, it creates comparison bar plots.
- Otherwise, it creates metric-vs-parameter plots (including an aggregated multi-metric plot with optional training-time axis).

### 6) Generate LaTeX tables

`table_builder.py` reads output CSVs and exports LaTeX tables (accuracy + time + precision/recall/F1) with the best rows highlighted.

Typical usage:

```bash
uv run python -m experiments.experiment.table_builder --input-dir experiment_output --output-dir tables
```

---

## Notes & limitations


- This project is primarily **research/educational** (university assignment).
- The tournament mechanism is a **knob**, not a guaranteed upgrade:
  - benefits depend heavily on dataset size, noise, and class imbalance.
- CART uses one-hot encoding for categoricals, which can inflate dimensionality and affect depth/compute.
