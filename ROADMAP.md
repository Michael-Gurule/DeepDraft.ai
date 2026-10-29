# DeepDraft.ai Roadmap

**From a folder of reference scripts to a top-tier data science and machine learning toolkit**

This roadmap turns the findings of the [repository review](docs/review/REPOSITORY_REVIEW.md) into an ordered implementation plan. Each phase lists its goal, work packages, deliverables, and exit criteria. Work packages are sized S (a day or less), M (a few days), or L (a week or more of focused work) so they can become GitHub issues directly.

---

## 1. Vision

DeepDraft.ai becomes the place a data scientist or ML engineer opens when they need to **understand, implement, or benchmark** a model quickly, and trust what they find. It has three layers:

| Layer | What it holds | Promise to the reader |
|-------|---------------|-----------------------|
| **Learn** (core library) | Canonical, tested reference implementations with a consistent walkthrough | "This runs, it is correct, and it explains itself." |
| **Lab** (experimental) | Your experimental models and research prototypes | "This is new and promising; here is the evidence and its maturity level." |
| **Data and Tools** | Dataset registry, benchmark harness, CLI, shared utilities | "You can reproduce every number on this site with one command." |

### Guiding principles

1. **Every module runs in CI.** A module that does not run does not ship.
2. **Every number comes from held-out data.** No tuning on the test split, no in-sample metrics presented as evaluation.
3. **Every claim matches the code and cites a source.**
4. **Reproducible by default:** pinned environments, seeds, cached datasets with checksums.
5. **No large data in git.** Data lives in a registry with licenses and checksums.
6. **Clear licensing** for code, documentation, and each dataset.

---

## 2. Target repository layout

The current layout (one folder per category, one script per model) is a good starting point. The target layout keeps that browsing experience but makes the code importable, testable, and shareable.

```
DeepDraft.ai/
├── pyproject.toml              # package metadata, optional-dependency extras per category
├── README.md                   # catalog, quickstart, learning paths
├── LICENSE                     # code license
├── CONTRIBUTING.md  CODE_OF_CONDUCT.md  SECURITY.md  CITATION.cff  CHANGELOG.md
├── src/deepdraft/
│   ├── core/                   # seed_everything, devices, plotting, logging, config base classes
│   ├── evaluation/             # splits (holdout, CV, rolling-origin, grouped), metrics, bootstrap CIs
│   ├── datasets/               # registry, loaders, synthetic generators with ground truth
│   ├── models/
│   │   ├── classic/  ensemble/  time_series/  anomaly_detection/  bayesian/
│   │   ├── causal_inference/  interpretability/  quantitative/
│   │   ├── reinforcement_learning/  multi_agent/  scientific_ml/
│   │   └── deep_learning/  nlp/  vision/  generative/  graph/  recsys/  survival/   (new, Phase 4)
│   ├── experimental/           # the Lab (Phase 5)
│   └── cli.py                  # `deepdraft list | run | bench | data`
├── tutorials/                  # jupytext-paired notebooks, one per module
├── benchmarks/                 # benchmark configs and stored results
├── docs/                       # mkdocs-material site sources
├── tests/
│   ├── smoke/                  # every module's main(fast=True)
│   └── reference/              # comparisons against established libraries
└── data/                       # local cache only, git-ignored
```

Migration note: keep the existing top-level category folders working during the transition by moving files into `src/deepdraft/models/` and leaving thin shims (or simply updating links in the README) until v0.3.

---

## 3. Phase overview

| Phase | Theme | Depends on | Milestone |
|-------|-------|------------|-----------|
| 0 | Foundation and triage | none | v0.1 "It runs" |
| 1 | Correctness and evaluation integrity | 0 | v0.2 "It's right" |
| 2 | Walkthrough and documentation overhaul | 0 (can overlap 1) | v0.3 "It teaches" |
| 3 | Shared core library and consistent API | 0 | v0.3 |
| 4 | Coverage expansion | 1, 2, 3 | v0.4 to v0.6 |
| 5 | Experimental models (the Lab) | 3 | v0.4 |
| 6 | Dataset collection | 3 | v0.5 |
| 7 | Tools: benchmark harness, CLI, tracking | 3, 6 | v0.6 |
| 8 | Community, quality, and releases | 0, ongoing | v1.0 |

Phases 0 and 1 come first because the review found that 16 of 56 modules crash or hang under current library releases and a further set produce misleading numbers. Adding content on top of that foundation multiplies the maintenance cost.

---

## Phase 0: Foundation and triage

**Status:** complete.

**Goal:** anyone can install the project, run any module, and see it pass in CI.

| ID | Work package | Size | Details |
|----|--------------|------|---------|
| 0.1 | Choose and add a LICENSE | S | Without a license, nobody can legally reuse the code. MIT or Apache-2.0 suit a learning toolkit (Apache-2.0 adds an explicit patent grant). Use CC-BY-4.0 for prose docs if you want attribution on text. |
| 0.2 | Write the README | M | Purpose, catalog table (category, model, status, dependencies), quickstart, how a module is structured, link to the roadmap and module standard. |
| 0.3 | Add `pyproject.toml` | M | Python >= 3.10. Core deps (numpy, scipy, pandas, scikit-learn, matplotlib). Optional extras: `[boosting]` (xgboost, lightgbm), `[timeseries]` (statsmodels, prophet), `[deep]` (torch), `[rl]` (gymnasium), `[causal]` (networkx), `[quant]` (arch), `[bayes]` (hmmlearn), `[all]`. Lock with `uv` for reproducible CI. |
| 0.4 | Add `.gitignore` | S | `*.png` outputs, `data/`, `runs/`, `.venv/`, `__pycache__/`, notebook checkpoints. |
| 0.5 | Fix the failing modules | M | See review section 5 for the file, line, error, and fix for each. Group into one PR per category. |
| 0.6 | Remove global warning suppression | S | Delete `warnings.filterwarnings("ignore")` from 49 files. Replace with targeted filters only where a warning is understood and harmless. This is how you will see the next deprecation before it becomes a crash. |
| 0.7 | Headless plotting and output directory | M | Add `save_figure(fig, name)` that writes to `--out` (default `runs/<module>/`) and only calls `plt.show()` when interactive. Replace 180 `savefig` + `show` pairs. |
| 0.8 | Fast mode for every `main()` | M | `main(fast=True)` or `DEEPDRAFT_FAST=1` shrinks samples, epochs, and grid sizes so each module finishes in under 30 s on CPU. |
| 0.9 | Offline fallbacks | S | `naive_bayes.py`, `lime_explainer.py`, `shap_values.py`, and the MNIST demo download data at runtime. Fall back to bundled or synthetic data when offline. |
| 0.10 | Smoke tests and CI | M | `tests/smoke/test_modules.py` parametrized over every module, running `main(fast=True)`. GitHub Actions matrix: Ubuntu, Python 3.10 to 3.13. Add a **nightly job against the latest releases** of all dependencies; the review found that dependency drift caused most crashes. |
| 0.11 | Pre-commit hooks | S | ruff (lint and format), codespell, nbstripout, check-yaml, end-of-file fixer. |
| 0.12 | Remove private references | S | Replace the 22 "Hyperion" references in 18 files with generic examples. Remove the dangling `frontier/vae.py` reference (or create the module in Phase 4). |
| 0.13 | Fix the calculator sandbox | S | `multi_agent_systems/react_agent.py` uses `eval` with emptied builtins, which is escapable. Replace with an AST whitelist evaluator. Teaching code gets copied into real agents. |

**Exit criteria (v0.1):** `pip install -e ".[all]"` succeeds; `pytest tests/smoke` passes in CI in under 15 minutes; no module suppresses warnings globally; a LICENSE and README exist.

---

## Phase 1: Correctness and evaluation integrity

**Status:** in progress.

**Goal:** every implementation is correct and every reported number is honest.

| ID | Work package | Size | Details |
|----|--------------|------|---------|
| 1.1 | Shared evaluation utilities | M | `deepdraft.evaluation`: train/validation/test splitter, rolling-origin backtest for time series, grouped splits, bootstrap confidence intervals, standard metric bundles per task. |
| 1.2 | Remove test-set selection | M | Ten modules choose a hyperparameter, threshold, or stopping point on the test set (decision tree ccp_alpha, KNN k, AdaBoost n_estimators, LightGBM categorical demo, isolation forest and autoencoder thresholds, DBSCAN eps, one-class SVM nu/gamma, stacking ranking by test AUC, N-BEATS). Move each decision to validation data or CV. |
| 1.3 | Remove in-sample evaluation | M | Anomaly detection modules score the same normal points they trained on; TFT evaluates on the full series; N-BEATS reports validation metrics as final results. Add proper held-out splits. |
| 1.4 | Fix algorithm bugs | L | Heston analytic pricer (returns NaN), DirectLiNGAM (wrong causal order and transposed adjacency), causal-graph metrics, QMIX recurrent training, DML standard errors, GARCH Student-t VaR scaling, Monte Carlo antithetic standard error, Naive Bayes comparisons and label order, Holt-Winters intervals, ETS auto-selection, stacking meta-learners, XGBoost learning-curve labels. Full list in review section 6. |
| 1.5 | Reference-comparison tests | L | For each from-scratch implementation, a test that compares against an established library or a closed form: Heston vs. an independent pricer, Kalman filter vs. a reference library or exact posterior, DML vs. DoubleML, PC vs. causal-learn, LiNGAM vs. `lingam`, permutation SHAP vs. `shap` on a small model, MCMC recovering a known Gaussian posterior, GARCH recovering simulation parameters, Black-Scholes put-call parity. |
| 1.6 | RL correctness | M | Treat `terminated` and `truncated` separately in DQN, PPO, A2C, and QMIX (SAC and TD3 already do). Seed environments and torch. Report returns over at least 5 seeds with confidence intervals. |
| 1.7 | Reproducibility | S | One `seed_everything()` covering `random`, NumPy, and torch (only 3 of 15 PyTorch modules seed torch today). Standardize on `np.random.default_rng`. |

**Exit criteria (v0.2):** every from-scratch implementation has at least one reference-comparison test; a lint rule or review checklist item blocks test-set tuning; RL modules report multi-seed results.

---

## Phase 2: Walkthrough and documentation overhaul

**Goal:** every module teaches, in the same order and voice, with verified claims.

| ID | Work package | Size | Details |
|----|--------------|------|---------|
| 2.1 | Adopt the module standard | S | Ratify [docs/MODULE_STANDARD.md](docs/MODULE_STANDARD.md). Add a docs lint that checks for the required docstring sections. |
| 2.2 | Rewrite module docstrings | L | Category by category, in learning-path order: classic, ensemble, time series, anomaly detection, Bayesian, causal, interpretability, quantitative, RL, multi-agent, scientific ML. Fix the factual errors and promise-versus-code mismatches listed in review section 7. |
| 2.3 | Companion notebooks | L | One jupytext-paired notebook per module under `tutorials/`, with rendered outputs and an "Open in Colab" badge. The `.py` file stays the source of truth. |
| 2.4 | Documentation site | M | mkdocs-material with mkdocstrings. Auto-generated API pages, a searchable catalog with tags (task, data type, difficulty, dependencies), and deployment to GitHub Pages on merge. |
| 2.5 | "Choose a model" guides | M | One decision guide per task: tabular classification, tabular regression, forecasting, anomaly detection, causal effect estimation, sequential decision-making. Each links to the relevant modules and baselines. |
| 2.6 | Learning paths | M | Curated sequences such as "Tabular ML in 10 modules", "Forecasting from ARIMA to N-BEATS", "Causal inference from DML to discovery", "Deep RL from DQN to SAC". |
| 2.7 | Glossary and notation guide | S | Shared symbols (n, d, X, y, theta, sigma) and terms, linked from every module. |
| 2.8 | Expected-output blocks | M | Each module states what the reader should see in fast mode, verified by the smoke test where possible. |

**Exit criteria (v0.3):** the docs lint passes for every module; the docs site deploys; at least three learning paths are published.

---

## Phase 3: Shared core library and consistent API

**Goal:** stop copying helpers between files and give every model the same interface.

| ID | Work package | Size | Details |
|----|--------------|------|---------|
| 3.1 | `deepdraft.core` | M | `seed_everything`, `get_device`, `save_figure`, plotting style, structured logging, `BaseConfig` dataclass with YAML override. |
| 3.2 | Deduplicate helpers | M | The review counted 14 copies of `plot_confusion_matrix`, 17 of `evaluate`, 11 of `cross_validate`, 10 each of the data loaders, and 3 identical anomaly-data generators. Move shared logic into `core`, `evaluation`, and `datasets`. |
| 3.3 | Uniform module interface | M | Each module exposes `Config`, `build_model(config)`, `train(...)`, `evaluate(...)`, and `main(fast=False)`. |
| 3.4 | scikit-learn-compatible estimators | M | Wrap from-scratch implementations (PC algorithm, LiNGAM, meta-learners, DML, permutation SHAP, LIME, genetic programming) as `BaseEstimator` subclasses so they work with Pipelines and model selection. |
| 3.5 | Typed configs | M | Replace the 516 `# Placeholder` comments with dataclass fields whose docstrings carry the hyperparameter table. |
| 3.6 | Type checking | S | Add mypy or pyright in CI for `core`, `evaluation`, and `datasets` first. |

**Exit criteria:** no duplicated helper remains in model files; every module follows the uniform interface.

---

## Phase 4: Coverage expansion

**Goal:** close the gaps a practitioner would notice first, in priority order. P1 items fill holes in the core learning paths; P2 items broaden coverage; P3 items are specialist content.

| Area | Present today | Add (priority) |
|------|---------------|----------------|
| **Foundations** | none | P1: gradient descent and optimizers from scratch, bias-variance and learning curves, cross-validation strategies, metrics and calibration, imbalanced learning, feature engineering and missing data, conformal prediction |
| **Deep learning fundamentals** | MLPs inside other modules | P1: MLP with autograd from scratch, CNN, RNN/LSTM/GRU, Transformer and attention from scratch, normalization and regularization, training diagnostics |
| **NLP and LLMs** | ReAct with a mock LLM | P1: tokenization, embeddings, transformer text classification, parameter-efficient fine-tuning (LoRA), retrieval-augmented generation, LLM evaluation, agents with native tool calling. P2: sequence-to-sequence, prompt optimization |
| **Computer vision** | Conv autoencoder demo | P2: ResNet, transfer learning, ViT, detection and segmentation primers |
| **Generative models** | none (VAE referenced but missing) | P1: VAE. P2: GAN, normalizing flows, diffusion (DDPM) |
| **Classic ML** | 9 modules | P2: GLMs (Poisson, Gamma, Tweedie), quantile regression, LDA/QDA classifiers, hierarchical, spectral, and HDBSCAN clustering, Ridge/Lasso regularization paths |
| **Time series** | ARIMA, ETS, Prophet, N-BEATS, TFT | P1: naive and seasonal-naive baselines, rolling-origin backtesting, MASE. P2: DeepAR, PatchTST, hierarchical reconciliation, intermittent demand (Croston). P3: forecasting foundation models as optional extras |
| **Causal inference** | DML, meta-learners, PC, LiNGAM | P1: IPW and AIPW, propensity matching, difference-in-differences, instrumental variables. P2: synthetic control, regression discontinuity, causal forest via EconML, uplift evaluation on randomized data |
| **Bayesian** | 7 modules | P2: probabilistic programming (PyMC or NumPyro hierarchical models), variational inference, Bayesian neural networks |
| **Reinforcement learning** | DQN, A2C, PPO, SAC, TD3 | P1: tabular methods (value and policy iteration, Q-learning, SARSA), REINFORCE, multi-armed bandits (UCB, Thompson sampling). P2: model-based RL, offline RL (CQL) |
| **Multi-agent** | QMIX, ReAct | P2: VDN, MAPPO and IPPO, PettingZoo environments. Move ReAct to an `agents/` or `nlp/` category |
| **Interpretability** | SHAP, LIME, IG | P1: permutation importance, PDP and ICE, ALE. P2: counterfactual explanations, Anchors, fairness metrics |
| **Anomaly detection** | IF, OCSVM, DBSCAN, AE | P2: LOF, ECOD and COPOD, time-series anomaly detection, a shared benchmark on ODDS or ADBench datasets |
| **Quantitative finance** | 5 modules | P2: Black-Litterman, hierarchical risk parity, factor models, backtesting with transaction costs, walk-forward evaluation |
| **Graph ML** | none | P2: GCN, GAT, GraphSAGE, node2vec |
| **Recommender systems** | none | P2: matrix factorization, ALS, two-tower retrieval, implicit-feedback evaluation |
| **Survival analysis** | none | P3: Kaplan-Meier, Cox proportional hazards, random survival forests |
| **Production ML** | none | P2: experiment tracking, model serialization (ONNX), drift detection, data validation, a minimal serving example |

**Process for each new module:** open an issue from a "new module" template, follow the module standard, add smoke and reference tests, add a notebook, and register it in the catalog.

**Exit criteria (per release):** each release from v0.4 to v0.6 closes all P1 items in at least three areas.

---

## Phase 5: Experimental models (the Lab)

**Goal:** a clearly separated home for your experimental models, with enough structure that readers can judge maturity and you can promote winners into the core library.

### Structure

```
src/deepdraft/experimental/<model_name>/
├── README.md        # model card: motivation, architecture diagram, status, results, limitations, citation
├── model.py         # the model
├── train.py         # training entry point
├── config.yaml      # default config, plus a `smoke` profile for CI
├── results/         # metrics JSON per benchmark run (committed); large artifacts stored elsewhere
└── tests/           # smoke test plus any unit tests
```

### Maturity levels

| Level | Badge | Requirements |
|-------|-------|--------------|
| Idea | `idea` | Model card describes the hypothesis; code may be incomplete. |
| Prototype | `prototype` | Runs end-to-end on a smoke config in CI. |
| Validated | `validated` | Results on at least one public benchmark from the dataset registry, compared against a core-library baseline over multiple seeds, with confidence intervals. |
| Graduated | `graduated` | Meets the module standard and moves into `deepdraft.models`. |

### Work packages

| ID | Work package | Size |
|----|--------------|------|
| 5.1 | Directory structure, model card template, and maturity badges | S |
| 5.2 | `deepdraft new-experiment <name>` scaffolding command | M |
| 5.3 | Separate optional-dependency group `[lab]` so experimental dependencies never burden core users | S |
| 5.4 | Results schema (JSON) and an auto-generated Lab leaderboard page in the docs | M |
| 5.5 | Graduation checklist and review process | S |

**Exit criteria (v0.4):** at least one of your experimental models reaches `prototype` with a model card and passes its smoke test in CI.

---

## Phase 6: Dataset collection

**Goal:** a curated, licensed, reproducible collection of datasets, from tiny teaching sets to large benchmarks, without bloating the git repository.

### Design

| Concern | Decision |
|---------|----------|
| Storage | Do not commit data to git. Host public datasets you are allowed to redistribute on the Hugging Face Hub (free hosting, versioning, streaming) or Zenodo (DOI for citation). Use DVC or git-lfs only for small curated files. |
| Download and cache | Use `pooch` (or `huggingface_hub`) to download on first use, cache under `~/.cache/deepdraft`, and verify SHA-256 checksums. |
| Registry | One entry per dataset: name, description, task, size, license, source URL, checksum, citation, schema, canonical splits, loader function. |
| Documentation | A dataset card per dataset following "Datasheets for Datasets" (Gebru et al., 2021): provenance, collection process, license, personal data, known biases, intended uses. |
| Formats | Parquet and Arrow for tabular data; streaming support for anything larger than memory; DuckDB or Polars examples for out-of-core work. |
| Licensing | Check redistribution rights for every dataset. When you cannot redistribute, ship only the loader and instructions. |
| Synthetic data | Consolidate the repository's synthetic generators (CATE, causal DAGs, GARCH, regime-switching, anomaly injection, PDE solutions) into `deepdraft.datasets.synthetic`. Known ground truth is a teaching asset most libraries lack. |
| CI | Every dataset has a small sample (or synthetic stand-in) so tests never hit the network. |

### Work packages

| ID | Work package | Size |
|----|--------------|------|
| 6.1 | Registry schema, `load(name)` API, cache, checksums | M |
| 6.2 | Dataset card template and contribution guide | S |
| 6.3 | Starter collection: the public datasets the modules already use (20 Newsgroups, California housing, digits, breast cancer, AirPassengers, MNIST) | M |
| 6.4 | Benchmark suites per task with canonical splits (tabular, forecasting, anomaly, causal with known effects) | L |
| 6.5 | Large-dataset support: streaming loaders, Parquet sharding, out-of-core examples | L |
| 6.6 | Your large datasets: upload to the chosen host, write cards, register | L |

**Exit criteria (v0.5):** every module loads data through the registry; each dataset has a card, a license entry, and a checksum.

---

## Phase 7: Tools

**Goal:** turn the library into a workbench.

| ID | Work package | Size | Details |
|----|--------------|------|---------|
| 7.1 | CLI | M | `deepdraft list`, `deepdraft run <category>.<module> [--fast] [--out DIR]`, `deepdraft data list|download|info`, `deepdraft bench <suite>`. |
| 7.2 | Benchmark harness | L | Run models against registered datasets with fixed splits and seeds; store results as JSON; publish leaderboards in the docs. Shared by the core library and the Lab. |
| 7.3 | Hyperparameter search wrapper | M | A thin Optuna integration that respects the validation-split rules from Phase 1. |
| 7.4 | Experiment tracking | S | Optional MLflow or Weights & Biases logging behind a flag. |
| 7.5 | Model card generator | S | Produce a model card from a config and a results JSON. |
| 7.6 | Dev environment | S | Devcontainer and Docker image with CPU and CUDA variants. |
| 7.7 | Interactive playground (optional) | M | A Streamlit or Gradio app to explore selected models (decision boundaries, forecasting horizons, option Greeks). |

**Exit criteria (v0.6):** a reader can reproduce any leaderboard number with one CLI command.

---

## Phase 8: Community, quality, and releases

**Goal:** make it easy and safe for others to contribute, and for anyone to cite and depend on the project.

| ID | Work package | Size |
|----|--------------|------|
| 8.1 | CONTRIBUTING guide (how to add a module, the module standard, testing, style) | S |
| 8.2 | Issue templates (bug, new module, dataset proposal, experimental model) and a PR template built from the module standard checklist | S |
| 8.3 | Conventional Commits and an automated CHANGELOG (current history mixes `feat:`, `feature:`, `Feature:`, `add:`, `build:`) | S |
| 8.4 | Semantic versioning, release automation, PyPI publishing (check name availability early) | M |
| 8.5 | CITATION.cff, badges (CI, coverage, docs, license, PyPI) | S |
| 8.6 | CODEOWNERS and review policy; label "good first issue" items drawn from this roadmap | S |
| 8.7 | SECURITY.md and a dependency update bot (Dependabot or Renovate) | S |

**Exit criteria (v1.0):** stable public API for `core`, `evaluation`, and `datasets`; documented deprecation policy; at least one external contributor merged.

---

## 4. Milestones

| Version | Name | Contents |
|---------|------|----------|
| v0.1 | It runs | Phase 0 complete |
| v0.2 | It's right | Phase 1 complete |
| v0.3 | It teaches | Phases 2 and 3 complete for existing modules |
| v0.4 | The Lab opens | Phase 5, plus Phase 4 P1 for foundations, deep learning, and generative |
| v0.5 | Data | Phase 6, plus Phase 4 P1 for NLP, time series, and causal |
| v0.6 | Workbench | Phase 7, plus remaining Phase 4 P1 items |
| v1.0 | Stable | Phase 8, stable APIs, documentation complete |

---

## 5. First ten actions

These are the highest-leverage steps, in order. Most take an afternoon.

1. Add a LICENSE.
2. Add `pyproject.toml` with optional extras and a `.gitignore`.
3. Write a minimal README with the module catalog.
4. Fix the nine one-line API-drift crashes (review section 5, group A).
5. Delete global `warnings.filterwarnings("ignore")` calls.
6. Add `main(fast=True)` and the smoke-test workflow with a nightly latest-dependencies job.
7. Replace the calculator `eval` in `react_agent.py`.
8. Remove the "Hyperion" references and the dangling `frontier/vae.py` link.
9. Fix the Heston analytic pricer and add a reference-value test.
10. Open GitHub issues for every item in review sections 6 and 7, labeled by phase and size.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Scope creep: Phase 4 is large | Gate each release on P1 items only; P2 and P3 items wait for contributors. |
| Dependency drift breaks modules again | Nightly CI against the latest releases; Dependabot; optional extras so heavy dependencies stay isolated. |
| Experimental code lowers trust in the core library | Strict separation in `experimental/`, maturity badges, and a graduation checklist. |
| Dataset licensing problems | License field required in the registry; loader-only entries for non-redistributable data. |
| Docs drift from code | Docs lint, expected-output checks in the smoke test, and doc pages generated from docstrings. |
| Solo-maintainer bandwidth | Prioritize Phases 0 and 1, automate everything possible, and convert review findings into well-scoped "good first issue" tickets. |
