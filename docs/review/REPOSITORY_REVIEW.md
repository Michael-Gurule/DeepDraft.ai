# DeepDraft.ai Repository Review

**Scope:** all 56 modules (27,716 lines) at commit `11f4472`
**Companion documents:** [ROADMAP.md](../../ROADMAP.md) (implementation plan) and [MODULE_STANDARD.md](../MODULE_STANDARD.md) (walkthrough template)

---

## 1. Executive summary

DeepDraft.ai already has the hardest part of a reference toolkit: breadth. Twelve categories cover classic ML, ensembles, forecasting, anomaly detection, Bayesian methods, causal inference, interpretability, quantitative finance, reinforcement learning, multi-agent systems, and scientific ML. Many modules implement algorithms from scratch (MCMC samplers, Kalman/EKF/UKF filters, particle filters, the PC algorithm, double machine learning, N-BEATS and N-HiTS, Fourier neural operators, PPO, SAC, TD3, QMIX, genetic-programming symbolic regression). The recurring "Use Cases" and "Fine-Tuning Recommendations" sections are a distinctive asset that most libraries lack.

The review found five problems that block the repository from being a trustworthy reference today:

1. **A third of the modules do not run.** 16 of 56 modules crash or hang under current library releases (scikit-learn 1.9, NumPy 2.5, matplotlib 3.11, xgboost 3.4). Most failures come from API removals that a global `warnings.filterwarnings("ignore")` hid until they became errors. One more module (Heston) runs but prints `nan` for every price.
2. **There is no project infrastructure.** The repository has no README, LICENSE, dependency file, tests, CI, or package structure. Without a license, nobody can legally reuse the code, which defeats the open-source goal.
3. **Evaluation often leaks.** At least ten modules pick a hyperparameter, threshold, or stopping point on the test set. The anomaly-detection modules, TFT, and N-BEATS report metrics on data the model trained on or tuned against. For a teaching repository, this is the most damaging category of issue, because readers copy the pattern.
4. **Several algorithms are wrong.** The Heston analytic pricer returns NaN; DirectLiNGAM recovers causal orders backwards; QMIX threads its recurrent state across agents instead of time; the DML standard error is not the robust estimator its comment claims; GARCH Student-t VaR uses the wrong quantile scale; three RL modules treat time-limit truncation as termination.
5. **The text promises more than the code delivers, and sometimes says the wrong thing.** Nine modules describe methods they never implement (Causal Forest, NUTS, Gibbs sampling, GES, A3C, the adjoint method, neural-guided symbolic regression, TFT static and known-future inputs, Periodic GP kernels). The review lists more than 40 factual errors, and 22 references to a private project ("Hyperion") will confuse outside readers.

None of these problems is hard to fix individually. Together they mean a reader cannot currently trust a module without re-deriving it. The roadmap's Phases 0 and 1 address them before any new content is added.

---

## 2. How this review was done

1. **Read every module**, including docstrings, configuration blocks, implementations, and `main()` demos.
2. **Ran every module** headless in a fresh Python 3.13 virtual environment with current releases: numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, statsmodels 0.15.0, xgboost 3.4.1, lightgbm 4.7.0, prophet 1.5.0, arch 8.0.0, hmmlearn 0.3.3, gymnasium 1.4.0, networkx 3.7, torch 2.15.0 (CPU). The harness forced the Agg backend, replaced `plt.show()`, **re-enabled the warnings each module suppresses**, and applied a 300 to 420 second timeout per module.
3. **Verified specific claims independently** where a bug was suspected:
   - Heston: compared the repository's analytic pricer against an independent "little trap" implementation and against the repository's own Monte Carlo simulator.
   - Holt-Winters intervals: called `prediction_interval()` directly and measured the interval width.
   - ReAct calculator: probed the `eval` sandbox with a harmless expression.
   - Library behavior: checked parameter signatures and documentation in the installed versions (for example, that arch's Student-t is standardized and that `fetch_20newsgroups` sorts its labels).
4. **Limitations:** the sandbox blocked dataset downloads, so modules that fetch data at runtime (20 Newsgroups, California housing) failed for that reason; the review flags them as "requires network, no offline fallback" rather than as code bugs. Long-running modules (RL training, LightGBM sweeps, Heston calibration) hit the timeout; the review only reports a failure when the module crashed or produced invalid output before the timeout.

---

## 3. Scorecard by category

| Category | Modules | Ran cleanly | Failed | Text quality | Biggest issue |
|----------|---------|-------------|--------|--------------|---------------|
| classic | 9 | 5 | 4 | Fair: practical tips, little math | Test-set tuning (decision tree, KNN); 3 API-drift crashes; Naive Bayes comparison bugs |
| ensemble | 6 | 3 | 3 | Fair: contradictions between XGBoost and LightGBM files | 3 crashes; broken `plot_staged_loss`; stacking meta-learner bugs |
| time_series | 5 | 4 | 1 | Good in ETS and N-BEATS; weak in TFT | No rolling-origin backtesting anywhere; TFT and N-BEATS evaluation leaks |
| anomaly_detection | 4 | 4 | 0 | Fair: several wrong claims | Evaluation on training normals; trivially separable synthetic anomalies |
| bayesian | 7 | 7 | 0 | Strong: math and references | Methods described but not implemented (NUTS, Gibbs, forward/Viterbi) |
| causal_inference | 3 | 2 | 1 | Strong in DML; uneven elsewhere | DirectLiNGAM wrong; Qini on confounded data; non-robust DML SE |
| interpretability | 3 | 0 | 3 | Good | All three fail (import error, NumPy 2 removal, network) |
| quantitative | 5 | 3 | 2 | Good formulas, some wrong claims | Heston pricer returns NaN; GARCH VaR scale; pie chart crash |
| reinforcement_learning | 5 | 5 | 0 | Good | Truncation treated as termination in 3 of 5; no multi-seed evaluation |
| multi_agent_systems | 2 | 2 | 0 | Fair | QMIX recurrent-state bug; unsafe calculator `eval`; mock LLM only |
| scientific_ml | 7 | 5 | 2 | Strong but over-claims | Adjoint and neural-guided SR claimed, not implemented; SINDy demo fails silently |

"Ran cleanly" means `main()` completed or reached the timeout without an error. It does not mean the output was correct; sections 6 and 7 cover correctness.

---

## 4. Repository-level findings

### 4.1 Missing project infrastructure

| Missing item | Why it matters |
|--------------|----------------|
| LICENSE | Code without a license is "all rights reserved" by default. Nobody can legally copy it into their work. |
| README | The GitHub landing page is empty. Readers cannot discover what exists or how to run it. |
| `pyproject.toml` / `requirements.txt` | Readers must guess 15+ dependencies and their versions. The review needed to discover them by grepping imports. |
| `.gitignore` | Running any module writes PNG files into the working directory; they will end up in commits. |
| Tests and CI | Nothing detects the API drift that broke a third of the modules. |
| Package structure (`__init__.py`) | Modules cannot import shared helpers, so every file re-implements them (see 4.3). |
| CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION | Needed before inviting contributors or asking people to cite the work. |

### 4.2 Global warning suppression hides breakage

49 of 56 files call `warnings.filterwarnings("ignore")` at import time. This hid the deprecation warnings for every API that has since been removed (`multi_class`, `n_iter`, `algorithm`, `early_stopping_rounds`, `labels=`, `np.trapz`, `ndarray.ptp`, `plt.cm.get_cmap`). With warnings re-enabled, the current runs still show the next wave:

- `SVC(probability=True)` is deprecated in scikit-learn 1.9 and will be removed in 1.11 (`classic/svm.py`, `ensemble/stacking_voting.py`).
- `LogisticRegression(penalty=...)` is deprecated in 1.8 and will be removed in 1.10 (`classic/logistic_regression.py` tuning grid).
- `adfuller` return type changes in statsmodels 0.16 (`time_series/arima.py`).
- hmmlearn triggers a NumPy 2.5 deprecation 469 times per run (`bayesian/hmm.py`).

It also hides `ConvergenceWarning`s that a learner should see (Gaussian process kernel bounds, ARIMA maximum likelihood), which are themselves teaching moments.

### 4.3 Duplicated helpers

The same helper appears in many files with small variations: `plot_confusion_matrix` (14 copies), `evaluate` (17), `cross_validate` (11), `load_classification_data` and `load_regression_data` (10 each), `evaluate_classifier` (9), and an identical `generate_anomaly_data` in three anomaly modules. Fixes made in one copy do not reach the others.

### 4.4 Plotting and outputs

Every module pairs `plt.savefig("<name>.png")` with `plt.show()` (180 of each). In a script, `plt.show()` blocks until the reader closes each window, and the PNG files land in whatever directory the reader ran from. There is no output directory, no `--no-show` flag, and no fast mode, so running a module end-to-end can take minutes and a dozen manual window closes.

### 4.5 Reproducibility

Random number handling mixes three APIs (`np.random.seed` in 15 files, `RandomState` in 17, `default_rng` in 13). Only 3 of the 15 PyTorch modules call `torch.manual_seed`. `interpretability/shap_values.py` uses the unseeded global NumPy generator. Results therefore change from run to run in ways the text does not mention.

### 4.6 Style consistency

- Three header styles: the original template (Use Cases / Assumptions / Fine-Tuning), the math-first style with References (12 files), and N-BEATS's style with `# ----` dividers.
- 516 `# Placeholder` markers, many rendered as `# # Placeholder:` with a doubled hash, carry no information beyond "you can change this".
- Mixed use of type hints; `list | None` syntax requires Python 3.10, which no file documents.
- Commit messages mix `feat:`, `feature:`, `Feature:`, `add:`, and `build:` prefixes.

### 4.7 Private and dangling references

- **22 references to "Hyperion"** in 18 files, describing a private project (for example "Hyperion's multi-agent swarm intercept platform" in `reinforcement_learning/ppo.py`, "Hyperion allocation signals" in `time_series/arima.py`). Outside readers cannot follow them, and some reveal details of private work. Files: `time_series/arima.py`, `time_series/tft.py`, `time_series/prophet_model.py`, `reinforcement_learning/actor_critic.py`, `reinforcement_learning/ppo.py`, `anomaly_detection/autoencoder_anomaly.py`, `anomaly_detection/isolation_forest.py`, `quantitative/monte_carlo.py`, `quantitative/portfolio_optimization.py`, `quantitative/garch.py`, `bayesian/bayesian_optimization.py`, `bayesian/particle_filter.py` (2), `bayesian/gmm.py`, `bayesian/hmm.py`, `bayesian/gaussian_process.py`, `scientific_ml/pinn.py`, `scientific_ml/deeponet.py`, `scientific_ml/hamiltonian_nn.py`, `scientific_ml/neural_ode.py`.
- `anomaly_detection/autoencoder_anomaly.py` points readers to `frontier/vae.py`, which does not exist.

### 4.8 Network-dependent demos without fallbacks

`classic/naive_bayes.py` (20 Newsgroups), `interpretability/lime_explainer.py` and `interpretability/shap_values.py` (California housing), and the MNIST demo in `anomaly_detection/autoencoder_anomaly.py` download data at runtime. The MNIST demo also needs `torchvision`, which nothing declares. Offline readers, CI runners, and corporate networks will see these fail.

### 4.9 Security: the "safe" calculator is not safe

`multi_agent_systems/react_agent.py` describes `_calculator` as a "safe arithmetic evaluator" and calls `eval(expr, {"__builtins__": {}}, allowed_names)`. Emptying `__builtins__` does not sandbox Python. The review confirmed that the expression `[c.__name__ for c in ().__class__.__base__.__subclasses__() if c.__name__=='Popen']` returns `['Popen']`, which gives access to process execution. In a ReAct agent, an LLM chooses the calculator input, and prompt injection can steer that choice. Because readers copy agent code into real systems, replace this with an AST-whitelist evaluator.

---

## 5. Verified runtime failures

Each entry below reproduced in the environment described in section 2.

### Group A: one-line API drift (fix first)

| File | Line | Error | Fix |
|------|------|-------|-----|
| `classic/logistic_regression.py` | 120 (config at 100) | `TypeError: unexpected keyword argument 'multi_class'` (removed in scikit-learn 1.7) | Delete `multi_class`; also move off `penalty`, which 1.10 removes |
| `classic/random_forest.py` | 275 | `TypeError: boxplot() got an unexpected keyword argument 'labels'` (matplotlib 3.9 renamed it) | `tick_labels=` |
| `classic/dimensionality_reduction.py` | 260 | `TypeError: TSNE.__init__() got an unexpected keyword argument 'n_iter'` | `max_iter=`; then `plt.cm.get_cmap` (line 368, removed in matplotlib 3.9) fails next: use `plt.get_cmap` or `matplotlib.colormaps` |
| `ensemble/adaboost.py` | 353 (config at 127) | `TypeError: unexpected keyword argument 'algorithm'` (removed in scikit-learn 1.8) | Delete `algorithm` from the config, the grid, and `plot_sample_weights_evolution` |
| `ensemble/xgboost_model.py` | 217 | `TypeError: fit() got an unexpected keyword argument 'early_stopping_rounds'` (moved to the constructor in xgboost 2.0) | Pass `early_stopping_rounds` to `XGBClassifier(...)`; remove `use_label_encoder` |
| `causal_inference/treatment_effects.py` | 377 | `AttributeError: module 'numpy' has no attribute 'trapz'` | `np.trapezoid` |
| `scientific_ml/deeponet.py` | 117 | `AttributeError: module 'numpy' has no attribute 'trapz'` (in data generation, so nothing runs) | `np.trapezoid` |
| `interpretability/shap_values.py` | 321 | `AttributeError: 'numpy.ndarray' object has no attribute 'ptp'` | `np.ptp(feat_vals)` |
| `interpretability/integrated_gradients.py` | 285 | `TypeError: unsupported operand type(s) for \|: 'builtin_function_or_method' and 'NoneType'` **at import time**, so the module cannot even be imported | `from collections.abc import Callable` and annotate `Callable \| None` |

### Group B: logic errors that crash

| File | Line | Error | Cause and fix |
|------|------|-------|---------------|
| `ensemble/stacking_voting.py` | 234 | `ValueError: Invalid format specifier '.4f if auc else 'N/A''` | A conditional cannot live inside a format spec. Compute the string first. Two more crashes wait behind it: `get_base_regressors` uses `KNeighborsClassifier` for regression (line 138), and `compare_meta_learners` uses `RidgeClassifier`, which has no `predict_proba`. |
| `quantitative/portfolio_optimization.py` | 412 | `ValueError: Wedge sizes 'x' must be non negative values` | Risk contributions can be negative; a pie chart cannot show them. Use a signed bar chart. |
| `scientific_ml/symbolic_regression.py` | 234 | `IndexError: list index out of range` in `subtree_crossover` | When a parent is a single leaf, `rng.integers(1, max(2, len(nodes)))` returns 1 for a one-element list. Allow the root as a cut point for single-node trees. |
| `time_series/tft.py` | 202 | `AttributeError: module 'torch' has no attribute 'elu'` on the first training batch | Use `torch.nn.functional.elu`. A second crash waits behind it at line 454: with 300 time steps the validation slice has 60 rows, fewer than `encoder_len + decoder_len = 64`, so `__len__` raises `ValueError: __len__() should return >= 0` (verified directly). `__len__` is also off by one. |

### Group C: silent failures and hangs

| File | Symptom | Cause |
|------|---------|-------|
| `quantitative/heston.py` | Prints `Heston Call: nan`, `Heston Put: nan`, `Put-call parity error: nan`, then the calibration step runs on a NaN objective until the timeout | See section 6.4. An independent pricer gives 23.87 / 8.93 / 1.29 for strikes 80 / 100 / 120 (parameters S0=100, T=1, r=0.03, V0=θ=0.04, κ=2, ξ=0.5, ρ=-0.7), and the repository's own Monte Carlo agrees (23.95 / 8.98 / 1.32). The repository's analytic pricer returns NaN for all three. |
| `time_series/exponential_smoothing.py` | `prediction_interval()` returns intervals of width exactly 0 | `HoltWintersResults` has no `get_prediction`; the bare `except` falls back to `lower = upper = forecast` without telling the reader. |
| `ensemble/gradient_boosting.py` | `plot_staged_loss` would crash if called (it is not called in `main`) | It calls a nonexistent `model.staged_score`, iterates over `... if False else []`, and builds an unused `X_all`. Rewrite or delete. |
| `classic/dimensionality_reduction.py` | `reconstruction_error(method="factor")` would crash | `FactorAnalysis` has no `inverse_transform`. |

### Group D: requires network, no offline fallback

`classic/naive_bayes.py` (20 Newsgroups) and `interpretability/lime_explainer.py` (California housing) failed in the sandbox with `URLError ... 403 Forbidden`. They will work on an open network but fail in CI and offline.

---

## 6. Correctness and methodology issues

### 6.1 Choosing on the test set (recurring pattern)

These functions pick a value by maximizing a score on the same data the demo then reports as the test result. The reported number is optimistically biased, and readers learn the wrong habit.

| File | Function | What it selects on test data |
|------|----------|------------------------------|
| `classic/decision_tree.py` | `find_optimal_ccp_alpha` | Pruning strength `ccp_alpha` |
| `classic/knn.py` | `plot_k_selection_curve` | Number of neighbors `k` |
| `ensemble/adaboost.py` | `plot_staged_accuracy` | `n_estimators` |
| `ensemble/lightgbm_model.py` | `demo_categorical_features` | Early-stopping round (uses `X_test` as the eval set) |
| `ensemble/stacking_voting.py` | `evaluate_classifiers` | Ranks models by test AUC |
| `anomaly_detection/isolation_forest.py` | `find_optimal_threshold` | Score threshold |
| `anomaly_detection/autoencoder_anomaly.py` | `find_threshold`, `find_optimal_threshold` | Threshold from evaluation-set scores |
| `anomaly_detection/dbscan.py` | `eps_sweep` | `eps` |
| `anomaly_detection/one_class_svm.py` | `nu_gamma_sweep` | `nu` and `gamma` |
| `time_series/n_beats.py` | `train_nbeats` then `evaluate` | Early stopping and final metrics use the same validation set |

**Fix:** a shared three-way split (train, validation, test) or cross-validation inside the training split, with the test split touched once. The roadmap's Phase 1.1 adds the utility; Phase 1.2 applies it.

### 6.2 Evaluating on training data

- **Anomaly detection (3 modules):** `isolation_forest.py`, `one_class_svm.py`, and `autoencoder_anomaly.py` train on `X_normal` and then evaluate on `X = X_normal + anomalies`, so every normal point in the evaluation set was also a training point.
- **TFT:** `evaluate_forecasts` builds windows over the entire series, including the training portion, and normalization statistics come from the full series.
- **Synthetic anomalies are too easy:** the shared generator draws anomalies uniformly from a box three times the data range, so almost any detector scores ROC-AUC near 1.0. The demo cannot show how detectors differ.

### 6.3 Termination versus truncation in reinforcement learning

Gymnasium separates `terminated` (the task ended) from `truncated` (a time limit cut the episode). Only termination should stop bootstrapping. `reinforcement_learning/dqn.py` (line 492), `ppo.py` (lines 684 and 819), and `actor_critic.py` (line 188) store `terminated or truncated` as `done`, which biases value estimates on CartPole-v1 (truncated at 500 steps). `sac.py` and `td3.py` store `term` correctly, so the category contradicts itself. `multi_agent_systems/qmix.py` has the same issue: its only episode end is a time limit, treated as terminal, and the state does not include time.

### 6.4 Algorithm and statistics bugs

| Severity | File | Issue |
|----------|------|-------|
| High | `quantitative/heston.py` | `heston_char_fn` sets `b = kappa - rho*xi*phi*i` and then subtracts `rho*xi*phi*i` again in `d`, `g`, `C`, and `D`, so the characteristic function is wrong and the integral returns NaN. `max(nan, 0.0)` returns NaN, so the error propagates silently into the surface and calibration. The docstring also claims the Albrecher et al. (2007) "little trap" form, but the code uses the original form (`g` with `+d`, `exp(+dT)`) that has branch-cut problems. |
| High | `causal_inference/causal_discovery.py` | `DirectLiNGAM.fit` picks, at each step, the variable whose regression residual on all others is most non-Gaussian. That criterion identifies sinks, yet the code appends the variable as the next root. The run recovers order X4, X2, X3, X5, X1 when the true roots are X1 and X2 and the sink is X5. The estimated adjacency is also transposed (`B[child, parent]`) relative to the true adjacency (`adj[parent, child]`). |
| High | `causal_inference/causal_discovery.py` | `evaluate_graph` counts each undirected skeleton edge twice (it printed TP=10 for 6 true edges), and its "SHD" counts a reversal as two errors and every undirected CPDAG edge as an error. Compare CPDAG to true CPDAG and use the standard SHD. The synthetic DAG also gives X3 and X5 Gaussian noise, which violates the LiNGAM identifiability assumption the docstring says the data supports. |
| High | `multi_agent_systems/qmix.py` | The update flattens episodes into independent transitions and then passes the GRU hidden state from agent i to agent i+1 (lines 521 to 529) instead of unrolling each agent over time. Acting carries hidden state through time; training never does, and agents share hidden state with each other. |
| High | `causal_inference/dml_estimation.py` | The standard error is labeled "sandwich" but computes the homoskedastic `sqrt(mean(e^2) / sum(D~^2))`. The DML (heteroskedasticity-robust) variance is `sum(D~^2 e^2) / (sum D~^2)^2`. Confidence intervals and p-values are invalid when effects are heteroskedastic. With a binary treatment and heterogeneous effects, the partially linear model estimates a variance-weighted average effect, not the ATE the docstring promises; the interactive regression model (AIPW) estimates the ATE. |
| High | `causal_inference/treatment_effects.py` | The Qini/AUUC evaluation uses global treatment shares on confounded observational data. Uplift curves are only unbiased under randomization (or with propensity weighting). The docstring recommends this metric "in real data", which is exactly where it fails. |
| Medium | `quantitative/garch.py` | arch's Student-t distribution is standardized to unit variance, but `calculate_var` uses the raw `stats.t.ppf(alpha, nu)`, which overstates VaR by `sqrt(nu/(nu-2))`. The simulator draws unstandardized `standard_t(6)` shocks (variance 1.5), so the "true" parameters cannot be recovered. The comment says "Kupiec test", but only a violation rate is printed. |
| Medium | `quantitative/monte_carlo.py` | With antithetic variates, the standard error treats each `z` and `-z` path as independent, which misstates the confidence interval. Average each pair first. |
| Medium | `classic/naive_bayes.py` | `compare_variants` fits GaussianNB on unscaled data but scores it on scaled data. BernoulliNB with `binarize=0.0` on MinMax-scaled data turns nearly every feature into 1. `fetch_20newsgroups` orders labels alphabetically, but `main()` passes an unsorted category list as class names, so the per-class feature plot mislabels classes. The "top discriminative features" plot shows the most frequent words, not discriminative ones. |
| Medium | `ensemble/xgboost_model.py` | `plot_learning_curve` labels the validation curve "Train" (the eval set only contains validation data) and then plots the same curve again as "Validation". |
| Medium | `ensemble/lightgbm_model.py` | The categorical-features demo generates a target that does not depend on the categorical columns, so it cannot show any benefit from native categorical handling. |
| Medium | `time_series/exponential_smoothing.py` | `fit_ets_auto` unpacks `(trend, seasonal, error)` but never uses `error` (the Holt-Winters API has no error component), so the "ETS_auto(error, ...)" names are fiction. Use statsmodels' `ETSModel`. |
| Medium | `time_series/tft.py` | With the default 300 steps, the validation slice has 60 rows, fewer than `encoder_len + decoder_len = 64`, so `__len__` goes negative. `__len__` is also off by one. The synthetic weekly dates all fall on Wednesday, so the `dow` feature is constant, and the covariates are pure noise. |
| Medium | `time_series/n_beats.py` | The benchmark scores neural models over every validation window but scores seasonal-naive on a single window. `load_m4_hourly_demo` returns synthetic data. |
| Medium | `anomaly_detection/isolation_forest.py` | The model trains on normal data only (novelty detection) with `contamination = 50/1050`, so it flags about 4.7% of normal points by construction. |
| Low | `classic/dimensionality_reduction.py` | The ICA demo overlays recovered and true sources by index, but ICA recovers sources in arbitrary order and sign; match by absolute correlation first. The embedding grid compares supervised (LDA, supervised UMAP) and unsupervised methods without saying so. |
| Low | `interpretability/shap_values.py` | `PermutationSHAP` calls the model once per row per feature per permutation; batch each permutation. Unseeded global RNG. |
| Low | `bayesian/mcmc.py` | `r_hat` is the classic non-split Gelman-Rubin statistic but uses the 1.01 threshold from rank-normalized split R-hat (Vehtari et al., 2021). The ESS comment claims Geyer's monotone sequence; the code truncates at the first negative autocorrelation. |

---

## 7. Text and walkthrough review

### 7.1 What already works

- **"Use Cases" and "Fine-Tuning Recommendations"** sections give practical guidance that library docs rarely do. Keep them.
- **Math-first docstrings** in the newer modules (ETS, N-BEATS, Kalman filter, MCMC, DML, Heston, FNO, SAC, TD3) explain the model before the code. Extend this style to the older modules.
- **"Replace with your own data" snippets** in every data loader lower the barrier to real use.
- **Synthetic data with known ground truth** (CATE, DAGs, GARCH, regime switching) lets readers check whether a method recovers the truth. This is rare and valuable.

### 7.2 Systemic issues and how to fix them

| # | Issue | Example | Fix |
|---|-------|---------|-----|
| 1 | **No consistent learning arc.** Older modules list facts (use cases, assumptions, tips) but never explain how the model works. | `classic/logistic_regression.py` never shows the sigmoid or log-loss; `classic/decision_tree.py` never defines Gini or entropy; `classic/svm.py` never shows the hinge loss or the kernel trick. | Adopt the twelve-section arc in [MODULE_STANDARD.md](../MODULE_STANDARD.md): summary, when to use, intuition, model, algorithm, assumptions, complexity, hyperparameters, evaluation protocol, expected output, pitfalls, references. |
| 2 | **Promises the code does not keep.** | See 7.3. | Implement, or say "not implemented here; see `<library>`". |
| 3 | **Factual errors.** | See 7.4. | Correct and cite. |
| 4 | **Contradictions** inside a file or between files. | Naive Bayes says it "naturally outputs calibrated probabilities" and two paragraphs later that it is "overconfident". AdaBoost calls SAMME the default and then SAMME.R the default. `xgboost_model.py` says LightGBM has "comparable or better AUC"; `lightgbm_model.py` says it is "often more accurate". | One source of truth per claim, ideally a shared comparison page. |
| 5 | **No "when not to use" and no baseline to beat.** | Forecasting modules never compare against naive or seasonal-naive forecasts (N-BEATS does, but incorrectly). | Add a baseline line to every module and to every demo. |
| 6 | **No code-to-math mapping.** Readers cannot tell which function implements which equation. | MCMC, Kalman filter, and DML have equations but no pointers into the code. | Number the algorithm steps and reference the step in code comments. |
| 7 | **Outputs are printed but never interpreted.** | Every module prints metrics and saves plots with no statement of what a reader should expect to see. | Add an "Expected output" block, and verify it in the smoke test where possible. |
| 8 | **Placeholder noise.** | 516 `# Placeholder` markers; many read `# # Placeholder:` with no content. | Replace with a hyperparameter table (name, effect, default with library version, range, tip). |
| 9 | **Missing references** in older files. | Only 12 of 56 modules have a References section. In `classic/`, only `dimensionality_reduction.py` has one; in `ensemble/`, only LightGBM and stacking do; `anomaly_detection/` has none. | Cite the original paper and the library docs. |
| 10 | **Silent demo failures.** | SINDy's "Cubic ODE 2D" demo returns R² of 0.23 with every coefficient non-zero; the particle-filter sweep shows no benefit from more particles (N=50 and N=1000 both give RMSE near 0.23); UKF scores worse than EKF on the pendulum. None of these outputs comes with an explanation. | Either fix the setup so the demo shows the intended effect, or explain why the method struggles. Failures are excellent teaching material when they are explained. |
| 11 | **Private references.** | 22 "Hyperion" mentions (section 4.7). | Replace with public, generic examples. |

### 7.3 Methods described but not implemented

| File | Described | Status in code |
|------|-----------|----------------|
| `causal_inference/treatment_effects.py` | Causal Forest (honesty, subsampling, asymptotic normality) under "Methods covered" | Not implemented |
| `causal_inference/causal_discovery.py` | GES; "Meek orientation rules" | GES not implemented; only Meek rule R1 |
| `bayesian/mcmc.py` | Gibbs sampling, NUTS | Neither implemented |
| `bayesian/hmm.py` | Forward-backward, Baum-Welch, Viterbi "key algorithms" | Delegated to hmmlearn; no from-scratch version |
| `bayesian/gaussian_process.py` | Periodic kernel, RBF times Periodic composites | Never imported or shown |
| `reinforcement_learning/actor_critic.py` | A3C | Implemented as synchronous rollouts, which is A2C |
| `scientific_ml/neural_ode.py` | "Gradients flow back through the solver via the adjoint method" | Backpropagates through a custom RK4; no adjoint |
| `scientific_ml/symbolic_regression.py` | Neural-guided symbolic regression under "Approaches implemented" | Not implemented |
| `time_series/tft.py` | Static covariates, known-future inputs, static enrichment, LSTM encoder-decoder | Encoder only; mean-pools over time into an MLP head |
| `time_series/exponential_smoothing.py` | ETS with error component | Holt-Winters only |
| `classic/dimensionality_reduction.py` | Truncated SVD, Factor Analysis | Not demonstrated (Factor Analysis path crashes) |
| `quantitative/heston.py` | "characteristic function + FFT" pricing | Uses `scipy.integrate.quad`; no FFT |
| `quantitative/monte_carlo.py` | Control variates, quasi-Monte Carlo | Only antithetic variates |
| `anomaly_detection/one_class_svm.py` | SGDOneClassSVM for streaming and large data | Not shown |
| `ensemble/gradient_boosting.py` | Ranking and survival analysis use cases | scikit-learn's GBM supports neither |

### 7.4 Factual corrections

| File | Current text (paraphrased) | Correction |
|------|----------------------------|------------|
| `classic/decision_tree.py` | Handles categorical data naturally | scikit-learn trees need numeric input; native categorical splits exist in HistGradientBoosting, LightGBM, and XGBoost |
| `classic/random_forest.py` | OOB score gives built-in uncertainty; supports proximity-based imputation | OOB is a generalization estimate, not uncertainty; scikit-learn has no proximity imputation |
| `classic/random_forest.py` | Permutation importance accounts for feature interactions | It is biased when features are correlated; mention the caveat and conditional or grouped alternatives |
| `classic/svm.py` | The decision-boundary plot shows margins and support vectors | It shows predicted regions only; plot `decision_function` levels -1, 0, 1 and circle `support_vectors_` |
| `classic/naive_bayes.py` | Naturally outputs calibrated probabilities | Naive Bayes is typically poorly calibrated (the same docstring later says so) |
| `classic/dimensionality_reduction.py` | UMAP preserves both local and global structure | Global structure preservation is limited and contested; say "better than t-SNE at moderate scales" with a citation |
| `ensemble/xgboost_model.py` | XGBoost uses level-wise (depth-first) growth | Level-wise growth is breadth-first; XGBoost also offers `grow_policy="lossguide"` |
| `ensemble/xgboost_model.py` | `max_delta_step` is a pruning mechanism | It caps leaf weight updates (useful for imbalanced logistic loss); `gamma` is the split-pruning control |
| `ensemble/lightgbm_model.py` | XGBoost categoricals need manual encoding; GPU via `tree_method=gpu` | XGBoost supports native categoricals (`enable_categorical=True`) since 1.6; GPU is `device="cuda"` since 2.0 |
| `ensemble/adaboost.py` | Does not require learning-rate tuning | The same file recommends tuning `learning_rate`; shrinkage matters |
| `ensemble/bagging.py` | `base_estimator` parameter | Renamed `estimator` in scikit-learn 1.2 |
| `ensemble/stacking_voting.py` | Test predictions average the K fold models | scikit-learn refits each base learner on the full training set for test predictions |
| `ensemble/stacking_voting.py` | Stacking outperforms voting | Often, not always; stacking can overfit small meta-feature sets |
| `time_series/arima.py` | `evaluate` uses dynamic one-step-ahead predictions | It produces a multi-step out-of-sample forecast |
| `time_series/arima.py` | The SARIMA plot is a "24-month forecast" | The model is fit on the training split, so the forecast covers the test window, not the future |
| `time_series/arima.py` | ARIMA is poor for long horizons relative to modern ML | Contested; in the M4 competition, combinations of statistical methods and statistical-ML hybrids beat pure ML methods. Cite evidence or soften |
| `time_series/exponential_smoothing.py` | Theta with θ → ∞ gives SES | Theta(2) is equivalent to SES with drift (Hyndman and Billah, 2003) |
| `time_series/prophet_model.py` | The synthetic series has a trend changepoint; Prophet is robust to outliers | The synthetic series has a level jump, which Prophet's continuous piecewise-linear trend cannot represent; Prophet docs recommend removing outliers |
| `time_series/tft.py` | GRN output = LayerNorm(x + GLU(...)) | The code gates with `sigmoid(W x)` on the input rather than a GLU on the hidden state |
| `anomaly_detection/isolation_forest.py` | Always scale features first | Isolation forest is invariant to per-feature monotone rescaling; scaling is unnecessary |
| `anomaly_detection/isolation_forest.py` | Handles high-dimensional data well, robust to irrelevant features | Irrelevant dimensions are a known weakness (see Extended Isolation Forest and feature bagging) |
| `anomaly_detection/isolation_forest.py` | Linear time complexity O(n log n) | Self-contradictory; training is O(t·ψ·log ψ) for t trees and subsample size ψ |
| `anomaly_detection/one_class_svm.py` | Many false positives: increase ν | Inverted. Increasing ν flags more training points as outliers; decrease ν to reduce false positives |
| `anomaly_detection/one_class_svm.py` | Scales poorly to high dimensions | The bottleneck is sample count: the kernel matrix needs O(n²) memory |
| `bayesian/hmm.py` | `n_init` parameter; `score_samples()` for online inference; MultinomialHMM for discrete data | GaussianHMM has no `n_init`; `score_samples` returns smoothed posteriors that use future observations; use `CategoricalHMM` for discrete symbols (hmmlearn 0.2.8 and later) |
| `bayesian/gaussian_process.py` | Multi-output needs `MultiOutputRegressor` | `GaussianProcessRegressor` accepts 2-D targets directly |
| `bayesian/mcmc.py` | MH target acceptance about 23% | 0.234 is the high-dimensional limit; about 0.44 is optimal in one dimension |
| `causal_inference/causal_discovery.py` | Data mixes Gaussian and Laplace noise "to support LiNGAM" | LiNGAM identifiability needs non-Gaussian noise on all but at most one variable |
| `interpretability/shap_values.py` | Consistency means a feature that increases output always gets positive SHAP | Consistency means that if a model changes so a feature's marginal contribution grows, its attribution does not decrease |
| `interpretability/integrated_gradients.py` | Completeness and Sensitivity are the two axioms, uniquely satisfied | The paper's axioms are Sensitivity and Implementation Invariance; uniqueness holds among path methods with the symmetry-preserving property |
| `quantitative/black_scholes.py` | The volatility smile comes from "market prices with stochastic vol" | The code injects a hand-written quadratic smile and inverts it, which is circular; generate prices with `heston.py` instead |
| `quantitative/black_scholes.py` | Add dividends by replacing S with S·e^(-qT) "in the d₁ formula" | The drift in d₁ becomes r - q, and the price uses S·e^(-qT); the code does this correctly, but the text does not |
| `quantitative/heston.py` | Reflection via `abs()` is the full-truncation variant | Reflection and full truncation are different schemes; the code mixes both. The price update is Euler on S (can go negative); log-Euler is standard |
| `quantitative/garch.py` | GARCH supports option pricing via a local volatility surface; standardized residuals should be N(0,1) | GARCH is not a local volatility model; with `dist="t"` the residuals should follow the fitted t distribution, so the QQ plot against a normal is the wrong check |
| `quantitative/monte_carlo.py` | Antithetic variates halve variance at no cost | Only for monotone payoffs; the gain depends on the payoff |
| `reinforcement_learning/ppo.py` | PPO underlies OpenAI Five and AlphaStar | OpenAI Five used PPO; AlphaStar used an IMPALA-style actor-critic with V-trace and UPGO |
| `reinforcement_learning/actor_critic.py` | MAPPO is many A2C actors with one centralized critic | MAPPO is PPO with a centralized value function |
| `reinforcement_learning/td3.py` | `policy_noise` is exploration noise during collection; TD3 needs epsilon-greedy exploration | `policy_noise` is target-policy smoothing noise; exploration uses Gaussian action noise (`expl_noise`), not epsilon-greedy |
| `multi_agent_systems/qmix.py` | For StarCraft, use MAPPO instead | SMAC is cooperative and is QMIX's standard benchmark |
| `multi_agent_systems/react_agent.py` | "Safe arithmetic evaluator" | Not safe (section 4.9) |
| `scientific_ml/neural_ode.py` | Adjoint is O(depth) memory vs O(depth²) | The adjoint method is O(1) in depth versus O(depth) for direct backpropagation (the same docstring states O(1) earlier) |
| `scientific_ml/fno.py` | Linear complexity in resolution: O(n log n) | O(n log n) is not linear; resolution invariance also needs a caveat and a demonstration |
| `scientific_ml/hamiltonian_nn.py` | Learned dynamics keep H exactly constant | Exact only for the learned H under exact integration; numerical integrators and model error introduce drift |

---

## Appendix A: per-file findings

Each entry lists the run result, the most important code fixes, and the most important text fixes. Items already covered in sections 5 to 7 are summarized briefly.

### classic

- **`linear_regression.py`** (ran). Text: lists normality "for inference" but never shows inference; add a statsmodels OLS summary with standard errors and confidence intervals, and explain the normal equations and the closed-form solution. Code: polynomial features come after scaling, so squared terms are unscaled; `plot_coefficients` loses meaning when `POLY_DEGREE > 1`.
- **`logistic_regression.py`** (crash, group A). Code: binary-only `plot_coefficients` and `plot_roc_curve` with no guard; `cross_validate` docstring mentions a train-versus-CV gap it never computes. Text: no sigmoid, log-loss, or odds-ratio interpretation; tips recommend threshold tuning and calibration that the demo never shows. See the rewrite example in MODULE_STANDARD.md.
- **`decision_tree.py`** (ran). Code: test-set tuning of `ccp_alpha` (6.1); `permutation_importance` imported but unused although the text recommends it. Text: define Gini and entropy, explain greedy split search and cost-complexity pruning; correct the categorical claim.
- **`random_forest.py`** (crash, group A). Code: `plot_n_estimators_curve` refits 30 forests; use `warm_start`, which the docstring already mentions. Text: correct the OOB, imputation, and permutation-importance claims.
- **`svm.py`** (ran; `probability=True` deprecated). Code: plot margins and support vectors; `LinearSVC` imported but unused. Text: explain the soft-margin objective, hinge loss, dual, and kernel trick; note that Platt-scaled `predict_proba` can disagree with `predict`.
- **`knn.py`** (ran). Code: test-set selection of `k`. Text: discuss ties, the curse of dimensionality (a small demo would help), and approximate nearest neighbors (FAISS, HNSW) for scale.
- **`kmeans.py`** (ran; solid). Code: `make_moons` is imported but unused; use it to show K-Means failing on non-convex clusters. Text: add a 15-line NumPy Lloyd's loop next to the scikit-learn call.
- **`naive_bayes.py`** (network). Code: the comparison and label-order bugs in 6.4; use `remove=("headers", "footers", "quotes")` as scikit-learn recommends to avoid leakage. Text: fix the calibration contradiction; explain the log-space computation and Laplace smoothing with a worked example.
- **`dimensionality_reduction.py`** (crash, group A, twice). Code: `fit_pca` is annotated as returning two values and returns three; `pca_as_preprocessing(X, y, X, y)` ignores its test arguments. Text: label supervised methods in the comparison; add Truncated SVD and Factor Analysis or remove them from the list; remove the doubled `# # Placeholder` markers.

### ensemble

- **`gradient_boosting.py`** (ran). Code: rewrite or delete `plot_staged_loss`; `permutation_importance` unused. Text: add a from-scratch residual-fitting loop (10 lines) to make "gradient descent in function space" concrete; remove the ranking and survival use cases or point to libraries that support them; show HistGradientBoosting's categorical support, which the text mentions.
- **`xgboost_model.py`** (crash, group A). Code: learning-curve labels; `feature_names` argument unused. Text: correct the growth-policy, `max_delta_step`, and calibration claims; explain the second-order (Newton) objective and the regularized leaf weight formula, which are what distinguish XGBoost.
- **`lightgbm_model.py`** (ran; slow under the timeout). Code: categorical demo target ignores the categorical columns and early-stops on test data; add a one-hot baseline to support the "1 to 3% AUC gain" claim or remove it. Text: fix the comparison table (including its misaligned border) and explain the `min_data_in_leaf` and `min_child_samples` aliases.
- **`adaboost.py`** (crash, group A). Code: test-set selection of `n_estimators`; `plot_sample_weights_evolution` is never called and refits one model per stage. Text: replace SAMME and SAMME.R statements with the current API; turn the sample-weight idea into a short from-scratch AdaBoost, which is the clearest way to teach it.
- **`bagging.py`** (ran). Text: update `base_estimator`; add the single-tree versus bagged-trees variance demo, which is the central lesson of bagging. Code: use `warm_start` for the OOB curve.
- **`stacking_voting.py`** (crash, group B). Code: the three crashes; stop ranking by test AUC. Text: correct the cross-fitting description and soften the "outperforms" claim.

### time_series

- **`arima.py`** (ran). Code: refit on the full series before forecasting the future; explain `trend="t"` for drift with `d=1`; avoid double differencing a deterministic sinusoid. Text: remove the private reference; fix the "one-step-ahead" wording; add rolling-origin evaluation, naive and seasonal-naive baselines, MASE, and an explanation of ADF regression options and KPSS.
- **`exponential_smoothing.py`** (ran). Code: zero-width intervals; fake ETS error component; misleading `ETSModel` import alias. Text: correct the Theta statement; keep the strong equations and references.
- **`prophet_model.py`** (ran). Code: overlay test actuals on the forecast plot; pass holidays to the constructor; use consistent mean or median CV metrics. Text: remove the private reference; explain that the synthetic level jump is something Prophet cannot model (a good teaching point once stated); soften the outlier claim.
- **`n_beats.py`** (ran until the timeout, no error). Code: fix the baseline comparison; add a test split; delete the unused `stack_forecast` tensor; rename `load_m4_hourly_demo`. Text: warn that sMAPE is unstable for series near zero, as in the synthetic data.
- **`tft.py`** (crash, group B: `torch.elu` does not exist, then the validation window is too short). Code: in-sample evaluation, normalization leakage, window length, constant `dow`. Text: either implement the decoder with known-future inputs and static covariates or rename the class to reflect a TFT-style encoder; fix the GRN description; remove the private reference.

### anomaly_detection

- **`isolation_forest.py`** (ran). Code: evaluation on training normals; contamination applied to normal-only training data; threshold tuned on evaluation labels. Text: correct the scaling, dimensionality, and complexity claims; explain outlier detection versus novelty detection.
- **`dbscan.py`** (ran). Code: the knee finder (maximum absolute second derivative on the descending curve) tends to land on the extreme-outlier head; use a Kneedle-style method or let the reader pick from the plot; `eps_sweep` tunes on labels. Text: show HDBSCAN, which the text recommends.
- **`one_class_svm.py`** (ran). Code: evaluation on training normals; sweep tunes on labels. Text: fix the inverted ν advice and the scaling claim; show `SGDOneClassSVM` with a Nystroem kernel approximation.
- **`autoencoder_anomaly.py`** (ran; MNIST demo needs `torchvision`). Code: thresholds come from evaluation-set scores; set them from held-out normal reconstruction errors. Text: remove the dangling `frontier/vae.py` reference and the private reference.

### bayesian

- **`mcmc.py`** (ran). Code: diagnostics notes in 6.4; finite-difference gradients in HMC (mention autodiff); no step-size adaptation. Text: implement or remove Gibbs and NUTS; add an ArviZ comparison.
- **`gaussian_process.py`** (ran; convergence warnings at kernel bounds). Text: show the Periodic kernel; explain that `predict(return_std=True)` includes the WhiteKernel noise (predictive versus latent uncertainty); correct the multi-output claim; surface the kernel-bound warnings as a lesson.
- **`gmm.py`** (ran). Text: state the K-Means limit precisely (shared isotropic covariance with variance going to zero); remove the private reference.
- **`hmm.py`** (ran). Text: correct `n_init`, `score_samples`, and MultinomialHMM; add a from-scratch forward algorithm and Viterbi (about 40 lines) since the text lists them as key algorithms; explain label switching when comparing to true states.
- **`kalman_filter.py`** (ran). Strong module. Text: the run shows UKF worse than EKF on the pendulum (0.095 versus 0.063 RMSE); explain why (sigma-point spread with α = 0.001 and κ = 0) or tune it, since readers will expect the opposite.
- **`particle_filter.py`** (ran). Text: the particle-count sweep shows no improvement from 50 to 1000 particles, so the demo cannot teach the effect; use a harder, multimodal model; remove the two private references.
- **`bayesian_optimization.py`** (ran). Text: remove the private reference; compare against random search at equal budget so the sample-efficiency claim has evidence.

### causal_inference

- **`treatment_effects.py`** (crash, group A). Code: Qini on confounded data; no cross-fitting for meta-learners. Text: implement or remove Causal Forest; fix the data-generating-process formula (it omits the `0.2·X2²` baseline term the code adds); add DR-learner and R-learner, which the tips mention.
- **`dml_estimation.py`** (ran). Code: robust standard error; IRM/AIPW for binary treatment; repeated cross-fitting with the median estimate as in Chernozhukov et al. (2018). Text: the FWL and orthogonality explanation is excellent; add a comparison with DoubleML or EconML.
- **`causal_discovery.py`** (ran, wrong output). Code: DirectLiNGAM, adjacency orientation, metrics, noise design (6.4). Text: say only Meek R1 is implemented; point to causal-learn and lingam for complete implementations.

### interpretability

- **`shap_values.py`** (crash, group A). Code: batching and seeding. Text: fix the consistency axiom; explain interventional versus conditional SHAP; validate against `shap.TreeExplainer`.
- **`lime_explainer.py`** (network). Solid from-scratch implementation. Text: add a comparison with the `lime` package. Category gaps: permutation importance, PDP and ICE, ALE, counterfactuals.
- **`integrated_gradients.py`** (crash at import, group A). Text: fix the axiom statement and the inconsistent completeness tolerance (1e-4 in one place, 1e-3 in another).

### quantitative

- **`black_scholes.py`** (ran). Formulas and Greeks are correct, including dividends. Text: fix the circular smile demo and the dividend wording; the implied-volatility lower bound should include the dividend term.
- **`heston.py`** (silent NaN, then hang). Code: characteristic function (6.4); log-Euler for the price; consistent naming for the variance scheme. Text: remove the FFT claim or add the FFT pricer; add a reference-value unit test.
- **`garch.py`** (ran). Code: VaR quantile scaling, simulator shock scaling, an actual Kupiec likelihood-ratio test, and an out-of-sample backtest. Text: fix the local-volatility and residual-normality claims.
- **`monte_carlo.py`** (ran). Code: antithetic standard error. Text: compare the Monte Carlo European price with the Black-Scholes closed form; add control variates and Sobol sequences, or remove them from the list.
- **`portfolio_optimization.py`** (crash, group B). Code: signed risk-contribution chart. Text: the module optimizes and evaluates on the same returns; add a walk-forward backtest with transaction costs; remove the private reference.

### reinforcement_learning

- **`dqn.py`**, **`ppo.py`**, **`actor_critic.py`** (no crash before the timeout). Code: separate termination from truncation. Text: fix the AlphaStar claim and the private reference in `ppo.py`; rename A3C to A2C or implement asynchronous workers; correct the MAPPO description.
- **`sac.py`**, **`td3.py`** (ran until the timeout, no error). Correct termination handling. Text: fix the `policy_noise` and epsilon-greedy statements in `td3.py`.
- **Category-wide:** seed torch and environments; evaluate over several seeds with confidence intervals; compare with published Stable-Baselines3 results so readers know whether an implementation reaches expected returns.

### multi_agent_systems

- **`react_agent.py`** (ran). Code: replace the calculator `eval`. Text: the mock LLM is a decision tree keyed on task strings, so `evaluate_agent` scores a script; add an adapter for a real model with native tool calling (keeping the mock for offline use). Move the module to an `agents/` category, since ReAct is a single-agent pattern.
- **`qmix.py`** (no crash before the timeout). Code: recurrent-state bug and truncation handling. Text: implement `td_lambda` or remove the tip; correct the SMAC statement.

### scientific_ml

- **`pinn.py`** (ran until the timeout, no error). Text: heat and Burgers demos report only the loss; compare against an analytic or reference solution (Burgers has a Cole-Hopf solution) as the Poisson demo already does; remove the private reference.
- **`neural_ode.py`** (ran until the timeout, no error). Text: fix the adjoint and memory statements.
- **`fno.py`** (ran until the timeout, no error). Text: fix the complexity wording; add the zero-shot super-resolution demo the text implies.
- **`deeponet.py`** (crash, group A). Text: remove the private reference.
- **`hamiltonian_nn.py`** (ran until the timeout, no error). Text: qualify the "exactly constant" claim; remove the private reference.
- **`sindy.py`** (ran). Lorenz and Van der Pol recover correct equations. The "Cubic ODE 2D" demo fails silently (R² 0.23, every coefficient non-zero); fix the setup or explain the failure.
- **`symbolic_regression.py`** (crash, group B). Text: implement or remove neural-guided search.
