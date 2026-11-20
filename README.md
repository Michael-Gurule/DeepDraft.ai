# DeepDraft.ai

Reference implementations of machine learning algorithms for data scientists and ML engineers. Each module is a single, self-contained Python file that explains a model, implements it (often from scratch), and runs an end-to-end demo with evaluation and plots.

Use it to:

- **Learn** how a model works, from intuition to math to code.
- **Copy** a working baseline into your own project and swap in your data.
- **Tune** with the practical "Fine-Tuning Recommendations" in every module.


---

## Quickstart

```bash
git clone https://github.com/Michael-Gurule/DeepDraft.ai.git
cd DeepDraft.ai
python -m venv .venv && source .venv/bin/activate

# Core dependencies only (numpy, scipy, pandas, scikit-learn, matplotlib)
pip install -e .

# Or everything, including PyTorch, boosting libraries, and forecasting tools
pip install -e ".[all]"

# Run a module end to end
python classic/logistic_regression.py
```

Install only the extras you need:

| Extra | Installs | Needed for |
|-------|----------|------------|
| *(none)* | numpy, scipy, pandas, scikit-learn, matplotlib | `classic/`, most of `bayesian/`, `causal_inference/`, `interpretability/`, `quantitative/` |
| `boosting` | xgboost, lightgbm | `ensemble/xgboost_model.py`, `ensemble/lightgbm_model.py` |
| `timeseries` | statsmodels, prophet | `time_series/arima.py`, `exponential_smoothing.py`, `prophet_model.py` |
| `deep` | torch | `scientific_ml/`, `time_series/n_beats.py`, `tft.py`, `anomaly_detection/autoencoder_anomaly.py`, `interpretability/integrated_gradients.py`, `multi_agent_systems/qmix.py` |
| `rl` | gymnasium, torch | `reinforcement_learning/` |
| `causal` | networkx | `causal_inference/causal_discovery.py` |
| `quant` | arch, statsmodels | `quantitative/garch.py` |
| `bayes` | hmmlearn | `bayesian/hmm.py` |
| `dev` | pytest, pytest-xdist, ruff, pre-commit | contributing |

Python 3.10 or newer is required.

### Running options

Modules read three optional environment variables:

| Variable | Effect |
|----------|--------|
| `DEEPDRAFT_FAST=1` | Shrinks data, iterations, and search grids so a demo finishes in seconds. Use it to check that a module runs; the metrics are not meaningful in this mode. |
| `DEEPDRAFT_OUTPUT_DIR=path` | Where figures are saved (default `./outputs`). |
| `DEEPDRAFT_NO_SHOW=1` | Save figures without opening plot windows. Windows are also skipped automatically with a non-interactive matplotlib backend such as `Agg`. |

```bash
# Quick headless check of one module
DEEPDRAFT_FAST=1 MPLBACKEND=Agg python ensemble/xgboost_model.py

# Run every module in fast mode (what CI does)
pip install -e ".[all,dev]"
pytest tests/smoke -n auto
```

---

## How a module is organized

Every module follows the same layout, so once you have read one you can navigate any of them:

1. **Docstring:** what the model is, when to use it, assumptions, and tuning advice.
2. **Data:** a synthetic or bundled dataset, plus a commented snippet showing how to load your own.
3. **Configuration:** a dictionary of hyperparameters with typical ranges.
4. **Model, training, and evaluation** functions.
5. **Visualization** helpers.
6. **`main()`:** runs the full demo when you execute the file.

The [module standard](docs/MODULE_STANDARD.md) describes the walkthrough structure every module is moving toward.

---

## Catalog

### Classic ML (`classic/`)

| Module | Description |
|--------|-------------|
| `linear_regression.py` | Ordinary least squares with Ridge, Lasso, and ElasticNet variants |
| `logistic_regression.py` | Linear classification with regularization tuning and ROC analysis |
| `decision_tree.py` | Classification and regression trees with cost-complexity pruning |
| `random_forest.py` | Bagged trees with OOB scoring and permutation importance |
| `svm.py` | Support vector classification and regression with kernels |
| `knn.py` | k-nearest neighbors for classification and regression |
| `kmeans.py` | K-Means and MiniBatch K-Means with K selection |
| `naive_bayes.py` | Gaussian, Multinomial, Bernoulli, and Complement Naive Bayes |
| `dimensionality_reduction.py` | PCA, ICA, LDA, t-SNE, UMAP, and Kernel PCA |

### Ensembles (`ensemble/`)

| Module | Description |
|--------|-------------|
| `bagging.py` | Bootstrap aggregation over arbitrary base learners |
| `adaboost.py` | Adaptive boosting for classification and regression |
| `gradient_boosting.py` | scikit-learn gradient boosting and HistGradientBoosting |
| `xgboost_model.py` | XGBoost with early stopping and randomized search |
| `lightgbm_model.py` | LightGBM with leaf-wise growth and native categoricals |
| `stacking_voting.py` | Voting and stacked generalization |

### Time series (`time_series/`)

| Module | Description |
|--------|-------------|
| `arima.py` | ARIMA and SARIMA with Box-Jenkins diagnostics |
| `exponential_smoothing.py` | SES, Holt, Holt-Winters, and Theta |
| `prophet_model.py` | Prophet with holidays and changepoints |
| `n_beats.py` | N-BEATS (generic and interpretable) and N-HiTS |
| `tft.py` | A simplified Temporal Fusion Transformer with quantile outputs |

### Anomaly detection (`anomaly_detection/`)

| Module | Description |
|--------|-------------|
| `isolation_forest.py` | Isolation-based outlier scoring |
| `one_class_svm.py` | Kernel boundary around normal data |
| `dbscan.py` | Density-based noise points as anomalies |
| `autoencoder_anomaly.py` | Reconstruction error from dense and convolutional autoencoders |

### Bayesian methods (`bayesian/`)

| Module | Description |
|--------|-------------|
| `gaussian_process.py` | GP regression and classification with kernel comparison |
| `bayesian_optimization.py` | GP surrogate with EI, UCB, and PI acquisition |
| `gmm.py` | Gaussian mixtures and Bayesian GMM |
| `hmm.py` | Gaussian hidden Markov models for regime detection |
| `kalman_filter.py` | Kalman, extended, and unscented filters with RTS smoothing |
| `particle_filter.py` | Bootstrap particle filter (sequential Monte Carlo) |
| `mcmc.py` | Metropolis-Hastings, Hamiltonian Monte Carlo, slice sampling, diagnostics |

### Causal inference (`causal_inference/`)

| Module | Description |
|--------|-------------|
| `dml_estimation.py` | Double machine learning with cross-fitting |
| `treatment_effects.py` | T-, S-, and X-learners for heterogeneous effects |
| `causal_discovery.py` | PC algorithm and LiNGAM |

### Interpretability (`interpretability/`)

| Module | Description |
|--------|-------------|
| `shap_values.py` | Permutation-based Shapley value estimation |
| `lime_explainer.py` | Local surrogate explanations for tabular models |
| `integrated_gradients.py` | Path-integrated gradient attributions for neural networks |

### Quantitative finance (`quantitative/`)

| Module | Description |
|--------|-------------|
| `black_scholes.py` | Option pricing, Greeks, and implied volatility |
| `heston.py` | Stochastic volatility pricing and calibration |
| `garch.py` | Volatility modeling and Value at Risk |
| `monte_carlo.py` | Path simulation, VaR/CVaR, and exotic option pricing |
| `portfolio_optimization.py` | Mean-variance, maximum Sharpe, and risk parity portfolios |

### Reinforcement learning (`reinforcement_learning/`)

| Module | Description |
|--------|-------------|
| `dqn.py` | DQN with Double, Dueling, and prioritized replay variants |
| `actor_critic.py` | Advantage actor-critic with parallel environments |
| `ppo.py` | Proximal policy optimization with clipping and adaptive KL |
| `sac.py` | Soft actor-critic with automatic entropy tuning |
| `td3.py` | Twin delayed deep deterministic policy gradient |

### Multi-agent systems (`multi_agent_systems/`)

| Module | Description |
|--------|-------------|
| `qmix.py` | Monotonic value factorization for cooperative MARL |
| `react_agent.py` | Reasoning-and-acting tool-use loop with a mock LLM backend |

### Scientific ML (`scientific_ml/`)

| Module | Description |
|--------|-------------|
| `pinn.py` | Physics-informed neural networks for PDEs and inverse problems |
| `neural_ode.py` | Continuous-depth networks and latent ODEs |
| `deeponet.py` | Operator learning with branch and trunk networks |
| `fno.py` | Fourier neural operators |
| `hamiltonian_nn.py` | Energy-conserving dynamics from learned Hamiltonians |
| `sindy.py` | Sparse identification of governing equations |
| `symbolic_regression.py` | Genetic programming for closed-form expressions |

---

## Contributing

Contributions are welcome. Until a full contributing guide lands:

1. Read the [module standard](docs/MODULE_STANDARD.md) before adding or rewriting a module.
2. Install the development tools with `pip install -e ".[dev]"` and run `pre-commit install`.
3. Make sure `pytest tests/smoke -n auto` passes. A new module needs fast-mode settings (`deepdraft.core.fast_or`) so its demo finishes in seconds, and should save figures with `deepdraft.core.save_figure`.
4. Keep each pull request focused on one module or one cross-cutting change.

## License

[MIT](LICENSE)
