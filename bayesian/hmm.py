"""
Hidden Markov Model (HMM)
==========================
A probabilistic sequence model with hidden (latent) states. At each time step,
the system occupies one hidden state and emits an observation. The underlying
states are not directly observed — they must be inferred.

Model components:
    π   = initial state distribution  (K,)
    A   = transition matrix            (K, K)  A[i,j] = P(s_t=j | s_{t-1}=i)
    B   = emission distribution        per state k: mean μ_k, covariance Σ_k (Gaussian)

Key algorithms:
    Forward-Backward: E-step — compute γ_t(k) = P(s_t=k | x_{1:T}, λ)
    Baum-Welch (EM): estimate π, A, B from unlabeled sequences
    Viterbi:         find most likely state sequence (argmax over all paths)
    Forward:         compute P(x_{1:T} | λ) for sequence likelihood

Use Cases:
    - Financial regime detection (bull/bear/sideways market states)
    - Time series segmentation (switching between operating modes)
    - Speech recognition (phoneme states)
    - Bioinformatics (gene/protein sequence analysis)
    - Activity recognition (sensor readings → walking/running/sitting)
    - Anomaly detection (low-likelihood sequences are anomalous)

Relationship to GMM:
    GMM: mixture of Gaussians with no temporal structure.
    HMM: GMM + Markov chain transitions over time.

Implementation note:
    We use hmmlearn.GaussianHMM which implements Gaussian emission distributions.
    For discrete observations: use hmmlearn.MultinomialHMM.
    For categorical time series: preprocess with label encoding.

Fine-Tuning Recommendations:
    - n_components (K): number of hidden states. Start with 2–5; use AIC/BIC
      to select. Plot log-likelihood vs K and pick the elbow.
    - covariance_type: 'full' (default) most flexible; 'diag' faster; 'tied'
      if states share covariance structure.
    - n_iter: EM iterations. 100 usually sufficient; use tol=1e-4 for convergence.
    - n_init: 1 (default). Multiple random restarts help avoid local optima.
    - Initialization: hmmlearn accepts manual startprob, transmat, means, covars.
      Good init speeds convergence — use KMeans to seed means.
    - Sequence length: each training sequence should be long enough to observe
      state transitions. Very short sequences → poor A estimation.
    - For online inference on new data: use model.score_samples() for
      forward-pass log-likelihoods per step.
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from deepdraft.core import fast_or, save_figure


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_regime_data(
    n_timesteps: int = 1000,
    n_features:  int = 2,
    seed:        int = 42,
) -> tuple:
    """
    Generate synthetic multi-regime time series data.

    Simulates 3 regimes (e.g., bull/bear/sideways market or normal/alert/critical):
        - Regime 0: low mean, low volatility
        - Regime 1: high mean, high volatility
        - Regime 2: near-zero mean, medium volatility

    Replace with real data:
        import pandas as pd
        df = pd.read_csv("market_returns.csv", index_col="date", parse_dates=True)
        X = df[["return", "volume_change"]].values.astype(np.float32)
        lengths = [len(X)]  # single sequence

    Returns:
        X:       observations (n_timesteps, n_features)
        true_states: ground truth latent states (n_timesteps,)
        lengths: list of sequence lengths (for hmmlearn batch fitting)
    """
    rng = np.random.RandomState(seed)

    # State definitions (mean, std)
    state_means = np.array([
        [-1.0, 0.5],    # regime 0: low mean, low vol
        [1.5,  1.5],    # regime 1: high mean, high vol
        [0.0,  0.8],    # regime 2: near-zero, medium vol
    ])
    state_stds = np.array([
        [0.3, 0.3],
        [0.8, 0.6],
        [0.5, 0.5],
    ])

    # Transition matrix
    trans = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.88, 0.07],
        [0.07, 0.05, 0.88],
    ])

    # Generate state sequence
    states = np.zeros(n_timesteps, dtype=int)
    states[0] = rng.choice(3, p=[0.33, 0.33, 0.34])
    for t in range(1, n_timesteps):
        states[t] = rng.choice(3, p=trans[states[t - 1]])

    # Generate observations
    X = np.zeros((n_timesteps, n_features), dtype=np.float32)
    for t in range(n_timesteps):
        s = states[t]
        X[t] = rng.normal(state_means[s, :n_features],
                           state_stds[s, :n_features])

    return X, states, [n_timesteps]


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune n_components (K) and covariance_type
HMM_CONFIG = {
    "n_components":    3,       # Placeholder: number of hidden states; select via AIC/BIC
    "covariance_type": "full",  # Placeholder: 'full', 'diag', 'tied', 'spherical'
    "n_iter":          fast_or(100, 15),  # Placeholder: EM iterations
    "tol":             1e-4,    # convergence threshold
    "random_state":    42,
}


def build_hmm(config: dict = None) -> hmm.GaussianHMM:
    """
    Build a Gaussian HMM.

    Args:
        config: HMM_CONFIG dict

    Returns:
        Unfitted GaussianHMM
    """
    if config is None:
        config = HMM_CONFIG
    return hmm.GaussianHMM(
        n_components=config["n_components"],
        covariance_type=config["covariance_type"],
        n_iter=config["n_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
    )


# ─────────────────────────────────────────────
# MODEL SELECTION (AIC / BIC)
# ─────────────────────────────────────────────

def select_n_states(X: np.ndarray, lengths: list,
                    k_range: range = range(2, fast_or(8, 5))) -> int:
    """
    Select optimal number of hidden states via BIC.

    BIC = -2 · log L + k · log n
    where k = number of free parameters, n = number of observations.

    Fit HMMs with different K, pick K with lowest BIC.
    """
    n_obs     = sum(lengths)
    n_feat    = X.shape[1]
    bics      = []
    log_likes = []

    for K in k_range:
        try:
            model = hmm.GaussianHMM(n_components=K, covariance_type="diag",
                                     n_iter=fast_or(100, 15), random_state=42)
            model.fit(X, lengths)
            ll = model.score(X, lengths)
            # Free parameters: K-1 (initial) + K(K-1) (trans) + K*d (means) + K*d (vars)
            n_params = (K - 1) + K * (K - 1) + K * n_feat + K * n_feat
            bic = -2 * ll + n_params * np.log(n_obs)
            bics.append(bic)
            log_likes.append(ll)
            print(f"  K={K}  LogL={ll:.2f}  BIC={bic:.2f}")
        except Exception as e:
            print(f"  K={K}  Failed: {e}")
            bics.append(np.inf)
            log_likes.append(-np.inf)

    best_k = list(k_range)[np.argmin(bics)]

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), bics, "o-", color="steelblue", label="BIC")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    plt.xlabel("Number of States K")
    plt.ylabel("BIC")
    plt.title("HMM — State Count Selection (BIC)")
    plt.legend()
    plt.tight_layout()
    save_figure("hmm_bic.png", dpi=150)

    print(f"\n  Best K by BIC: {best_k}")
    return best_k


# ─────────────────────────────────────────────
# DECODING & SCORING
# ─────────────────────────────────────────────

def decode_states(model: hmm.GaussianHMM, X: np.ndarray,
                  lengths: list) -> np.ndarray:
    """
    Viterbi decoding: find most likely hidden state sequence.

    Returns:
        state_sequence: (n_timesteps,) array of inferred state indices
    """
    log_prob, states = model.decode(X, lengths=lengths, algorithm="viterbi")
    print(f"  Viterbi log-probability: {log_prob:.4f}")
    return states


def anomaly_scores(model: hmm.GaussianHMM, X: np.ndarray,
                   lengths: list) -> np.ndarray:
    """
    Per-step log-likelihood as anomaly score.

    score_samples() returns log P(x_t | x_{1:t-1}, model) using the forward algorithm.
    Low (very negative) log-likelihood → anomalous step.

    Negate so higher = more anomalous.
    """
    _, posteriors = model.score_samples(X, lengths)
    # Per-step log-likelihood approximation: sum over states of log P(x|state) * posterior
    log_likes_per_step = np.zeros(len(X))
    for k in range(model.n_components):
        # Gaussian log-likelihood for state k
        diff = X - model.means_[k]
        cov  = model.covars_[k]
        if model.covariance_type == "diag":
            log_p = -0.5 * np.sum(diff**2 / cov + np.log(2 * np.pi * cov), axis=1)
        else:
            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)
            log_p = -0.5 * (np.einsum("ij,jk,ik->i", diff, inv_cov, diff) +
                            logdet + X.shape[1] * np.log(2 * np.pi))
        log_likes_per_step += posteriors[:, k] * log_p
    return -log_likes_per_step  # negate: higher = more anomalous


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_state_sequence(X: np.ndarray, inferred_states: np.ndarray,
                        true_states: np.ndarray = None) -> None:
    """
    Plot time series colored by inferred (and optionally true) hidden state.
    """
    n = len(X)
    t = np.arange(n)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(X.shape[1] + 1, 1, figsize=(14, 3 + 2 * X.shape[1]),
                              sharex=True)

    for i, ax in enumerate(axes[:-1]):
        ax.plot(t, X[:, i], linewidth=0.7, alpha=0.8, color="steelblue")
        ax.set_ylabel(f"Feature {i}")

    # State color bar
    ax = axes[-1]
    for state in range(max(inferred_states) + 1):
        mask = inferred_states == state
        ax.fill_between(t, 0, 1, where=mask, alpha=0.7,
                        color=cmap(state), label=f"State {state}")
    if true_states is not None:
        for state in range(max(true_states) + 1):
            mask = true_states == state
            ax.fill_between(t, -1, 0, where=mask, alpha=0.7,
                            color=cmap(state))
        ax.set_yticks([-0.5, 0.5])
        ax.set_yticklabels(["True", "Inferred"])
    else:
        ax.set_ylim(0, 1)
    ax.set_ylabel("States")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time Step")

    plt.suptitle("HMM — Inferred State Sequence", y=1.01)
    plt.tight_layout()
    save_figure("hmm_states.png", dpi=150)


def plot_transition_matrix(model: hmm.GaussianHMM) -> None:
    """Heatmap of learned transition matrix."""
    A = model.transmat_
    plt.figure(figsize=(5, 4))
    im = plt.imshow(A, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, label="Transition Probability")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=10)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title("HMM — Learned Transition Matrix")
    plt.tight_layout()
    save_figure("hmm_transition.png", dpi=150)


def plot_emission_distributions(model: hmm.GaussianHMM, X: np.ndarray) -> None:
    """Plot emission distributions (means ± 2σ) overlaid on data."""
    if X.shape[1] != 1:
        print("  plot_emission_distributions: only shown for 1D data")
        return
    x_range = np.linspace(X.min() - 1, X.max() + 1, 300)
    plt.figure(figsize=(9, 4))
    cmap = plt.get_cmap("tab10")
    for k in range(model.n_components):
        mu = model.means_[k, 0]
        if model.covariance_type == "diag":
            std = np.sqrt(model.covars_[k, 0])
        elif model.covariance_type == "full":
            std = np.sqrt(model.covars_[k, 0, 0])
        else:
            std = np.sqrt(model.covars_[k])
        plt.plot(x_range, norm.pdf(x_range, mu, std),
                 color=cmap(k), linewidth=2, label=f"State {k}")
    plt.hist(X.ravel(), bins=50, density=True, alpha=0.3, color="gray")
    plt.xlabel("Observation")
    plt.ylabel("Density")
    plt.title("HMM — Emission Distributions")
    plt.legend()
    plt.tight_layout()
    save_figure("hmm_emissions.png", dpi=150)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Hidden Markov Model — Regime Detection ===")

    # Placeholder: replace with your time series data
    X, true_states, lengths = generate_regime_data(n_timesteps=fast_or(1000, 300), n_features=2)
    print(f"Sequence length: {X.shape[0]} steps | {X.shape[1]} features")
    print(f"True state distribution: {np.bincount(true_states)}")

    # Select number of states via BIC
    print("\nState selection (BIC):")
    best_k = select_n_states(X, lengths)

    # Fit model with best K
    config = {**HMM_CONFIG, "n_components": best_k}
    model = build_hmm(config)
    model.fit(X, lengths)
    print(f"\nFitted HMM | LogL: {model.score(X, lengths):.4f}")

    # Decode states
    inferred = decode_states(model, X, lengths)
    print(f"Inferred state distribution: {np.bincount(inferred)}")

    # Anomaly scoring
    scores = anomaly_scores(model, X, lengths)
    print(f"Anomaly score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Visualizations
    plot_state_sequence(X, inferred, true_states)
    plot_transition_matrix(model)
    plot_emission_distributions(model, X[:, :1])


if __name__ == "__main__":
    main()
