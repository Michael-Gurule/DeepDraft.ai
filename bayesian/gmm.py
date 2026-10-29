"""
Gaussian Mixture Model (GMM)
==============================
A probabilistic generative model representing data as a weighted sum of K
Gaussian components. Each component has its own mean and covariance.

Model:
    p(x) = Σ_k π_k · N(x | μ_k, Σ_k)

    π_k  = mixture weight (prior probability of component k)
    μ_k  = component mean
    Σ_k  = component covariance

Fitting via EM (Expectation-Maximization):
    E-step: compute responsibilities r_{ik} = P(z_i=k | x_i, θ)
    M-step: update π_k, μ_k, Σ_k using weighted MLE

    Converges to a local maximum of the log-likelihood.

Relationship to K-Means:
    K-Means is a hard-assignment GMM with spherical covariances and equal weights.
    GMM provides soft assignments (probabilistic cluster membership) and
    handles elliptical, rotated clusters.

Relationship to HMM:
    GMM models the marginal distribution with no temporal structure.
    HMM adds Markov transitions between mixture components over time.

Use Cases:
    - Density estimation and generative modeling
    - Soft clustering (probabilistic membership)
    - Anomaly detection (low likelihood = anomaly)
    - Market regime identification (multi-modal return distributions)
    - EM initialization for HMM (seed state means)
    - Speaker recognition, image segmentation

Fine-Tuning Recommendations:
    - n_components: select via BIC or AIC. BIC prefers parsimony; AIC allows
      more components. Plot both and look for the elbow.
    - covariance_type:
        'full'      — most flexible, can overfit with few samples
        'tied'      — all components share one covariance; stable
        'diag'      — per-component diagonal; good middle ground
        'spherical' — fastest; assumes isotropic clusters
    - n_init: 5–10 random restarts (avoid local optima in EM).
    - max_iter: 200 usually sufficient with tol=1e-3.
    - warm_start: False (default). True reuses previous fit; useful for
      incremental fitting or hyperparameter search.
    - Regularization: reg_covar (default 1e-6) prevents singular covariances.
      Increase to 1e-3 for small samples.
    - BayesianGaussianMixture: automatically infers number of active components.
      Use when K is unknown and you want a principled Bayesian prior on weights.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    roc_auc_score, average_precision_score, silhouette_score,
    adjusted_rand_score
)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_cluster_data(
    n_samples:   int = 1000,
    n_features:  int = 10,
    n_clusters:  int = 4,
    n_anomalies: int = 30,
):
    """
    Generate multi-cluster data with injected anomalies.

    Replace with your own dataset:
        import pandas as pd
        df = pd.read_csv("returns.csv")
        X = df.values.astype(np.float32)

    Returns:
        X:    all samples
        y:    labels (0=normal, 1=anomaly) — used only for evaluation
        X_normal: normal-only data for unsupervised fitting
    """
    X_normal, true_labels = make_blobs(n_samples=n_samples, n_features=n_features,
                                        centers=n_clusters, cluster_std=0.8,
                                        random_state=42)
    rng = np.random.RandomState(7)
    X_anomalies = rng.uniform(
        low=X_normal.min(axis=0) * 3,
        high=X_normal.max(axis=0) * 3,
        size=(n_anomalies, n_features),
    )
    X = np.vstack([X_normal, X_anomalies]).astype(np.float32)
    y = np.array([0] * n_samples + [1] * n_anomalies)
    return X, y, X_normal.astype(np.float32), true_labels


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

# Placeholder: tune n_components (K) and covariance_type
GMM_CONFIG = {
    "n_components":    4,       # Placeholder: select via BIC/AIC
    "covariance_type": "full",  # Placeholder: 'full', 'tied', 'diag', 'spherical'
    "n_init":          5,       # Placeholder: 5–10 random restarts
    "max_iter":        200,
    "tol":             1e-3,
    "reg_covar":       1e-6,    # Placeholder: increase to 1e-3 for small samples
    "random_state":    42,
}


def build_gmm(config: dict = None) -> GaussianMixture:
    """Build a GaussianMixture model with given config."""
    if config is None:
        config = GMM_CONFIG
    return GaussianMixture(**config)


# ─────────────────────────────────────────────
# MODEL SELECTION (BIC / AIC)
# ─────────────────────────────────────────────

def select_n_components(X_scaled: np.ndarray,
                         k_range: range = range(2, 12)) -> int:
    """
    Select optimal K via BIC and AIC.

    BIC = -2 log L + k log n     (penalizes complexity more)
    AIC = -2 log L + 2k          (allows more components)

    Returns the K with lowest BIC.
    """
    bics = []
    aics = []

    for k in k_range:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=3, random_state=42)
        gm.fit(X_scaled)
        bics.append(gm.bic(X_scaled))
        aics.append(gm.aic(X_scaled))
        print(f"  K={k:2d}  BIC={bics[-1]:.1f}  AIC={aics[-1]:.1f}")

    best_k_bic = list(k_range)[np.argmin(bics)]
    best_k_aic = list(k_range)[np.argmin(aics)]

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), bics, "o-", color="steelblue", label="BIC")
    plt.plot(list(k_range), aics, "s--", color="tomato",   label="AIC")
    plt.axvline(best_k_bic, color="steelblue", linestyle=":", alpha=0.7,
                label=f"Best BIC: K={best_k_bic}")
    plt.axvline(best_k_aic, color="tomato", linestyle=":", alpha=0.7,
                label=f"Best AIC: K={best_k_aic}")
    plt.xlabel("Number of Components K")
    plt.ylabel("Information Criterion")
    plt.title("GMM — Component Count Selection (BIC / AIC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gmm_bic_aic.png", dpi=150)
    plt.show()

    print(f"\n  Best K (BIC): {best_k_bic}  |  Best K (AIC): {best_k_aic}")
    return best_k_bic


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_clustering(model: GaussianMixture, X_scaled: np.ndarray,
                         true_labels: np.ndarray = None) -> dict:
    """
    Evaluate GMM clustering quality.

    Metrics:
        Log-likelihood: overall fit
        BIC/AIC: penalized fit (lower = better)
        Silhouette: cluster separation (higher = better)
        ARI: agreement with ground truth labels (if available)
    """
    labels   = model.predict(X_scaled)
    log_like = model.score(X_scaled) * len(X_scaled)
    bic      = model.bic(X_scaled)
    aic      = model.aic(X_scaled)
    sil      = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0

    print(f"  Log-likelihood: {log_like:.2f}")
    print(f"  BIC: {bic:.2f}  |  AIC: {aic:.2f}")
    print(f"  Silhouette:     {sil:.4f}")

    ari = None
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        print(f"  ARI (vs truth): {ari:.4f}")

    return {"log_likelihood": log_like, "bic": bic, "aic": aic,
            "silhouette": sil, "ari": ari}


def evaluate_anomaly_detection(model: GaussianMixture, scaler: StandardScaler,
                                X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate GMM as anomaly detector using log-likelihood scores.

    Low log-likelihood (very negative) → anomaly.
    We negate scores so higher = more anomalous.
    """
    X_scaled = scaler.transform(X)
    scores   = -model.score_samples(X_scaled)  # negate: higher = more anomalous

    contamination = y.mean()
    thresh        = np.percentile(scores, 100 * (1 - contamination))
    y_pred        = (scores > thresh).astype(int)

    auc = roc_auc_score(y, scores)
    ap  = average_precision_score(y, scores)

    from sklearn.metrics import confusion_matrix
    cm   = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}")
    return {"roc_auc": auc, "avg_precision": ap, "precision": prec, "recall": rec}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_clusters_2d(model: GaussianMixture, X: np.ndarray,
                     y: np.ndarray = None) -> None:
    """
    2D PCA projection showing GMM soft assignments and ellipses.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca      = PCA(n_components=2, random_state=42)
    X_2d     = pca.fit_transform(X_scaled) if X.shape[1] > 2 else X_scaled

    labels       = model.predict(X_2d if X.shape[1] <= 2 else
                                 scaler.transform(X))
    probs        = model.predict_proba(X_2d if X.shape[1] <= 2 else
                                       scaler.transform(X))

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(9, 7))
    for k in range(model.n_components):
        mask = labels == k
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[cmap(k)], s=10, alpha=0.5, label=f"Component {k}")

    if y is not None:
        anom_mask = y == 1
        plt.scatter(X_2d[anom_mask, 0], X_2d[anom_mask, 1],
                    c="red", s=40, alpha=0.9, marker="x", label="True Anomaly", zorder=5)

    plt.title("GMM — Cluster Assignments (PCA 2D)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig("gmm_clusters.png", dpi=150)
    plt.show()


def plot_anomaly_scores(model: GaussianMixture, scaler: StandardScaler,
                        X: np.ndarray, y: np.ndarray) -> None:
    """Histogram of log-likelihood anomaly scores."""
    scores = -model.score_samples(scaler.transform(X))
    plt.figure(figsize=(9, 4))
    plt.hist(scores[y == 0], bins=50, alpha=0.6, label="Normal",  color="steelblue")
    plt.hist(scores[y == 1], bins=50, alpha=0.6, label="Anomaly", color="tomato")
    thresh = np.percentile(scores, 100 * (1 - y.mean()))
    plt.axvline(thresh, color="black", linestyle="--", label=f"Threshold ({thresh:.3f})")
    plt.xlabel("Anomaly Score (negated log-likelihood)")
    plt.ylabel("Count")
    plt.title("GMM — Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gmm_scores.png", dpi=150)
    plt.show()


def plot_responsibilities(model: GaussianMixture, scaler: StandardScaler,
                           X: np.ndarray, n_plot: int = 200) -> None:
    """
    Stacked bar showing soft component responsibilities per sample.
    Shows the probabilistic nature of GMM vs. hard K-Means assignments.
    """
    X_scaled = scaler.transform(X[:n_plot])
    probs    = model.predict_proba(X_scaled)  # (n, K)

    cmap    = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 3))
    bottom  = np.zeros(n_plot)
    for k in range(model.n_components):
        ax.bar(range(n_plot), probs[:, k], bottom=bottom,
               color=cmap(k), label=f"Component {k}", width=1.0)
        bottom += probs[:, k]
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Responsibility")
    ax.set_title("GMM — Soft Component Responsibilities")
    ax.legend(loc="upper right", fontsize=7, ncol=model.n_components)
    plt.tight_layout()
    plt.savefig("gmm_responsibilities.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# BAYESIAN GMM (automatic K selection)
# ─────────────────────────────────────────────

def fit_bayesian_gmm(X_scaled: np.ndarray,
                     max_components: int = 10) -> BayesianGaussianMixture:
    """
    Variational Bayesian GMM — automatically determines active components.

    Sets n_components to an upper bound; the model shrinks unused components
    to near-zero weight via the Dirichlet concentration prior.

    Placeholder: tune weight_concentration_prior. Smaller → fewer active components.
    """
    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="full",
        weight_concentration_prior=1e-2,   # Placeholder: 1e-3 to 1.0
        n_init=3,
        max_iter=200,
        random_state=42,
    )
    bgmm.fit(X_scaled)
    active = np.sum(bgmm.weights_ > 0.01)
    print(f"  Bayesian GMM: {active} active components (of {max_components} max)")
    print(f"  Weights: {np.round(bgmm.weights_, 3)}")
    return bgmm


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Gaussian Mixture Model ===")

    # Placeholder: replace with your dataset
    X, y, X_normal, true_labels = generate_cluster_data()
    print(f"Dataset: {X.shape[0]} samples ({y.sum()} anomalies) | {X.shape[1]} features")

    scaler   = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    # Select K via BIC
    print("\nComponent selection (BIC/AIC):")
    best_k = select_n_components(X_normal_scaled)

    # Fit GMM
    config = {**GMM_CONFIG, "n_components": best_k}
    model  = build_gmm(config)
    model.fit(X_normal_scaled)

    print("\nClustering evaluation:")
    evaluate_clustering(model, X_normal_scaled, true_labels)

    print("\nAnomaly detection evaluation:")
    evaluate_anomaly_detection(model, scaler, X, y)

    # Bayesian GMM (auto K)
    print("\nBayesian GMM:")
    fit_bayesian_gmm(X_normal_scaled)

    # Visualizations
    plot_clusters_2d(model, X, y)
    plot_anomaly_scores(model, scaler, X, y)
    plot_responsibilities(model, scaler, X)


if __name__ == "__main__":
    main()
