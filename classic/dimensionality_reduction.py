"""
Dimensionality Reduction
========================

Techniques for projecting high-dimensional data into lower-dimensional spaces,
preserving structure relevant to the task.

Methods covered:
  PCA  — Principal Component Analysis (linear, global variance)
  ICA  — Independent Component Analysis (non-Gaussian sources)
  LDA  — Linear Discriminant Analysis (supervised, maximizes class separation)
  t-SNE — t-distributed Stochastic Neighbor Embedding (nonlinear, local structure)
  UMAP — Uniform Manifold Approximation and Projection (nonlinear, global + local)
  Kernel PCA — PCA in an RKHS (nonlinear via kernel trick)
  Truncated SVD — PCA for sparse data (TF-IDF matrices, no centering)
  Factor Analysis — probabilistic generative model; latent + noise decomposition

PCA math:
  X = U Σ V^T   (SVD)
  Components: V  (directions of maximum variance)
  Scores:     XV (projected data)
  Variance explained by k components: Σ_{i=1}^k σ_i² / Σ_j σ_j²

t-SNE key idea:
  In high-D: p_{ij} ∝ Gaussian similarity between x_i and x_j
  In low-D:  q_{ij} ∝ Student-t similarity between y_i and y_j (heavier tail)
  Minimize KL(P||Q) via gradient descent.
  Perplexity ≈ effective number of neighbors; not linear — use only for visualization.

UMAP key idea:
  Builds a fuzzy topological graph in high-D; optimizes low-D representation
  to match the graph. Preserves both local and global structure.
  Faster than t-SNE and supports transform() for new data.

Fine-tuning recommendations:
  PCA:
    - n_components (#Placeholder): choose by explained variance threshold (e.g., 95%)
    - whiten (#Placeholder: False): True normalizes components to unit variance (useful for SVM/GMM)
  t-SNE:
    - perplexity (#Placeholder: 30): effective neighbors (5–50); lower = local, higher = global
    - learning_rate (#Placeholder: "auto"): use "auto" in sklearn; or 10–1000
    - n_iter (#Placeholder: 1000): more = better convergence
  UMAP:
    - n_neighbors (#Placeholder: 15): local vs global balance (5–200)
    - min_dist (#Placeholder: 0.1): tightness of embedding (0.0–0.99)
    - metric (#Placeholder: "euclidean"): "cosine" for text, "euclidean" for numeric
  LDA:
    - n_components (#Placeholder): max is min(n_classes-1, n_features)
    - solver (#Placeholder: "svd"): "lsqr" + "shrinkage" for regularized LDA

References:
  Pearson (1901) — PCA
  Hyvarinen & Oja (2000) — Independent Component Analysis
  Fisher (1936) — LDA
  van der Maaten & Hinton (2008) — Visualizing Data using t-SNE
  McInnes et al. (2018) — UMAP: Uniform Manifold Approximation and Projection
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_digits, load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import (
    PCA, FastICA, KernelPCA, TruncatedSVD, FactorAnalysis,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "pca_n_components":     50,      # # Placeholder: or float for variance threshold
    "pca_whiten":           False,   # # Placeholder:
    "ica_n_components":     10,      # # Placeholder:
    "lda_n_components":     None,    # # Placeholder: None = max (n_classes - 1)
    "tsne_perplexity":      30,      # # Placeholder:
    "tsne_n_iter":          1000,    # # Placeholder:
    "tsne_n_components":    2,       # # Placeholder:
    "umap_n_neighbors":     15,      # # Placeholder:
    "umap_min_dist":        0.1,     # # Placeholder:
    "umap_n_components":    2,       # # Placeholder:
    "kpca_kernel":          "rbf",   # # Placeholder: "rbf" | "poly" | "cosine" | "sigmoid"
    "kpca_gamma":           None,    # # Placeholder: None = 1/n_features
    "seed":                 42,
}


# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────

def fit_pca(X: np.ndarray, cfg: dict) -> tuple[PCA, np.ndarray]:
    """
    Fit PCA. If n_components is a float (0 < n < 1), sklearn selects the
    minimum number of components to explain that fraction of variance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=cfg["pca_n_components"], whiten=cfg["pca_whiten"],
              random_state=cfg["seed"])
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, scaler


def explained_variance_plot(pca: PCA, threshold: float = 0.95):
    """Scree plot + cumulative explained variance."""
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_thresh = int(np.searchsorted(cum_var, threshold)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PCA — Explained Variance", fontsize=12)

    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].plot(cum_var, color="steelblue", lw=2)
    axes[1].axhline(threshold, color="red", linestyle="--",
                    label=f"{threshold*100:.0f}% threshold")
    axes[1].axvline(n_thresh, color="orange", linestyle="--",
                    label=f"{n_thresh} components")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title(f"Cumulative Variance ({n_thresh} comps → {threshold*100:.0f}%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_explained_variance.png", dpi=150)
    plt.show()
    print(f"Components needed for {threshold*100:.0f}% variance: {n_thresh}")
    return n_thresh


def pca_as_preprocessing(X_train, y_train, X_test, y_test,
                          n_components_list: list = None) -> dict:
    """
    Evaluate classification accuracy with PCA preprocessing.
    Shows the trade-off between dimensionality and downstream task performance.
    """
    n_components_list = n_components_list or [5, 10, 20, 40, None]  # # Placeholder:
    results = {}
    for nc in n_components_list:
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=nc, random_state=42) if nc is not None else PCA(),
            LogisticRegression(max_iter=500),
        )
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        results[nc] = {"mean": scores.mean(), "std": scores.std()}
        label = f"n={nc}" if nc else "all"
        print(f"  PCA({label:>4}): CV Acc = {scores.mean():.4f} ± {scores.std():.4f}")
    return results


# ─────────────────────────────────────────────
# ICA
# ─────────────────────────────────────────────

def fit_ica(X: np.ndarray, cfg: dict) -> tuple[FastICA, np.ndarray]:
    """
    FastICA separates independent non-Gaussian signals.
    Classic use: blind source separation (cocktail party problem).
    Unlike PCA, components are statistically independent (not just uncorrelated).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ica = FastICA(n_components=cfg["ica_n_components"], random_state=cfg["seed"],
                  max_iter=500, tol=0.01)
    X_ica = ica.fit_transform(X_scaled)
    return ica, X_ica


def ica_cocktail_party_demo(n_samples: int = 2000, seed: int = 42) -> dict:
    """
    Classic ICA demo: 3 independent sources mixed by unknown matrix A.
    ICA recovers original sources from mixed observations.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8, n_samples)

    # Independent sources
    s1 = np.sin(2 * t)              # sinusoid
    s2 = np.sign(np.sin(3 * t))    # square wave
    s3 = rng.standard_normal(n_samples) * 0.5  # Gaussian noise

    S = np.stack([s1, s2, s3], axis=1)
    S /= S.std(axis=0)

    # Random mixing matrix
    A = rng.uniform(size=(3, 3))
    X_mixed = S @ A.T

    ica = FastICA(n_components=3, random_state=seed)
    S_recovered = ica.fit_transform(X_mixed)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("ICA — Cocktail Party Source Separation", fontsize=12)
    for i, (src, rec, name) in enumerate(zip(S.T, S_recovered.T,
                                              ["Sin", "Square", "Noise"])):
        axes[i].plot(t[:300], src[:300], "k-", lw=1, alpha=0.5, label=f"True {name}")
        axes[i].plot(t[:300], rec[:300] / rec[:300].std(), "r-", lw=1.2,
                     label="ICA recovered")
        axes[i].legend(fontsize=8); axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig("ica_cocktail_party.png", dpi=150)
    plt.show()
    print("Saved: ica_cocktail_party.png")
    return {"S_true": S, "S_recovered": S_recovered, "A": A}


# ─────────────────────────────────────────────
# LDA
# ─────────────────────────────────────────────

def fit_lda(X: np.ndarray, y: np.ndarray, cfg: dict) -> tuple[LinearDiscriminantAnalysis, np.ndarray]:
    """
    Supervised dimensionality reduction. Finds directions that maximize
    between-class variance / within-class variance (Fisher criterion).
    Max components = n_classes - 1.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=cfg["lda_n_components"],
                                      solver="svd")
    X_lda = lda.fit_transform(X_scaled, y)
    print(f"LDA: {X_scaled.shape[1]} → {X_lda.shape[1]} dimensions")
    print(f"  Explained variance ratio: {lda.explained_variance_ratio_}")
    return lda, X_lda


# ─────────────────────────────────────────────
# t-SNE
# ─────────────────────────────────────────────

def fit_tsne(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    t-SNE: nonlinear visualization (2D/3D only).
    IMPORTANT: t-SNE has no transform() — must refit for new data.
    Run PCA first (50 dims) when X has many features to speed up and stabilize.
    """
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=cfg["seed"]).fit_transform(X)
    tsne = TSNE(
        n_components=cfg["tsne_n_components"],
        perplexity=cfg["tsne_perplexity"],
        learning_rate="auto",
        n_iter=cfg["tsne_n_iter"],
        random_state=cfg["seed"],
        n_jobs=-1,
    )
    return tsne.fit_transform(X)


# ─────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────

def fit_umap(X: np.ndarray, y=None, cfg: dict = None) -> np.ndarray:
    """
    UMAP: faster than t-SNE, supports transform() for new data.
    y can be passed for supervised UMAP (better class separation).
    Install: pip install umap-learn
    """
    if not HAS_UMAP:
        print("UMAP not installed: pip install umap-learn")
        return np.zeros((len(X), 2))
    cfg = cfg or CONFIG
    reducer = umap.UMAP(
        n_components=cfg["umap_n_components"],
        n_neighbors=cfg["umap_n_neighbors"],
        min_dist=cfg["umap_min_dist"],
        random_state=cfg["seed"],
    )
    return reducer.fit_transform(X, y=y)


# ─────────────────────────────────────────────
# Kernel PCA
# ─────────────────────────────────────────────

def fit_kernel_pca(X: np.ndarray, cfg: dict) -> tuple[KernelPCA, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kpca = KernelPCA(
        n_components=2,
        kernel=cfg["kpca_kernel"],
        gamma=cfg["kpca_gamma"],
        fit_inverse_transform=True,
        random_state=cfg["seed"],
    )
    X_kpca = kpca.fit_transform(X_scaled)
    return kpca, X_kpca


# ─────────────────────────────────────────────
# Reconstruction Error
# ─────────────────────────────────────────────

def reconstruction_error(
    X: np.ndarray,
    method: str = "pca",
    n_components_range: list = None,
) -> dict:
    """
    Plot reconstruction error vs number of components.
    For PCA: error = ||X - X_reconstructed||_F²
    Useful for choosing dimensionality.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components_range = n_components_range or list(range(1, min(X.shape[1] + 1, 51)))

    errors = []
    for nc in n_components_range:
        if method == "pca":
            model = PCA(n_components=nc, random_state=42)
        elif method == "factor":
            model = FactorAnalysis(n_components=nc, random_state=42)
        else:
            raise ValueError(method)
        X_red = model.fit_transform(X_scaled)
        X_rec = model.inverse_transform(X_red)
        err = np.mean((X_scaled - X_rec) ** 2)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_components_range, errors, color="steelblue", lw=2)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Mean Squared Reconstruction Error")
    ax.set_title(f"{method.upper()} Reconstruction Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{method}_reconstruction_error.png", dpi=150)
    plt.show()
    return {"n_components": n_components_range, "errors": errors}


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_2d_embeddings(embeddings: dict, y: np.ndarray, title: str = ""):
    """Grid of 2D scatter plots for each embedding method."""
    methods = list(embeddings.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Dimensionality Reduction — {title}", fontsize=12)

    cmap = plt.cm.get_cmap("tab10", len(np.unique(y)))
    for ax, name in zip(axes, methods):
        Z = embeddings[name]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, s=10, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.2)

    plt.colorbar(sc, ax=axes[-1], label="Class")
    plt.tight_layout()
    fname = f"dim_reduction_{title.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def plot_pca_components(pca: PCA, feature_names: list = None, top_k: int = 6):
    """Visualize first top_k PCA component loadings."""
    n_comp = min(top_k, pca.components_.shape[0])
    fig, axes = plt.subplots(1, n_comp, figsize=(3 * n_comp, 3))
    fig.suptitle("PCA Component Loadings", fontsize=11)
    if n_comp == 1:
        axes = [axes]
    names = feature_names or [f"f{i}" for i in range(pca.components_.shape[1])]
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        order = np.argsort(np.abs(loadings))[::-1][:15]
        ax.barh([names[j] for j in order[::-1]], loadings[order[::-1]],
                color=["steelblue" if v > 0 else "darkorange" for v in loadings[order[::-1]]])
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"PC{i+1}")
    plt.tight_layout()
    plt.savefig("pca_loadings.png", dpi=150)
    plt.show()
    print("Saved: pca_loadings.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    np.random.seed(CONFIG["seed"])
    print(f"Config: {CONFIG}\n")

    # Load Digits dataset (8x8 images, 10 classes, 64 features)
    digits = load_digits()
    X, y = digits.data, digits.target
    feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
    print(f"Dataset: Digits | shape={X.shape} | n_classes={len(np.unique(y))}")

    # ── PCA ────────────────────────────────────────────────────────
    print("\n=== PCA ===")
    pca_full = PCA(random_state=CONFIG["seed"])
    pca_full.fit(StandardScaler().fit_transform(X))
    n_95 = explained_variance_plot(pca_full, threshold=0.95)

    pca, X_pca, scaler = fit_pca(X, {**CONFIG, "pca_n_components": 2})
    plot_pca_components(pca_full, feature_names=feature_names)
    reconstruction_error(X, method="pca", n_components_range=list(range(1, 40)))
    pca_as_preprocessing(X, y, X, y)

    # ── LDA ────────────────────────────────────────────────────────
    print("\n=== LDA ===")
    lda, X_lda = fit_lda(X, y, CONFIG)

    # ── ICA ────────────────────────────────────────────────────────
    print("\n=== ICA — Cocktail Party Demo ===")
    ica_cocktail_party_demo(seed=CONFIG["seed"])
    ica, X_ica = fit_ica(X, {**CONFIG, "ica_n_components": 2})

    # ── t-SNE ──────────────────────────────────────────────────────
    print("\n=== t-SNE ===")
    X_tsne = fit_tsne(X, CONFIG)

    # ── Kernel PCA ─────────────────────────────────────────────────
    print("\n=== Kernel PCA ===")
    kpca, X_kpca = fit_kernel_pca(X, CONFIG)

    # ── UMAP ───────────────────────────────────────────────────────
    print(f"\n=== UMAP {'(installed)' if HAS_UMAP else '(not installed — pip install umap-learn)'} ===")
    X_umap = fit_umap(StandardScaler().fit_transform(X), y=y, cfg=CONFIG)

    # ── Compare all embeddings ─────────────────────────────────────
    embeddings = {
        "PCA":        X_pca,
        "LDA":        X_lda[:, :2] if X_lda.shape[1] >= 2 else np.hstack([X_lda, X_lda]),
        "t-SNE":      X_tsne,
        "Kernel PCA": X_kpca,
        "ICA":        X_ica,
    }
    if HAS_UMAP:
        embeddings["UMAP"] = X_umap

    plot_2d_embeddings(embeddings, y, title="Digits Dataset")


if __name__ == "__main__":
    main()
