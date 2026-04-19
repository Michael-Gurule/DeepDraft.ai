"""
Naive Bayes
===========
A family of probabilistic classifiers based on Bayes' theorem with the
"naive" assumption of conditional independence between features given the class.
Despite this strong assumption, it works surprisingly well in practice,
especially for text classification.

Variants:
    GaussianNB       — continuous features assumed to follow Gaussian distribution
    MultinomialNB    — discrete count features (e.g., word counts in documents)
    BernoulliNB      — binary features (feature present/absent)
    ComplementNB     — improved MultinomialNB for imbalanced text datasets

Use Cases:
    - Text classification: spam filtering, sentiment analysis, topic labeling
    - Real-time classification (very fast inference)
    - High-dimensional data with limited training samples
    - Baseline classifier for any classification problem
    - Probabilistic reasoning when calibrated probabilities matter

Advantages:
    - Extremely fast training and inference (O(n))
    - Works well with small training data
    - Naturally outputs calibrated probabilities
    - Handles high-dimensional data (e.g., bag-of-words with 100k features)
    - No hyperparameters for most variants

Limitations:
    - Independence assumption rarely holds — correlated features hurt performance
    - Gaussian NB assumes normal distribution (fails for skewed data)
    - Known to be a poor estimator of class probabilities (overconfident)
    - Cannot capture feature interactions

Fine-Tuning Recommendations:
    - GaussianNB: var_smoothing controls numerical stability; rarely needs tuning.
      Consider log-transforming skewed features before use.
    - MultinomialNB: alpha (Laplace smoothing) prevents zero probabilities for
      unseen words. alpha=1.0 (default) is standard; tune between 0.01 and 10.
    - BernoulliNB: alpha same as MultinomialNB. binarize threshold controls
      which values count as "present".
    - Feature selection: NB benefits more from removing irrelevant features
      than most models — use chi-squared or mutual information filtering.
    - For text: use TF-IDF with MultinomialNB for better results than raw counts.
    - ComplementNB often outperforms MultinomialNB on imbalanced text tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, fetch_20newsgroups
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_continuous_data(n_samples: int = 800, n_features: int = 8):
    """
    Generate synthetic continuous classification data for GaussianNB.

    GaussianNB is appropriate when features are (approximately) normally
    distributed. Log-transform heavily skewed features before use.

    Returns:
        X: feature matrix (continuous values)
        y: binary class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


def load_text_data(categories: list | None = None):
    """
    Load 20 Newsgroups text dataset for MultinomialNB / ComplementNB demo.

    The 20 Newsgroups dataset is the canonical benchmark for text classification.
    Subset to a few categories for speed.

    Replace with your own text data:
        texts = ["email body 1", "email body 2", ...]
        labels = [0, 1, ...]  # e.g., spam=1, not-spam=0

    Args:
        categories: list of newsgroup categories to load (None = all 20)

    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    if categories is None:
        categories = ["sci.space", "rec.sport.hockey", "talk.politics.misc", "comp.graphics"]

    train = fetch_20newsgroups(subset="train", categories=categories, remove=("headers",))
    test  = fetch_20newsgroups(subset="test",  categories=categories, remove=("headers",))

    return train.data, test.data, train.target, test.target


def load_binary_data(n_samples: int = 600, n_features: int = 10):
    """
    Generate synthetic binary feature data for BernoulliNB.

    BernoulliNB works on presence/absence of features.
    Common use case: bag-of-words with binary (present/absent) encoding.

    Returns:
        X: binary feature matrix (0/1 values)
        y: class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        random_state=42,
    )
    X = (X > 0).astype(float)  # binarize to 0/1
    return X, y


# ─────────────────────────────────────────────
# MODEL CONFIGURATIONS
# ─────────────────────────────────────────────

# Placeholder: adjust smoothing parameters per variant
GAUSSIAN_CONFIG = {
    "var_smoothing": 1e-9,   # numerical stability for variance; rarely needs tuning
}

MULTINOMIAL_CONFIG = {
    "alpha": 1.0,            # Laplace smoothing; try [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
}

BERNOULLI_CONFIG = {
    "alpha": 1.0,            # Laplace smoothing
    "binarize": 0.0,         # threshold for binarizing features; None = already binary
}

COMPLEMENT_CONFIG = {
    "alpha": 1.0,            # Laplace smoothing
    "norm": True,            # normalize feature weights; recommended for long documents
}


def build_text_pipeline(vectorizer_type: str = "tfidf",
                         classifier=None) -> Pipeline:
    """
    Build a text vectorizer → Naive Bayes pipeline.

    Text pipeline options:
        CountVectorizer  → MultinomialNB: uses raw term counts
        TfidfVectorizer  → MultinomialNB: weights rare terms higher (usually better)
        TfidfVectorizer  → ComplementNB:  best for imbalanced classes

    Fine-tuning tip: TF-IDF + ComplementNB often outperforms all other NB
    variants on imbalanced text classification tasks.

    Args:
        vectorizer_type: 'tfidf' or 'count'
        classifier:      NB classifier instance; defaults to MultinomialNB

    Returns:
        sklearn Pipeline
    """
    if classifier is None:
        classifier = MultinomialNB(**MULTINOMIAL_CONFIG)

    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=20000,      # limit vocab size for speed
            min_df=2,                # ignore terms appearing in fewer than 2 docs
            max_df=0.95,             # ignore terms appearing in > 95% of docs (stop words)
            sublinear_tf=True,       # use log(1 + tf) to dampen high frequencies
            ngram_range=(1, 2),      # unigrams + bigrams; tune to (1,1) for speed
        )
    else:
        vectorizer = CountVectorizer(max_features=20000, min_df=2, max_df=0.95)

    return Pipeline([
        ("vectorizer", vectorizer),
        ("model", classifier),
    ])


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    """
    Compute accuracy and classification report.

    For text classification with NB, pay special attention to per-class
    F1 scores — NB can perform very differently across categories.

    Args:
        model:  fitted pipeline or classifier
        X_test: test features (raw text or feature matrix)
        y_test: true labels

    Returns:
        dict of metrics
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc}


def cross_validate(pipeline: Pipeline, X, y, cv: int = 5) -> None:
    """
    K-fold cross-validation.

    Fine-tuning tip: NB models train so fast that you can afford
    10-fold CV. If performance is highly variable across folds,
    the dataset is likely too small for the number of features.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"\n{cv}-Fold CV Accuracy: {scores.round(4)}")
    print(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")


# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────

def tune_multinomial(X_train_text: list, y_train: np.ndarray) -> Pipeline:
    """
    Grid search over alpha (smoothing) and vectorizer settings.

    Alpha is the primary knob for MultinomialNB:
        alpha=0  → no smoothing (risk: zero probabilities for unseen words)
        alpha=1  → Laplace smoothing (standard)
        alpha>1  → more smoothing (similar to class prior)

    Fine-tuning tip: also search over n-gram range and max_features —
    these often have as much impact as alpha.

    Args:
        X_train_text: list of raw text strings
        y_train:      training labels

    Returns:
        best fitted pipeline
    """
    pipeline = build_text_pipeline()
    param_grid = {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__max_features": [10000, 20000, None],
        "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0],
    }
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train_text, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, title: str = "Naive Bayes") -> None:
    """Display confusion matrix."""
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
    plt.title(f"{title} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_confusion.png", dpi=150)
    plt.show()


def plot_top_features_per_class(pipeline: Pipeline, class_names: list,
                                 top_n: int = 10) -> None:
    """
    Show the top discriminative words/features for each class in a text NB model.

    The feature log-probability matrix (coef_ or feature_log_prob_) shows which
    words are most associated with each class.

    Most useful for: debugging, auditing, detecting bias in text classifiers.

    Args:
        pipeline:    fitted text pipeline with MultinomialNB or ComplementNB
        class_names: list of class name strings
        top_n:       number of top features to show per class
    """
    vectorizer = pipeline.named_steps["vectorizer"]
    clf = pipeline.named_steps["model"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    if not hasattr(clf, "feature_log_prob_"):
        print("Model does not expose feature_log_prob_. Skipping.")
        return

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        top_idx = np.argsort(clf.feature_log_prob_[i])[-top_n:][::-1]
        ax.barh(feature_names[top_idx][::-1], clf.feature_log_prob_[i][top_idx][::-1],
                color="steelblue", edgecolor="k")
        ax.set_title(f"Class: {class_name}")
        ax.set_xlabel("Log Probability")

    plt.suptitle("Top Discriminative Features per Class")
    plt.tight_layout()
    plt.savefig("naive_bayes_top_features.png", dpi=150)
    plt.show()


def compare_variants(X_test: np.ndarray, y_test: np.ndarray,
                      models: dict) -> None:
    """
    Compare accuracy of multiple NB variants in a bar chart.

    Args:
        X_test:  test features
        y_test:  true labels
        models:  dict mapping model name → fitted model
    """
    names = list(models.keys())
    accs  = [accuracy_score(y_test, m.predict(X_test)) for m in models.values()]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, accs, color="steelblue", edgecolor="k")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title("Naive Bayes Variant Comparison")
    plt.tight_layout()
    plt.savefig("naive_bayes_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── GaussianNB on continuous data ───────────────────────────────
    print("=== GaussianNB (Continuous Features) ===")
    X, y = load_continuous_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    gnb = GaussianNB(**GAUSSIAN_CONFIG)
    gnb.fit(X_train, y_train)
    evaluate(gnb, X_test, y_test)
    cross_validate(gnb, X, y)

    # ── MultinomialNB on text data ───────────────────────────────────
    print("\n=== MultinomialNB (Text Classification) ===")
    train_texts, test_texts, y_train_text, y_test_text = load_text_data()
    categories = ["sci.space", "rec.sport.hockey", "talk.politics.misc", "comp.graphics"]

    mnb_pipeline = build_text_pipeline(vectorizer_type="tfidf")
    mnb_pipeline.fit(train_texts, y_train_text)
    evaluate(mnb_pipeline, test_texts, y_test_text)

    print("\n=== Hyperparameter Tuning (MultinomialNB) ===")
    best_mnb = tune_multinomial(train_texts, y_train_text)
    print("\nBest MultinomialNB — Test Metrics:")
    evaluate(best_mnb, test_texts, y_test_text)

    # ── BernoulliNB on binary data ───────────────────────────────────
    print("\n=== BernoulliNB (Binary Features) ===")
    Xb, yb = load_binary_data()
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=42, stratify=yb
    )
    bnb = BernoulliNB(**BERNOULLI_CONFIG)
    bnb.fit(Xb_train, yb_train)
    evaluate(bnb, Xb_test, yb_test)

    # ── Variant comparison ──────────────────────────────────────────
    print("\n=== Variant Comparison (Continuous Features) ===")
    # Scale to [0, 1] for MultinomialNB compatibility
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        "GaussianNB":    GaussianNB().fit(X_train, y_train),
        "MultinomialNB": MultinomialNB(alpha=1.0).fit(X_train_scaled, y_train),
        "BernoulliNB":   BernoulliNB(alpha=1.0).fit(X_train_scaled, y_train),
    }
    compare_variants(X_test_scaled, y_test, models)

    # Text visualizations
    plot_confusion_matrix(mnb_pipeline, test_texts, y_test_text, title="MultinomialNB Text")
    plot_top_features_per_class(mnb_pipeline, categories[:4])


if __name__ == "__main__":
    main()
