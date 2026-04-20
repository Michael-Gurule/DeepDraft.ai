"""
Causal Discovery
=================
Algorithms for learning causal graph structure from observational data —
recovering directed acyclic graphs (DAGs) that represent causal relationships
among variables without running experiments.

Key distinction: correlation ≠ causation.
    Causal discovery attempts to identify which correlations reflect causal
    mechanisms rather than confounding or selection bias.

Constraint-based approach (PC algorithm):
    1. Start with a complete undirected graph over variables.
    2. Remove edges by testing conditional independence (CI tests).
       X ⊥ Y | Z  ⟹  remove edge X-Y (given separating set Z).
    3. Orient edges by detecting v-structures (colliders): X → Z ← Y
       where X ⊥ Y but X ⊬ Y | Z.
    4. Apply Meek orientation rules to propagate orientations.
    Assumes: causal Markov condition, faithfulness, no hidden confounders (CFC).

Score-based approach (GES — Greedy Equivalence Search):
    1. Forward phase: greedily add edges that maximize a score (BIC, BDeu).
    2. Backward phase: greedily remove edges.
    3. Returns a Completed Partially Directed DAG (CPDAG) — the Markov
       equivalence class of the true DAG.

Linear non-Gaussian (LiNGAM):
    Assumes: X = B X + e, where B is a strictly lower-triangular matrix
    and noise e_i are non-Gaussian, independent.
    Uses ICA to recover B directly — fully identifies the DAG (not just CPDAG).
    Key result (Shimizu 2006): non-Gaussianity is sufficient for full identifiability.

Faithfulness assumption (shared by PC and GES):
    Every conditional independence in the distribution is reflected in the graph
    via d-separation. Violated when true path cancellations make dependent
    variables appear independent.

Use Cases:
    - Genomics: gene regulatory network reconstruction
    - Economics: identifying policy levers in macroeconomic models
    - Medicine: discovering disease pathways from EHR data
    - Climate science: causal links between climate variables

Advantages:
    - Learns structure without randomized experiments
    - PC and GES have polynomial complexity (in practice)
    - LiNGAM fully identifies the DAG under non-Gaussianity

Limitations:
    - PC/GES return equivalence classes, not unique DAGs
    - All assume no hidden confounders (CFC) — violated in most real data
    - Scale poorly beyond ~50 variables
    - CI tests have low power in high dimensions

Fine-Tuning Recommendations:
    1. alpha (CI test significance): 0.01–0.05. Lower → sparser graphs (fewer false edges).
    2. CI test: Fisher-Z for Gaussian data; KCI or HSIC for nonlinear/non-Gaussian.
    3. max_cond_set: cap the conditioning set size to speed up PC at the cost of completeness.
    4. Score (GES): BIC works well for continuous data; BDeu for discrete.
    5. LiNGAM: check non-Gaussianity assumption (Shapiro-Wilk test) before applying.
    6. Bootstrap: run discovery on N bootstrap samples; keep only edges in >50% of runs.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def generate_linear_dag_data(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Generate observational data from a known linear DAG.

    True DAG structure (5 variables):
        X1 → X3 → X5
        X2 → X3
        X2 → X4 → X5
        X1 → X4

    Noise: mixed Gaussian + Laplace to support LiNGAM recovery.

    Args:
        n_samples: number of observations
        seed:      random seed

    Returns:
        X:         (n_samples, 5) data matrix
        true_graph: dict with edge list and adjacency matrix
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    # Structural equations with non-Gaussian noise
    X1 = rng.laplace(0, 1, n)
    X2 = rng.laplace(0, 1, n)
    X3 = 0.8 * X1 + 0.6 * X2 + rng.normal(0, 0.5, n)
    X4 = 0.5 * X1 + 0.7 * X2 + rng.laplace(0, 0.5, n)
    X5 = 0.9 * X3 + 0.4 * X4 + rng.normal(0, 0.3, n)

    X = np.column_stack([X1, X2, X3, X4, X5])
    X = StandardScaler().fit_transform(X)

    true_edges = [(0, 2), (1, 2), (1, 3), (0, 3), (2, 4), (3, 4)]
    adj = np.zeros((5, 5), dtype=int)
    for i, j in true_edges:
        adj[i, j] = 1

    true_graph = {"edges": true_edges, "adjacency": adj, "n_nodes": 5}
    return X, true_graph


# ─────────────────────────────────────────────
# CONDITIONAL INDEPENDENCE TESTS
# ─────────────────────────────────────────────

def partial_correlation_test(
    data: np.ndarray,
    x: int,
    y: int,
    cond_set: list[int],
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """
    Fisher-Z partial correlation test for conditional independence.

    Tests H0: X ⊥ Y | Z via partial correlation under Gaussian assumption.
    Partial correlation is computed by regressing out Z from both X and Y.

    Test statistic: z = 0.5 * log((1+r)/(1-r)) * sqrt(n - |Z| - 3)
    where r is the partial correlation and |Z| is the size of the conditioning set.

    Args:
        data:     (n_samples, n_vars) data matrix
        x:        index of variable X
        y:        index of variable Y
        cond_set: indices of conditioning variables Z
        alpha:    significance level

    Returns:
        is_independent: True if X ⊥ Y | Z at level alpha
        p_value:        p-value of the test
    """
    n = data.shape[0]

    if not cond_set:
        # Zero-order: standard Pearson correlation test
        r, p = stats.pearsonr(data[:, x], data[:, y])
        return p > alpha, p

    # Partial correlation via residuals
    Z = data[:, cond_set]
    reg = LinearRegression()
    res_x = data[:, x] - reg.fit(Z, data[:, x]).predict(Z)
    res_y = data[:, y] - reg.fit(Z, data[:, y]).predict(Z)
    r, _ = stats.pearsonr(res_x, res_y)
    r = np.clip(r, -0.9999, 0.9999)

    # Fisher-Z transform
    df = n - len(cond_set) - 3
    if df < 1:
        return False, 1.0
    z_stat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(df)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return p_value > alpha, p_value


# ─────────────────────────────────────────────
# PC ALGORITHM
# ─────────────────────────────────────────────

class PCAlgorithm:
    """
    PC (Peter-Clark) constraint-based causal discovery algorithm.

    Recovers the Markov Equivalence Class (CPDAG) of the true DAG.
    A CPDAG contains both directed and undirected edges:
        - Directed edges are common to all DAGs in the class
        - Undirected edges can be oriented either way consistently

    Implementation:
        Phase 1 (skeleton): remove edges via CI tests with increasing |Z|
        Phase 2 (v-structures): orient X-Z-Y as X→Z←Y if X⊬Z and Y⊬Z
                                  but X⊥Y | Z\{...} for some set not containing Z
        Phase 3 (Meek rules): propagate orientations to avoid new v-structures
                               or directed cycles

    Args:
        alpha:        significance level for CI tests
        max_cond_set: maximum conditioning set size (cap for scalability)
    """

    def __init__(
        self,
        alpha: float = 0.05,        # Placeholder: 0.01 for sparser, 0.1 for denser graphs
        max_cond_set: int = 4,      # Placeholder: increase for more accurate (slower) results
    ):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.sep_sets: dict = {}    # sep_sets[(i,j)] = separating set for i,j
        self.skeleton_: np.ndarray | None = None
        self.cpdag_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAlgorithm":
        """
        Run PC algorithm on data X.

        Args:
            X: (n_samples, n_vars) data matrix (standardized recommended)

        Returns:
            self (fitted)
        """
        n_vars = X.shape[1]
        # Phase 1: skeleton
        skeleton = self._learn_skeleton(X, n_vars)
        self.skeleton_ = skeleton.copy()
        # Phase 2 + 3: orientation
        self.cpdag_ = self._orient_edges(skeleton, n_vars)
        return self

    def _learn_skeleton(self, X: np.ndarray, n_vars: int) -> np.ndarray:
        """
        Remove edges by testing conditional independence with increasing |Z|.

        Starts with complete undirected graph and iteratively tests pairs
        conditioned on growing subsets of their neighbors.
        """
        adj = np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(adj, 0)

        for cond_size in range(self.max_cond_set + 1):
            pairs = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars) if adj[i, j]]
            for i, j in pairs:
                # Candidate conditioning sets: neighbors of i excluding j
                neighbors = [k for k in range(n_vars) if adj[i, k] and k != j]
                if len(neighbors) < cond_size:
                    continue
                for Z in itertools.combinations(neighbors, cond_size):
                    indep, _ = partial_correlation_test(X, i, j, list(Z), self.alpha)
                    if indep:
                        adj[i, j] = adj[j, i] = 0
                        self.sep_sets[(i, j)] = self.sep_sets[(j, i)] = list(Z)
                        break
        return adj

    def _orient_edges(self, skeleton: np.ndarray, n_vars: int) -> np.ndarray:
        """
        Orient v-structures and apply Meek rules.

        cpdag[i,j]=1, cpdag[j,i]=0  → i → j  (directed)
        cpdag[i,j]=1, cpdag[j,i]=1  → i — j  (undirected)
        """
        cpdag = skeleton.copy()

        # V-structure orientation: X — Z — Y where X,Y not adjacent
        for z in range(n_vars):
            parents_z = [k for k in range(n_vars) if skeleton[k, z]]
            for x, y in itertools.combinations(parents_z, 2):
                if skeleton[x, y]:   # x and y are adjacent — not a v-structure
                    continue
                sep = self.sep_sets.get((x, y), [])
                if z not in sep:
                    # X → Z ← Y (collider)
                    cpdag[z, x] = 0
                    cpdag[z, y] = 0

        # Meek Rule R1: orient Z-Y as Z→Y if X→Z and X,Y not adjacent
        changed = True
        while changed:
            changed = False
            for z in range(n_vars):
                for y in range(n_vars):
                    if cpdag[z, y] == 1 and cpdag[y, z] == 1:  # undirected Z-Y
                        for x in range(n_vars):
                            if cpdag[x, z] == 1 and cpdag[z, x] == 0 and not skeleton[x, y]:
                                cpdag[y, z] = 0   # orient as Z→Y
                                changed = True
                                break
        return cpdag

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Directed adjacency matrix of the estimated CPDAG."""
        return self.cpdag_


# ─────────────────────────────────────────────
# LiNGAM
# ─────────────────────────────────────────────

class DirectLiNGAM:
    """
    Direct LiNGAM (Shimizu et al., 2011) for fully identifiable causal discovery.

    Assumes: X = B X + e (linear acyclic model) with non-Gaussian, independent noise.
    Under these assumptions, the causal order and B are uniquely identifiable.

    Algorithm (Direct LiNGAM):
        1. Repeatedly find the variable most independent of its residuals
           after regressing out all others → this variable has no parents (root).
        2. Remove that variable; repeat on residuals to recover full causal order.
        3. Estimate B by regressing each variable on its discovered parents.

    Independence measure: we use negative absolute kurtosis as a proxy for
    non-Gaussianity. A Gaussian residual has kurtosis ≈ 0; non-Gaussian noise
    has non-zero excess kurtosis.

    Args:
        threshold: minimum edge weight to include in the estimated DAG
    """

    def __init__(
        self,
        threshold: float = 0.1,     # Placeholder: 0.05–0.3; higher = sparser graph
    ):
        self.threshold = threshold
        self.causal_order_: list[int] = []
        self.adjacency_matrix_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DirectLiNGAM":
        """
        Fit LiNGAM to data X.

        Args:
            X: (n_samples, n_vars) standardized data

        Returns:
            self (fitted)
        """
        n_vars = X.shape[1]
        remaining = list(range(n_vars))
        causal_order = []
        residuals = X.copy()

        for _ in range(n_vars - 1):
            # Find variable most independent of its residuals
            best_var, best_score = None, -np.inf
            for var in remaining:
                others = [v for v in remaining if v != var]
                if not others:
                    score = self._non_gaussianity(residuals[:, var])
                else:
                    # Regress var on others, measure non-Gaussianity of residual
                    R = np.column_stack([residuals[:, v] for v in others])
                    coef, _, _, _ = np.linalg.lstsq(R, residuals[:, var], rcond=None)
                    res = residuals[:, var] - R @ coef
                    score = self._non_gaussianity(res)
                if score > best_score:
                    best_score = score
                    best_var = var

            causal_order.append(best_var)
            remaining.remove(best_var)

            # Update residuals: remove influence of best_var from remaining vars
            for var in remaining:
                r = residuals[:, best_var]
                cov = np.dot(residuals[:, var], r) / (np.dot(r, r) + 1e-10)
                residuals[:, var] -= cov * r

        causal_order.append(remaining[0])
        self.causal_order_ = causal_order

        # Estimate B: regress each variable on its causal predecessors
        B = np.zeros((n_vars, n_vars))
        for i, var in enumerate(causal_order):
            predecessors = causal_order[:i]
            if predecessors:
                X_pred = X[:, predecessors]
                coefs, _, _, _ = np.linalg.lstsq(X_pred, X[:, var], rcond=None)
                for pred, coef in zip(predecessors, coefs):
                    B[var, pred] = coef

        # Threshold small coefficients
        B[np.abs(B) < self.threshold] = 0.0
        adj = (B != 0).astype(int)
        self.adjacency_matrix_ = adj
        self.B_ = B
        return self

    @staticmethod
    def _non_gaussianity(x: np.ndarray) -> float:
        """Excess kurtosis as a non-Gaussianity measure (Gaussian → 0)."""
        x = (x - x.mean()) / (x.std() + 1e-10)
        return float(abs(stats.kurtosis(x, fisher=True)))


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_graph(
    estimated_adj: np.ndarray,
    true_adj: np.ndarray,
) -> dict[str, float]:
    """
    Compute structural metrics comparing estimated and true DAG.

    Metrics:
        SHD (Structural Hamming Distance): number of edge additions, deletions,
            and reversals needed to transform estimated into true graph.
            Lower is better; 0 = perfect recovery.
        Precision: of all recovered edges, fraction that are true edges.
        Recall:    of all true edges, fraction that were recovered.
        F1:        harmonic mean of precision and recall.

    Args:
        estimated_adj: (n_vars, n_vars) binary estimated adjacency (directed)
        true_adj:      (n_vars, n_vars) binary true adjacency (directed)

    Returns:
        dict of evaluation metrics
    """
    n = true_adj.shape[0]

    # Undirected skeleton comparison (ignore orientation for skeleton metrics)
    est_skel = ((estimated_adj + estimated_adj.T) > 0).astype(int)
    np.fill_diagonal(est_skel, 0)
    true_skel = ((true_adj + true_adj.T) > 0).astype(int)
    np.fill_diagonal(true_skel, 0)

    tp = int(np.sum((est_skel == 1) & (true_skel == 1)))
    fp = int(np.sum((est_skel == 1) & (true_skel == 0)))
    fn = int(np.sum((est_skel == 0) & (true_skel == 1)))

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)

    # SHD: count wrong edges in directed graph
    shd = int(np.sum(estimated_adj != true_adj))

    return {
        "SHD":       shd,
        "precision": precision,
        "recall":    recall,
        "F1":        f1,
        "TP":        tp,
        "FP":        fp,
        "FN":        fn,
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_graph(
    adj: np.ndarray,
    title: str = "Causal Graph",
    node_names: list[str] | None = None,
    true_adj: np.ndarray | None = None,
    filename: str = "causal_graph.png",
) -> None:
    """
    Draw the directed causal graph.

    Correct edges (vs. true_adj) are green; incorrect edges are red.
    Undirected edges (both directions present) are shown as dashed.

    Args:
        adj:        (n_vars, n_vars) adjacency matrix
        title:      plot title
        node_names: list of variable names (default: X1, X2, ...)
        true_adj:   if provided, color edges by correctness
        filename:   output file path
    """
    n = adj.shape[0]
    node_names = node_names or [f"X{i+1}" for i in range(n)]

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)
    labels = {i: node_names[i] for i in range(n)}

    if true_adj is not None:
        edge_colors = []
        for u, v in G.edges():
            if true_adj[u, v] == 1:
                edge_colors.append("green")
            else:
                edge_colors.append("red")
    else:
        edge_colors = "steelblue"

    plt.figure(figsize=(7, 5))
    nx.draw_networkx(
        G, pos=pos, labels=labels,
        node_color="lightblue", node_size=800,
        edge_color=edge_colors, width=2,
        arrows=True, arrowsize=20,
    )
    if true_adj is not None:
        plt.title(f"{title}\n(green=correct, red=incorrect edge)")
    else:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_metrics_comparison(metrics_pc: dict, metrics_lingam: dict) -> None:
    """
    Bar chart comparing PC and LiNGAM on precision, recall, F1, and SHD.
    """
    keys = ["precision", "recall", "F1"]
    pc_vals     = [metrics_pc[k]     for k in keys]
    lingam_vals = [metrics_lingam[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, pc_vals,     width, label=f"PC (SHD={metrics_pc['SHD']})",     color="steelblue")
    ax.bar(x + width/2, lingam_vals, width, label=f"LiNGAM (SHD={metrics_lingam['SHD']})", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Causal Discovery — PC vs. LiNGAM")
    ax.legend()
    plt.tight_layout()
    plt.savefig("causal_discovery_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== Causal Discovery — PC Algorithm & LiNGAM ===")

    # 1. Generate data from known DAG
    X, true_graph = generate_linear_dag_data(n_samples=500)
    n_vars = X.shape[1]
    node_names = [f"X{i+1}" for i in range(n_vars)]
    print(f"Data shape: {X.shape}")
    print(f"True edges: {true_graph['edges']}")

    # 2. Plot true graph
    plot_graph(true_graph["adjacency"], title="True Causal Graph",
               node_names=node_names, filename="true_causal_graph.png")

    # 3. PC Algorithm
    print("\n--- PC Algorithm ---")
    pc = PCAlgorithm(alpha=0.05, max_cond_set=3)
    pc.fit(X)
    metrics_pc = evaluate_graph(pc.adjacency_matrix, true_graph["adjacency"])
    print(f"Metrics: {metrics_pc}")
    plot_graph(pc.adjacency_matrix, title="PC Algorithm — CPDAG",
               node_names=node_names, true_adj=true_graph["adjacency"],
               filename="pc_graph.png")

    # 4. LiNGAM
    print("\n--- Direct LiNGAM ---")
    lingam = DirectLiNGAM(threshold=0.15)
    lingam.fit(X)
    print(f"Causal order: {[node_names[i] for i in lingam.causal_order_]}")
    metrics_lingam = evaluate_graph(lingam.adjacency_matrix_, true_graph["adjacency"])
    print(f"Metrics: {metrics_lingam}")
    plot_graph(lingam.adjacency_matrix_, title="LiNGAM — Estimated DAG",
               node_names=node_names, true_adj=true_graph["adjacency"],
               filename="lingam_graph.png")

    # 5. Compare
    plot_metrics_comparison(metrics_pc, metrics_lingam)


if __name__ == "__main__":
    main()
