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

from deepdraft.core import save_figure


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

    Noise: Laplace and uniform, never Gaussian. LiNGAM identifies the DAG
    only when the noise is non-Gaussian (at most one Gaussian term), so every
    structural equation here gets non-Gaussian noise.

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
    X3 = 0.8 * X1 + 0.6 * X2 + rng.uniform(-1, 1, n)
    X4 = 0.5 * X1 + 0.7 * X2 + rng.laplace(0, 0.5, n)
    X5 = 0.9 * X3 + 0.4 * X4 + rng.uniform(-0.6, 0.6, n)

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
                                  but X⊥Y | Z\\{...} for some set not containing Z
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
        1. Find the exogenous variable: the x_j that is most independent of
           the residuals r_i = x_i - b_ij x_j of regressing every other
           variable on it. Only a variable with no parents leaves residuals
           independent of itself; regressing on a variable that has parents
           leaves its noise mixed into the residuals.
        2. Regress x_j out of every remaining variable and repeat on the
           residuals. The order in which variables are removed is the
           causal order.
        3. Estimate B by regressing each variable on its causal predecessors.

    Independence measure: the pairwise likelihood ratio of Hyvarinen and
    Smith (2013), the default in the reference `lingam` package. For each
    pair it compares the entropy of "x_j causes x_i" against "x_i causes
    x_j", using a maximum-entropy approximation of differential entropy.

    Conventions:
        B_[i, j]           coefficient of x_j in the equation for x_i (j → i)
        adjacency_matrix_  adjacency_matrix_[i, j] = 1 means i → j, the same
                           convention as the true graph and the PC output

    Args:
        threshold: minimum |coefficient| to keep an edge in the estimated DAG
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
            X: (n_samples, n_vars) data matrix

        Returns:
            self (fitted)
        """
        n_vars = X.shape[1]
        remaining = list(range(n_vars))
        causal_order = []
        residuals = X - X.mean(axis=0)

        while remaining:
            root = self._most_exogenous(residuals, remaining)
            causal_order.append(root)
            remaining.remove(root)
            # Remove the root's influence from every remaining variable
            for var in remaining:
                residuals[:, var] = self._residual(residuals[:, var], residuals[:, root])

        self.causal_order_ = causal_order

        # Estimate B: regress each variable on its causal predecessors
        B = np.zeros((n_vars, n_vars))
        for i, var in enumerate(causal_order):
            predecessors = causal_order[:i]
            if predecessors:
                reg = LinearRegression().fit(X[:, predecessors], X[:, var])
                B[var, predecessors] = reg.coef_

        # Threshold small coefficients
        B[np.abs(B) < self.threshold] = 0.0
        self.B_ = B
        # B_[i, j] != 0 means j → i; transpose to the i → j convention
        self.adjacency_matrix_ = (B != 0).T.astype(int)
        return self

    def _most_exogenous(self, X: np.ndarray, candidates: list[int]) -> int:
        """Pick the candidate whose regressions on the others look most like cause → effect."""
        if len(candidates) == 1:
            return candidates[0]
        scores = []
        for j in candidates:
            xj = self._standardize(X[:, j])
            penalty = 0.0
            for i in candidates:
                if i == j:
                    continue
                xi = self._standardize(X[:, i])
                ri_j = self._residual(xi, xj)    # x_i with x_j regressed out
                rj_i = self._residual(xj, xi)    # x_j with x_i regressed out
                # Lower total entropy means higher likelihood. h_ji scores the
                # model x_j → x_i, h_ij the model x_i → x_j.
                h_ji = self._entropy(xj) + self._entropy(self._standardize(ri_j))
                h_ij = self._entropy(xi) + self._entropy(self._standardize(rj_i))
                # Only evidence against x_j being the cause (h_ij < h_ji) is penalized.
                penalty += min(0.0, h_ij - h_ji) ** 2
            scores.append(-penalty)
        return candidates[int(np.argmax(scores))]

    @staticmethod
    def _standardize(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-12)

    @staticmethod
    def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """Residual of regressing xi on xj (both centered)."""
        return xi - (np.dot(xi, xj) / (np.dot(xj, xj) + 1e-12)) * xj

    @staticmethod
    def _entropy(u: np.ndarray) -> float:
        """
        Maximum-entropy approximation of the differential entropy of a
        standardized variable (Hyvarinen, 1998). A Gaussian has the largest
        entropy for a given variance; non-Gaussian variables score lower.
        """
        k1, k2, gamma = 79.047, 7.4129, 0.37457
        return float((1 + np.log(2 * np.pi)) / 2
                     - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2
                     - k2 * np.mean(u * np.exp(-u ** 2 / 2)) ** 2)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_graph(
    estimated_adj: np.ndarray,
    true_adj: np.ndarray,
) -> dict[str, float]:
    """
    Compute structural metrics comparing an estimated graph to the true DAG.

    Adjacency convention: adj[i, j] = 1 means i → j. An undirected edge (as in
    a PC CPDAG) has adj[i, j] = adj[j, i] = 1.

    Metrics:
        SHD (Structural Hamming Distance): the number of node pairs whose
            edge differs: a missing edge, an extra edge, or a wrong or
            missing orientation each cost 1. Lower is better; 0 = perfect.
        Precision, recall, F1: on the skeleton (edges ignoring direction),
            counting each node pair once.
        Orientation accuracy: of the true edges found in the skeleton, the
            fraction with the correct direction. An undirected edge counts
            as not oriented.

    Args:
        estimated_adj: (n_vars, n_vars) binary estimated adjacency
        true_adj:      (n_vars, n_vars) binary true adjacency (a DAG)

    Returns:
        dict of evaluation metrics
    """
    est = np.asarray(estimated_adj) != 0
    true = np.asarray(true_adj) != 0
    n = true.shape[0]
    iu = np.triu_indices(n, k=1)                 # each unordered pair once

    est_skel = (est | est.T)[iu]
    true_skel = (true | true.T)[iu]
    tp = int(np.sum(est_skel & true_skel))
    fp = int(np.sum(est_skel & ~true_skel))
    fn = int(np.sum(~est_skel & true_skel))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Per-pair edge type: (i→j present, j→i present). Any difference costs 1.
    est_type  = np.stack([est[iu],  est.T[iu]],  axis=1)
    true_type = np.stack([true[iu], true.T[iu]], axis=1)
    shd = int(np.sum(np.any(est_type != true_type, axis=1)))

    found = est_skel & true_skel
    oriented_ok = np.all(est_type == true_type, axis=1) & found
    orientation_acc = float(oriented_ok.sum() / found.sum()) if found.any() else 0.0

    return {
        "SHD":       shd,
        "precision": precision,
        "recall":    recall,
        "F1":        f1,
        "orientation_accuracy": orientation_acc,
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
    save_figure(filename, dpi=150)


def plot_metrics_comparison(metrics_pc: dict, metrics_lingam: dict) -> None:
    """
    Bar chart comparing PC and LiNGAM on skeleton precision, recall, F1,
    orientation accuracy, and SHD.
    """
    keys = ["precision", "recall", "F1", "orientation_accuracy"]
    pc_vals     = [metrics_pc[k]     for k in keys]
    lingam_vals = [metrics_lingam[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, pc_vals,     width, label=f"PC (SHD={metrics_pc['SHD']})",     color="steelblue")
    ax.bar(x + width/2, lingam_vals, width, label=f"LiNGAM (SHD={metrics_lingam['SHD']})", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(["precision", "recall", "F1", "orientation"])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Causal Discovery — PC vs. LiNGAM")
    ax.legend()
    plt.tight_layout()
    save_figure("causal_discovery_comparison.png", dpi=150)


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
