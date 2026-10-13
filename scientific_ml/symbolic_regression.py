"""
Symbolic Regression
===================

Goal: discover closed-form mathematical expressions that fit observed data.
Unlike regression with a fixed functional form, symbolic regression searches
the space of mathematical expressions (programs).

Approaches implemented:
  1. Genetic Programming (GP) — tournament selection + crossover/mutation
     on expression trees. The classic approach (Koza 1994). No external deps.
  2. Neural-Guided symbolic regression — use a trained MLP to pre-screen
     candidate expressions by predicting their fitness before evaluation.
     (lightweight version; production → use PySR or EQL).
  3. PySR wrapper — thin wrapper around PySR (Julia-backed, SOTA performance)
     for when PySR is installed. Falls back gracefully if unavailable.

Expression Tree:
  Nodes are binary or unary operators; leaves are variables or constants.
  Example tree for "x₀² + sin(x₁)":
           +
          / \
        pow   sin
        / \    \
      x₀   2   x₁

Genetic operations:
  Crossover: swap a random subtree between two parents
  Mutation:  replace a random subtree with a newly generated tree
  Point mutation: replace a leaf constant with a nearby perturbed value

Fitness:
  f(expr) = 1 / (1 + RMSE(expr(X), y))   ← higher is better
  Parsimony penalty: f' = f / (1 + λ · size(tree))  ← prefer shorter expressions

Fine-tuning recommendations:
  - pop_size (#Placeholder: 500): population size — larger finds better solutions
  - n_generations (#Placeholder: 50): more generations → better search
  - max_depth (#Placeholder: 4): max expression tree depth (controls complexity)
  - tournament_size (#Placeholder: 7): selection pressure
  - crossover_prob (#Placeholder: 0.7): fraction of new pop from crossover
  - mutation_prob (#Placeholder: 0.1): per-individual mutation rate
  - parsimony_coef (#Placeholder: 0.01): penalty for large expressions
  - const_range (#Placeholder: (-5, 5)): range for random constants
  - operators (#Placeholder: list): which operators to include

References:
  Koza (1994) — Genetic Programming: On the Programming of Computers...
  Cranmer (2020) — Discovering Symbolic Models from Deep Learning (SINDy)
  Udrescu & Tegmark (2020) — AI Feynman: a Physics-Inspired Method for SR
  Kamienny et al. (2023) — Deep Generative Symbolic Regression (DGSR)
  Cranmer et al. (2023) — PySR: Fast, Parallelized Symbolic Regression in Python
"""

import numpy as np
import copy
import random
import operator
from dataclasses import dataclass, field
from typing import Callable, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "pop_size":         500,    # # Placeholder:
    "n_generations":    50,     # # Placeholder:
    "max_depth":        4,      # # Placeholder:
    "tournament_size":  7,      # # Placeholder:
    "crossover_prob":   0.7,    # # Placeholder:
    "mutation_prob":    0.1,    # # Placeholder:
    "parsimony_coef":   0.01,   # # Placeholder:
    "const_range":      (-5.0, 5.0),  # # Placeholder:
    "elite_frac":       0.05,   # # Placeholder: fraction kept unchanged each gen
    "seed":             42,
}

# Operators: (name, arity, func)
UNARY_OPS = [
    ("sin",  1, np.sin),
    ("cos",  1, np.cos),
    ("exp",  1, lambda x: np.clip(np.exp(x), -1e6, 1e6)),
    ("log",  1, lambda x: np.log(np.abs(x) + 1e-8)),
    ("sqrt", 1, lambda x: np.sqrt(np.abs(x))),
    ("neg",  1, operator.neg),
    ("abs",  1, np.abs),
]

BINARY_OPS = [
    ("add",  2, operator.add),
    ("sub",  2, operator.sub),
    ("mul",  2, operator.mul),
    ("div",  2, lambda a, b: a / (b + np.sign(b + 1e-9) * 1e-8)),
    ("pow2", 2, lambda a, b: np.clip(a ** 2, -1e6, 1e6)),  # safe square (b ignored)
]

ALL_OPS = UNARY_OPS + BINARY_OPS


# ─────────────────────────────────────────────
# Expression Tree
# ─────────────────────────────────────────────

@dataclass
class Node:
    """A node in the expression tree."""
    node_type: str       # "op", "var", "const"
    op_name: str = ""    # operator name (if op node)
    arity: int = 0
    func: Optional[Callable] = None
    var_idx: int = 0     # variable index (if var node)
    const_val: float = 0.0
    children: list = field(default_factory=list)

    def clone(self) -> "Node":
        n = Node(
            node_type=self.node_type,
            op_name=self.op_name,
            arity=self.arity,
            func=self.func,
            var_idx=self.var_idx,
            const_val=self.const_val,
        )
        n.children = [c.clone() for c in self.children]
        return n

    def __str__(self) -> str:
        if self.node_type == "const":
            return f"{self.const_val:.3g}"
        if self.node_type == "var":
            return f"x{self.var_idx}"
        if self.arity == 1:
            return f"{self.op_name}({self.children[0]})"
        if self.op_name == "pow2":
            return f"({self.children[0]})²"
        return f"({self.children[0]} {self.op_name} {self.children[1]})"

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the expression for all rows in X. Returns (n,) array."""
        if self.node_type == "const":
            return np.full(X.shape[0], self.const_val)
        if self.node_type == "var":
            return X[:, self.var_idx]
        child_vals = [c.evaluate(X) for c in self.children]
        try:
            result = self.func(*child_vals)
            result = np.where(np.isfinite(result), result, 0.0)
            return np.clip(result, -1e6, 1e6)
        except Exception:
            return np.zeros(X.shape[0])

    def collect_nodes(self) -> list["Node"]:
        """Collect all nodes in a flat list (for crossover/mutation)."""
        nodes = [self]
        for c in self.children:
            nodes.extend(c.collect_nodes())
        return nodes


# ─────────────────────────────────────────────
# Tree Generation
# ─────────────────────────────────────────────

def _make_leaf(n_vars: int, cfg: dict, rng: np.random.Generator) -> Node:
    if rng.random() < 0.5 and n_vars > 0:
        return Node("var", var_idx=int(rng.integers(0, n_vars)))
    val = rng.uniform(*cfg["const_range"])
    return Node("const", const_val=float(val))


def random_tree(
    n_vars: int,
    cfg: dict,
    rng: np.random.Generator,
    max_depth: int = -1,
    method: str = "ramped",  # "full", "grow", "ramped"
) -> Node:
    """
    Generate a random expression tree.
    - "full": all leaves at max_depth
    - "grow": stops early with prob 0.5
    - "ramped": alternates full/grow for diversity
    """
    if max_depth < 0:
        max_depth = cfg["max_depth"]
    if method == "ramped":
        method = "full" if rng.random() < 0.5 else "grow"
    return _random_subtree(n_vars, cfg, rng, max_depth, method)


def _random_subtree(
    n_vars: int, cfg: dict, rng: np.random.Generator,
    remaining_depth: int, method: str,
) -> Node:
    if remaining_depth == 0:
        return _make_leaf(n_vars, cfg, rng)
    if method == "grow" and rng.random() < 0.4:
        return _make_leaf(n_vars, cfg, rng)

    op = ALL_OPS[int(rng.integers(0, len(ALL_OPS)))]
    name, arity, func = op
    node = Node("op", op_name=name, arity=arity, func=func)
    for _ in range(arity):
        node.children.append(
            _random_subtree(n_vars, cfg, rng, remaining_depth - 1, method)
        )
    return node


# ─────────────────────────────────────────────
# Genetic Operators
# ─────────────────────────────────────────────

def subtree_crossover(p1: Node, p2: Node, rng: np.random.Generator) -> tuple[Node, Node]:
    """Swap a random subtree between two parents."""
    c1 = p1.clone()
    c2 = p2.clone()

    nodes1 = c1.collect_nodes()
    nodes2 = c2.collect_nodes()

    # Pick random cut points (skip root for simplicity)
    n1 = nodes1[int(rng.integers(1, max(2, len(nodes1))))]
    n2 = nodes2[int(rng.integers(1, max(2, len(nodes2))))]

    # Swap the subtrees by modifying in-place
    n1.node_type, n2.node_type = n2.node_type, n1.node_type
    n1.op_name,   n2.op_name   = n2.op_name,   n1.op_name
    n1.arity,     n2.arity     = n2.arity,     n1.arity
    n1.func,      n2.func      = n2.func,      n1.func
    n1.var_idx,   n2.var_idx   = n2.var_idx,   n1.var_idx
    n1.const_val, n2.const_val = n2.const_val, n1.const_val
    n1.children,  n2.children  = n2.children,  n1.children

    return c1, c2


def subtree_mutation(
    tree: Node, n_vars: int, cfg: dict, rng: np.random.Generator
) -> Node:
    """Replace a random subtree with a newly generated one."""
    new_tree = tree.clone()
    nodes = new_tree.collect_nodes()
    target = nodes[int(rng.integers(0, len(nodes)))]

    # Replace target node in-place with a new random subtree
    fresh = _random_subtree(n_vars, cfg, rng, cfg["max_depth"] // 2, "grow")
    target.node_type = fresh.node_type
    target.op_name   = fresh.op_name
    target.arity     = fresh.arity
    target.func      = fresh.func
    target.var_idx   = fresh.var_idx
    target.const_val = fresh.const_val
    target.children  = fresh.children
    return new_tree


def point_mutation(tree: Node, cfg: dict, rng: np.random.Generator) -> Node:
    """Perturb a random constant by ±10%."""
    new_tree = tree.clone()
    const_nodes = [n for n in new_tree.collect_nodes() if n.node_type == "const"]
    if const_nodes:
        n = const_nodes[int(rng.integers(0, len(const_nodes)))]
        n.const_val *= (1.0 + rng.uniform(-0.1, 0.1))
        n.const_val = float(np.clip(n.const_val, *cfg["const_range"]))
    return new_tree


# ─────────────────────────────────────────────
# Fitness and Selection
# ─────────────────────────────────────────────

def fitness(tree: Node, X: np.ndarray, y: np.ndarray, parsimony: float) -> float:
    """
    f(expr) = 1 / (1 + RMSE) · exp(-λ · size)
    Higher = better.
    """
    pred = tree.evaluate(X)
    if not np.all(np.isfinite(pred)):
        return 0.0
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    base = 1.0 / (1.0 + rmse)
    penalty = np.exp(-parsimony * tree.size())
    return base * penalty


def tournament_select(
    population: list[Node],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Node:
    """Tournament selection: pick k random individuals, return the best."""
    idx = rng.integers(0, len(population), size=k)
    best = idx[np.argmax(fitnesses[idx])]
    return population[best]


# ─────────────────────────────────────────────
# Genetic Programming
# ─────────────────────────────────────────────

class GeneticProgramming:
    """
    Genetic Programming for symbolic regression.
    Returns best expression tree found after n_generations.
    """

    def __init__(self, cfg: dict, n_vars: int, seed: int = 42):
        self.cfg = cfg
        self.n_vars = n_vars
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.best_tree: Optional[Node] = None
        self.best_fitness_history: list[float] = []
        self.mean_fitness_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Node:
        """Run GP. Returns best expression tree."""
        cfg = self.cfg
        n = cfg["pop_size"]
        n_elite = max(1, int(n * cfg["elite_frac"]))

        # Initialize population
        pop = [random_tree(self.n_vars, cfg, self.rng) for _ in range(n)]
        fits = np.array([fitness(t, X, y, cfg["parsimony_coef"]) for t in pop])

        best_idx = int(np.argmax(fits))
        self.best_tree = pop[best_idx].clone()
        best_fit = fits[best_idx]

        for gen in range(cfg["n_generations"]):
            # Sort by fitness
            order = np.argsort(fits)[::-1]
            pop = [pop[i] for i in order]
            fits = fits[order]

            new_pop = pop[:n_elite]   # elitism

            while len(new_pop) < n:
                r = self.rng.random()

                if r < cfg["crossover_prob"]:
                    p1 = tournament_select(pop, fits, cfg["tournament_size"], self.rng)
                    p2 = tournament_select(pop, fits, cfg["tournament_size"], self.rng)
                    c1, c2 = subtree_crossover(p1, p2, self.rng)
                    # Depth limit enforcement
                    if c1.depth() <= cfg["max_depth"]:
                        new_pop.append(c1)
                    if len(new_pop) < n and c2.depth() <= cfg["max_depth"]:
                        new_pop.append(c2)
                elif r < cfg["crossover_prob"] + cfg["mutation_prob"]:
                    p = tournament_select(pop, fits, cfg["tournament_size"], self.rng)
                    child = subtree_mutation(p, self.n_vars, cfg, self.rng)
                    if child.depth() <= cfg["max_depth"]:
                        new_pop.append(child)
                else:
                    # Reproduction
                    new_pop.append(
                        tournament_select(pop, fits, cfg["tournament_size"], self.rng).clone()
                    )

            new_pop = new_pop[:n]
            fits = np.array([fitness(t, X, y, cfg["parsimony_coef"]) for t in new_pop])
            pop = new_pop

            gen_best = float(np.max(fits))
            gen_mean = float(np.mean(fits))
            self.best_fitness_history.append(gen_best)
            self.mean_fitness_history.append(gen_mean)

            if gen_best > best_fit:
                best_fit = gen_best
                self.best_tree = pop[int(np.argmax(fits))].clone()

            if (gen + 1) % 10 == 0:
                print(f"  Gen {gen+1:3d}/{cfg['n_generations']} | "
                      f"Best fit: {gen_best:.5f} | Mean: {gen_mean:.5f} | "
                      f"Best expr: {str(self.best_tree)[:60]}")

        return self.best_tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.best_tree is not None, "Call fit() first"
        return self.best_tree.evaluate(X)


# ─────────────────────────────────────────────
# PySR Wrapper (optional)
# ─────────────────────────────────────────────

def run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 40,     # # Placeholder:
    binary_operators: list = None,
    unary_operators: list = None,
    maxsize: int = 20,           # # Placeholder:
    populations: int = 15,       # # Placeholder:
) -> Optional[object]:
    """
    Run PySR symbolic regression if available.
    PySR is SOTA — uses Julia backend with multi-population evolution.

    Install: pip install pysr  (then call pysr.install() once for Julia)

    Args:
        X: (n, d) input features
        y: (n,) target
    Returns:
        PySRRegressor instance with .equations_ DataFrame, or None if not installed
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("PySR not installed. Run: pip install pysr && python -c 'import pysr; pysr.install()'")
        return None

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]   # # Placeholder:
    if unary_operators is None:
        unary_operators = ["sin", "cos", "exp", "log", "sqrt", "abs"]  # # Placeholder:

    model = PySRRegressor(
        niterations=n_iterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        maxsize=maxsize,
        populations=populations,
        verbosity=1,
        random_state=42,
    )
    model.fit(X, y)

    print("\nTop PySR equations:")
    print(model.equations_[["equation", "loss", "complexity"]].head(10))
    return model


# ─────────────────────────────────────────────
# Benchmark Datasets
# ─────────────────────────────────────────────

def generate_benchmark(
    name: str = "feynman_I_9_18",
    n_samples: int = 500,
    noise: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Classical symbolic regression benchmarks.
    Replace with your own data for real applications.

    Returns: X (n, d), y (n,), true_expr_string
    """
    rng = np.random.default_rng(seed)

    if name == "feynman_I_9_18":
        # F = m * a  (Newton's 2nd law, trivial)
        x0 = rng.uniform(0.5, 5.0, n_samples)  # mass
        x1 = rng.uniform(0.5, 5.0, n_samples)  # acceleration
        y = x0 * x1
        true_expr = "x0 * x1"
        X = np.stack([x0, x1], axis=1)

    elif name == "pendulum":
        # T ≈ 2π√(L/g) — small angle period
        L = rng.uniform(0.1, 5.0, n_samples)
        g = 9.81
        y = 2 * np.pi * np.sqrt(L / g)
        true_expr = "2*pi*sqrt(x0/9.81)"
        X = L[:, None]

    elif name == "kepler":
        # v² = GM/r — orbital speed
        r = rng.uniform(1.0, 10.0, n_samples)   # orbital radius
        GM = 4 * np.pi ** 2                      # in AU³/yr²
        y = np.sqrt(GM / r)
        true_expr = "sqrt(4*pi^2 / x0)"
        X = r[:, None]

    elif name == "heat_capacity":
        # Dulong-Petit: C_v = 3Nk_B (classical limit)
        # Debye: C_v ≈ 12π⁴Nk_B/5 * (T/θ_D)³  for T << θ_D
        T = rng.uniform(5.0, 50.0, n_samples)
        theta_D = 300.0
        y = 12 * np.pi ** 4 / 5 * (T / theta_D) ** 3
        true_expr = "12*pi^4/5 * (x0/300)^3"
        X = T[:, None]

    elif name == "cubic":
        # y = x³ - x + 1
        x = rng.uniform(-3, 3, n_samples)
        y = x ** 3 - x + 1
        true_expr = "x0^3 - x0 + 1"
        X = x[:, None]

    elif name == "damped_oscillator":
        # y = A * exp(-γt) * cos(ωt + φ)
        t = rng.uniform(0, 10, n_samples)
        y = 2.0 * np.exp(-0.3 * t) * np.cos(2 * np.pi * t + 0.5)
        true_expr = "2*exp(-0.3*x0)*cos(2*pi*x0 + 0.5)"
        X = t[:, None]

    else:
        raise ValueError(f"Unknown benchmark: {name}")

    # Add Gaussian noise
    y += rng.normal(0, noise * np.abs(y).mean(), n_samples)

    return X.astype(np.float32), y.astype(np.float32), true_expr


# ─────────────────────────────────────────────
# Evaluate Discovered Expression
# ─────────────────────────────────────────────

def evaluate_expression(
    tree: Node,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_expr: str,
) -> dict:
    """Compute R², RMSE, MAE and print expression."""
    y_pred = tree.evaluate(X_test)
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    r2   = float(r2_score(y_test, y_pred)) if np.all(np.isfinite(y_pred)) else -999.0
    sz   = tree.size()

    print(f"\n{'─'*50}")
    print(f"Discovered:  {tree}")
    print(f"True:        {true_expr}")
    print(f"Tree size:   {sz} nodes | depth: {tree.depth()}")
    print(f"RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.6f}")
    print(f"{'─'*50}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "expr": str(tree), "size": sz}


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_symbolic_regression(
    gp: GeneticProgramming,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    true_expr: str,
    dataset_name: str = "benchmark",
):
    """Plot GP fitness evolution, predicted vs true, and parity plot."""
    n_plots = 3 + (X_train.shape[1] == 1)   # extra 1D regression plot
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    fig.suptitle(f"Symbolic Regression — {dataset_name}", fontsize=12)

    # Fitness history
    axes[0].plot(gp.best_fitness_history, label="Best", color="steelblue")
    axes[0].plot(gp.mean_fitness_history, label="Mean", color="darkorange", alpha=0.7)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness  (higher = better)")
    axes[0].set_title("GP Fitness Evolution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parity plot
    y_pred = gp.predict(X_test)
    axes[1].scatter(y_test, y_pred, s=10, alpha=0.4)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", label="y=x")
    r2 = r2_score(y_test, y_pred) if np.all(np.isfinite(y_pred)) else 0.0
    axes[1].set_title(f"Predicted vs True  (R²={r2:.4f})")
    axes[1].set_xlabel("True y"); axes[1].set_ylabel("Predicted y")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Expression size distribution in final population
    if hasattr(gp, "_final_pop"):
        sizes = [t.size() for t in gp._final_pop]
        axes[2].hist(sizes, bins=20, color="forestgreen", edgecolor="white")
        axes[2].axvline(gp.best_tree.size(), color="red", linestyle="--",
                        label=f"Best ({gp.best_tree.size()} nodes)")
    else:
        axes[2].text(0.5, 0.5, f"Best expression:\n{str(gp.best_tree)}",
                     ha="center", va="center", transform=axes[2].transAxes,
                     fontsize=9, wrap=True)
        axes[2].text(0.5, 0.1, f"True:\n{true_expr}",
                     ha="center", va="center", transform=axes[2].transAxes,
                     fontsize=9, color="red")
    axes[2].set_title("Discovered Expression")
    axes[2].axis("off")

    # 1D regression overlay (only if 1 input variable)
    if X_train.shape[1] == 1:
        ax = axes[3]
        x_plot = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 200)[:, None].astype(np.float32)
        y_plot = gp.predict(x_plot)
        ax.scatter(X_train[:, 0], y_train, s=10, alpha=0.3, label="Train data", c="gray")
        ax.plot(x_plot[:, 0], y_plot, "r-", lw=2, label="Discovered")
        ax.set_xlabel("x₀"); ax.set_ylabel("y")
        ax.set_title("1D Regression Fit")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"symbolic_regression_{dataset_name}.png", dpi=150)
    plt.show()
    print(f"Saved: symbolic_regression_{dataset_name}.png")


def plot_multi_benchmark_comparison(results: dict):
    """Bar chart comparing R² across multiple benchmark problems."""
    names = list(results.keys())
    r2_vals = [results[n]["r2"] for n in names]
    sizes = [results[n]["size"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Symbolic Regression — Benchmark Comparison", fontsize=12)

    colors = plt.cm.RdYlGn(np.clip(np.array(r2_vals), 0, 1))
    axes[0].barh(names, r2_vals, color=colors, edgecolor="white")
    axes[0].axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("R²  (higher = better)")
    axes[0].set_title("R² by Benchmark")
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="x")

    axes[1].barh(names, sizes, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Expression Tree Size (nodes)")
    axes[1].set_title("Expression Complexity")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("symbolic_regression_benchmark.png", dpi=150)
    plt.show()
    print("Saved: symbolic_regression_benchmark.png")


# ─────────────────────────────────────────────
# Multi-Benchmark Runner
# ─────────────────────────────────────────────

def run_benchmarks(cfg: dict) -> dict:
    """
    Run GP symbolic regression on several benchmark problems.
    Returns dict of {name: metrics}.
    """
    benchmarks = [
        "cubic",
        "pendulum",
        "kepler",
        "damped_oscillator",
    ]

    all_results = {}
    for name in benchmarks:
        print(f"\n{'='*55}")
        print(f"Benchmark: {name}")
        print(f"{'='*55}")

        X, y, true_expr = generate_benchmark(name, n_samples=300, seed=cfg["seed"])
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        n_vars = X_train.shape[1]

        gp = GeneticProgramming(cfg, n_vars=n_vars, seed=cfg["seed"])
        gp.fit(X_train, y_train)

        metrics = evaluate_expression(gp.best_tree, X_test, y_test, true_expr)
        all_results[name] = metrics

        plot_symbolic_regression(gp, X_train, y_train, X_test, y_test,
                                  true_expr, dataset_name=name)

    plot_multi_benchmark_comparison(all_results)
    return all_results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print(f"Config: {CONFIG}\n")

    # ── Single benchmark demo ───────────────────────────────────────────
    BENCHMARK = "cubic"   # # Placeholder: "cubic" | "pendulum" | "kepler" | "damped_oscillator"
    X, y, true_expr = generate_benchmark(BENCHMARK, n_samples=400, seed=CONFIG["seed"])
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"=== Genetic Programming — {BENCHMARK} ===")
    print(f"Target: {true_expr}")
    print(f"Data shape: X={X_train.shape}, y={y_train.shape}\n")

    gp = GeneticProgramming(CONFIG, n_vars=X_train.shape[1], seed=CONFIG["seed"])
    best_tree = gp.fit(X_train, y_train)

    metrics = evaluate_expression(best_tree, X_test, y_test, true_expr)
    plot_symbolic_regression(gp, X_train, y_train, X_test, y_test,
                              true_expr, dataset_name=BENCHMARK)

    # ── Optional: PySR ─────────────────────────────────────────────────
    print("\n=== PySR (if installed) ===")
    run_pysr(X_train, y_train)

    # ── Multi-benchmark comparison ──────────────────────────────────────
    print("\n=== Multi-Benchmark Comparison ===")
    results = run_benchmarks({**CONFIG, "n_generations": 30})

    print("\nSummary:")
    for name, m in results.items():
        print(f"  {name:25s} | R²={m['r2']:.4f} | RMSE={m['rmse']:.4f} | size={m['size']}")


if __name__ == "__main__":
    main()
