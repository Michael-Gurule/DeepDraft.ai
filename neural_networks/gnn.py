"""
Graph Neural Network (GNN)
============================
A class of deep learning models designed for graph-structured data. Unlike CNNs
or Transformers which operate on grids or sequences, GNNs operate on graphs:
    G = (V, E)
    V = node set (entities, agents, atoms, sensors)
    E = edge set (relationships, communication links, bonds)

Message Passing Neural Network (MPNN) framework:
    For each iteration (layer) l:
        1. Message:    m_{ij}^l = φ_m(h_i^l, h_j^l, e_{ij})
        2. Aggregate:  M_i^l    = ⊕_{j ∈ N(i)} m_{ij}^l       (sum/mean/max)
        3. Update:     h_i^{l+1} = φ_u(h_i^l, M_i^l)

    where:
        h_i^l     = hidden state of node i at layer l
        N(i)      = neighbors of i
        e_{ij}    = edge features between i and j (optional)
        φ_m, φ_u = learnable MLP transforms
        ⊕         = aggregation function (sum, mean, max)

Architecture variants implemented here:
    GCNConv (Kipf & Welling 2017): normalized adjacency message passing
    GraphSAGE: sample-and-aggregate with learnable aggregators
    GATConv (Veličković 2018): attention-weighted message passing
    GINConv (Xu et al. 2019): maximally expressive MPNN (as powerful as 1-WL test)

Graph-level tasks:
    Node classification:  predict label for each node (e.g., fraud node)
    Edge prediction:      predict missing/future edges (e.g., link prediction)
    Graph classification: classify entire graph (e.g., molecule toxicity)

Relationship to Hyperion SwarmGNN:
    Hyperion's SwarmGNN uses attention-based message passing where each UAV/agent
    is a node, communication links are edges, and the model learns distributed
    coordination policies. The GATConv attention mechanism weights neighbor
    contributions based on relevance — critical for multi-agent consensus.

Use Cases:
    - Multi-agent coordination (Hyperion swarm communication)
    - Social network analysis (node classification, link prediction)
    - Molecule property prediction (drug discovery)
    - Knowledge graph completion
    - Recommendation systems (user-item bipartite graphs)
    - Traffic flow prediction (road network graph)
    - Point cloud processing (3D sensor data)

Fine-Tuning Recommendations:
    - n_layers: 2–4. More layers → over-smoothing (all nodes become similar).
      Use residual connections or JUMPING KNOWLEDGE to mitigate.
    - hidden_dim: 64–256.
    - dropout: 0.2–0.5 for node classification (graphs can overfit).
    - Aggregation: sum > mean for degree-sensitive tasks; mean for normalization.
    - GATConv heads: 4–8. Multi-head attention improves stability.
    - For large graphs: use neighbor sampling (GraphSAGE style) or cluster-GCN.
    - Batch normalization: apply after each message-passing layer.
    - For graph classification: readout = sum/mean pooling over nodes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────────
# DATA: SYNTHETIC GRAPH
# ─────────────────────────────────────────────

class SyntheticGraph:
    """
    Synthetic graph for node classification.

    Two-class community graph:
        - n nodes split into 2 communities
        - Intra-community edges denser than inter-community
        - Node features: random + community signal
        - Labels: community membership (0 or 1)

    Replace with real graph data:
        Using PyG (torch_geometric):
            from torch_geometric.datasets import Planetoid
            dataset = Planetoid(root="/tmp/Cora", name="Cora")
            data = dataset[0]
            # data.x, data.edge_index, data.y

        Or build from adjacency matrix:
            adj = scipy.sparse.load_npz("adj.npz")
            edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long)
            x = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
    """

    def __init__(self, n_nodes: int = 200, n_features: int = 16,
                 n_classes: int = 2, edge_prob_intra: float = 0.15,
                 edge_prob_inter: float = 0.01, seed: int = 42):
        self.n     = n_nodes
        self.n_f   = n_features
        self.n_c   = n_classes
        rng        = np.random.RandomState(seed)

        # Node labels (community)
        nodes_per_class = n_nodes // n_classes
        self.y = np.array([i // nodes_per_class for i in range(n_nodes)])
        self.y = np.clip(self.y, 0, n_classes - 1).astype(np.int64)

        # Node features: class-discriminative signal + noise
        class_means = rng.randn(n_classes, n_features) * 2
        self.x = class_means[self.y] + rng.randn(n_nodes, n_features) * 0.5
        self.x = self.x.astype(np.float32)

        # Edges (stochastic block model)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                p = edge_prob_intra if self.y[i] == self.y[j] else edge_prob_inter
                if rng.uniform() < p:
                    edges.append([i, j])
                    edges.append([j, i])  # undirected: add both directions

        if not edges:
            # Ensure connectivity: add chain edges
            for i in range(n_nodes - 1):
                edges.extend([[i, i+1], [i+1, i]])

        self.edge_index = np.array(edges, dtype=np.int64).T  # (2, n_edges)

        # Split: 60% train, 20% val, 20% test
        idx = rng.permutation(n_nodes)
        n_tr = int(0.6 * n_nodes)
        n_v  = int(0.2 * n_nodes)
        self.train_mask = np.zeros(n_nodes, dtype=bool)
        self.val_mask   = np.zeros(n_nodes, dtype=bool)
        self.test_mask  = np.zeros(n_nodes, dtype=bool)
        self.train_mask[idx[:n_tr]]          = True
        self.val_mask[idx[n_tr:n_tr + n_v]]  = True
        self.test_mask[idx[n_tr + n_v:]]     = True

    def to_tensors(self):
        """Convert to torch tensors."""
        x          = torch.tensor(self.x,          device=DEVICE)
        y          = torch.tensor(self.y,          device=DEVICE)
        edge_index = torch.tensor(self.edge_index, device=DEVICE)
        train_mask = torch.tensor(self.train_mask, device=DEVICE)
        val_mask   = torch.tensor(self.val_mask,   device=DEVICE)
        test_mask  = torch.tensor(self.test_mask,  device=DEVICE)
        return x, y, edge_index, train_mask, val_mask, test_mask

    @property
    def n_edges(self):
        return self.edge_index.shape[1]


# ─────────────────────────────────────────────
# GRAPH CONVOLUTION LAYERS (pure PyTorch, no PyG dep)
# ─────────────────────────────────────────────

def build_adj_normalized(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """
    Build symmetric normalized adjacency: Â = D^{-1/2} A D^{-1/2}

    Used in GCN for stable, degree-normalized message passing.
    Self-loops added: A = A + I
    """
    # Build sparse adjacency
    row, col = edge_index
    # Add self-loops
    self_loops = torch.arange(n_nodes, device=edge_index.device)
    row = torch.cat([row, self_loops])
    col = torch.cat([col, self_loops])

    # Degree
    deg = torch.zeros(n_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(len(row), device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    # Normalized values
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Dense adjacency (works for small graphs; use sparse for large graphs)
    A = torch.zeros(n_nodes, n_nodes, device=edge_index.device)
    A[row, col] = norm
    return A


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling 2017).

    H^{l+1} = σ(Â H^l W^l)
    where Â = D^{-1/2}(A + I)D^{-1/2}
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias   = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        node features (n_nodes, in_dim)
            adj_norm: normalized adjacency (n_nodes, n_nodes)

        Returns:
            output: (n_nodes, out_dim)
        """
        support = x @ self.weight
        out     = adj_norm @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Veličković et al. 2018).

    Multi-head attention where each neighbor gets an attention weight α_{ij}:

        α_{ij} = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
        h_i^{l+1} = σ(Σ_j α_{ij} W h_j^l)

    Multi-head: average K attention heads.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 4, dropout: float = 0.2,
                 alpha: float = 0.2):
        super().__init__()
        self.n_heads   = n_heads
        self.out_dim   = out_dim
        self.head_dim  = out_dim // n_heads
        self.dropout   = nn.Dropout(dropout)
        self.leaky     = nn.LeakyReLU(alpha)

        # Linear transform per head
        self.W    = nn.Parameter(torch.empty(in_dim, out_dim))
        # Attention params: [Wh_i || Wh_j] → scalar
        self.a    = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          node features (n_nodes, in_dim)
            edge_index: (2, n_edges) source/target indices

        Returns:
            output: (n_nodes, out_dim)
        """
        n_nodes = x.shape[0]
        Wh = self.dropout(x) @ self.W   # (n_nodes, out_dim)
        Wh = Wh.view(n_nodes, self.n_heads, self.head_dim)  # (n, H, d_h)

        src, dst = edge_index  # source and destination nodes

        # Concatenate source-destination feature pairs for attention
        Wh_src = Wh[src]  # (n_edges, H, d_h)
        Wh_dst = Wh[dst]  # (n_edges, H, d_h)
        pair   = torch.cat([Wh_src, Wh_dst], dim=-1)  # (n_edges, H, 2*d_h)

        # Compute attention coefficients
        e = self.leaky((pair * self.a.unsqueeze(0)).sum(-1))  # (n_edges, H)

        # Softmax over neighbors (using scatter softmax manually)
        # For simplicity: dense softmax (works for small graphs)
        alpha_mat = torch.full((n_nodes, n_nodes, self.n_heads), float("-inf"),
                                device=x.device)
        alpha_mat[src, dst] = e
        alpha_mat = torch.softmax(alpha_mat, dim=1)  # normalize over source neighbors

        # Aggregate: weighted sum
        # out[i] = Σ_j α[i,j] Wh[j]  (for each head)
        out = torch.einsum("ijh,jhd->ihd", alpha_mat, Wh)  # (n, H, d_h)
        out = out.reshape(n_nodes, self.out_dim)
        return out


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer (Xu et al. 2019).

    Maximally expressive MPNN: as powerful as 1-WL graph isomorphism test.

    h_i^{l+1} = MLP((1 + ε) h_i^l + Σ_{j ∈ N(i)} h_j^l)

    Key: ε is a learnable parameter that weights self vs. neighbor contribution.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 eps: float = 0.0, train_eps: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.eps = nn.Parameter(torch.tensor([eps])) if train_eps else eps

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          node features (n_nodes, in_dim)
            edge_index: (2, n_edges)

        Returns:
            output: (n_nodes, out_dim)
        """
        n    = x.shape[0]
        src, dst = edge_index

        # Aggregate neighbor features
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])

        return self.mlp((1 + self.eps) * x + agg)


# ─────────────────────────────────────────────
# GNN MODELS
# ─────────────────────────────────────────────

# Placeholder: tune hidden_dim, n_layers, dropout, gnn_type
GNN_CONFIG = {
    "gnn_type":   "gcn",    # Placeholder: 'gcn', 'gat', 'gin'
    "hidden_dim": 64,       # Placeholder: 32–256
    "n_layers":   3,        # Placeholder: 2–4 (more → over-smoothing)
    "dropout":    0.3,      # Placeholder: 0.0–0.5
    "n_heads":    4,        # (GAT only) Placeholder: 2–8
    "lr":         1e-3,
    "epochs":     200,
    "patience":   20,
}


class GCN(nn.Module):
    """Graph Convolutional Network for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [n_classes]
        self.convs = nn.ModuleList([
            GCNLayer(dims[i], dims[i+1])
            for i in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes  = x.shape[0]
        adj_norm = build_adj_normalized(edge_index, n_nodes)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_norm)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, adj_norm)
        return x  # logits


class GAT(nn.Module):
    """Graph Attention Network for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        dims    = [in_dim] + [hidden_dim] * (n_layers - 1) + [n_classes]
        n_heads_ = [n_heads] * (n_layers - 1) + [1]
        self.convs = nn.ModuleList([
            GATLayer(dims[i], dims[i+1], n_heads=n_heads_[i], dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [n_classes]
        self.convs = nn.ModuleList([
            GINLayer(dims[i], dims[i+1], dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x


def build_gnn(n_features: int, n_classes: int, config: dict = None) -> nn.Module:
    """Factory function to build GCN, GAT, or GIN."""
    if config is None:
        config = GNN_CONFIG
    gnn_type   = config["gnn_type"]
    hidden_dim = config["hidden_dim"]
    n_layers   = config["n_layers"]
    dropout    = config["dropout"]

    if gnn_type == "gcn":
        return GCN(n_features, hidden_dim, n_classes, n_layers, dropout)
    elif gnn_type == "gat":
        return GAT(n_features, hidden_dim, n_classes, n_layers,
                   config.get("n_heads", 4), dropout)
    elif gnn_type == "gin":
        return GIN(n_features, hidden_dim, n_classes, n_layers, dropout)
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_gnn(graph: SyntheticGraph, config: dict = None) -> tuple:
    """
    Train GNN for node classification (semi-supervised).

    Semi-supervised: only train_mask nodes contribute to loss;
    all nodes participate in message passing (transductive learning).
    """
    if config is None:
        config = GNN_CONFIG

    x, y, edge_index, train_mask, val_mask, test_mask = graph.to_tensors()
    n_classes = int(y.max().item()) + 1

    model = build_gnn(graph.n_f, n_classes, config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                  weight_decay=5e-4)

    history    = {"train_loss": [], "val_acc": []}
    best_val   = 0.0
    patience_c = 0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        logits = model(x, edge_index)
        loss   = F.cross_entropy(logits[train_mask], y[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            val_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            val_acc  = accuracy_score(val_true, val_pred)

        history["train_loss"].append(loss.item())
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= config["patience"]:
                print(f"  Early stopping at epoch {epoch}  (best val: {best_val:.4f})")
                break

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | loss: {loss.item():.4f} | val_acc: {val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: nn.Module, graph: SyntheticGraph) -> dict:
    """Evaluate on test nodes."""
    x, y, edge_index, _, _, test_mask = graph.to_tensors()

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs  = torch.softmax(logits[test_mask], dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        true   = y[test_mask].cpu().numpy()

    acc = accuracy_score(true, preds)
    try:
        auc = roc_auc_score(true, probs[:, 1]) if probs.shape[1] == 2 else 0.0
    except Exception:
        auc = 0.0

    n_correct  = int((preds == true).sum())
    print(f"  Test Acc: {acc:.4f}  ({n_correct}/{len(true)} correct)")
    if auc > 0:
        print(f"  Test AUC: {auc:.4f}")
    return {"accuracy": acc, "auc": auc}


def compare_architectures(graph: SyntheticGraph) -> None:
    """Compare GCN, GAT, GIN on the same graph."""
    results = {}
    for gnn_type in ["gcn", "gat", "gin"]:
        print(f"\n  Training {gnn_type.upper()}...")
        cfg   = {**GNN_CONFIG, "gnn_type": gnn_type, "epochs": 200}
        model, _ = train_gnn(graph, cfg)
        m = evaluate(model, graph)
        results[gnn_type.upper()] = m["accuracy"]

    print("\n  Architecture comparison:")
    for name, acc in results.items():
        print(f"    {name}: Test Acc = {acc:.4f}")


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """Plot training loss and validation accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], color="steelblue")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(history["val_acc"], color="tomato")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    plt.suptitle("GNN — Training Curves")
    plt.tight_layout()
    plt.savefig("gnn_training.png", dpi=150)
    plt.show()


def plot_node_embeddings(model: nn.Module, graph: SyntheticGraph) -> None:
    """
    t-SNE visualization of learned node embeddings.
    Shows how well the GNN separates node classes.
    """
    from sklearn.manifold import TSNE

    x, y, edge_index, _, _, _ = graph.to_tensors()

    # Extract penultimate layer embeddings
    model.eval()
    with torch.no_grad():
        if isinstance(model, GCN):
            adj_norm = build_adj_normalized(edge_index, x.shape[0])
            emb = x
            for conv in model.convs[:-1]:
                emb = F.relu(conv(emb, adj_norm))
        elif isinstance(model, GAT):
            emb = x
            for conv in model.convs[:-1]:
                emb = F.elu(conv(emb, edge_index))
        else:
            emb = x
            for conv in model.convs[:-1]:
                emb = F.relu(conv(emb, edge_index))

    emb_np = emb.cpu().numpy()
    y_np   = y.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z    = tsne.fit_transform(emb_np)

    cmap   = plt.get_cmap("tab10")
    plt.figure(figsize=(7, 6))
    for cls in range(int(y_np.max()) + 1):
        mask = y_np == cls
        plt.scatter(z[mask, 0], z[mask, 1], c=[cmap(cls)],
                    s=15, alpha=0.6, label=f"Class {cls}")
    plt.title("GNN — Node Embeddings (t-SNE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gnn_embeddings.png", dpi=150)
    plt.show()


def plot_graph_structure(graph: SyntheticGraph, max_nodes: int = 100) -> None:
    """
    Visualize graph structure (subgraph for large graphs).
    Nodes colored by label.
    """
    try:
        import networkx as nx
    except ImportError:
        print("  networkx not installed; skipping graph plot")
        return

    n    = min(max_nodes, graph.n)
    G    = nx.DiGraph()
    G.add_nodes_from(range(n))
    ei   = graph.edge_index.T
    for src, dst in ei:
        if src < n and dst < n:
            G.add_edge(int(src), int(dst))

    pos    = nx.spring_layout(G, seed=42)
    colors = [plt.cm.tab10(graph.y[i]) for i in range(n)]

    plt.figure(figsize=(8, 7))
    nx.draw_networkx(G, pos=pos, node_color=colors, node_size=80,
                     with_labels=False, arrows=False, alpha=0.8,
                     edge_color="gray", width=0.3)
    plt.title(f"Graph Structure (first {n} nodes, colored by class)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("gnn_graph.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"=== Graph Neural Network ===  (device: {DEVICE})")

    # Placeholder: replace with your graph dataset
    graph = SyntheticGraph(n_nodes=300, n_features=16, n_classes=2,
                            edge_prob_intra=0.12, edge_prob_inter=0.01)
    print(f"Graph: {graph.n} nodes | {graph.n_edges} edges | "
          f"{graph.n_f} features | {int(graph.y.max())+1} classes")
    print(f"  Train: {graph.train_mask.sum()} | "
          f"Val: {graph.val_mask.sum()} | "
          f"Test: {graph.test_mask.sum()}")

    # Train GCN
    print(f"\nTraining {GNN_CONFIG['gnn_type'].upper()}...")
    model, history = train_gnn(graph)

    print("\nTest evaluation:")
    evaluate(model, graph)

    # Compare architectures
    print("\nArchitecture comparison:")
    compare_architectures(graph)

    # Visualizations
    plot_training_curves(history)
    plot_node_embeddings(model, graph)
    plot_graph_structure(graph)


if __name__ == "__main__":
    main()
