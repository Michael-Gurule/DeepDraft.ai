"""
Equivariant Graph Neural Network (EGNN / SE(3)-Equivariant)
============================================================

Motivation:
  Physical systems (molecules, proteins, point clouds) exist in 3D space.
  Standard GNNs are not invariant to rigid-body transformations — rotating
  a molecule should not change its predicted energy. Equivariant GNNs bake
  in the symmetry as an inductive bias, drastically reducing the amount of
  data needed and improving generalization.

Symmetry hierarchy:
  Invariant:    f(Rx + t) = f(x)         (output is a scalar, unchanged)
  Equivariant:  f(Rx + t) = R·f(x) + t  (output transforms with the input)

EGNN (Satorras et al. 2021):
  Message passing that updates both node features h_i and coordinates x_i.
  Coordinate update is equivariant by construction:
    m_{ij} = φ_e(h_i, h_j, ||x_i - x_j||², a_{ij})       ← edge MLP
    h_i'   = φ_h(h_i, Σ_{j∈N(i)} m_{ij})                   ← node MLP
    x_i'   = x_i + Σ_{j∈N(i)} (x_i - x_j) · φ_x(m_{ij})   ← equivariant coord update

  The relative displacement (x_i - x_j) is equivariant under rotation;
  multiplying by a scalar φ_x(m_{ij}) preserves equivariance.
  The squared distance ||x_i - x_j||² is rotation/translation invariant.

SchNet-style radial basis:
  e_k(r) = exp(-γ (r - μ_k)²)  with K Gaussian centers {μ_k}
  Provides a smooth, expressive encoding of interatomic distance.

Applications:
  - Molecular property prediction (QM9 dataset)
  - Protein structure analysis
  - Particle physics simulations
  - 3D object recognition / point cloud processing

Fine-tuning recommendations:
  - n_layers (#Placeholder: 4): more layers = larger receptive field in graph
  - hidden_dim (#Placeholder: 128): node feature dimension
  - edge_hidden (#Placeholder: 64): edge MLP hidden units
  - cutoff (#Placeholder: 5.0): Angstrom cutoff for neighbor graph construction
  - n_rbf (#Placeholder: 16): number of radial basis functions
  - coord_update_norm (#Placeholder: True): normalize coordinate updates by neighbor count
  - dropout (#Placeholder: 0.1): applied to node MLP
  - update_coords (#Placeholder: True): set False for invariant-only (no coord update)

References:
  Satorras et al. (2021) — E(n) Equivariant Graph Neural Networks (EGNN)
  Schütt et al. (2018) — SchNet: A continuous-filter convolutional neural network
  Thomas et al. (2018) — Tensor Field Networks (SE(3)-equivariant)
  Fuchs et al. (2020) — SE(3)-Transformers
  Batzner et al. (2022) — E(3)-Equivariant Graph Neural Networks for Data-Efficient
                           and Accurate Interatomic Potentials (NequIP)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from typing import Optional

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "n_layers":         4,      # # Placeholder: EGNN message-passing layers
    "hidden_dim":       128,    # # Placeholder: node feature dimension
    "edge_hidden":      64,     # # Placeholder: edge MLP hidden size
    "n_rbf":            16,     # # Placeholder: radial basis functions
    "cutoff":           3.0,    # # Placeholder: neighbor cutoff distance
    "update_coords":    True,   # # Placeholder: equivariant coord updates
    "coord_norm":       True,   # # Placeholder: normalize coord updates by degree
    "dropout":          0.1,    # # Placeholder:
    "lr":               1e-3,   # # Placeholder:
    "weight_decay":     1e-5,   # # Placeholder:
    "batch_size":       32,     # # Placeholder:
    "n_epochs":         100,    # # Placeholder:
    "clip_grad":        1.0,    # # Placeholder:
    "n_atoms":          10,     # # Placeholder: atoms per molecule in synthetic data
    "n_node_features":  5,      # # Placeholder: one-hot element type + charge, etc.
    "n_molecules":      2000,   # # Placeholder: synthetic dataset size
    "seed":             42,
}


# ─────────────────────────────────────────────
# Radial Basis Functions
# ─────────────────────────────────────────────

class GaussianRBF(nn.Module):
    """
    Gaussian radial basis:  e_k(r) = exp(-γ (r - μ_k)²)
    Centers μ_k uniformly spaced in [0, cutoff]; γ fixed.
    Provides a smooth encoding of pairwise distance.
    """

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = n_rbf / cutoff ** 2     # width parameter

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (n_edges,) → (n_edges, n_rbf)"""
        diff = dist.unsqueeze(1) - self.centers.unsqueeze(0)   # (E, K)
        return torch.exp(-self.gamma * diff.pow(2))


def cosine_cutoff(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth envelope that goes to 0 at r = cutoff: 0.5(cos(πr/r_c)+1)."""
    env = 0.5 * (torch.cos(np.pi * dist / cutoff) + 1.0)
    return env * (dist < cutoff).float()


# ─────────────────────────────────────────────
# EGNN Layer
# ─────────────────────────────────────────────

class EGNNLayer(nn.Module):
    """
    One layer of E(n)-Equivariant GNN (Satorras et al. 2021).

    Updates:
      Edge:  m_{ij} = φ_e(h_i, h_j, ||x_i-x_j||², edge_attr)
      Node:  h_i'   = φ_h(h_i, agg_m_i)
      Coord: x_i'   = x_i + Σ_j (x_i - x_j) φ_x(m_{ij}) / (n_neighbors + 1)

    The coordinate update is equivariant because:
      - (x_i - x_j) → R(x_i-x_j) under rotation R
      - φ_x(m_{ij}) is a scalar (rotation invariant)
      - The sum of equivariant vectors is equivariant
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_hidden: int,
        n_rbf: int,
        dropout: float = 0.0,
        update_coords: bool = True,
        coord_norm: bool = True,
    ):
        super().__init__()
        self.update_coords = update_coords
        self.coord_norm = coord_norm

        # Edge MLP: (h_i, h_j, rbf_encoding) → m_{ij}
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_rbf, edge_hidden),
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.SiLU(),
        )

        # Node MLP: (h_i, aggregated_messages) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_hidden, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

        # Coordinate MLP: m_{ij} → scalar weight for displacement
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(edge_hidden, edge_hidden),
                nn.SiLU(),
                nn.Linear(edge_hidden, 1),
                nn.Tanh(),    # bound the update magnitude
            )

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim)
        x: torch.Tensor,           # (N, 3) — 3D positions
        edge_index: torch.Tensor,  # (2, E) — [src, dst]
        rbf: torch.Tensor,         # (E, n_rbf)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]  # (E,)

        # ── Edge messages ──────────────────────────────────────
        h_src = h[src]          # (E, hidden)
        h_dst = h[dst]          # (E, hidden)
        edge_input = torch.cat([h_src, h_dst, rbf], dim=-1)
        m = self.edge_mlp(edge_input)                             # (E, edge_hidden)

        # ── Node update ────────────────────────────────────────
        # Aggregate messages per destination node
        agg = torch.zeros(h.shape[0], m.shape[1], device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(m), m)    # (N, edge_hidden)

        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_norm(h + self.node_mlp(node_input))    # residual + norm

        # ── Coordinate update (equivariant) ────────────────────
        if self.update_coords:
            diff = x[src] - x[dst]                               # (E, 3) — equivariant
            # Clamp large displacements for numerical stability
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            diff_norm = diff / dist                              # unit direction

            weight = self.coord_mlp(m)                           # (E, 1)
            coord_msg = diff_norm * weight                       # (E, 3) — equivariant

            coord_agg = torch.zeros_like(x)
            coord_agg.scatter_add_(0, src.unsqueeze(1).expand_as(coord_msg), coord_msg)

            if self.coord_norm:
                # Normalize by degree to avoid scale explosion in dense graphs
                degree = torch.zeros(x.shape[0], device=x.device)
                degree.scatter_add_(0, src, torch.ones(src.shape[0], device=x.device))
                degree = degree.clamp(min=1).unsqueeze(1)
                coord_agg = coord_agg / degree

            x_new = x + coord_agg
        else:
            x_new = x

        return h_new, x_new


# ─────────────────────────────────────────────
# EGNN Model
# ─────────────────────────────────────────────

class EGNN(nn.Module):
    """
    Full E(n)-Equivariant Graph Neural Network.

    Architecture:
      1. Embed node features: node_embed(x_feat) → h ∈ ℝ^hidden
      2. Compute pairwise RBF encoding of distances
      3. Apply n_layers EGNN layers
      4. Readout: global mean pooling → output MLP → scalar (energy/property)

    Invariance for scalar prediction:
      Since the final readout uses invariant features (h, not x) via mean pooling,
      the output is SE(3)-invariant regardless of update_coords.
    """

    def __init__(
        self,
        n_node_features: int,
        hidden_dim: int,
        n_layers: int,
        edge_hidden: int,
        n_rbf: int,
        cutoff: float,
        output_dim: int = 1,
        dropout: float = 0.0,
        update_coords: bool = True,
        coord_norm: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.node_embed = nn.Linear(n_node_features, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_hidden=edge_hidden,
                n_rbf=n_rbf,
                dropout=dropout,
                update_coords=update_coords,
                coord_norm=coord_norm,
            )
            for _ in range(n_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def _build_edge_index(self, pos: torch.Tensor, batch: torch.Tensor) -> tuple:
        """
        Build neighbor graph: all pairs within molecule with dist < cutoff.
        Operates molecule-by-molecule using batch assignment vector.

        Args:
            pos:   (N_total, 3) — all atom positions in the batch
            batch: (N_total,)   — molecule index per atom

        Returns:
            edge_index: (2, E)
            dist:       (E,)
        """
        src_list, dst_list = [], []
        unique_mols = batch.unique()

        for mol_id in unique_mols:
            mask = batch == mol_id
            idx = mask.nonzero(as_tuple=True)[0]
            pos_mol = pos[idx]                     # (n_atoms, 3)
            n = pos_mol.shape[0]

            # All pairs (excluding self)
            i_idx, j_idx = torch.meshgrid(
                torch.arange(n, device=pos.device),
                torch.arange(n, device=pos.device),
                indexing="ij",
            )
            mask_pair = i_idx != j_idx
            i_flat = i_idx[mask_pair]
            j_flat = j_idx[mask_pair]

            dist_mat = (pos_mol[i_flat] - pos_mol[j_flat]).norm(dim=-1)
            within_cutoff = dist_mat < self.cutoff

            # Map back to global indices
            src_list.append(idx[i_flat[within_cutoff]])
            dst_list.append(idx[j_flat[within_cutoff]])

        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
        dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
        return edge_index, dist

    def forward(
        self,
        node_feat: torch.Tensor,  # (N, n_node_features)
        pos: torch.Tensor,         # (N, 3)
        batch: torch.Tensor,       # (N,) — molecule index
    ) -> torch.Tensor:
        """
        Returns:
            out: (n_molecules, output_dim) — per-molecule prediction
        """
        # Build graph and RBF
        edge_index, dist = self._build_edge_index(pos, batch)
        rbf = self.rbf(dist) * cosine_cutoff(dist, self.cutoff).unsqueeze(1)

        # Node embedding
        h = F.silu(self.node_embed(node_feat))

        # Message passing
        x = pos.clone()
        for layer in self.layers:
            h, x = layer(h, x, edge_index, rbf)

        # Global mean pooling per molecule
        n_mols = batch.max().item() + 1
        h_mol = torch.zeros(int(n_mols), h.shape[1], device=h.device)
        count = torch.zeros(int(n_mols), 1, device=h.device)
        mol_idx = batch.unsqueeze(1).expand_as(h)
        h_mol.scatter_add_(0, mol_idx, h)
        count.scatter_add_(0, batch.unsqueeze(1), torch.ones(batch.shape[0], 1, device=h.device))
        h_mol = h_mol / count.clamp(min=1)

        return self.readout(h_mol)


# ─────────────────────────────────────────────
# Synthetic Molecular Dataset
# ─────────────────────────────────────────────

class SyntheticMoleculeDataset(Dataset):
    """
    Synthetic molecular graph dataset.
    Replace with real data (e.g., QM9) for serious applications.

    Each "molecule" is n_atoms random atoms in a unit cube.
    Target energy: sum of pairwise Lennard-Jones potentials (ground truth).

    E = Σ_{i<j} 4ε [ (σ/r)^12 - (σ/r)^6 ]
    """

    def __init__(
        self,
        n_molecules: int,
        n_atoms: int,
        n_node_features: int,
        eps: float = 1.0,
        sigma: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.molecules = []

        for mol_id in range(n_molecules):
            pos = rng.uniform(-1.5, 1.5, (n_atoms, 3)).astype(np.float32)
            feat = rng.dirichlet(np.ones(n_node_features), n_atoms).astype(np.float32)

            # Lennard-Jones energy
            energy = 0.0
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r = np.linalg.norm(pos[i] - pos[j])
                    r = max(r, 0.1)   # avoid explosion
                    sr6 = (sigma / r) ** 6
                    energy += 4 * eps * (sr6 ** 2 - sr6)

            self.molecules.append({
                "pos": torch.tensor(pos),
                "feat": torch.tensor(feat),
                "energy": torch.tensor([energy], dtype=torch.float32),
            })

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        return self.molecules[idx]


def collate_molecules(batch: list[dict]) -> dict:
    """Batch molecules by concatenating atoms and tracking batch assignment."""
    all_pos, all_feat, all_energy, batch_idx = [], [], [], []
    for mol_id, mol in enumerate(batch):
        n = mol["pos"].shape[0]
        all_pos.append(mol["pos"])
        all_feat.append(mol["feat"])
        all_energy.append(mol["energy"])
        batch_idx.append(torch.full((n,), mol_id, dtype=torch.long))

    return {
        "pos":    torch.cat(all_pos, dim=0),
        "feat":   torch.cat(all_feat, dim=0),
        "energy": torch.cat(all_energy, dim=0),
        "batch":  torch.cat(batch_idx, dim=0),
    }


# ─────────────────────────────────────────────
# Equivariance Tests
# ─────────────────────────────────────────────

def test_equivariance(model: EGNN, n_atoms: int = 8, n_features: int = 5):
    """
    Verify SE(3) invariance:
      E(R·x + t) ≈ E(x)  for random rotation R and translation t.
    """
    model.eval()
    model.to(DEVICE)

    pos = torch.randn(n_atoms, 3, device=DEVICE)
    feat = torch.randn(n_atoms, n_features, device=DEVICE)
    feat = F.softmax(feat, dim=-1)
    batch = torch.zeros(n_atoms, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        e_orig = model(feat, pos, batch)

    # Random rotation (QR decomposition of random matrix)
    R_raw = torch.randn(3, 3, device=DEVICE)
    R, _ = torch.linalg.qr(R_raw)
    if torch.linalg.det(R) < 0:
        R[:, 0] *= -1   # ensure det=+1 (proper rotation)

    t = torch.randn(3, device=DEVICE)
    pos_rot = (R @ pos.T).T + t.unsqueeze(0)

    with torch.no_grad():
        e_rot = model(feat, pos_rot, batch)

    error = (e_orig - e_rot).abs().max().item()
    print(f"SE(3) Invariance Test | Max |ΔE|: {error:.6f}  (should be < 1e-4)")
    return error


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_egnn(cfg: dict) -> tuple[EGNN, dict]:
    """
    Train EGNN on synthetic Lennard-Jones energy prediction.
    Returns trained model and metrics.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    dataset = SyntheticMoleculeDataset(
        n_molecules=cfg["n_molecules"],
        n_atoms=cfg["n_atoms"],
        n_node_features=cfg["n_node_features"],
        seed=cfg["seed"],
    )

    n_train = int(0.8 * len(dataset))
    n_val = int(0.1 * len(dataset))
    n_test = len(dataset) - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                               shuffle=True, collate_fn=collate_molecules)
    val_loader   = DataLoader(val_set,   batch_size=cfg["batch_size"],
                               shuffle=False, collate_fn=collate_molecules)
    test_loader  = DataLoader(test_set,  batch_size=cfg["batch_size"],
                               shuffle=False, collate_fn=collate_molecules)

    model = EGNN(
        n_node_features=cfg["n_node_features"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        edge_hidden=cfg["edge_hidden"],
        n_rbf=cfg["n_rbf"],
        cutoff=cfg["cutoff"],
        output_dim=1,
        dropout=cfg["dropout"],
        update_coords=cfg["update_coords"],
        coord_norm=cfg["coord_norm"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(cfg["n_epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            pos   = batch["pos"].to(DEVICE)
            feat  = batch["feat"].to(DEVICE)
            target = batch["energy"].to(DEVICE)
            bidx  = batch["batch"].to(DEVICE)

            pred = model(feat, pos, bidx)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pos   = batch["pos"].to(DEVICE)
                feat  = batch["feat"].to(DEVICE)
                target = batch["energy"].to(DEVICE)
                bidx  = batch["batch"].to(DEVICE)
                pred = model(feat, pos, bidx)
                val_loss += F.mse_loss(pred, target).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['n_epochs']} | "
                  f"Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")

    model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            pos   = batch["pos"].to(DEVICE)
            feat  = batch["feat"].to(DEVICE)
            target = batch["energy"].to(DEVICE)
            bidx  = batch["batch"].to(DEVICE)
            pred = model(feat, pos, bidx)
            all_pred.append(pred.cpu())
            all_true.append(target.cpu())

    y_pred = torch.cat(all_pred).numpy().ravel()
    y_true = torch.cat(all_true).numpy().ravel()
    test_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    test_r2 = r2_score(y_true, y_pred)
    test_mae = np.mean(np.abs(y_pred - y_true))

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }
    print(f"\nTest Results: RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | R²={test_r2:.4f}")

    return model, metrics


# ─────────────────────────────────────────────
# Ablation: EGNN vs Standard GNN vs MLP
# ─────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """
    Non-equivariant baseline: flatten all atom positions + features → MLP.
    Does NOT generalize across rotations — should underperform EGNN.
    """

    def __init__(self, n_atoms: int, n_node_features: int, hidden_dim: int):
        super().__init__()
        input_dim = n_atoms * (3 + n_node_features)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        n_mols = int(batch.max().item()) + 1
        n_atoms = pos.shape[0] // n_mols
        x = torch.cat([pos, node_feat], dim=-1)
        x = x.view(n_mols, -1)
        return self.net(x)


def compare_with_baseline(cfg: dict):
    """Compare EGNN vs MLP on test set, especially under rotation."""
    print("Training EGNN...")
    egnn_model, egnn_metrics = train_egnn(cfg)

    # MLP baseline (non-equivariant)
    print("\nTraining baseline MLP...")
    dataset = SyntheticMoleculeDataset(
        n_molecules=cfg["n_molecules"],
        n_atoms=cfg["n_atoms"],
        n_node_features=cfg["n_node_features"],
        seed=cfg["seed"],
    )
    n_train = int(0.8 * len(dataset))
    n_rest = len(dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_rest],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                               shuffle=True, collate_fn=collate_molecules)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_molecules)

    mlp = BaselineMLP(cfg["n_atoms"], cfg["n_node_features"], cfg["hidden_dim"]).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=cfg["lr"])
    for epoch in range(cfg["n_epochs"]):
        mlp.train()
        for batch in train_loader:
            pred = mlp(batch["feat"].to(DEVICE), batch["pos"].to(DEVICE),
                       batch["batch"].to(DEVICE))
            loss = F.mse_loss(pred, batch["energy"].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

    mlp.eval()
    mlp_preds, mlp_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = mlp(batch["feat"].to(DEVICE), batch["pos"].to(DEVICE),
                       batch["batch"].to(DEVICE))
            mlp_preds.append(pred.cpu()); mlp_true.append(batch["energy"])
    mlp_y = torch.cat(mlp_preds).numpy().ravel()
    mlp_t = torch.cat(mlp_true).numpy().ravel()
    mlp_rmse = np.sqrt(np.mean((mlp_y - mlp_t) ** 2))
    print(f"\nBaseline MLP: RMSE={mlp_rmse:.4f}")

    print(f"\nEGNN:         RMSE={egnn_metrics['test_rmse']:.4f} | R²={egnn_metrics['test_r2']:.4f}")
    print(f"Baseline MLP: RMSE={mlp_rmse:.4f}")
    print("\nNote: EGNN generalizes across rotations; MLP does not.")

    return egnn_model, egnn_metrics, mlp_rmse


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_results(model: EGNN, metrics: dict, cfg: dict):
    """Plot training curves, predictions, and equivariance test."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("EGNN — SE(3)-Equivariant GNN", fontsize=13)

    # Training curves
    axes[0].plot(metrics["train_losses"], label="Train MSE", color="steelblue")
    axes[0].plot(metrics["val_losses"],   label="Val MSE",   color="darkorange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predicted vs true energy on test set
    dataset = SyntheticMoleculeDataset(
        n_molecules=cfg["n_molecules"],
        n_atoms=cfg["n_atoms"],
        n_node_features=cfg["n_node_features"],
        seed=cfg["seed"] + 1,
    )
    loader = DataLoader(dataset[:200], batch_size=cfg["batch_size"],
                        shuffle=False, collate_fn=collate_molecules)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["feat"].to(DEVICE), batch["pos"].to(DEVICE),
                         batch["batch"].to(DEVICE))
            preds.append(pred.cpu()); trues.append(batch["energy"])
    y_pred = torch.cat(preds).numpy().ravel()
    y_true = torch.cat(trues).numpy().ravel()

    axes[1].scatter(y_true, y_pred, alpha=0.4, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", label="y=x")
    axes[1].set_xlabel("True Energy")
    axes[1].set_ylabel("Predicted Energy")
    axes[1].set_title(f"Pred vs True  (R²={metrics['test_r2']:.3f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SE(3) invariance error across multiple random rotations
    errors = []
    for _ in range(50):
        err = test_equivariance(model, n_atoms=cfg["n_atoms"],
                                 n_features=cfg["n_node_features"])
        errors.append(err)

    axes[2].hist(errors, bins=20, color="forestgreen", edgecolor="white")
    axes[2].axvline(1e-4, color="red", linestyle="--", label="1e-4 threshold")
    axes[2].set_xlabel("Max |ΔE| under rotation")
    axes[2].set_ylabel("Count")
    axes[2].set_title("SE(3) Invariance Test (50 rotations)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("egnn_results.png", dpi=150)
    plt.show()
    print("Saved: egnn_results.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")

    # ── Train EGNN ─────────────────────────────────────────────────────
    print("=== Training EGNN ===")
    model, metrics = train_egnn(CONFIG)

    # ── SE(3) Invariance Test ──────────────────────────────────────────
    print("\n=== SE(3) Invariance Test ===")
    test_equivariance(model, n_atoms=CONFIG["n_atoms"],
                       n_features=CONFIG["n_node_features"])

    # ── Visualize ─────────────────────────────────────────────────────
    plot_results(model, metrics, CONFIG)

    # ── Compare vs baseline ────────────────────────────────────────────
    print("\n=== EGNN vs Non-equivariant Baseline ===")
    compare_with_baseline({**CONFIG, "n_epochs": 50, "n_molecules": 500})


if __name__ == "__main__":
    main()
