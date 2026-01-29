#!/usr/bin/env python3
"""
GNN Baselines for CIF → Nanoparticle Property Prediction

This module implements Graph Neural Network baselines:
1. SchNet - Continuous-filter convolutional neural network
2. CGCNN - Crystal Graph Convolutional Neural Network  
3. E3NN - E(3)-equivariant neural network

Data Split Logic:
    R values: 10, 11, 12, 13, ..., 30
    
    TRAIN: R values NOT in ID or OOD = [12, 14, 16, 18, 19, 21, 22, 23, 25, 26, 28]
    ID (Test): [13, 15, 17, 20, 24, 27]  - In-distribution test
    OOD (Test): [10, 11, 29, 30]          - Out-of-distribution test

Directory structure:
    {base_dir}/{Material}/R{value}/xyz/rot_0.xyz
    Example: /Users/jp/SCALAR/scalar/quaternions/Ag/R10/xyz/rot_0.xyz

Usage:
    python gnn_baselines.py --data benchmark.jsonl --xyz-dir /path/to/quaternions/ --output results/
"""

import argparse
import time
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check for torch_geometric
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
    from torch_geometric.nn import radius_graph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not available. Install with: pip install torch-geometric")

# Check for e3nn
try:
    import e3nn
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet
    from e3nn.o3 import Irreps, spherical_harmonics
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    warnings.warn("e3nn not available. Install with: pip install e3nn")

# Check for ASE
try:
    from ase.io import read as ase_read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available. Install with: pip install ase")


# =============================================================================
# Data Split Configuration
# =============================================================================

# R value splits - THIS IS THE KEY CONFIGURATION
R_VALUES_ALL = list(range(10, 31))  # R10 to R30

R_SPLITS = {
    "ID": [13, 15, 17, 20, 24, 27],      # In-distribution TEST
    "OOD": [10, 11, 29, 30],              # Out-of-distribution TEST
}

# TRAIN = everything NOT in ID or OOD
R_SPLITS["TRAIN"] = [r for r in R_VALUES_ALL if r not in R_SPLITS["ID"] and r not in R_SPLITS["OOD"]]
# TRAIN = [12, 14, 16, 18, 19, 21, 22, 23, 25, 26, 28]


def print_data_split_info():
    """Print information about data splits."""
    logger.info("=" * 60)
    logger.info("DATA SPLIT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"All R values: {R_VALUES_ALL}")
    logger.info(f"TRAIN R values ({len(R_SPLITS['TRAIN'])}): {R_SPLITS['TRAIN']}")
    logger.info(f"ID TEST R values ({len(R_SPLITS['ID'])}): {R_SPLITS['ID']}")
    logger.info(f"OOD TEST R values ({len(R_SPLITS['OOD'])}): {R_SPLITS['OOD']}")
    logger.info("=" * 60)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    hidden_channels: int = 128
    num_layers: int = 4
    num_rbf: int = 50
    cutoff: float = 5.0
    max_neighbors: int = 32
    
    # Training
    batch_size: int = 8  # Smaller for large nanoparticles
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 20
    
    # Validation split (from TRAIN data)
    val_fraction: float = 0.15
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Properties to predict
    target_properties: List[str] = field(default_factory=lambda: [
        'num_atoms', 'mass_amu', 'convex_hull_volume', 'density'
    ])


# =============================================================================
# Atomic Data
# =============================================================================

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
}

MAX_ATOMIC_NUMBER = 100


# =============================================================================
# XYZ File Parsing
# =============================================================================

def parse_xyz_file(xyz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse XYZ file and return atomic numbers and positions.
    
    XYZ format:
        Line 1: Number of atoms
        Line 2: Comment (ignored)
        Line 3+: Element X Y Z
    
    Returns:
        atomic_numbers: (N,) array of atomic numbers
        positions: (N, 3) array of positions in Angstroms
    """
    logger.debug(f"Parsing XYZ file: {xyz_path}")
    
    if ASE_AVAILABLE:
        try:
            atoms = ase_read(xyz_path)
            atomic_numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            logger.debug(f"  → Loaded {len(atomic_numbers)} atoms via ASE")
            return atomic_numbers, positions
        except Exception as e:
            logger.warning(f"ASE failed to read {xyz_path}: {e}, using fallback parser")
    
    # Fallback: manual parsing
    atomic_numbers = []
    positions = []
    
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    
    for i in range(2, min(2 + n_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 4:
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            z_num = ATOMIC_NUMBERS.get(element, 0)
            if z_num == 0:
                try:
                    z_num = int(element)
                except ValueError:
                    z_num = 1
            
            atomic_numbers.append(z_num)
            positions.append([x, y, z])
    
    logger.debug(f"  → Loaded {len(atomic_numbers)} atoms via fallback parser")
    return np.array(atomic_numbers), np.array(positions)


def get_xyz_path(base_dir: str, material: str, r_value: int) -> Path:
    """
    Construct XYZ file path from components.
    
    Pattern: {base_dir}/{Material}/R{r_value}/xyz/rot_0.xyz
    Example: /Users/jp/SCALAR/scalar/quaternions/Ag/R10/xyz/rot_0.xyz
    """
    return Path(base_dir) / material / f"R{r_value}" / "xyz" / "rot_0.xyz"


# =============================================================================
# Dataset
# =============================================================================

class NanoparticleDataset(Dataset):
    """
    Dataset for nanoparticle structures.
    
    Each item contains:
        - XYZ structure (atoms + positions)
        - Ground truth properties
        - Metadata (material, r_value, split)
    """
    
    def __init__(
        self,
        data_items: List[Dict],
        xyz_base_dir: str,
        target_properties: List[str],
        cutoff: float = 5.0,
        split_name: str = "unknown",
    ):
        self.data_items = data_items
        self.xyz_base_dir = Path(xyz_base_dir)
        self.target_properties = target_properties
        self.cutoff = cutoff
        self.split_name = split_name
        
        logger.info(f"Creating {split_name} dataset with {len(data_items)} items")
        
        # Verify XYZ files exist
        self._verify_files()
        
        # Compute normalization statistics
        self._compute_stats()
        
    def _verify_files(self):
        """Verify all XYZ files exist."""
        missing = []
        for item in self.data_items:
            xyz_path = get_xyz_path(
                self.xyz_base_dir, 
                item['material'], 
                item['r_value']
            )
            if not xyz_path.exists():
                missing.append(str(xyz_path))
        
        if missing:
            logger.warning(f"Missing {len(missing)} XYZ files in {self.split_name} dataset:")
            for path in missing[:5]:
                logger.warning(f"  - {path}")
            if len(missing) > 5:
                logger.warning(f"  ... and {len(missing) - 5} more")
        else:
            logger.info(f"  ✓ All {len(self.data_items)} XYZ files verified")
    
    def _compute_stats(self):
        """Compute mean and std for target normalization."""
        self.target_means = {}
        self.target_stds = {}
        
        for prop in self.target_properties:
            values = [item['ground_truth'].get(prop, 0) for item in self.data_items]
            self.target_means[prop] = np.mean(values)
            self.target_stds[prop] = np.std(values) + 1e-8
            
        logger.info(f"  Target statistics for {self.split_name}:")
        for prop in self.target_properties:
            logger.info(f"    {prop}: mean={self.target_means[prop]:.2f}, std={self.target_stds[prop]:.2f}")
    
    def set_normalization(self, means: Dict, stds: Dict):
        """Set normalization stats (use training stats for val/test)."""
        self.target_means = means
        self.target_stds = stds
        logger.info(f"  Using external normalization stats for {self.split_name}")
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # Get XYZ path
        xyz_path = get_xyz_path(
            self.xyz_base_dir,
            item['material'],
            item['r_value']
        )
        
        # Parse structure
        atomic_numbers, positions = parse_xyz_file(str(xyz_path))
        
        # Convert to tensors
        z = torch.tensor(atomic_numbers, dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float32)
        
        # Get normalized targets
        targets = []
        for prop in self.target_properties:
            val = item['ground_truth'].get(prop, 0)
            val_norm = (val - self.target_means[prop]) / self.target_stds[prop]
            targets.append(val_norm)
        
        y = torch.tensor(targets, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(z=z, pos=pos, y=y)
        data.material = item['material']
        data.r_value = item['r_value']
        data.split = item['split']
        data.num_atoms_actual = len(atomic_numbers)
        
        # Store original targets for evaluation
        data.y_original = torch.tensor(
            [item['ground_truth'].get(prop, 0) for prop in self.target_properties],
            dtype=torch.float32
        )
        
        return data
    
    def denormalize(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        denorm = predictions.clone()
        for i, prop in enumerate(self.target_properties):
            denorm[:, i] = predictions[:, i] * self.target_stds[prop] + self.target_means[prop]
        return denorm


def collate_fn(data_list):
    """Custom collate function for PyG data."""
    return Batch.from_data_list(data_list)


# =============================================================================
# SchNet Model
# =============================================================================

class GaussianSmearing(nn.Module):
    """Gaussian smearing of distances for radial basis functions."""
    
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        
    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CFConv(MessagePassing):
    """Continuous-filter convolution layer."""
    
    def __init__(self, in_channels, out_channels, num_filters, nn_module, cutoff):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = nn_module
        self.cutoff = cutoff
        
    def forward(self, x, edge_index, edge_weight, edge_attr):
        W = self.nn(edge_attr)
        C = 0.5 * (torch.cos(edge_weight * np.pi / self.cutoff) + 1.0)
        C = C * (edge_weight < self.cutoff).float()
        W = W * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x
    
    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(nn.Module):
    """SchNet interaction block."""
    
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = nn.SiLU()
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(nn.Module):
    """
    SchNet: Continuous-filter convolutional neural network.
    
    Paper: Schütt et al., J. Chem. Phys. 148, 241722 (2018)
    
    Architecture:
        1. Atom embedding (atomic number → vector)
        2. Gaussian distance expansion (distance → radial basis)
        3. Interaction blocks (message passing with learned filters)
        4. Global sum pooling
        5. Output MLP
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_targets: int = 4,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        # Step 1: Atom embedding
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER, hidden_channels)
        
        # Step 2: Distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        # Step 3: Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ])
        
        # Step 5: Output network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, num_targets),
        )
        
        logger.info("SchNet initialized:")
        logger.info(f"  hidden_channels={hidden_channels}, num_interactions={num_interactions}")
        logger.info(f"  cutoff={cutoff}Å, num_gaussians={num_gaussians}")
        
    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        
        # Build radius graph (connect atoms within cutoff)
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors
        )
        
        # Compute edge features
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        
        # Embedding
        x = self.embedding(z)
        
        # Interaction blocks (with residual connections)
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)
        
        # Global pooling (sum over all atoms in each graph)
        x = global_add_pool(x, batch)
        
        # Output
        out = self.output_network(x)
        
        return out


# =============================================================================
# CGCNN Model
# =============================================================================

class CGConv(MessagePassing):
    """Crystal Graph Convolutional layer with gated activation."""
    
    def __init__(self, channels, edge_dim):
        super().__init__(aggr='add')
        self.lin_src = nn.Linear(channels, channels)
        self.lin_dst = nn.Linear(channels, channels)
        self.lin_edge = nn.Linear(edge_dim, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        z = self.lin_src(x_i) + self.lin_dst(x_j) + self.lin_edge(edge_attr)
        z = self.bn1(z)
        z_filter = torch.sigmoid(z)
        z_core = F.softplus(z)
        return z_filter * z_core
    
    def update(self, aggr_out, x):
        return F.softplus(self.bn2(aggr_out + x))


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.
    
    Paper: Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018)
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_targets: int = 4,
    ):
        super().__init__()
        
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.convs = nn.ModuleList([
            CGConv(hidden_channels, num_gaussians)
            for _ in range(num_layers)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Softplus(),
            nn.Linear(hidden_channels // 2, num_targets),
        )
        
        logger.info("CGCNN initialized:")
        logger.info(f"  hidden_channels={hidden_channels}, num_layers={num_layers}")
        
    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors
        )
        
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        
        x = self.embedding(z)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        x = global_mean_pool(x, batch)
        out = self.output(x)
        
        return out


# =============================================================================
# E3NN Model (Simplified)
# =============================================================================

class E3NNConvSimple(nn.Module):
    """Simplified E(3)-equivariant convolution (scalar features only)."""
    
    def __init__(self, in_channels, out_channels, num_basis=10, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        
        self.message_net = nn.Sequential(
            nn.Linear(in_channels + num_basis, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        
    def forward(self, x, edge_index, edge_vec, edge_length):
        row, col = edge_index
        
        # Radial basis
        edge_basis = self._radial_basis(edge_length)
        
        # Cutoff
        cutoff_val = 0.5 * (torch.cos(edge_length * np.pi / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (edge_length < self.cutoff).float()
        
        # Message
        msg_input = torch.cat([x[col], edge_basis], dim=-1)
        messages = self.message_net(msg_input) * cutoff_val.unsqueeze(-1)
        
        # Aggregate
        out = torch.zeros(x.shape[0], messages.shape[-1], device=x.device)
        out.index_add_(0, row, messages)
        
        return out
    
    def _radial_basis(self, x):
        centers = torch.linspace(0, self.cutoff, self.num_basis, device=x.device)
        width = (centers[1] - centers[0]) * 0.5
        return torch.exp(-((x.unsqueeze(-1) - centers) / width) ** 2)


class E3NNModel(nn.Module):
    """
    Simplified E(3)-equivariant neural network.
    
    Uses scalar features only for simplicity.
    """
    
    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_basis: int = 10,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_targets: int = 4,
        **kwargs,  # Ignore extra args like lmax
    ):
        super().__init__()
        
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER, hidden_channels)
        
        self.convs = nn.ModuleList([
            E3NNConvSimple(hidden_channels, hidden_channels, num_basis, cutoff)
            for _ in range(num_layers)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, num_targets),
        )
        
        logger.info("E3NN (simplified) initialized:")
        logger.info(f"  hidden_channels={hidden_channels}, num_layers={num_layers}")
        
    def forward(self, data):
        z, pos, batch = data.z, data.pos, data.batch
        
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch,
            max_num_neighbors=self.max_num_neighbors
        )
        
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_length = edge_vec.norm(dim=-1)
        
        x = self.embedding(z)
        
        for conv in self.convs:
            x = x + conv(x, edge_index, edge_vec, edge_length)
        
        x = global_mean_pool(x, batch)
        out = self.output(x)
        
        return out


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch with detailed logging."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        loss = criterion(pred, batch.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
        
        if batch_idx % 10 == 0:
            logger.debug(f"  Batch {batch_idx}: loss={loss.item():.4f}, "
                        f"graphs={batch.num_graphs}, atoms={batch.z.shape[0]}")
    
    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, loader, dataset, device, split_name="test"):
    """Evaluate model and compute metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_splits = []
    all_materials = []
    all_r_values = []
    
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        
        # Denormalize predictions
        pred_denorm = dataset.denormalize(pred.cpu())
        
        all_preds.append(pred_denorm)
        all_targets.append(batch.y_original.cpu())
        
        # Handle batch attributes
        for i in range(batch.num_graphs):
            all_splits.append(batch.split[i] if hasattr(batch, 'split') else 'unknown')
            all_materials.append(batch.material[i] if hasattr(batch, 'material') else 'unknown')
            all_r_values.append(batch.r_value[i] if hasattr(batch, 'r_value') else 0)
    
    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics
    metrics = compute_metrics(preds, targets, all_splits, dataset.target_properties)
    
    return metrics, preds, targets, all_r_values


def compute_metrics(preds, targets, splits, property_names):
    """Compute evaluation metrics by split."""
    metrics = defaultdict(lambda: defaultdict(dict))
    splits = np.array(splits)
    
    for split_name in ['ID', 'OOD', 'TRAIN', 'all']:
        if split_name == 'all':
            mask = np.ones(len(splits), dtype=bool)
        else:
            mask = splits == split_name
        
        if mask.sum() == 0:
            continue
        
        for i, prop in enumerate(property_names):
            y_pred = preds[mask, i]
            y_true = targets[mask, i]
            
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            mape = np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8))) * 100
            
            metrics[split_name][prop] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'n_samples': int(mask.sum()),
            }
    
    return dict(metrics)


def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    val_dataset,
    config: TrainingConfig,
    model_name: str,
    output_dir: Path,
):
    """Train model with early stopping and detailed logging."""
    device = torch.device(config.device)
    model = model.to(device)
    
    optimizer = Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {model_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"Learning rate: {config.learning_rate}, Batch size: {config.batch_size}")
    
    for epoch in range(config.num_epochs):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, val_dataset, device, "val")
        
        # Compute validation loss
        val_mae = np.mean([
            val_metrics.get('all', {}).get(prop, {}).get('mae', 0)
            for prop in config.target_properties
        ])
        
        scheduler.step(val_mae)
        
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                f"train_loss={train_loss:.4f} | val_mae={val_mae:.4f} | "
                f"lr={current_lr:.2e} | time={epoch_time:.1f}s"
            )
        
        # Early stopping
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f'{model_name}_best.pt')
            logger.debug(f"  → New best model saved (val_mae={val_mae:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / f'{model_name}_best.pt'))
    logger.info(f"Loaded best model with val_mae={best_val_loss:.4f}")
    
    return model, best_metrics


# =============================================================================
# Data Loading and Splitting
# =============================================================================

def load_benchmark_data(jsonl_path: str) -> Dict[str, List[Dict]]:
    """
    Load benchmark data and split by R value.
    
    Returns:
        Dict with keys 'TRAIN', 'ID', 'OOD', each containing list of items
    """
    logger.info(f"Loading data from {jsonl_path}")
    
    all_items = {'TRAIN': [], 'ID': [], 'OOD': []}
    materials_seen = set()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            material = data['material']
            materials_seen.add(material)
            
            for item in data['items']:
                r_value = item['r_value']
                
                # Determine split based on R value
                if r_value in R_SPLITS['ID']:
                    split = 'ID'
                elif r_value in R_SPLITS['OOD']:
                    split = 'OOD'
                elif r_value in R_SPLITS['TRAIN']:
                    split = 'TRAIN'
                else:
                    logger.warning(f"R value {r_value} not in any split, skipping")
                    continue
                
                item_data = {
                    'material': material,
                    'r_value': r_value,
                    'split': split,
                    'ground_truth': item['ground_truth'],
                }
                all_items[split].append(item_data)
    
    logger.info(f"Loaded {len(materials_seen)} materials")
    logger.info(f"  TRAIN: {len(all_items['TRAIN'])} items")
    logger.info(f"  ID:    {len(all_items['ID'])} items")
    logger.info(f"  OOD:   {len(all_items['OOD'])} items")
    
    return all_items


def discover_xyz_files(xyz_base_dir: str, materials: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Discover XYZ files from directory structure.
    
    Use this when you don't have a benchmark JSONL file.
    
    Directory structure: {base_dir}/{Material}/R{value}/xyz/rot_0.xyz
    """
    logger.info(f"Discovering XYZ files from {xyz_base_dir}")
    
    base_path = Path(xyz_base_dir)
    all_items = {'TRAIN': [], 'ID': [], 'OOD': []}
    
    # Find all material directories
    if materials is None:
        materials = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    for material in materials:
        mat_dir = base_path / material
        if not mat_dir.exists():
            continue
        
        for r_dir in mat_dir.iterdir():
            if not r_dir.is_dir() or not r_dir.name.startswith('R'):
                continue
            
            try:
                r_value = int(r_dir.name[1:])
            except ValueError:
                continue
            
            xyz_file = r_dir / 'xyz' / 'rot_0.xyz'
            if not xyz_file.exists():
                continue
            
            # Determine split
            if r_value in R_SPLITS['ID']:
                split = 'ID'
            elif r_value in R_SPLITS['OOD']:
                split = 'OOD'
            elif r_value in R_SPLITS['TRAIN']:
                split = 'TRAIN'
            else:
                continue
            
            # Compute ground truth from XYZ file
            atomic_numbers, positions = parse_xyz_file(str(xyz_file))
            num_atoms = len(atomic_numbers)
            
            # Simple ground truth (you may want to compute more properties)
            ground_truth = {
                'num_atoms': num_atoms,
                'mass_amu': num_atoms * 107.87,  # Placeholder, assumes Ag
                'convex_hull_volume': (4/3) * np.pi * (r_value ** 3),  # Approximate
                'density': num_atoms / ((4/3) * np.pi * (r_value ** 3)),
            }
            
            item_data = {
                'material': material,
                'r_value': r_value,
                'split': split,
                'ground_truth': ground_truth,
            }
            all_items[split].append(item_data)
    
    logger.info(f"Discovered {len(materials)} materials")
    logger.info(f"  TRAIN: {len(all_items['TRAIN'])} items")
    logger.info(f"  ID:    {len(all_items['ID'])} items")
    logger.info(f"  OOD:   {len(all_items['OOD'])} items")
    
    return all_items


# =============================================================================
# Main Runner
# =============================================================================

def run_gnn_baselines(
    data_path: str,
    xyz_base_dir: str,
    output_dir: str,
    models_to_run: List[str] = None,
    config: TrainingConfig = None,
    discover_mode: bool = False,
):
    """Run GNN baselines with proper train/test split."""
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric required. Install with: pip install torch-geometric")
    
    if config is None:
        config = TrainingConfig()
    
    if models_to_run is None:
        models_to_run = ['schnet', 'cgcnn', 'e3nn']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Print split configuration
    print_data_split_info()
    
    # Load data
    if discover_mode:
        all_items = discover_xyz_files(xyz_base_dir)
    else:
        all_items = load_benchmark_data(data_path)
    
    # Split TRAIN into train/val
    train_items = all_items['TRAIN']
    np.random.seed(42)
    np.random.shuffle(train_items)
    
    n_val = int(len(train_items) * config.val_fraction)
    val_items = train_items[:n_val]
    train_items = train_items[n_val:]
    
    # Mark val items
    for item in val_items:
        item['split'] = 'VAL'
    
    logger.info("\nFinal data splits:")
    logger.info(f"  Train: {len(train_items)} items")
    logger.info(f"  Val:   {len(val_items)} items")
    logger.info(f"  ID Test:  {len(all_items['ID'])} items")
    logger.info(f"  OOD Test: {len(all_items['OOD'])} items")
    
    # Create datasets
    train_dataset = NanoparticleDataset(
        train_items, xyz_base_dir, config.target_properties, 
        config.cutoff, split_name="TRAIN"
    )
    val_dataset = NanoparticleDataset(
        val_items, xyz_base_dir, config.target_properties,
        config.cutoff, split_name="VAL"
    )
    
    # Use training stats for val/test normalization
    val_dataset.set_normalization(train_dataset.target_means, train_dataset.target_stds)
    
    # Combined test set (ID + OOD)
    test_items = all_items['ID'] + all_items['OOD']
    test_dataset = NanoparticleDataset(
        test_items, xyz_base_dir, config.target_properties,
        config.cutoff, split_name="TEST"
    )
    test_dataset.set_normalization(train_dataset.target_means, train_dataset.target_stds)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # Model definitions
    model_classes = {
        'schnet': lambda: SchNet(
            hidden_channels=config.hidden_channels,
            num_filters=config.hidden_channels,
            num_interactions=config.num_layers,
            num_gaussians=config.num_rbf,
            cutoff=config.cutoff,
            max_num_neighbors=config.max_neighbors,
            num_targets=len(config.target_properties),
        ),
        'cgcnn': lambda: CGCNN(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            num_gaussians=config.num_rbf,
            cutoff=config.cutoff,
            max_num_neighbors=config.max_neighbors,
            num_targets=len(config.target_properties),
        ),
        'e3nn': lambda: E3NNModel(
            hidden_channels=config.hidden_channels // 2,
            num_layers=config.num_layers,
            num_basis=config.num_rbf // 5,
            cutoff=config.cutoff,
            max_num_neighbors=config.max_neighbors,
            num_targets=len(config.target_properties),
        ),
    }
    
    all_results = {}
    
    # Train and evaluate each model
    for model_name in models_to_run:
        if model_name not in model_classes:
            logger.warning(f"Unknown model: {model_name}")
            continue
        
        try:
            # Create model
            model = model_classes[model_name]()
            
            # Train
            model, val_metrics = train_model(
                model, train_loader, val_loader,
                train_dataset, val_dataset,
                config, model_name, output_path
            )
            
            # Evaluate on test set
            logger.info(f"\nEvaluating {model_name} on test set...")
            test_metrics, preds, targets, r_values = evaluate(
                model, test_loader, test_dataset, config.device, "test"
            )
            
            # Print results
            logger.info(f"\n{'='*70}")
            logger.info(f"{model_name.upper()} TEST RESULTS")
            logger.info(f"{'='*70}")
            logger.info(f"{'Property':<20} {'ID MAE':>10} {'ID MAPE%':>10} {'OOD MAE':>10} {'OOD MAPE%':>10}")
            logger.info('-' * 70)
            
            for prop in config.target_properties:
                id_mae = test_metrics.get('ID', {}).get(prop, {}).get('mae', float('nan'))
                id_mape = test_metrics.get('ID', {}).get(prop, {}).get('mape', float('nan'))
                ood_mae = test_metrics.get('OOD', {}).get(prop, {}).get('mae', float('nan'))
                ood_mape = test_metrics.get('OOD', {}).get(prop, {}).get('mape', float('nan'))
                
                logger.info(f"{prop:<20} {id_mae:>10.2f} {id_mape:>10.2f} {ood_mae:>10.2f} {ood_mape:>10.2f}")
            
            all_results[model_name] = test_metrics
            
            # Save model
            torch.save(model.state_dict(), output_path / f'{model_name}_final.pt')
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_file = output_path / 'gnn_baseline_metrics.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary comparison
    logger.info(f"\n{'='*70}")
    logger.info("GNN BASELINE COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"\n{'Model':<15} {'num_atoms ID':>15} {'num_atoms OOD':>15}")
    logger.info('-' * 50)
    
    for model_name, metrics in all_results.items():
        id_mape = metrics.get('ID', {}).get('num_atoms', {}).get('mape', float('nan'))
        ood_mape = metrics.get('OOD', {}).get('num_atoms', {}).get('mape', float('nan'))
        logger.info(f"{model_name:<15} {id_mape:>14.2f}% {ood_mape:>14.2f}%")
    
    logger.info(f"\nResults saved to {results_file}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNN baselines for nanoparticle property prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Split:
    TRAIN R values: [12, 14, 16, 18, 19, 21, 22, 23, 25, 26, 28]
    ID TEST R values: [13, 15, 17, 20, 24, 27]
    OOD TEST R values: [10, 11, 29, 30]

Examples:
    # With benchmark JSONL file
    python gnn_baselines.py --data benchmark.jsonl --xyz-dir /path/to/quaternions/
    
    # Discover XYZ files from directory (no JSONL needed)
    python gnn_baselines.py --xyz-dir /path/to/quaternions/ --discover
    
    # Train specific model
    python gnn_baselines.py --data benchmark.jsonl --xyz-dir quaternions/ --model schnet
    
    # Custom hyperparameters
    python gnn_baselines.py --data benchmark.jsonl --xyz-dir quaternions/ \\
        --hidden 256 --layers 6 --epochs 200 --batch-size 4
        """
    )
    
    parser.add_argument('--data', '-d', help='Path to benchmark JSONL file')
    parser.add_argument('--xyz-dir', '-x', required=True, help='Base directory for XYZ files')
    parser.add_argument('--output', '-o', default='gnn_results', help='Output directory')
    parser.add_argument('--model', '-m', nargs='+', default=None,
                        choices=['schnet', 'cgcnn', 'e3nn'],
                        help='Models to train (default: all)')
    parser.add_argument('--discover', action='store_true',
                        help='Discover XYZ files from directory instead of using JSONL')
    
    # Hyperparameters
    parser.add_argument('--hidden', type=int, default=128, help='Hidden channels')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff distance (Å)')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    args = parser.parse_args()
    
    if not args.data and not args.discover:
        parser.error("Either --data or --discover is required")
    
    # Create config
    config = TrainingConfig(
        hidden_channels=args.hidden,
        num_layers=args.layers,
        cutoff=args.cutoff,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
    
    # Run
    run_gnn_baselines(
        data_path=args.data,
        xyz_base_dir=args.xyz_dir,
        output_dir=args.output,
        models_to_run=args.model,
        config=config,
        discover_mode=args.discover,
    )


if __name__ == '__main__':
    main()