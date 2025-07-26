# bipartite_data.py
import torch
from torch import Tensor, device
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import Self
import config as cfg


class BipartiteData():
    """
    Heterogeneous bipartite graph modeling PFS galaxy evolution exposures. 
    """
    def __init__(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                 edge_attr: Tensor, x_u: Tensor) -> None: 
        self.x_s = x_s
        self.x_t = x_t
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x_u = x_u

    def to(self, d: device) -> Self:
        self.x_s = self.x_s.to(d)
        self.x_t = self.x_t.to(d)
        self.edge_index = self.edge_index.to(d)
        self.edge_attr = self.edge_attr.to(d)
        self.x_u = self.x_u.to(d)

        return self


class Fossil():
    """
    Stores original state information for calculations. 
    """
    def __init__(self, edge_rank: Tensor, class_info: Tensor, 
                 class_labels: Tensor, time_req: Tensor, 
                 time_spent: Tensor) -> None:
        self.edge_rank = edge_rank
        self.class_info = class_info
        self.class_labels = class_labels
        self.time_req = time_req
        self.time_spent = time_spent
    
    def to(self, d: device) -> Self: 
        self.edge_rank = self.edge_rank.to(d)
        self.class_info = self.class_info.to(d)
        self.class_labels = self.class_labels.to(d)
        self.time_req = self.time_req.to(d)
        self.time_spent = self.time_spent.to(d)


def construct_data(num_src: int, num_tgt: int, class_info: Tensor,
                   prob_edges: Tensor) -> BipartiteData:
    """
    Constructs bipartite graph with stochastic node features. 

    Args: 
        num_src:    number of source nodes. 
        num_tgt:    number of target nodes. 
        class_info: (M, 2) tensor, where M is number of distinct classes. 
                    First column is time requirement per class.
                    Second column is number of galaxies per class. 
        prob_edges: probability vector. ith element denotes the (i.i.d.) 
                    probability that any target node has i edges. 

    Returns: 
        BipartiteData object with PFS problem parameters.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # === Source node feature construction. ===
    # Uniformly distribute positions in the unit disk using Vogel's method.
    # Resulting `src_pos` has size (num_src, 2). 
    golden_ratio = (1 + math.sqrt(5)) / 2
    golden_angle = 2 * math.pi * (1 - 1 / golden_ratio)
    src_idx = torch.arange(1, num_src + 1, dtype=torch.float, device=cfg.device)
    src_mod = torch.sqrt(src_idx / num_src)
    src_arg = src_idx * golden_angle
    src_x = src_mod * torch.cos(src_arg)
    src_y = src_mod * torch.sin(src_arg)
    src_pos = torch.stack([src_x, src_y], dim=1)

    # Set an inner and outer radius. Source nodes cannot observe galaxies 
    # which are outside the annulus defined by these radii. 
    r_inner, r_outer = cfg.annulus
    r_inner = torch.tensor(r_inner, device=cfg.device)
    r_outer = torch.tensor(r_outer, device=cfg.device)
    inner_radii = r_inner.expand(num_src, 1)
    outer_radii = r_outer.expand(num_src, 1)

    # Combine into source nodes. Resulting size of (num_src, 4). 
    x_s = torch.cat([src_pos, inner_radii, outer_radii], dim=1)

    # === Target node feature construction. ===
    # Label the galaxies according to their class number. 
    # Resulting `class_labels` has size (num_tgt,1). 
    repeats = [int(cnt.item()) // cfg.num_fields for cnt in class_info[:-1,1]]
    repeats.append(num_tgt - sum(repeats))
    class_labels = torch.repeat_interleave(
        torch.arange(cfg.num_classes, device=cfg.device).to(torch.float),
        torch.tensor(repeats, device=cfg.device)
    ).unsqueeze(1)

    # Required number of exposures to completion. Initialized to class values. 
    # Resulting `requirements` has size (num_tgt, 1).
    time_req = torch.repeat_interleave(
        class_info[:,0].to(torch.float),
        torch.tensor(repeats, device=cfg.device)
    ).unsqueeze(1)

    # Number of exposures already received by target nodes. Initialized to zeros.
    time_spent = torch.zeros((num_tgt, 1), device=cfg.device)

    # Random positions in the unit disk. 
    # Resulting `tgt_pos` has size (num_tgt, 2). 
    tgt_mod = torch.sqrt(torch.rand(num_tgt, device=cfg.device))
    tgt_arg = 2 * math.pi * torch.rand(num_tgt, device=cfg.device)
    tgt_x = tgt_mod * torch.cos(tgt_arg)
    tgt_y = tgt_mod * torch.sin(tgt_arg)
    tgt_pos = torch.stack([tgt_x, tgt_y], dim=1)

    # The intra-class priority of each galaxy. Sampled uniformly from [0,1). 
    priority = torch.rand((num_tgt, 1), device=cfg.device)

    # Combine into target nodes. Resulting size of (num_tgt, 6). 
    x_t = torch.cat([class_labels, time_req, time_spent, tgt_pos, priority], dim=1)

    # === Edge connectivity construction. === 
    # Compute pairwise distances (vectorized). 
    dists = torch.cdist(src_pos, tgt_pos)
    valid = (dists >= r_inner)  & (dists <= r_outer)
    observable = dists.masked_fill(~valid, float('inf'))

    # For each target, get k nearest source candidates. 
    k = prob_edges.size(0)
    neighbors, idx_topk = observable.topk(k, dim=0, largest=False)

    # Sample number of edges per target according to prob_edges. 
    choices = np.arange(k)
    prob_edges_cpu = prob_edges.cpu().numpy()
    edges_per_tgt = np.random.choice(choices, size=num_tgt, p=prob_edges_cpu)
    edges_per_tgt = torch.from_numpy(edges_per_tgt).to(cfg.device)

    # Build a mask of which (rank, target) pairs to include. 
    rank_idx = torch.arange(k, device=cfg.device).unsqueeze(1).expand(k, num_tgt)
    mask = (rank_idx < edges_per_tgt.unsqueeze(0)) & (neighbors != float('inf'))

    # Combine into edge lists. edge_pairs is shape (E, 2) holding [rank, tgt]. 
    edge_pairs = mask.nonzero(as_tuple=False)
    edge_rank = edge_pairs[:,0]
    tgt_index = edge_pairs[:,1]
    src_index = idx_topk[edge_rank, tgt_index]

    E = src_index.size(0)
    edge_index = torch.stack([src_index, tgt_index], dim=0)
    edge_attr = torch.clamp(torch.normal(
        mean=0.5, std=0.15, size=(E, cfg.total_exposures), device=cfg.device
    ), min=0.0)
    
    # === Global node feature construction. === 
    x_u = torch.full((1, cfg.global_dim), cfg.optimal, device=cfg.device)

    # === Build bipartite graph. ===
    data = BipartiteData(x_s, x_t, edge_index, edge_attr, x_u)

    # Store original state data. 
    fossil = Fossil(edge_rank, class_info, class_labels, time_req, time_spent)

    return data, fossil
