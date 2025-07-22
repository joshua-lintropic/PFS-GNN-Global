import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import math
import numpy as np
import os
import config as cfg

def build_bipartite(num_src: int, num_tgt: int, class_info: Tensor, 
                    prob_edges: Tensor, seed: int = None) -> HeteroData:
    """
    Create a heterogeneous bipartite graph modeling PFS galaxy evolution exposures.
    """
    if seed is not None: 
        torch.manual_seed(seed)

    # === Source node feature construction. ===
    # Uniformly distribute positions in the unit disk using Vogel's method.
    # Resulting `src_pos` has size (num_src, 2). 
    golden_ratio = (1 + math.sqrt(5)) / 2
    golden_angle = 2 * math.pi * (1 - 1 / golden_ratio)
    src_idx = torch.arange(1, num_src + 1, dtype=torch.float)
    src_mod = torch.sqrt(src_idx / num_src)
    src_arg = src_idx * golden_angle
    src_x = src_mod * torch.cos(src_arg)
    src_y = src_mod * torch.sin(src_arg)
    src_pos = torch.stack([src_x, src_y], dim=1)

    # Set an inner radius. Source nodes cannot observe galaxies which are 
    # strictly closer than the inner radius. 
    r_inner = 0.001
    inner_radii = torch.full((num_src, 1), r_inner)

    # Set an outer radius. Source nodes cannot observe galaxies which are 
    # strictly further than the outer radius. 
    r_outer = 0.1
    outer_radii = torch.full((num_src, 1), r_outer)

    # Combine into source nodes. Resulting size of (num_src, 4). 
    src_nodes = torch.cat([src_pos, inner_radii, outer_radii], dim=1)

    # === Target node feature construction. ===
    # Evenly distribute class labels among the target nodes by interleaving. 
    # Resulting `labels` has size (num_tgt, 1). 
    labels = torch.cat([
        torch.full((int(count / cfg.num_fields),), float(time_req))
        for time_req, count in class_info
    ]).unsqueeze(1)

    # Number of exposures already received by target nodes. Initialized to zeros.
    exposures = torch.zeros((num_tgt, 1))

    # Random positions in the unit disk. 
    # Resulting `tgt_pos` has size (num_tgt, 2). 
    tgt_mod = torch.sqrt(torch.rand(num_tgt))
    tgt_arg = 2 * math.pi * torch.rand(num_tgt)
    tgt_x = tgt_mod * torch.cos(tgt_arg)
    tgt_y = tgt_mod * torch.sin(tgt_arg)
    tgt_pos = torch.stack([tgt_x, tgt_y], dim=1)

    # The intra-class priority of each galaxy. Sampled uniformly from [0,1). 
    priority = torch.rand((num_tgt, 1))

    # Combine into target nodes. Resulting size of (num_tgt, 5). 
    tgt_nodes = torch.cat([labels, exposures, tgt_pos, priority], dim=1)

    # === Edge connectivity construction. === 
    edge_src = []
    edge_tgt = []
    k = prob_edges.size(0)
    for t in range(num_tgt): 
        # Filter by distance constraints. Must be within the annulus of observation.
        dist = torch.norm(src_pos - tgt_pos[t], dim=1)
        valid = ((r_inner <= dist) & (dist <= r_outer)).nonzero(as_tuple=False).squeeze()
        if valid.numel() == 0: 
            continue
        # Sort valid by ascending distance, and take k nearest neighbors. 
        valid = valid[dist[valid].argsort()]
        k_nearest = valid[:k]
        for e, s in enumerate(k_nearest):
            if torch.rand(1).item() <= prob_edges[e].item():
                edge_src.append(s.item())
                edge_tgt.append(t)
        
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)
        edge_attr = torch.zeros((edge_index.size(1), cfg.total_exposures))
    else: 
        raise ValueError("No edges were generated")
    
    # === Global node feature construction. === 
    global_x = torch.zeros((cfg.num_rounds, cfg.lifted_dim))
    
    # === Build heterogeneous graph. === 
    data = HeteroData()
    data['src'].x = src_nodes
    data['tgt'].x = tgt_nodes
    data['src', 'to', 'tgt'].edge_index = edge_index
    data['src', 'to', 'tgt'].edge_attr = edge_attr
    data['global'].x = global_x

    return data


def main():
    class_info = np.loadtxt(os.path.join(cfg.data_dir, cfg.class_file), delimiter=',')
    class_info = torch.tensor(class_info)
    prob_edges = torch.tensor([0.0, 0.65, 0.3, 0.05])
    data = build_bipartite(num_src=cfg.num_fibers, 
                           num_tgt=int(cfg.num_galaxies/cfg.num_fields), 
                           class_info=class_info, 
                           prob_edges=prob_edges, 
                           seed=cfg.seed)
    torch.save(data, os.path.join(cfg.data_dir, cfg.graph_file))


if __name__ == '__main__':
    main()
