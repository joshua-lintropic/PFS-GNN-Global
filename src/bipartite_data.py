import torch
from torch import Tensor
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import math
import numpy as np
from argparse import ArgumentParser
from os.path import join
import config as cfg


def parse_args():
    parser = ArgumentParser(description="Train bipartite graph & optionally visualize.")
    parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), 
                        default=True, help='Toggle data saving (True/False)')
    parser.add_argument('--visualize', type=lambda x: (str(x).lower() == 'true'), 
                        default=False, help='Toggle visualization (True/False)')
    return parser.parse_args()


class BipartiteData(HeteroData):
    """
    Heterogeneous bipartite graph modeling PFS galaxy evolution exposures. 
    """
    def __init__(self) -> None:
        super().__init__()
        self['edge_rank'] = torch.empty(0)
        self['class_labels'] = torch.empty(0)
        self['class_info'] = torch.empty(0)
        self['time_req'] = torch.empty(0)
        self['time_spent'] = torch.empty(0)

        self.optimal = {
            'loss': np.inf, 
            'objective': -np.inf, 
            'epoch': -1, 
            'history': np.empty(0),
            'plan': np.empty(0)
        }

    def construct(self, num_src: int, num_tgt: int, class_info: Tensor, 
                prob_edges: Tensor, seed: int = None) -> HeteroData:
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
            seed:       controls random output. 
        """
        if seed is not None: 
            torch.manual_seed(seed)

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
        src_nodes = torch.cat([src_pos, inner_radii, outer_radii], dim=1)

        # === Target node feature construction. ===
        # Label the galaxies according to their class number. 
        # Resulting `labels` has size (num_tgt, 1). 
        labels = torch.cat([
            torch.full((int(count / cfg.num_fields),), float(i), device=cfg.device)
            for i, count in enumerate(class_info[:,1])
        ]).unsqueeze(1)

        # Required number of exposures to completion. Initialized to class values. 
        # Resulting `requirements` has size (num_tgt, 1).
        time_req = torch.cat([
            torch.full((int(count / cfg.num_fields),), float(time_req), device=cfg.device)
            for time_req, count in class_info
        ]).unsqueeze(1)

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
        tgt_nodes = torch.cat([labels, time_req, time_spent, tgt_pos, priority], dim=1)

        # === Edge connectivity construction. === 
        # Compute pairwise distances (vectorized). 
        dists = torch.cdist(src_pos, tgt_pos)
        valid = (dists >= r_inner)  & (dists <= r_outer)
        observable = dists.clone()
        observable[~valid] = float('inf')

        # For each target, get k nearest source candidates. 
        k = prob_edges.size(0)
        neighbors, idx_topk = observable.topk(k, dim=0, largest=False)

        # Sample number of edges per target according to prob_edges. 
        choices = np.arange(k)
        prob_edges_cpu = prob_edges.cpu().nump()
        edges_per_tgt = np.random.choice(choices, size=num_tgt, p=prob_edges_cpu)
        edges_per_tgt = torch.from_numpy(edges_per_tgt).to(cfg.device)

        # Build a mask of which (rank, target) pairs to include. 
        rank_idx = torch.arange(k, device=cfg.device).unsqueeze(1).expand(k, num_tgt)
        mask = (rank_idx < edges_per_tgt.unsqueeze(0)) & (neighbors != float('inf'))

        # Combine into edge lists. edge_pairs is shape (E, 2) holding [rank, tgt]. 
        edge_pairs = mask.nonzero(as_tuple=False)
        edge_rank = edge_pairs[:,0]
        tgt_index = edge_pairs[:, 1]
        src_index = idx_topk[edge_rank, tgt_indices]

        edge_index = torch.stack([src_index, tgt_index], dim=0)
        edge_attr = torch.rand((edge_index.size(1), cfg.total_exposures), device=cfg.device)
        
        # === Global node feature construction. === 
        global_x = torch.rand((1, cfg.global_dim), device=cfg.device)
        
        # === Build heterogeneous graph. === 
        self['src'].x = src_nodes
        self['tgt'].x = tgt_nodes
        self['src', 'to', 'tgt'].edge_index = edge_index
        self['src', 'to', 'tgt'].edge_attr = edge_attr
        self['global'].x = global_x

        self['edge_rank'] = torch.tensor(edge_rank, dtype=torch.long)
        self['class_labels'] = labels.to(torch.long)
        self['class_info'] = class_info
        self['time_req'] = time_req
        self['time_spent'] = time_spent

        self.optimal['history'] = np.zeros((cfg.num_histories, cfg.num_epochs))
        self.optimal['plan'] = np.zeros((num_src, num_tgt))
    
    def visualize(self, max_edges: int, edge_alpha: float, src_size: int,
                  tgt_size: int, figsize: tuple, path: str) -> None:
        """
        Scatter-plot src and tgt nodes at their 2D positions and draw (sampled) edges.

        Args:
            max_edges:  maximum number of edges to plot (randomly sampled).
            edge_alpha: transparency for edge lines.
            node_size:  marker size for node scatter.
            figsize:    size of the matplotlib figure.
        """
        src_pos = self['src'].x[:, :2].cpu().numpy()
        tgt_pos = self['tgt'].x[:, 3:5].cpu().numpy()

        # Sample edges if necessary.
        edge_index = self['src', 'to', 'tgt'].edge_index.cpu().numpy()
        edge_rank = self['edge_rank'].detach().cpu().numpy().astype(int)
        n_edges = edge_index.shape[1]
        if n_edges > max_edges:
            print(f'{n_edges} edges is too dense, truncating to {max_edges}')
            idx = np.random.choice(n_edges, max_edges, replace=False)
            edge_index = edge_index[:, idx]
            edge_rank = edge_rank[idx]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw sampled edges.
        edge_cmap = plt.get_cmap('viridis')
        unique_ranks = np.unique(edge_rank)
        edge_colors = {r: edge_cmap(i / (len(unique_ranks))) 
                for i, r in enumerate(unique_ranks)}
        for (s, t, r) in zip(edge_index[0], edge_index[1], edge_rank):
            x0, y0 = src_pos[s]
            x1, y1 = tgt_pos[t]
            ax.plot([x0, x1], [y0, y1], lw=0.5, alpha=edge_alpha, 
                    color=edge_colors[r], zorder=1)

        # Draw source nodes.
        node_cmap = plt.get_cmap('tab20')
        src_color = node_cmap(0)
        ax.scatter(src_pos[:,0], src_pos[:,1], c=[src_color], s=src_size, 
                marker='o', label='Fibers', zorder=2)

        # Color target nodes by their class label.
        class_labels = self['tgt'].x[:, 0].cpu().numpy()
        unique_labels = np.unique(class_labels)
        for idx, label in enumerate(unique_labels):
            mask = class_labels == label
            label_color = node_cmap((idx+1) / len(unique_labels))
            ax.scatter(tgt_pos[mask, 0], tgt_pos[mask, 1], s=tgt_size, 
                    c=[label_color], label=f'Class {int(label)}', 
                    edgecolor='k', linewidth=0.2, alpha=0.9, zorder=3)

        # Plot node legend. 
        node_handles, node_labels = ax.get_legend_handles_labels()
        node_legend = ax.legend(node_handles, node_labels, loc='upper right', 
                                fontsize='small')
        ax.add_artist(node_legend)

        edge_handles = [plt.Line2D([0],[0], color=edge_colors[r], lw=2)
                for r in unique_ranks]
        endings = ['th', 'st', 'nd', 'rd', 'th']
        edge_labels = [f'{r+1}{endings[min(r+1, len(endings)-1)]}-nearest' 
                    for r in unique_ranks]
        ax.legend(edge_handles, edge_labels, title='kth nearest source', 
                loc='upper left', fontsize='small')
        ax.set_title('PFS Fiber-Galaxy Spatial Visualization with Connectivity', fontsize=20)
        plt.tight_layout()
        plt.savefig(path, dpi=cfg.dpi)


def main(args): 
    # Construct and save bipartite graph. 
    if args.save: 
        class_info = np.loadtxt(join(cfg.data_dir, cfg.class_file), delimiter=',')
        class_info = torch.tensor(class_info)
        prob_edges = torch.tensor([0.0, 0.65, 0.3, 0.05])
        data = BipartiteData()
        data.construct(num_src=cfg.num_fibers, num_tgt=int(cfg.num_galaxies/cfg.num_fields), 
                    class_info=class_info, prob_edges=prob_edges, device=cfg.device, 
                    seed=cfg.seed)
        torch.save(data, join(cfg.data_dir, cfg.data_file))
    
    # Visualize bipartite graph via 2D positions.
    if args.visualize:
        data.visualize(max_edges=50_000, edge_alpha=1.0, src_size=30, tgt_size=10, 
                       figsize=(16,16), path=join(cfg.data_dir, cfg.viz_file))


if __name__ == '__main__': 
    args = parse_args()
    main(args)
