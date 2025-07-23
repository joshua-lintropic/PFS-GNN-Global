import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_scatter import scatter
from argparse import ArgumentParser
from os.path import join

from bipartite_data import BipartiteData
import config as cfg

def parse_args():
    parser = ArgumentParser(description="Train bipartite graph & optionally visualize.")
    parser.add_argument("--visualize", type=lambda x: (str(x).lower() == "true"), 
                        default=False, help="Toggle visualization (True/False)")
    return parser.parse_args()

def main(args): 
    # Construct and save bipartite graph. 
    class_info = np.loadtxt(join(cfg.data_dir, cfg.class_file), delimiter=',')
    class_info = torch.tensor(class_info)
    prob_edges = torch.tensor([0.0, 0.65, 0.3, 0.05])
    data = BipartiteData(num_src=cfg.num_fibers, 
                          num_tgt=int(cfg.num_galaxies/cfg.num_fields), 
                          class_info=class_info, 
                          prob_edges=prob_edges, 
                          device=cfg.device,
                          seed=cfg.seed)
    # torch.save(data, join(cfg.data_dir, cfg.data_file))

    # Visualize bipartite graph via 2D positions.
    if args.visualize:
        data.visualize(max_edges=50_000, 
                        edge_alpha=1.0, 
                        src_size=30, 
                        tgt_size=10, 
                        figsize=(16,16))

if __name__ == '__main__':
    args = parse_args()
    main(args)
