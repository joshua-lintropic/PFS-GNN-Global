import torch
from os.path import join

import config as cfg


def main(): 
    data = torch.load(join(cfg.data_dir, cfg.data_file), device=cfg.device)


if __name__ == '__main__': 
    main()
