import torch
import numpy as np
import matploblit.pyplot as plt
from torch_scatter import scatter
from model import MPNN, BipartiteData
from config import *

def soft_floor(x, sharpness=20, noiselevel=0.3):
    noise = noiselevel * (torch.rand_like(x) - 0.5)
    x = x + noise
    sharpness = x.new_tensor(sharpness)
    pi = x.new_tensor(np.pi)
    r = torch.where(sharpness == 0, torch.tensor(0.0, device=x.device), torch.exp(-1/sharpness))
    return x + 1 / pi * (torch.arctan(r * torch.sin(2 * pi * x) / (1 - r * torch.cos(2 * pi * x))) - torch.arctan(r / (torch.ones_like(r) - r)))

def loss(graph, galaxy_info, penalties, sharpness=0.5, finaloutput=False):
    src, tgt = graph.edge_index
    galaxy_classes = galaxy_info[:,0] # first index will be class number
    class_counts = scatter(galaxy_classes, torch.arange(NUM_CLASSES), dim_size=NUM_CLASSES, reduce='sum')
    time_pred = MPNN

