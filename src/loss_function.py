import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter
import numpy as np

def soft_floor(x: Tensor, sharpness: float, 
               noise_level: float = 0.3) -> Tensor:
    """
    Differentiable soft floor function, constructed via complex 
    logarithm. See https://www.desmos.com/calculator/qccszslg9b. 
    """
    noise = noise_level * (torch.rand_like(x) - 0.5)
    x = x + noise
    sharpness = x.new_tensor(sharpness)
    pi = x.new_tensor(np.pi)
    r = torch.where(sharpness == 0, 
                    torch.tensor(0.0, device=x.device), 
                    torch.exp(-1.0/sharpness))
    num = r * torch.sin(2*pi*x)
    den = 1 - r * torch.cos(2*pi*x)
    return x + 1/pi * torch.arctan(num/den) - torch.arctan(r/(1-r))


def compute_loss(data: HeteroData, class_info: Tensor, sharp: float):
    """
    Sub-differentiable loss function for the graph network. 
    """
    src, tgt = data['src', 'to', 'tgt'].edge_index

    # TODO: Compute the amount of time provided to each galaxy.