import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_scatter import scatter_sum, scatter_max 
import numpy as np

from bipartite_data import BipartiteData
import config as cfg


def soft_round(x: Tensor, sharpness: float, 
               noise_level: float = 0.3) -> Tensor:
    """
    Differentiable soft round function, motivated by the complex
    logarithm. See https://www.desmos.com/calculator/i42op1wlrp.

    Tends to identity as sharpness -> 0. 
    Tends to round as sharpness -> infinity. 
    """
    noise = noise_level * (torch.rand_like(x) - 0.5)
    x = x + noise
    sharpness = x.new_tensor(sharpness)
    pi = x.new_tensor(np.pi)
    r = torch.where(sharpness == 0, 
                    torch.tensor(0.0, device=x.device), 
                    torch.exp(-1.0/sharpness))
    num = r * torch.sin(2*pi*x)
    den = 1 + r * torch.cos(2*pi*x)
    return x - 1/pi * torch.arctan(num/den)


def soft_floor(x: Tensor, sharpness: float, 
               noise_level: float = 0.3) -> Tensor:
    """
    Differentiable soft floor function, motivated by the complex 
    logarithm. See https://www.desmos.com/calculator/qccszslg9b. 

    Tends to identity as sharpness -> 0. 
    Tends to floor as sharpness -> infinity. 
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


def discretize(edge_attr: Tensor, src: LongTensor) -> Tensor: 
    """
    Hard selection: for each source node s and each column j, pick the 
    edge e with max edge_attr[e,j] (softmax beforehand) and set 
    mask[e,j] = 1, else 0.

    Args: 
        edge_attr:  Values in [0,1], normalized by softmax per (s,j).  
        src:        src[e] is the source-node of edge e. 

    Returns: 
        One-hot encoding per (s, j). Same shape as edge_attr.
    """
    _, edge_dim = edge_attr.shape
    device = edge_attr.device

    num_src = int(src.max().item() + 1) 
    _, argmax = scatter_max(edge_attr, src, dim=0, dim_size=num_src)

    flat_s = argmax.reshape(-1)
    flat_j = torch.arange(edge_dim, device=device)\
        .unsqueeze(0).expand(num_src, edge_dim).reshape(-1)
    mask = torch.zeros_like(edge_attr)
    mask[flat_s, flat_j] = 1.0
    return mask


def straight_through_estimate(edge_attr: Tensor, src: LongTensor) -> Tensor: 
    """
    Apply a Straight-Through Estimator to backpropagate binary decisions. 

    Args: 
        edge_attr:  Values in [0,1], normalized by softmax per (s,j).  
        src:        src[e] is the source-node of edge e. 
    
    Returns: 
        Discretized one-hot selection with identity function gradients. 
    """
    discrete_mask = discretize(edge_attr, src)
    return discrete_mask.detach() - edge_attr.detach() + edge_attr


def compute_loss(data: BipartiteData, sharpness: float):
    """
    Sub-differentiable loss function for the graph network. 
    """
    src, tgt = data['src', 'to', 'tgt'].edge_index
    edge_attr = data['src', 'to', 'tgt'].edge_attr
    galaxy_requirement = F.leaky_relu(
        data['time_req'] - data['time_spent'],
        negative_slope=cfg.leaky_slope
    )

    # Compute the fraction of galaxies completed in each class.
    edge_attr = straight_through_estimate(edge_attr, src)
    observations = scatter_sum(
        edge_attr, tgt, dim=0, dim_size=data['tgt'].x.size(0)
    ).sum(dim=1) / galaxy_requirement
    observations.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    observations = soft_floor(observations, sharpness=sharpness)
    class_counts = scatter_sum(observations, data.class_labels)
    class_completion = class_counts / data['class_info'][:,1]
    min_completion = class_completion.min()

    # Compute the overtime by each source node. 
    fiber_overtime = scatter_sum(
        edge_attr, src, dim=0, dim_size=data['src'].x.size(0)
    )
    fiber_overtime = F.leaky_relu(
        fiber_overtime - torch.ones_like(fiber_overtime), 
        negative_slope=cfg.leaky_slope
    )**2

    # Compute loss and objective. 
    loss = cfg.weights['objective'] * min_completion + \
        cfg.weights['overtime'] * fiber_overtime
    return loss, min_completion, observations 


def compute_upper_bound(data: BipartiteData) -> float:
    """
    Compute upper bound on the minimum completion per class. 
    Assumes no discretization + all classes receive equal time.
    """
    total_required = torch.clamp(
        data['time_req'] - data['time_spent'], 
        min=0.0
    ).sum()
    total_available = cfg.total_exposures * cfg.num_fibers
    return total_available / total_required
    