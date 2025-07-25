# loss_function.py
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
import numpy as np

from bipartite_data import BipartiteData, Fossil
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


def smooth_min(x: Tensor, beta: float) -> Tensor:
    return  -1.0/beta * torch.logsumexp(-beta * x, dim=0)


def compute_loss(data: BipartiteData, fossil: Fossil, 
    sharpness: float, tau: float) -> tuple[Tensor]:
    """
    Smooth loss function that rewards minimum class completion and 
    punishes fiber overtime. 

    Args: 
        data:       The bipartite graph to evaluate. 
        fossil:     Saved galaxy data for benchmarking. 
        sharpness:  The level of discreteness for soft_round. 
    
    Returns: 
        Metrics for evaluating loss and progress. 
    """
    src, tgt = data.edge_index
    edge_attr = data.edge_attr
    galaxy_requirement = F.leaky_relu(
        fossil.time_req - fossil.time_spent,
        negative_slope=cfg.leaky_slope
    ).squeeze()

    """
    Compute the fraction of galaxies completed in each class. 
    Sizes: 
        fiber_action:       (num_edges, total_exposures)
        completion_mask:    (num_tgt,)
        class_*:            (num_classes,)
        min_completion:     (1,) 
    """
    logits = scatter_softmax(edge_attr, src, dim=0)
    fiber_action = F.gumbel_softmax(logits, tau=tau,  hard=True, dim=1)
    completion_mask = scatter_add(
        fiber_action, tgt, dim=0, dim_size=data.x_t.size(0)
    ).sum(dim=1) / (galaxy_requirement + cfg.eps)
    completion_mask = soft_floor(completion_mask, sharpness)
    completion_mask = torch.clamp(completion_mask, min=0.0, max=1.0)
    class_labels = fossil.class_labels.squeeze().to(torch.long)
    class_counts = scatter_add(
        completion_mask, class_labels, dim=0, dim_size=cfg.num_classes
    )
    class_completion = class_counts / fossil.class_info[:,1]
    class_completion = torch.clamp(class_completion, min=0.0, max=1.0)
    min_completion = smooth_min(class_completion, beta=cfg.beta)

    """
    Punishes fibers for trying to observe >1 galaxy per exposure. 
    Sizes:
        fiber_overtime:     (num_src, edge_dim)
    """
    # fiber_overtime = scatter_add(
    #     fiber_action, src, dim=0, dim_size=data.x_s.size(0)
    # )
    # fiber_overtime = torch.sum(F.leaky_relu(
    #     fiber_overtime - torch.ones_like(fiber_overtime), 
    #     negative_slope=cfg.leaky_slope
    # )**2)

    # Compute loss and objective. 
    # loss = cfg.weights['objective'] * min_completion + \
    #     cfg.weights['overtime'] * fiber_overtime
    loss = - min_completion
    return loss, min_completion, fiber_overtime, class_completion


def compute_upper_bound(fossil: Fossil) -> float:
    """
    Compute upper bound on the minimum completion per class. 
    Assumes no discretization + all classes receive equal time.

    Args: 
        fossil:     Fossil object with galaxy data. 
    
    Returns: 
        A non-discretized upper-bound on minimum completion. 
    """
    total_required = torch.clamp(
        fossil.time_req - fossil.time_spent, 
        min=0.0
    ).sum()
    total_available = cfg.total_exposures * cfg.num_fibers
    upper_bound = total_available / total_required
    return upper_bound.cpu().numpy()
    
