import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import numpy as np
from tqdm import trange, tqdm
from datetime import datetime
import os

from models import GraphNetwork
from bipartite_data import BipartiteData, Fossil, construct_data
from loss_function import compute_loss, compute_upper_bound
from visualize import plot_history
import config as cfg


def train_step(data: BipartiteData, fossil: Fossil, model: GraphNetwork, 
               optimizer: AdamW, epoch: int, history: dict, 
               optimal: dict) -> tuple[np.generic]: 
    # Backpropagate the loss function. 
    optimizer.zero_grad()
    data_ = model(data)
    try: 
        sharpness = cfg.sharps[0] + (cfg.sharps[1]-cfg.sharps[0]) \
            * (epoch-1)/(cfg.num_epochs-1)
    except ZeroDivisionError: 
        sharpness = cfg.sharps[0]
    loss, objective = compute_loss(data_, fossil, sharpness)
    loss.backward()
    optimizer.step()

    # Store history for analysis. 
    loss_cpu = loss.detach().cpu().numpy()
    objective_cpu = objective.detach().cpu().numpy()
    history['loss'][epoch-1] = loss_cpu
    history['objective'][epoch-1] = objective_cpu

    # Checkpoint best-performing model. 
    if objective_cpu >= optimal['objective']:
        optimal['loss'] = loss_cpu
        optimal['objective'] = objective_cpu
        optimal['epoch'] = epoch
        torch.save(
            model.state_dict(), 
            os.path.join(cfg.models_dir, cfg.checkpoint_file)
        )
    
    return loss_cpu, objective_cpu


def train(): 
    # Create the message-passing network and lift the data. 
    class_info = np.loadtxt(os.path.join(
        cfg.data_dir, cfg.class_file), delimiter=','
    )
    class_info = torch.tensor(class_info, device=cfg.device)
    prob_edges = torch.tensor(cfg.prob_edges, device=cfg.device)
    data, fossil = construct_data(
        num_src = cfg.num_fibers,
        num_tgt = cfg.num_galaxies // cfg.num_fields,
        class_info = class_info, 
        prob_edges = prob_edges
    )
    data = data.to(cfg.device)
    model = GraphNetwork(
        num_blocks = cfg.num_blocks, 
        src_dim = data.x_s.size(1),
        tgt_dim = data.x_t.size(1),
        edge_dim = data.edge_attr.size(1),
        lifted_src_dim = cfg.lifted_src_dim, 
        lifted_tgt_dim = cfg.lifted_tgt_dim,
        lifted_edge_dim = cfg.lifted_edge_dim, 
        global_dim = cfg.global_dim
    )
    model = model.to(cfg.device)

    # Perform the initial encoding into a higher-dimensional space. 
    with torch.no_grad():
        data = model.encode(data)

    # Initialize information tracking. 
    history = {
        'loss': np.zeros(cfg.num_epochs),
        'objective': np.zeros(cfg.num_epochs),
    }
    optimal = {
        'loss': np.inf, 
        'objective': -np.inf,
        'epoch': -1,
    }

    # Begin training and optimization. 
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    desc = 'Training Neural Message Passing for Galaxy Evolution'
    progress_bar = trange(1, cfg.num_epochs + 1, desc=desc)
    for epoch in progress_bar: 
        loss, objective = train_step(data, fossil, model, optimizer, 
                                     epoch, history, optimal)
        progress_bar.set_postfix(loss=loss, objective=objective)

    return data, fossil, model, history, optimal
    

def main(): 
    # Create models directory if it exists. 
    os.makedirs(cfg.models_dir, exist_ok=True)

    data, fossil, model, history, optimal = train()
    time = datetime.now()
    time = time.strftime("%B %-d, %Y @ %I:%M:%S %p")

    # Calculate relative optimality.
    upper_bound = compute_upper_bound(fossil)
    ratio = optimal['objective'] / upper_bound
    
    # Write optimal to log file. 
    os.makedirs(cfg.results_dir, exist_ok=True)
    log_path = os.path.join(cfg.results_dir, cfg.log_file)
    excluded_keys = ['plan']
    with open(log_path, 'w') as file: 
        file.write(f'TIME: {time}\n')
        file.write(f'Upper Bound: {upper_bound}\n')
        file.write(f'Optimality Ratio: {ratio}\n')
        for key, val in optimal.items(): 
            if key in excluded_keys: 
                continue
            file.write(f'Optimal {key}: {val}\n')
    
    # Plot the results. 
    plot_history(history, optimal)

if __name__ == '__main__': 
    main()
