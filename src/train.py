import torch
from torch import Tensor
from torch.optim import Adam
from bipartite_data import BipartiteData
from models import GraphNetwork
import os
from tqdm import trange
from datetime import datetime

from loss_function import compute_loss, compute_upper_bound
import config as cfg


def to_scalar(x): 
    return x.item() if isinstance(x, Tensor) else x


def train_step(data: BipartiteData, model: GraphNetwork, 
               optimizer: Adam, epoch: int) -> None: 
    # Backpropagate the loss function. 
    optimizer.zero_grad()
    data = model(data)
    try: 
        param = (epoch - 1) / (cfg.num_epochs - 1) 
    except ZeroDivisionError: 
        param = 0
    sharpness = cfg.sharps[0] + param * (cfg.sharps[1] - cfg.sharps[0])
    loss, objective, observations = compute_loss(data, sharpness)
    loss.backward()
    optimizer.step()

    # Detach for the next iteration. 
    data['src'].x = data['src'].x.detach()
    data['tgt'].x = data['tgt'].x.detach()
    data['src', 'to', 'tgt'].edge_attr = data['src', 'to', 'tgt']\
        .edge_attr.detach()
    data['global'].x = data['global'].x.detach()

    # Store history for analysis. 
    loss_cpu = loss.detach().cpu().numpy()
    objective_cpu = objective.detach().cpu().numpy()
    data.optimal['history'][0][epoch] = loss_cpu
    data.optimal['history'][1][epoch] = objective_cpu

    # Checkpoint best-performing model. 
    if to_scalar(objective) >= data.optimal['objective']:
        data.optimal['loss'] = loss_cpu
        data.optimal['objective'] = objective_cpu
        data.optimal['epoch'] = epoch
        data['plan'] = observations.detach().cpu().numpy()
        if sharpness >= cfg.min_sharp or epoch == 1:
            torch.save(data, os.path.join(cfg.models_dir, cfg.checkpoint_file))
    
    return data


def train(): 
    # Create the message-passing network and lift the data. 
    data = torch.load(os.path.join(cfg.data_dir, cfg.data_file), weights_only=False)
    data = data.to(cfg.device)
    model = GraphNetwork(
        num_blocks = cfg.num_blocks, 
        src_dim = data['src'].x.size(1),
        tgt_dim = data['tgt'].x.size(1),
        edge_dim = data['src', 'to', 'tgt'].edge_attr.size(1),
        lifted_src_dim = cfg.lifted_src_dim, 
        lifted_tgt_dim = cfg.lifted_tgt_dim,
        lifted_edge_dim = cfg.lifted_edge_dim, 
        global_dim = cfg.global_dim
    )
    model = model.to(cfg.device)
    with torch.no_grad():
        data = model.encode(data)
    model.train()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    # Begin training loop. 
    desc = 'Training [Neural Message Passing for Galaxy Evolution]'
    for epoch in trange(1, cfg.num_epochs + 1, desc=desc): 
        data = train_step(data, model, optimizer, epoch)

    return data, model
    

def main(): 
    # Create models directory if it exists. 
    os.makedirs(cfg.models_dir, exist_ok=True)

    data, model = train()
    time = datetime.now()
    time = time.strftime("%B %-d, %Y @ %I:%M:%S %p")

    # Calculate relative optimality.
    space = max(len(field) for field in data.optimal.keys()) + 4
    upper_bound = compute_upper_bound(data)
    ratio = data.optimal['objective'] / upper_bound
    
    # Write to log file. 
    os.makedirs(cfg.results_dir, exist_ok=True)
    log_path = os.path.join(cfg.results_dir, cfg.log_file)
    with open(log_path, 'w') as file: 
        file.write(f'TIME: {time}\n')
        file.write(f'Upper Bound: {upper_bound}\n')
        file.write(f'Optimality Ratio: {ratio}\n')
        for key, val in data.optimal.items():
            file.write(f'{(key):<{space}} @ Optimal Objective: {val}\n')

if __name__ == '__main__': 
    main()
