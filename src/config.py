import torch
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    raise Exception('CUDA not found, please install a supported version')

# project directories
data_dir    = os.path.expanduser('~/PFS-GNN-Global/data/')
models_dir  = os.path.expanduser('~/PFS-GNN-Global/models/')
results_dir = os.path.expanduser('~/PFS-GNN-Global/results/')

# load and save points 
class_file = 'class_info.csv'
data_file = 'bipartite_data.pt'
graph_file = 'bipartite_graph.png'
history_file = 'history.png'
checkpoint_file = 'model_gnn_dev.pt'
pretrained_file = 'model_gnn_core.pt'
log_file = 'log.txt'

# problem parameters
num_galaxies = 338_900  # number of distinct galaxies (all fields)
num_classes = 12        # number of galaxy classes
num_fibers = 2_000      # number of available fibers
num_fields = 10         # partitioning of the sky
num_pointings = 3       # number of positions with different fiber views
num_blocks = 4          # number of message-passing rounds
total_exposures = 42    # number of observations stages
annulus = (0.0, 2.0)    # annulus of observation for each fiber
prob_edges = [          # chance that any galaxy is connected to k edges
    0.0, 
    0.65, 
    0.3, 
    0.05
]

# model specification
lifted_src_dim = 10
lifted_tgt_dim = 10
lifted_edge_dim = 50
global_dim = 10

# hyperparameters
retrain = False
num_epochs = 40_000
num_histories = 2
learning_rate = 5e-4
leaky_slope = 0.1
weights = {
    'objective': -2000.0, 
    'overtime': 0.1
}
sharps = (0.0, 10.0)
min_sharp = 5.0

# miscellaneous 
seed = 42
dpi = 600
eps = 1e-6
