import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# project directories
data_dir = '../data/'
figures_dir = '../figures/'
models_dir = '../models/'
results_dir = '../results/'

# load and save points 
class_file = 'class_info.csv'
data_file = 'bipartite_data.pt'
viz_file = 'bipartite_graph.png'
checkpoint_file = 'model_gnn_dev.pt'
pretrained_file = 'model_gnn_core.pt'
log_file = 'log.txt'

# problem parameters
num_galaxies = 338_900 # number of target nodes
num_fibers = 2_000 # number of source nodes
num_fields = 10 # partitioning of the sky
num_blocks = 4 # number of message-passing rounds
total_exposures = 42 # number of observations stages
annulus = (1e-6, 1e-1)

# model specification
lifted_src_dim = 10
lifted_tgt_dim = 10
lifted_edge_dim = 50
global_dim = 10

# hyperparameters
retrain = False
num_epochs = 20_000
num_histories = 2
learning_rate = 5e-4
leaky_slope = 0.1
weights = {
    'objective': 10.0, 
    'overtime': 0.1
}
min_completion_weight = 10.0
fiber_overtime_weight = 0.1
sharps = (0.0, 100.0)
min_sharp = 50.0

# miscellaneous 
seed = 42
dpi = 600
eps = 1e-6
