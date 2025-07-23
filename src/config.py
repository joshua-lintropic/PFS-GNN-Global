import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# project directories
data_dir = '../data/'
models_dir = '../models/'
figures_dir = '../figures/'

# load and save points 
class_file = 'class_info.csv'
data_file = 'bipartite_graph.pt'
viz_file = 'bipartite_graph.png'
checkpoint_file = 'model_gnn_dev.pt'
pretrained_file = 'model_gnn_core.pt'

# problem parameters
num_galaxies = 338_900 # number of target nodes
num_fibers = 2_000 # number of source nodes
num_fields = 10 # partitioning of the sky
num_rounds = 4 # rounds of message-passing
total_exposures = 42 # number of observations stages
annulus = (1e-6, 1e-1)

# hyperparameters
retrain = False
num_epochs = 10_000
lifted_dim = 10
learning_rate = 5e-4
leaky_slope = 0.1
p_class = 0.1
p_fiber = 0.1
w_utils = 2000.0
w_var = 1.0
sharps = (0.0, 100.0)
min_sharp = 50.0

# miscellaneous 
seed = 42
dpi = 600
eps = 1e-6
