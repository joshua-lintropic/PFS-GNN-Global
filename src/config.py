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
graph_file = 'bipartite_graph.pt'
checkpoint_file = 'model_gnn_dev.pt'
pretrained_file = 'model_gnn_core.pt'

# problem parameters
num_galaxies = 338_900 
num_fibers = 2_000
num_fields = 10
total_exposures = 42

# hyperparameters
retrain = False
num_epochs = 10_000
lifted_dim = 10
learning_rate = 5e-4
weights = {
    'class_overtime': -0.1,
    'fiber_overtime': -0.1,
    'min_completion': 2000.0,
    'fiber_variance': 1.0,
}
sharps = (0.0, 100.0)
min_sharp = 50.0
