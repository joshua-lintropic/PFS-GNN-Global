# Neural Message Passing for Galaxy Evolution

This repo implements a deep learning pipeline to predict connectivity properties in bipartite graphs using PyTorch. It is designed to solve the allocation problem which the Prime Focus Spectrograph faces: a high-dimensional combinatorial optimization between fibers on the PFS telescope and galaxies in its fields of observation. 

For the class-based bipartite model, see [joshua-lintropic/PFS-Class-Optimizer](https://github.com/joshua-lintropic/PFS-Class-Optimizer).

![Bipartite Graph](data/bipartite_graph.png)

## Project Structure

- `bipartite_data.py`: Loads and prepares bipartite graph data, defines `BipartiteDataModule` using PyTorch Geometric.
- `config.py`: Stores configuration parameters used across the project, such as dataset paths, model dimensions, and training options.
- `loss_function.py`: Defines a custom loss function for handling class imbalance during training.
- `models.py`: Contains neural network model definitions based on the message-passing framework, as in the paper [Relational inductive bias, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).
- `train.py`: Main script for training and evaluating the model.

## Getting Started

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Construct the data:
    ```bash
    python bipartite_data.py --save=True --visualize=False
    ```

## ️ Configuration

Modify `config.py` to adjust:
- Dataset path
- Model architecture (e.g., hidden dimensions)
- Training settings (epochs, batch size, etc.)
