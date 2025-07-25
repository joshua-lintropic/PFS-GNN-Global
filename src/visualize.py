import matplotlib.pyplot as plt
import os

import config as cfg

import matplotlib.pyplot as plt
import os

import config as cfg

def plot_history(history: dict, optimal: dict) -> None:
    """
    Plot the loss and objective values over training epochs.

    Args:
        history (dict): Dictionary containing 'loss' and 'objective' arrays.
        optimal (dict): Dictionary containing optimal training metrics, including 'epoch'.
    """
    epochs = range(1, len(history['loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot loss over epochs
    ax1.plot(epochs, history['loss'], marker='o', color='red', label='Loss')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Draw vertical line at the epoch with optimal objective.
    ax1.axvline(x=optimal['epoch'], color='black', linestyle='--', label='Optimal Epoch')
    ax1.legend()

    # Plot objective over epochs.
    ax2.plot(epochs, history['objective'], marker='o', color='blue', label='Objective')
    ax2.set_title('Objective Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Objective')
    ax2.grid(True)
    
    # Draw vertical line at the epoch with optimal objective.
    ax2.axvline(x=optimal['epoch'], color='black', linestyle='--', label='Optimal Epoch')
    ax2.legend()

    # Plot overtime over epochs.
    ax3.plot(epochs, history['overtime'], marker='o', color='green', label='Overtime')
    ax3.set_title('Fiber Overtime Over Epochs')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Overtime')
    ax3.grid(True)
    
    # Draw vertical line at the epoch with optimal objective.
    ax3.axvline(x=optimal['epoch'], color='black', linestyle='--', label='Optimal Epoch')
    ax3.legend()

    # Plot completion for each class over epochs.
    num_classes = history['completion'].shape[0]
    cmap = plt.get_cmap('tab20')
    for i in range(num_classes):
        ax4.plot(epochs, history['completion'][i, :], color=cmap(i), label=f'Class {i}')
    ax4.set_title('Class Completion Over Epochs')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Completion')
    ax4.grid(True)
    
    # Draw vertical line at the epoch with optimal objective.
    ax4.axvline(x=optimal['epoch'], color='black', linestyle='--', label='Optimal Epoch')
    ax4.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, cfg.history_file), dpi=cfg.dpi)


# def visualize_data(data: BipartiteData, edge_rank: Tensor, class_labels: Tensor, 
#                    max_edges: int, edge_alpha: float, src_size: int, tgt_size: int, 
#                    figsize: tuple, path: str) -> None:
#     """
#     Scatter-plot src and tgt nodes at their 2D positions and draw (sampled) edges.

#     Args:
#         max_edges:  maximum number of edges to plot (randomly sampled).
#         edge_alpha: transparency for edge lines.
#         node_size:  marker size for node scatter.
#         figsize:    size of the matplotlib figure.

#     Returns: 
#         None
    
#     TODO: recompute edge_rank at runtime. 
#     """
#     src_pos = data.x_s[:, :2].cpu().numpy()
#     tgt_pos = data.x_t[:, 3:5].cpu().numpy()

#     # Sample edges if necessary.
#     edge_index = data.edge_index.cpu().numpy()
#     edge_rank = edge_rank.detach().cpu().numpy().astype(int)
#     n_edges = edge_index.shape[1]
#     if n_edges > max_edges:
#         print(f'{n_edges} edges is too dense, truncating to {max_edges}')
#         idx = np.random.choice(n_edges, max_edges, replace=False)
#         edge_index = edge_index[:, idx]
#         edge_rank = edge_rank[idx]

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_aspect('equal')
#     ax.axis('off')

#     # Draw sampled edges.
#     edge_cmap = plt.get_cmap('viridis')
#     unique_ranks = np.unique(edge_rank)
#     edge_colors = {r: edge_cmap(i / (len(unique_ranks))) 
#             for i, r in enumerate(unique_ranks)}
#     for (s, t, r) in zip(edge_index[0], edge_index[1], edge_rank):
#         x0, y0 = src_pos[s]
#         x1, y1 = tgt_pos[t]
#         ax.plot([x0, x1], [y0, y1], lw=0.5, alpha=edge_alpha, 
#                 color=edge_colors[r], zorder=1)

#     # Draw source nodes.
#     node_cmap = plt.get_cmap('tab20')
#     src_color = node_cmap(0)
#     ax.scatter(src_pos[:,0], src_pos[:,1], c=[src_color], s=src_size, 
#             marker='o', label='Fibers', zorder=2)

#     # Color target nodes by their class label.
#     class_labels = class_labels.cpu().numpy()
#     unique_labels = np.unique(class_labels)
#     for idx, label in enumerate(unique_labels):
#         mask = class_labels == label
#         label_color = node_cmap((idx+1) / len(unique_labels))
#         ax.scatter(tgt_pos[mask, 0], tgt_pos[mask, 1], s=tgt_size, 
#                 c=[label_color], label=f'Class {int(label)}', 
#                 edgecolor='k', linewidth=0.2, alpha=0.9, zorder=3)

#     # Plot node legend. 
#     node_handles, node_labels = ax.get_legend_handles_labels()
#     node_legend = ax.legend(node_handles, node_labels, loc='upper right', 
#                             fontsize='small')
#     ax.add_artist(node_legend)

#     edge_handles = [plt.Line2D([0],[0], color=edge_colors[r], lw=2)
#             for r in unique_ranks]
#     endings = ['th', 'st', 'nd', 'rd', 'th']
#     edge_labels = [f'{r+1}{endings[min(r+1, len(endings)-1)]}-nearest' 
#                 for r in unique_ranks]
#     ax.legend(edge_handles, edge_labels, title='kth nearest source', 
#             loc='upper left', fontsize='small')
#     ax.set_title('PFS Fiber-Galaxy Spatial Visualization with Connectivity', fontsize=20)
#     plt.tight_layout()
#     plt.savefig(path, dpi=cfg.dpi)
#     plt.closefig()