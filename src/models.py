# models.py
import torch
from torch import Tensor
from torch.nn import Linear, LeakyReLU, Embedding, RMSNorm
from torch.nn import Sequential, Module, ModuleList
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

from bipartite_data import BipartiteData
import config as cfg


class MessagePassingEdgeModel(Module):
    """
    Edge update: takes node features for sources and targets with their 
    corresponding edges to update edge features. 
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None: 
        super().__init__()
        message_dim = lifted_src_dim + lifted_tgt_dim + lifted_edge_dim + global_dim
        self.update_mlp = Sequential(
            Linear(message_dim, message_dim),
            LeakyReLU(cfg.leaky_slope),
            Linear(message_dim, lifted_edge_dim)
        )
        self.norm = RMSNorm(lifted_edge_dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor,
        edge_attr: Tensor, x_u: Tensor) -> Tensor:
        src, tgt = edge_index
        E = edge_attr.size(0)
        h = torch.cat([x_s[src], x_t[tgt], edge_attr, x_u.expand(E,-1)], dim=-1)
        return self.norm(self.update_mlp(h))

class AttentionEdgeModel(Module):
    """
    Graph attention to udpate edge features. 
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()
        # Project each term into the same edge space. 
        self.lin_src  = Linear(lifted_src_dim, lifted_edge_dim, bias=False)
        self.lin_tgt  = Linear(lifted_tgt_dim, lifted_edge_dim, bias=False)
        self.lin_edge = Linear(lifted_edge_dim, lifted_edge_dim, bias=False)

        # Scalar attention score
        self.attn     = Linear(lifted_edge_dim, 1, bias=False)
        self.leaky    = LeakyReLU(0.2)
        self.norm     = RMSNorm(lifted_edge_dim)
    
    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor: 
        src, tgt = edge_index

        # Project into shared space. 
        h_src  = self.lin_src(x_s[src])
        h_tgt  = self.lin_tgt(x_t[tgt])
        h_edge = self.lin_edge(edge_attr)
        h      = h_src + h_tgt + h_edge

        # Compute un‐normalized attention per edge. 
        e = self.leaky(self.attn(h)).squeeze(-1)

        # Normalize across all edges sharing the same source node. 
        alpha = scatter_softmax(e, src, dim=0)

        # Weight the combined features. 
        h = h * alpha.unsqueeze(-1)

        return self.norm(h)

class SourceModel(Module):
    """
    Source-node update: aggregates incoming edge messages (with statistics)
    and updates source-node embeddings.
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()
        message_dim = lifted_tgt_dim + lifted_edge_dim
        self.message_mlp = Sequential(
            Linear(message_dim, message_dim),
            LeakyReLU(cfg.leaky_slope),
            Linear(message_dim, message_dim)
        )
        update_dim = lifted_src_dim + 4 * message_dim + global_dim
        self.update_mlp = Sequential(
            Linear(update_dim, update_dim),
            LeakyReLU(cfg.leaky_slope),
            Linear(update_dim, lifted_src_dim)
        )
        self.norm = RMSNorm(lifted_src_dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Aggregates statistical data from target nodes and edges. 
        """
        # Compute statistical moments. 
        src, tgt = edge_index
        msg = torch.cat([x_t[tgt], edge_attr], dim=1)
        msg = self.message_mlp(msg)

        mean = scatter_mean(msg, src, dim=0, dim_size=x_s.size(0))
        mean = torch.nan_to_num(mean, nan=0.0)

        var = F.leaky_relu(scatter_mean(msg**2, src, dim=0, dim_size=x_s.size(0)) - mean**2)
        var = torch.nan_to_num(var,  nan=0.0)
        std = torch.sqrt(var + 1e-6)

        skew = scatter_mean((msg - mean[src])**3, src, dim=0, dim_size=x_s.size(0)) / std**3
        skew = torch.nan_to_num(skew, nan=0.0)

        kurt = scatter_mean((msg - mean[src])**4, src, dim=0, dim_size=x_s.size(0)) / std**4
        kurt = torch.nan_to_num(kurt, nan=0.0)

        h = torch.cat([x_s, mean, std, skew, kurt, x_u.expand(len(x_s), -1)], dim=-1)
        return self.norm(self.update_mlp(h))


class TargetModel(Module):
    """
    Target-node update: sums incoming edge messages and updates target-node embeddings.
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()
        message_dim = lifted_src_dim + lifted_edge_dim
        self.message_mlp = Sequential(
            Linear(message_dim, message_dim),
            LeakyReLU(cfg.leaky_slope),
            Linear(message_dim, message_dim)
        )
        update_dim = lifted_tgt_dim + message_dim + global_dim
        self.update_mlp = Sequential(
            Linear(update_dim, update_dim), 
            LeakyReLU(cfg.leaky_slope),
            Linear(update_dim, lifted_tgt_dim)
        )
        self.norm = RMSNorm(lifted_tgt_dim)


    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Aggregates data from source nodes and edges. 
        """
        src, tgt = edge_index
        msg = torch.cat([x_s[src], edge_attr], dim=1)
        msg = self.message_mlp(msg)
        agg = scatter_add(msg, tgt, dim=0, dim_size=x_t.size(0))
        h = torch.cat([x_t, agg, x_u.expand(len(x_t),-1)], dim=-1)
        return self.norm(self.update_mlp(h))


class GlobalModel(Module):
    """
    Graph-level update: pools node embeddings to update global features.
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()
        update_dim = global_dim + lifted_src_dim + lifted_tgt_dim
        self.mlp = Sequential(
            Linear(update_dim, update_dim), 
            LeakyReLU(cfg.leaky_slope),
            Linear(update_dim, global_dim)
        )
        self.norm = RMSNorm(global_dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Graph-level averaging from source and target nodes. 
        """
        src_mean = x_s.mean(dim=0, keepdim=True)
        tgt_mean = x_t.mean(dim=0, keepdim=True)
        h = torch.cat([x_u, src_mean, tgt_mean], dim=-1)
        return self.norm(self.mlp(h))


class Block(Module):
    """
    A single MetaLayer block combining edge, source-node, target-node,
    and global update modules.
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()
        self.src_model = SourceModel(lifted_src_dim, lifted_tgt_dim, 
                                     lifted_edge_dim, global_dim)
        self.tgt_model = TargetModel(lifted_src_dim, lifted_tgt_dim, 
                                     lifted_edge_dim, global_dim)
        self.edge_model = AttentionEdgeModel(lifted_src_dim, lifted_tgt_dim, 
                                    lifted_edge_dim, global_dim)
        self.global_model = GlobalModel(lifted_src_dim, lifted_tgt_dim, 
                                        lifted_edge_dim, global_dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Sequentially applies: edge -> source -> target -> global updates.
        Includes residual connections for faster convergence. 
        """
        edge_res = self.edge_model(x_s, x_t, edge_index, edge_attr, x_u)
        edge_attr = edge_attr + edge_res
        src_res = self.src_model(x_s, x_t, edge_index, edge_attr, x_u)
        x_s = x_s + src_res
        tgt_res = self.tgt_model(x_s, x_t, edge_index, edge_attr, x_u)
        x_t = x_t + tgt_res
        global_res = self.global_model(x_s, x_t, edge_index, edge_attr, x_u)
        x_u = x_u + global_res
        return x_s, x_t, edge_index, edge_attr, x_u


class GraphNetwork(Module):
    """
    Full Message-Passing Neural Network stacking multiple MetaLayer-style blocks 
    to predict a discrete time value per edge via a differentiable rounding scheme.
    """
    def __init__(self, num_blocks: int, src_dim: int, tgt_dim: int, 
                 edge_dim: int, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()

        # Encode node features into higher-dimensional representation.
        self.src_encoder = Sequential(
            Linear(src_dim, lifted_src_dim),
            LeakyReLU(cfg.leaky_slope),
            Linear(lifted_src_dim, lifted_src_dim)
        )
        self.tgt_encoder = Sequential(
            Linear(tgt_dim + lifted_tgt_dim, lifted_tgt_dim), 
            LeakyReLU(cfg.leaky_slope),
            Linear(lifted_tgt_dim, lifted_tgt_dim)
        )
        self.edge_encoder = Sequential(
            Linear(edge_dim, lifted_edge_dim),
            LeakyReLU(cfg.leaky_slope),
            Linear(lifted_edge_dim, lifted_edge_dim)
        )

        # Class embedding. 
        self.class_embedding = Embedding(cfg.num_classes, lifted_tgt_dim)

        # Apply several rounds of message-passing blocks. 
        self.msg_pass_blocks = ModuleList([
            Block(lifted_src_dim, lifted_tgt_dim, lifted_edge_dim, global_dim)
            for _ in range(cfg.num_blocks)
        ])

        # Decode node features into original-dimension representations. 
        self.src_decoder = Sequential(
            Linear(lifted_src_dim, lifted_src_dim), 
            LeakyReLU(cfg.leaky_slope),
            Linear(lifted_src_dim, src_dim)
        )
        self.tgt_decoder = Sequential(
            Linear(lifted_tgt_dim, lifted_tgt_dim), 
            LeakyReLU(cfg.leaky_slope),
            Linear(lifted_tgt_dim, tgt_dim)
        )
        self.edge_decoder = Sequential(
            Linear(lifted_edge_dim, lifted_edge_dim), 
            LeakyReLU(cfg.leaky_slope),
            Linear(lifted_edge_dim, cfg.total_exposures)
        )

    def forward(self, data: BipartiteData) -> BipartiteData:
        """
        Forward pass through MetaLayer-style message passing blocks.
        """
        x_s, x_t, edge_index, edge_attr, x_u = data.x_s, data.x_t, \
            data.edge_index, data.edge_attr, data.x_u

        # Encode features into a higher-dimensional representation. 
        labels = data.x_t[:,0].long()
        ce = self.class_embedding(labels)
        x_s = self.src_encoder(x_s)
        x_t = self.tgt_encoder(torch.cat([x_t, ce], dim=1))
        edge_attr = self.edge_encoder(edge_attr)

        # Save initial embeddings for global residual connection. 
        x_s0, x_t0, edge_attr0, x_u0 = x_s, x_t, edge_attr, x_u

        # Apply several rounds of MetaLayer-style message passing on graph. 
        for block in self.msg_pass_blocks: 
            x_s, x_t, edge_index, edge_attr, x_u = block(
                x_s, x_t, edge_index, edge_attr, x_u
            )
        src, _ = edge_index

        # Add global residual connection. 
        x_s = x_s + x_s0
        x_t = x_t + x_t0
        edge_attr = edge_attr + edge_attr0
        x_u = x_u + x_u0

        # Prepare for exposure-wise softmax grouped by source nodes. 
        edge_attr = self.edge_decoder(edge_attr)

        return BipartiteData(x_s, x_t, edge_index, edge_attr, x_u)

