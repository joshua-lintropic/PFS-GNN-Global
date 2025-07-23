import torch
from torch import Tensor
from torch.nn import Linear, LeakyReLU, BatchNorm1d, RMSNorm, Sequential, Module
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_scatter import scatter_add, scatter_mean, scatter_softmax
import config as cfg


class EdgeModel(Sequential):
    """
    Small MLP to learn edge embeddings. 
    """
    def __init__(self, lifted_dim: int) -> None:
        message_dim = 4 * lifted_dim
        super().__init__(
            Linear(message_dim, message_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(message_dim, lifted_dim)
        )
        self.norm = torch.nn.BatchNorm1d(lifted_dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        src, tgt = edge_index
        E = edge_attr.size(0)
        h = torch.cat([x_s[src], x_t[tgt], edge_attr, x_u.expand(E,-1)], dim=-1)
        return self.norm(super().forward(h))


class SourceModel(torch.nn.Module):
    """
    Source-node update: aggregates incoming edge messages (with statistics)
    and updates source-node embeddings.
    """
    def __init__(self, lifted_dim: int) -> None:
        super().__init__()
        message_dim = 2 * lifted_dim
        self.message_mlp = Sequential(
            Linear(message_dim, message_dim),
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(message_dim, message_dim)
        )
        update_dim = 4 * message_dim + 2 * lifted_dim
        self.update_mlp = Sequential(
            Linear(update_dim, update_dim),
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(update_dim, lifted_dim)
        )
        self.norm = BatchNorm1d(lifted_dim)

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
        var = F.leaky_relu(scatter_mean(msg**2, src, dim=0, dim_size=x_s.size(0)) - mean**2)
        skew = scatter_mean((msg - mean[src])**3, src, dim=0, dim_size=x_s.size(0)) / std**3
        kurt = scatter_mean((msg - mean[src])**4, src, dim=0, dim_size=x_s.size(0)) / std**4

        # Zero out undefined values. 
        mean = torch.nan_to_num(mean, nan=0.0)
        var = torch.nan_to_num(var,  nan=0.0)
        std = torch.sqrt(var + 1e-6)
        skew = torch.nan_to_num(skew, nan=0.0)
        kurt = torch.nan_to_num(kurt, nan=0.0)

        h = torch.cat([x_s, mean, std, skew, kurt, x_u.expand(len(x_s), -1)], dim=-1)
        return self.norm(self.update_mlp(h))


class TargetModel(Module):
    """
    Target-node update: sums incoming edge messages and updates target-node embeddings.
    """
    def __init__(self, lifted_dim: int) -> None:
        super().__init__()
        message_dim = 2 * lifted_dim
        self.message_mlp = Sequential(
            Linear(message_dim, message_dim),
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(message_dim, message_dim)
        )
        update_dim = 4 * lifted_dim
        self.update_mlp = Sequential(
            Linear(update_dim, update_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(update_dim, lifted_dim)
        )
        self.norm = BatchNorm1d(lifted_dim)


    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Aggregates data from source nodes and edges. 
        """
        src, tgt = edge_index
        msg = torch.cat([x_s[src], edge_attr], dim=1)
        msg = self.message_mlp(msg)
        agg = scatter_add(msg, tgt, dim=0, dim_size=x_t.size(0))
        h_cat = torch.cat([x_t, agg, x_u.expand(len(x_t),-1)], dim=-1)
        return self.norm(self.update_mlp(h_cat))


class GlobalModel(Sequential):
    """
    Graph-level update: pools node embeddings to update global features.
    """
    def __init__(self, dim: int) -> None:
        message_dim = 3 * dim 
        super().__init__(
            Linear(message_dim, message_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(message_dim, dim)
        )
        self.norm = RMSNorm(dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Graph-level averaging from source and target nodes. 
        """
        src_mean = x_s.mean(dim=0, keepdim=True)
        tgt_mean = x_t.mean(dim=0, keepdim=True)
        h = torch.cat([x_u, src_mean, tgt_mean], dim=-1)
        return self.norm(super().forward(h))


class Block(Module):
    """
    A single MetaLayer block combining edge, source-node, target-node,
    and global update modules.
    """
    def __init__(self, lifted_src_dim: int, lifted_tgt_dim: int, 
                 lifted_edge_dim: int, global_dim: int) -> None:
        super().__init__()
        self.src_model = SourceModel(lifted_src_dim)
        self.tgt_model = TargetModel(lifted_tgt_dim)
        self.edge_model = EdgeModel(lifted_edge_dim)
        self.global_model = GlobalModel(global_dim)

    def forward(self, x_s: Tensor, x_t: Tensor, edge_index: Tensor, 
                edge_attr: Tensor, x_u: Tensor) -> Tensor:
        """
        Sequentially applies: edge -> source -> target -> global updates.
        """
        edge_attr = self.edge_model(x_s, x_t, edge_index, edge_attr, x_u)
        x_s = self.src_model(x_s, x_t, edge_index, edge_attr, x_u)
        x_t = self.tgt_model(x_s, x_t, edge_index, edge_attr, x_u)
        x_u = self.global_model(x_s, x_t, edge_index, edge_attr, x_u)
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
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(lifted_src_dim, lifted_src_dim)
        )
        self.tgt_encoder = Sequential(
            Linear(tgt_dim, lifted_tgt_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(lifted_tgt_dim, lifted_tgt_dim)
        )
        self.edge_encoder = Sequential(
            Linear(edge_dim, lifted_edge_dim),
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(lifted_edge_dim, lifted_edge_dim)
        )

        # Apply several rounds of message-passing blocks. 
        self.msg_pass_blocks = Sequential(*(Block(
            lifted_src_dim, lifted_tgt_dim, lifted_edge_dim, global_dim
        ) for _ in range(num_blocks)))

        # Decode node features into original-dimension representations. 
        self.src_decoder = Sequential(
            Linear(lifted_src_dim, lifted_src_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(lifted_src_dim, src_dim)
        )
        self.tgt_decoder = Sequential(
            Linear(lifted_tgt_dim, lifted_tgt_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(lifted_tgt_dim, tgt_dim)
        )
        self.edge_decoder = Sequential(
            Linear(lifted_edge_dim, lifted_edge_dim), 
            LeakyReLU(negative_slope=cfg.leaky_slope),
            Linear(lifted_edge_dim, cfg.total_exposures)
        )

    def forward(self, data: HeteroData) -> HeteroData:
        """
        Forward pass through MetaLayer-style message passing blocks.
        """
        x_s = data['src'].x
        x_t = data['tgt'].x
        edge_index = data['src', 'to', 'tgt'].edge_index
        edge_attr = data['src', 'to', 'tgt'].edge_attr
        x_u = data['global'].x

        # Pass bipartite nodes through encoders. 
        x_s = self.src_encoder(x_s)
        x_t = self.tgt_encoder(x_t)

        # Apply several rounds of MetaLayer-style message passing on graph. 
        x_s, x_t, (src, tgt), edge_attr, x_u = self.msg_pass_blocks(
            x_s, x_t, edge_index, edge_attr, x_u
        )

        data['src'].x = self.src_decoder(x_s)
        data['tgt'].x = self.tgt_decoder(x_t)
        data['src', 'to', 'tgt'].edge_attr = scatter_softmax(
            self.edge_decoder(edge_attr), src, dim=0
        )
        data['global'].x = x_u

        return data
