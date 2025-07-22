import torch
from torch.nn import Linear, LeakyReLU, Sequential, BatchNorm1d, RMSNorm, Module
from torch.nn.functional import leaky_relu, softplus
from torch_geometric.data import Data
from torch_scatter import scatter
import config as cfg

class BipartiteData(Data):
    def __init__(self, edge_index, x_s, x_t, x_e, x_u):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index.to(cfg.device)
        self.x_s = x_s.to(cfg.device)
        self.x_t = x_t.to(cfg.device)
        self.x_e = x_e.to(cfg.device)
        self.x_u = x_u.to(cfg.device)

class MLP(Sequential):
    def __init__(self, dimensions, negative_slopes=0.1):
        if type(negative_slopes) == float:
            negative_slopes = [negative_slopes for _ in range(len(dimensions) - 1)]
        if len(dimensions) != len(negative_slopes) + 1:
            raise IndexError("dimensions should be one longer than negative_slopes")
        layers = []
        for i in range(len(dimensions) - 1):
            in_dim = dimensions[i]
            out_dim = dimensions[i+1]
            layers.append(Linear(in_dim, out_dim))
            if i < len(dimensions) - 2:
                layers.append(LeakyReLU(negative_slopes[i]))
        super(MLP, self).__init__(*layers)

class EdgeModel(MLP):
    def __init__(self, lifted_dim):
        message_dim = 4 * lifted_dim
        super(EdgeModel, self).__init__(message_dim, message_dim, lifted_dim)
        self.norm = BatchNorm1d(lifted_dim)

    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        src, tgt = edge_index
        E = edge_attr.size(0)
        h = torch.cat([x_s[src], x_t[tgt], edge_attr, u.expand(E, -1)], dim=-1)
        return self.norm(super().forward(h))

class SourceModel(Module):
    def __init__(self, lifted_dim=10, normed=True):
        super(SourceModel, self).__init__()
        message_dim_1 = 2 * lifted_dim
        message_dim_2 = 4 * message_dim_1 + 2 * lifted_dim
        self.node_mlp_1 = MLP([message_dim_1, message_dim_1, message_dim_1])
        self.node_mlp_2 = MLP([message_dim_2, message_dim_2, lifted_dim])
        self.norm = BatchNorm1d(lifted_dim) if normed else lambda x: x

    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        src, tgt = edge_index
        in_msg = torch.cat([x_t[tgt], edge_attr], dim=1)
        out_msg = self.node_mlp_1(in_msg)

        # aggregate incoming message statistics
        mean = scatter(msg, src, dim=0, dim_size=x_s.size(0), reduce='mean')
        var = leaky_relu(scatter(msg**2, src, dim=0, dim_size=x_s.size(0), reduce='mean') - mean**2)
        std = torch.sqrt(var + 1e-6)
        skew = torch.nan_to_num(skew, nan=0.0)
        kurt = torch.nan_to_num(kurt, nan=0.0)

        h_cat = torch.cat([x_s, mean, std, skew, kurt, u.expand(len(x_s), -1)], dim=-1)
        return self.norm(self.node_mlp_2(h_cat))

class TargetModel(Module):
    def __init__(self, lifted_dim=10, normed=True):
        super(TargetModel, self).__init__()
        message_dim_1 = 2 * lifted_dim
        message_dim_2 = 4 * lifted_dimd
        self.node_mlp_1 = MLP([message_dim_1, message_dim_1, message_dim_1])
        self.node_mlp_2 = MLP([message_dim_2, message_dim_2, lifted_dim])
        self.norm = BatchNorm1d(lifted_dim) if normed else lambda x: x

    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        src, tgt = edge_index
        in_msg = torch.cat([x_s[src], edge_attr], dim=1)
        out_msg = self.node_mlp_1(in_msg)
        aggregation = scatter(msg, tgt, dim=0, dim_size=x_t.size(0), reduce='sum')
        h_cat = torch.cat([x_t, aggregation, u.expand(len(x_t), -1)], dim=-1)
        return self.norm(self.node_mlp_2(h_cat))

class GlobalModel(MLP):
    def __init__(self, lifted_dim=10, normed=True):
        message_dim = 3 * lifted_dim
        super(GlobalModel, self).__init__([message_dim, message_dim, lifted_dim])
        self.norm = RMSNorm(lifted_dim) if normed else lambda x: x

    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        s_mean = x_s.mean(dim=0, keepdim=True)
        t_mean = x_t.mean(dim=0, keepdim=True)
        h_cat = torch.cat([u, s_mean, t_mean], dim=-1)
        return self.norm(super().forward(h_cat))

class Block(Module):
    def __init__(self, lifted_dim=10, normed=True):
        super(Block, self).__init__()
        self.edge_model = EdgeModel(lifted_dim, normed=normed)
        self.source_model = SourceModel(lifted_dim, normed=normed)
        self.target_model = TargetModel(lifted_dim, normed=normed)
        self.global_model = GlobalModel(lifted_dim, normed=normed)

    def forward(self, edge_index, x_s, x_t, x_e, x_u):
        x_e = self.edge_model(x_s, x_t, edge_index, x_e, x_u)
        x_s = self.source_model(x_s, x_t, edge_index, x_e, x_u)
        x_t = self.target_model(x_s, x_t, edge_index, x_e, x_u)
        x_u = self.global_model(x_s, x_t, edge_index, x_e, x_u)
        return edge_index, x_s, x_t, x_e, x_u

class MPNN(Module):
    def __init__(self, num_blocks=4, lifted_dim=16, decoder_dim=num_galaxies, source_dim=1, target_dim=1, normed=True):
        super(MPNN, self).__init__()
        self.source_encoder = MLP(source_dim, lifted_dim, lifted_dim)
        self.target_encoder = MLP(target_dim, lifted_dim, lifted_dim)
        self.message_blocks = Sequential(*(Block(lifted_dim, normed=normed) for _ in range(num_blocks)))
        self.edge_decoder = MLP(lifted_dim, lifted_dim, 1)
        self.source_decoder = MLP(target_dim, lifted_dim, lifted_dim)

    def forward(self, graph):
        x_s = graph.x_s
        x_t = graph.x_t
        edge_index = graph.edge_index
        x_e = graph.x_e
        x_u = graph.x_u

        x_s = self.source_encoder(x_s)
        x_t = self.target_encoder(x_t)

        _, x_s, x_t, x_e, x_u = self.message_blocks(edge_index, x_s, x_t, x_e, x_u)

        return BipartiteData(edge_index, x_s, x_t, x_e, x_u)

    def edge_prediction(self, x_e, scale=1):
        pred = self.edge_decoder(x_e)
        pred = softplus(pred) * scale
        return pred

    def node_prediction(self, x_s, scale=1):
        pred = self.source_decoder(x_s)
        time = torch.softmax(pred, dim=-1) * scale
        return time
