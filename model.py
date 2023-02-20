from torch_geometric.nn import MessagePassing, MetaLayer
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EdgeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super().__init__()
        self.residuals = residuals
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_features + n_edge_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(hiddens + n_features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)
        if self.residuals:
            out = out + x
        return out


def build_layer(self, n_hiddens, batchnorm, residuals):
    return MetaLayer(
        edge_model=EdgeModel(n_hiddens, n_hiddens, n_hiddens, n_hiddens, residuals=residuals),
        node_model=NodeModel(n_hiddens, n_hiddens, n_hiddens, n_hiddens, residuals=residuals),
    )