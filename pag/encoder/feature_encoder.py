import torch
from torch import nn
from torch_geometric.graphgym.models.layer import (
    new_layer_config,
    BatchNorm1dNode,
    BatchNorm1dEdge,
)

from pag.encoder.type_dict_encoder import TypeDictNodeEncoder, TypeDictEdgeEncoder

# from torch_geometric.graphgym.config import cfg


class FeatureEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.node_encoder = TypeDictNodeEncoder(node_in_dim, dim)
        self.node_encoder = nn.Linear(node_in_dim, dim)

        if edge_in_dim is not None:
            # self.edge_encoder = TypeDictEdgeEncoder(edge_in_dim, dim)
            self.edge_encoder = nn.Linear(edge_in_dim, dim)

        # self.node_bn = BatchNorm1dNode(
        #     new_layer_config(dim, -1, -1, has_act=False, has_bias=False, cfg=None)
        # )
        # if edge_in_dim is not None:
        #     self.edge_bn = BatchNorm1dEdge(
        #         new_layer_config(dim, -1, -1, has_act=False, has_bias=False, cfg=None)
        #     )

    def forward(self, data):
        # for module in self.children():
        #     data = module(data)
        # return data
        data.x = self.node_encoder(data.x)
        return data
