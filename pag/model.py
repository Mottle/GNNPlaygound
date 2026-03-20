import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from rum.models import RUMModel
from graph_ae.gnn import GNN
from torch_geometric.nn import global_mean_pool
from pag.connection import HyperConnection
from pag.layer.block import PathAttentionBlock
from torch.nn import LayerNorm
from pag.encoder.feature_encoder import FeatureEncoder
from pag.encoder.rrwp_encoder import RRWPLinearEdgeEncoder, RRWPLinearNodeEncoder
from torch_geometric.nn import LayerNorm


class PathAttentionGraphormer(Module):
    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        channels: int,
        pa_layers: list[PathAttentionBlock],
        hc_rate: int = 4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.hc_rate = hc_rate
        self.num_pa_layers = len(pa_layers)

        self.feature_encoder = FeatureEncoder(
            node_in_channels, edge_in_channels, channels
        )
        self.node_rrwp_encoder = RRWPLinearNodeEncoder(8, channels)

        if edge_in_channels is not None:
            self.edge_rrwp_encoder = RRWPLinearEdgeEncoder(edge_in_channels, channels)

        # self.layer_norm0 = LayerNorm()
        # self.layer_norm1 = LayerNorm()
        # self.layer_norm2 = LayerNorm()
        self.node_layer_norms = nn.ModuleList(
            [LayerNorm(channels) for _ in range(self.num_pa_layers - 1)]
        )

        self.pa_blocks = nn.ModuleList(pa_layers)
        self.node_hc = nn.ModuleList([
            HyperConnection(channels, hc_rate) for _ in range(self.num_pa_layers)
        ])
        self.feature_hc = nn.ModuleList([
            HyperConnection(channels, hc_rate) for _ in range(self.num_pa_layers)
        ])

    # @torch.compile
    def forward(self, data):
        data = self.feature_encoder(data)
        data = self.node_rrwp_encoder(data)
        if (
            hasattr(data, "edge_attr")
            and data.edge_attr is not None
            and hasattr(self, "edge_rrwp_encoder")
        ):
            data = self.edge_rrwp_encoder(data)

        # h
        h = data.x.unsqueeze(1).repeat(1, self.hc_rate, 1)
        g = global_mean_pool(data.x, data.batch)
        g = g.unsqueeze(1).repeat(1, self.hc_rate, 1)

        in_model_loss = 0

        for idx in range(self.num_pa_layers):
            mix_h, beta_h = self.node_hc[idx].width_connection(h)
            mix_g, beta_g = self.feature_hc[idx].width_connection(g)

            data.x = mix_h[..., 0, :].squeeze(0)

            if idx != 0:
                data.x = self.node_layer_norms[idx - 1](data.x)

            h, g, a_w, loss = self.pa_blocks[idx](data)

            h = self.node_hc[idx].depth_connection(mix_h, h, beta_h)
            g = self.feature_hc[idx].depth_connection(mix_g, g, beta_g)
            in_model_loss = in_model_loss + loss

        return g[..., 0, :].squeeze(0), h[..., 0, :].squeeze(0), a_w, in_model_loss
