import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from rum.models import RUMModel
from graph_ae.gnn import GNN
from torch_geometric.nn import global_mean_pool
from pag.connection import HyperConnection
from pag.block import PathAttentionBlock
from torch.nn import LayerNorm


class PathAttentionGraphormer(Module):
    def __init__(
        self,
        channels: int,
        num_rw_layers: int,
        num_rw_samples: int,
        num_rw_length: int,
        num_global_encoder_layers: int,
        rw_dropout: float = 0.2,
        global_encoder_dropout: float = 0.2,
        attention_dropout: float = 0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.rw_dropout = rw_dropout
        self.global_encoder_dropout = global_encoder_dropout
        self.attention_dropout = attention_dropout

        # self.layer_norm0 = LayerNorm()
        # self.layer_norm1 = LayerNorm()
        # self.layer_norm2 = LayerNorm()

        self.pab0 = PathAttentionBlock(
            channels,
            num_me_layers=2,
            num_le_depth=2,
            num_le_samples=5,
            le_rw_length=2,
            me_dropout=0.2,
            le_dropout=0.2,
            pa_dropout=0.4,
        )
        self.pab1 = PathAttentionBlock(
            channels,
            num_me_layers=2,
            num_le_depth=2,
            num_le_samples=5,
            le_rw_length=4,
            me_dropout=0.2,
            le_dropout=0.2,
            pa_dropout=0.4,
        )
        self.pab2 = PathAttentionBlock(
            channels,
            num_me_layers=1,
            num_le_depth=1,
            num_le_samples=3,
            le_rw_length=8,
            me_dropout=0.2,
            le_dropout=0.2,
            pa_dropout=0.4,
        )

        self.hc0 = HyperConnection(channels, 4)
        self.hc1 = HyperConnection(channels, 4)
        # self.hc2 = HyperConnection(channels, 4)

    # @torch.compile
    def forward(self, data):
        ori_h = data.h
        h = ori_h.unsqueeze(1).repeat(1, 4, 1)

        mix_h, beta = self.hc0.width_connection(h)
        h, g0, _, loss0 = self.pab0(mix_h[..., 0, :].squeeze(0), data)
        h = self.hc0.depth_connection(mix_h, h, beta)

        mix_h, beta = self.hc1.width_connection(h)
        h, g1, _, loss1 = self.pab1(mix_h[..., 0, :].squeeze(0), data)
        h = self.hc1.depth_connection(mix_h, h, beta)

        # mix_h, beta = self.hc2.width_connection(h)
        h, g2, attn_weights, loss2 = self.pab0(mix_h[..., 0, :].squeeze(0), data)
        # h = self.hc2.depth_connection(mix_h, h, beta)

        fused_global_feature = g0 + g1 + g2
        loss = loss0 + loss1 + loss2

        return fused_global_feature, h, attn_weights, loss
