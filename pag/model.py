import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from rum.models import RUMModel
from graph_ae.gnn import GNN
from torch_geometric.nn import global_mean_pool
from pag.path_attention import PathAttention, LocalPathAttention

# class PreEncoder(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, out_channels)

#     def forward(self, data):
#         if hasattr(data, "h"):
#             return data.h
#         return self.linear(data.x)


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

        self.local_encoders = nn.ModuleList(
            [
                RUMModel(
                    in_features=channels,
                    out_features=channels,
                    hidden_features=channels,
                    edge_features=channels,
                    depth=3,
                    num_samples=num_rw_samples,
                    length=2,
                    dropout=rw_dropout,
                    binary=False,
                ),
                RUMModel(
                    in_features=channels,
                    out_features=channels,
                    hidden_features=channels,
                    edge_features=channels,
                    depth=2,
                    num_samples=num_rw_samples,
                    length=4,
                    dropout=rw_dropout,
                    binary=False,
                ),
                RUMModel(
                    in_features=channels,
                    out_features=channels,
                    hidden_features=channels,
                    edge_features=channels,
                    depth=2,
                    num_samples=num_rw_samples,
                    length=8,
                    dropout=rw_dropout,
                    binary=False,
                ),
            ]
        )

        self.global_encoders = nn.ModuleList(
            [
                GNN(
                    channels,
                    num_global_encoder_layers // 3 + 1,
                    backbone="gin",
                    dropout=global_encoder_dropout,
                    norm="layer_norm",
                ),
                GNN(
                    channels,
                    num_global_encoder_layers // 3 + 1,
                    backbone="gin",
                    dropout=global_encoder_dropout,
                    norm="layer_norm",
                ),
                GNN(
                    channels,
                    num_global_encoder_layers // 3,
                    backbone="gin",
                    dropout=global_encoder_dropout,
                    norm="layer_norm",
                ),
            ]
        )

        self.path_attentions = nn.ModuleList(
            [
                PathAttention(channels, attention_dropout),
                PathAttention(channels, attention_dropout),
                PathAttention(channels, attention_dropout),
            ]
        )

        self.node_compress_linears = nn.ModuleList(
            [nn.Linear(2 * channels, channels) for _ in range(3)]
        )

        self.feature_compress_linears = nn.ModuleList(
            [nn.Linear(2 * channels, channels) for _ in range(3)]
        )

    def forward_global_encoder(self, data, idx):
        h = data.h
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        h = self.global_encoders[idx](
            h, edge_index=data.edge_index, edge_attr=edge_attr, batch=data.batch
        )
        return h, 0

    def forward_rw(self, data, idx):
        h = data.h
        out, ss_loss = self.local_encoders[idx](data, h, e=data.edge_attr)
        return out, ss_loss

    def forward(self, data):
        loss = 0
        fused_global_feature = 0
        for idx in range(3):
            global_encoder_features, loss_global = self.forward_global_encoder(
                data, idx
            )
            global_encoder_feature = global_mean_pool(
                global_encoder_features, data.batch
            )

            rw_global_features, loss_rw = self.forward_rw(data, idx)
            attn_rw_feature, attn_weights, attn_entropy_loss = self.path_attentions[
                idx
            ](global_encoder_feature, rw_global_features, data.batch)

            rw_global_features = rw_global_features.sum(dim=0)
            data.h = self.node_compress_linears[idx](
                torch.cat([global_encoder_features, rw_global_features], dim=-1)
            )

            fused_global_feature = fused_global_feature + self.feature_compress_linears[
                idx
            ](torch.cat([global_encoder_feature, attn_rw_feature], dim=-1))

            loss = loss + loss_global + loss_rw + attn_entropy_loss

        return fused_global_feature, data.h, attn_weights, loss
