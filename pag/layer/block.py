import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from rum.models import RUMModel
from graph_ae.gnn import GNN
from pag.path_attention import PathAttention
from pag.fusion import AFF, IAFF
from pag.layer.grit_layer import GritTransformerLayer


class PathAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_me_layers: int,
        num_le_depth: int,
        num_le_samples: int,
        le_rw_length: int,
        me_dropout: float,
        le_dropout: float,
        pa_dropout: float,
        pa_temp: float = 1.0,
        pooler=global_mean_pool,
        global_fuser=AFF,
        node_fuser=AFF,
        grit_num_heads: int = 4,
        grit_deg_scaler: bool = True,
        grit_signed_sqrt: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.macro_encoder = GritTransformerLayer(
            channels,
            channels,
            grit_num_heads,
            me_dropout,
            me_dropout,
            deg_scaler=grit_deg_scaler,
            signed_sqrt=grit_signed_sqrt,
        )

        self.local_encoder = RUMModel(
            in_features=channels,
            out_features=channels,
            hidden_features=channels,
            edge_features=channels,
            depth=num_le_depth,
            num_samples=num_le_samples,
            length=le_rw_length,
            dropout=le_dropout,
            binary=False,
        )

        self.path_attention = PathAttention(
            channels, dropout=pa_dropout, temp=pa_temp, lambda_entropy=0.0
        )
        self.pooler = pooler
        self.global_fuser = global_fuser(channels)
        self.node_fuser = node_fuser(channels)

    def forward_local(self, h, data):
        out, ss_loss = self.local_encoder(data, h, data.edge_attr)
        return out, ss_loss

    def forward_macro(self, h, data):
        # h = self.macro_encoder(
        #     h, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch
        # )
        batch = self.macro_encoder(data)
        h = batch.x  # GritTransformerLayer 返回 batch 对象，我们需要提取 x
        return h, 0

    def readout(self, h, data):
        return self.pooler(h, data.batch)

    def forward(self, data):
        h = data.x
        struct_features, ss_loss = self.forward_local(h, data)
        macro_features, _ = self.forward_macro(h, data)
        global_feature = self.readout(macro_features, data)

        attn_struct_feature, attn_weights, attn_entropy_loss = self.path_attention(
            global_feature, struct_features, data.batch
        )

        fused_global_feature = self.global_fuser(global_feature, attn_struct_feature)
        struct_features = struct_features.mean(dim=0)
        fused_h = self.node_fuser(macro_features, struct_features)

        loss = attn_entropy_loss + ss_loss

        return fused_h, fused_global_feature, attn_weights, loss
