import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from rum.models import RUMModel
from graph_ae.gnn import GNN
from torch_geometric.nn import global_mean_pool


import torch
import torch.nn as nn
import torch.nn.functional as F

# class PreEncoder(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.linear = nn.Linear(in_channels, out_channels)

#     def forward(self, data):
#         if hasattr(data, "h"):
#             return data.h
#         return self.linear(data.x)


class PathAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2, lambda_entropy=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.lambda_entropy = lambda_entropy

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)

    def attention_entropy_loss(self, attn_weights, eps=1e-8):
        """
        计算注意力权重的熵极小化损失。

        参数:
            attn_weights: Tensor, shape [N, W] 或 [B, N, W]，经过 Softmax 归一化的注意力得分。
            eps: float, 用于维持数值稳定性的极小值。

        返回:
            loss_entropy: 标量 Tensor, 表示当前 Batch 内所有节点注意力分布的平均熵。
        """
        # 限制最小值为 eps，防止 log(0) 产生 -Inf 进而导致梯度 NaN
        attn_weights_safe = torch.clamp(attn_weights, min=eps)

        # 计算每个节点的熵: -sum(p * log(p))
        # 假设输入 shape 为 [N, W]，dim=-1 即在路径维度 W 上求和
        entropy_per_node = -torch.sum(
            attn_weights * torch.log(attn_weights_safe), dim=-1
        )

        # 对所有节点取平均
        loss_entropy = entropy_per_node.mean()

        return loss_entropy

    def forward(self, global_feature, rw_features, batch):
        """
        参数:
            global_feature: Tensor, shape [B, D] (图级全局特征, B为Batch Size)
            rw_features: Tensor, shape [W, N, D] 或 [N, W, D] (节点级多路径游走特征)
            batch: Tensor, shape [N] (节点到图的映射索引)
        返回:
            final_out: Tensor, shape [N, D] (融合后的节点级特征)
            attn_weights: Tensor, shape [N, W] (用于 Motif 分析的注意力得分)
        """
        # 1. 维度广播 (Broadcasting)
        # 利用 batch 索引，将 [B, D] 的图级 Query 展开为 [N, D]
        # 确保每个节点都能获取其所属分子的宏观上下文
        Q_base = global_feature[batch]
        Q = self.q_proj(Q_base).unsqueeze(1)  # [N, 1, D]

        # 2. 游走特征维度对齐
        # 由于此前使用了 out.mean(dim=0)，说明 RUMModel 的默认输出是 [W, N, D]
        # Attention 需要以节点为 batch 进行 bmm，因此需将其转置为 [N, W, D]
        if rw_features.dim() == 3 and rw_features.shape[0] != Q_base.shape[0]:
            rw_features = rw_features.transpose(0, 1).contiguous()

        K = self.k_proj(rw_features)  # [N, W, D]
        V = self.v_proj(rw_features)  # [N, W, D]

        # 3. 计算注意力得分 (Motif Discovery 核心)
        # Q: [N, 1, D], K^T: [N, D, W] -> scores: [N, 1, W]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale

        # 在游走路径维度上归一化，找出当前节点最重要的路径
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, 1, W]
        attn_weights_dropped = self.dropout(attn_weights)

        # 4. 局部特征聚合
        # [N, 1, W] bmm [N, W, D] -> [N, 1, D] -> [N, D]
        fused_local = torch.bmm(attn_weights_dropped, V).squeeze(1)

        # 5. 全局与局部的特征融合
        # 将节点的全局指导向量与提纯后的局部 Motif 特征相加
        # final_out = Q_base + fused_local  # [N, D]
        final_out = fused_local
        attn_weights = attn_weights.squeeze(1)
        # loss = self.attention_entropy_loss(attn_weights)
        loss = 0

        return final_out, attn_weights, loss * self.lambda_entropy


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

        self.rum = RUMModel(
            in_features=channels,
            out_features=channels,
            hidden_features=channels,
            edge_features=channels,
            depth=num_rw_layers,
            num_samples=num_rw_samples,
            length=num_rw_length,
            dropout=rw_dropout,
            binary=False,
        )

        self.global_encoder = GNN(
            channels,
            num_global_encoder_layers,
            backbone="gin",
            dropout=global_encoder_dropout,
            norm="layer_norm",
        )

        self.path_attention = PathAttention(channels, attention_dropout)

    def forward_global_encoder(self, data):
        h = data.h
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        h = self.global_encoder(
            h, edge_index=data.edge_index, edge_attr=edge_attr, batch=data.batch
        )
        return h, 0

    def forward_rw(self, data):
        h = data.h
        out, ss_loss = self.rum(data, h, e=data.edge_attr)
        return out, ss_loss

    def forward(self, data):
        global_encoder_features, loss_global = self.forward_global_encoder(data)
        global_encoder_feature = global_mean_pool(global_encoder_features, data.batch)

        rw_global_features, loss_rw = self.forward_rw(data)
        attn_rw_features, attn_weights, attn_entropy_loss = self.path_attention(
            global_encoder_feature, rw_global_features, data.batch
        )
        attn_rw_global_feature = global_mean_pool(attn_rw_features, data.batch)

        fused_node_features = torch.stack(
            [global_encoder_features, attn_rw_features], dim=1
        )
        fused_global_feature = torch.cat(
            [global_encoder_feature, attn_rw_global_feature], dim=-1
        )
        loss = loss_global + loss_rw + attn_entropy_loss

        return fused_global_feature, fused_node_features, attn_weights, loss
