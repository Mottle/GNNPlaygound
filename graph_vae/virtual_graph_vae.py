import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv


class GNN(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        backbone: str = ["gcn", "gin", "gat"],
        dropout: float = 0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = backbone

        self.layers = nn.ModuleList([self.build_conv() for _ in range(self.num_layers)])

    def build_conv(self):
        if self.backbone == "gcn":
            return GCNConv(self.channels, self.channels)
        elif self.backbone == "gat":
            return GATConv(self.channels, self.channels)
        elif self.backbone == "gin":
            return GINConv(
                nn.Sequential(
                    nn.Linear(self.channels, self.channels),
                    nn.LeakyReLU(),
                    nn.Linear(self.channels, self.channels),
                )
            )
        else:
            raise Exception()

    def forward(self, x, edge_index, batch = None):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.leaky_relu(x)
        return x


class VirtualNodeGraphVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_gnn_layers: int = 3,
        backbone: str = ["gcn", "gin", "gat"],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.backbone = backbone
        self.dropout = dropout

        self.encoder = GNN(
            hidden_channels, num_gnn_layers, backbone=backbone, dropout=dropout
        )

        self.decoder = GNN(
            hidden_channels, num_gnn_layers, backbone=backbone, dropout=dropout
        )

        # 虚拟节点的可学习初始向量 (替代全0)
        self.virtual_embedding = nn.Embedding(1, hidden_channels)

        # 简单的特征映射 (假设输入已经是 hidden_dim)
        # 如果 NCI1 输入是 One-Hot，这里可以用 Linear
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        # self.output_proj = nn.Linear(hidden_channels, in_channels)

    def forward(self, data):
        x = data.x
        edge_index_enc = data.edge_index_enc
        edge_index_dec = data.edge_index_dec
        mask = data.virtual_node_mask

        # 1. Input Project
        h = self.input_proj(x)
        
        # 2. 初始化虚拟节点特征
        # 将 mask 为 True 的位置替换为 Embedding
        num_virtual = mask.sum().item()
        if num_virtual > 0:
            h[mask] = self.virtual_embedding.weight.repeat(num_virtual, 1)
            
        # -------------------------
        # Encoder 阶段 (Unified)
        # -------------------------
        # 此时 edge_index_enc 混合了局部连接和全局汇聚
        # 经过 num_layers 层后，h[mask] 里的特征已经聚合了全图信息
        z_all = self.encoder(h, edge_index_enc)
        
        # 提取 Latent Code (只取虚拟节点部分)
        z = z_all[mask] # [Batch, Hidden]
        
        # -------------------------
        # Decoder 阶段 (Unified)
        # -------------------------
        # 准备 Decoder 输入：
        # 原节点 -> 全0 (Blind，迫使模型从 z 恢复)
        # 虚拟节点 -> z
        h_decode = torch.zeros_like(h)
        h_decode[mask] = z 
        
        # 运行 Decoder GNN
        # 此时 edge_index_dec 混合了局部连接(原图结构)和全局广播(S->V)
        # 每一层 GNN，S 的信息都会流入 V，同时 V 之间互相平滑
        recon_features = self.decoder(h_decode, edge_index_dec)
        
        # 3. 还原维度
        # out = self.output_proj(recon_features)
        
        # 返回 (预测的原节点, 真实的原节点)
        return recon_features[~mask], h[~mask]