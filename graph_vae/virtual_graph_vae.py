import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from gnn import GNN
from typing import Optional

class VirtualNodeGraphVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_gnn_layers: int = 3,
        backbone: str = ["gcn", "gin", "gat"],
        dropout: float = 0.2,  
        num_atom_features: Optional[int] = None,
        num_bond_features: Optional[int] = None,
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

        if num_atom_features is None:
            self.atom_embedding = nn.Linear(in_channels, hidden_channels)
        else:
            self.atom_embedding = nn.Embedding(num_atom_features, hidden_channels)

        if num_bond_features is not None:
            self.bond_embedding = nn.Linear(num_bond_features, hidden_channels)
    
    def save_encoder_and_decoder(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

        if not str.endswith(path, '/'):
            self.encoder.save(f'{path}/encoder.pth')
            self.decoder.save(f'{path}/decoder.pth')
        else:
            self.encoder.save(f'{path}encoder.pth')
            self.decoder.save(f'{path}decoder.pth')

    def forward(self, data):
        x = data.x
        edge_index_enc = data.edge_index_enc
        edge_index_dec = data.edge_index_dec
        edge_attr_enc = data.edge_attr_enc
        edge_attr_dec = data.edge_attr_dec
        mask = data.virtual_node_mask

        ori_h = self.atom_embedding(x)
        h = F.dropout(ori_h, self.dropout, self.training)
        
        h_bond_enc = self.bond_embedding(edge_attr_enc)
        h_bond_enc = F.dropout(h_bond_enc, self.dropout, self.training)

        h_bond_dec = self.bond_embedding(edge_attr_dec)
        h_bond_dec = F.dropout(h_bond_dec, self.dropout, self.training)
        
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
        z_all = self.encoder(h, edge_index_enc, h_bond_enc)
        
        # 提取 Latent Code (只取虚拟节点部分)
        z = z_all[mask] # [Batch, Hidden]
        z = F.dropout(z, self.dropout, self.training)
        
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
        recon_features = self.decoder(h_decode, edge_index_dec, h_bond_dec)
        
        
        # 返回 (预测的原节点, 真实的原节点)
        return recon_features[~mask], ori_h[~mask]