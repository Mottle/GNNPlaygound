import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from graph_vae.gnn import GNN
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

        self.atom_embedding = nn.Linear(num_atom_features, hidden_channels)
        self.bond_embedding = nn.Linear(num_bond_features, hidden_channels)

        self.classifier = nn.Sequential(nn.Linear(hidden_channels, in_channels))
    
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
        out = self.classifier(recon_features)
        
        # 返回 (预测的原节点, 真实的原节点)
        return z, out[~mask], data.x[~mask]
    


def mask_nodes(x, batch, mask_ratio=0.15):
    device = x.device
    num_nodes = x.size(0)
    batch_size = batch.max().item() + 1
    
    # 1. 定位虚拟节点（每个 batch 的最后一个节点）
    # 在 PyG 平铺格式中，当 batch[i] != batch[i+1] 时，i 是当前图的最后一个节点
    # 我们构造一个 diff 向量，并在末尾补 1 确保最后一个图的最后节点也被选中
    diff = torch.cat([batch[1:] - batch[:-1], torch.tensor([1], device=device)])
    is_virtual_node = diff > 0
    maskable_nodes_mask = ~is_virtual_node

    # 2. 准备掩码容器
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    
    # 3. 逐图处理掩码逻辑
    # 虽然这里使用了循环，但在 batch_size=128 的量级下，耗时远小于 GNN 运算
    for i in range(batch_size):
        # 仅选择当前 batch 内且非虚拟节点的索引
        current_batch_mask = (batch == i) & maskable_nodes_mask
        idx = current_batch_mask.nonzero(as_tuple=True)[0]
        
        if len(idx) == 0:
            continue
        
        # 计算该图需要掩码的数量（向上取整确保至少有一个点被盖住，增强扰动）
        num_to_mask = torch.ceil(torch.tensor(len(idx), device=device) * mask_ratio).long()
        
        # 随机采样并标记
        perm = torch.randperm(len(idx), device=device)[:num_to_mask]
        node_mask[idx[perm]] = True
    
    # 4. 生成掩码后的特征
    x_masked = x.clone()
    x_masked[node_mask] = 0.0
    
    return x_masked, node_mask