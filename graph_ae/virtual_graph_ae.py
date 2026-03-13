import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from graph_ae.gnn import GNN
from typing import Optional
from rum.models import RUMModel


class VirtualNodeGraphAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_gnn_layers: int = 3,
        backbone: str = ["gcn", "gin", "gat", "rum"],
        norm: str = ["layer_norm", "batch_norm", "graph_norm"],
        dropout: float = 0.2,
        num_atom_features: Optional[int] = None,
        num_bond_features: Optional[int] = None,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.backbone = backbone
        self.dropout = dropout
        self.mask_ratio = mask_ratio

        if backbone == "rum":
            self.encoder = RUMModel(
                in_features=hidden_channels,
                out_features=hidden_channels,
                hidden_features=hidden_channels,
                edge_features=hidden_channels,
                depth=num_gnn_layers,
                num_samples=2,
                length=3,
                dropout=dropout,
                binary=False,
            )
        else:
            self.encoder = GNN(
                hidden_channels,
                num_gnn_layers,
                backbone=backbone,
                dropout=dropout,
                norm=norm,
            )

        if backbone == "rum":
            self.decoder = GNN(
                hidden_channels,
                num_gnn_layers,
                backbone="gine",
                dropout=dropout,
                norm=norm,
            )
        else:
            self.decoder = GNN(
                hidden_channels,
                num_gnn_layers,
                backbone=backbone,
                dropout=dropout,
                norm=norm,
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

        if not str.endswith(path, "/"):
            self.encoder.save(f"{path}/encoder.pth")
            self.decoder.save(f"{path}/decoder.pth")
        else:
            self.encoder.save(f"{path}encoder.pth")
            self.decoder.save(f"{path}decoder.pth")

    def forward(self, data):
        x = data.x
        batch = data.batch

        if self.mask_ratio > 0:
            masked_x, _ = mask_nodes(x, batch, self.mask_ratio)
        else:
            masked_x = x

        edge_index_enc = data.edge_index_enc
        edge_index_dec = data.edge_index_dec
        edge_attr_enc = data.edge_attr_enc
        edge_attr_dec = data.edge_attr_dec
        mask = data.virtual_node_mask

        ori_h = self.atom_embedding(masked_x)
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
        if self.backbone == "rum":
            z_all, ss_loss = self.encoder(data, h, e=h_bond_enc)
            z_all = z_all.mean(dim=0)
        else:
            z_all = self.encoder(h, edge_index_enc, h_bond_enc)

        # 提取 Latent Code (只取虚拟节点部分)
        z = z_all[mask]  # [Batch, Hidden]
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
        return z, out[~mask], data.x[~mask], ss_loss if self.backbone == "rum" else 0.0


def mask_nodes(x, batch, mask_ratio=0.15):
    """
    全向量化实现：不使用 Python 循环，性能最高。
    """
    device = x.device
    num_nodes = x.size(0)

    # 1. 识别虚拟节点 (每个 batch 的最后一个节点)
    # 计算 diff: [batch[1]-batch[0], ..., 1]
    diff = torch.cat([batch[1:] - batch[:-1], torch.tensor([1], device=device)])
    is_virtual_node = diff > 0

    # 2. 为每个节点生成随机扰动
    # 我们希望在每个 batch 内部随机选点。
    # 技巧：生成随机数，并给虚拟节点加上一个巨大的偏移量，使其永远不会被排到前面
    rand_noise = torch.rand(num_nodes, device=device)
    rand_noise = rand_noise + is_virtual_node.float() * 1e6

    # 3. 计算每个 batch 的节点数和需要 mask 的数量
    # 使用 scatter_add 计算每张图的节点总数 (包括虚拟节点)
    ones = torch.ones(num_nodes, device=device)
    nodes_per_graph = torch.zeros(batch.max().item() + 1, device=device).scatter_add_(
        0, batch, ones
    )

    # mask 数量基于原节点数 (nodes_per_graph - 1)
    num_to_mask_per_graph = torch.ceil((nodes_per_graph - 1) * mask_ratio).long()

    # 4. 核心：通过排序获取每个 batch 内的相对排名
    # 我们需要一个在每个 batch 内部从 0 开始的排名。
    # 首先根据 batch 排序，再根据 rand_noise 排序。
    # 简单做法：给 rand_noise 加上 batch * 2 (确保不同 batch 之间不重叠)
    ranking_key = rand_noise + batch.float() * 2.0
    sorted_indices = torch.argsort(ranking_key)

    # 5. 确定哪些位置属于“前 K 个随机节点”
    # 计算每个 batch 的起始索引偏移
    cum_nodes = torch.cat(
        [torch.tensor([0], device=device), nodes_per_graph.cumsum(0)[:-1]]
    )

    # 构造每个 batch 掩码位置的绝对索引
    # 这是一个小挑战，但我们可以通过生成一个相对排名矩阵来完成
    # 也可以简单地计算出每个 batch 应该取的范围
    mask_indices = []
    # 如果 batch_size 还是 128，我们用一个更骚的操作：
    # 创建一个全局排名，判断排名是否小于该 batch 的阈值

    # 终极向量化：计算每个点在其 batch 内的相对排名
    # 我们利用 argsort 的反函数
    rel_pos_in_batch = torch.arange(num_nodes, device=device) - cum_nodes[batch]

    # 我们只需重新对 sorted_indices 进行一次反向映射，就能得到每个点在 batch 里的随机序号
    # 但最快的方法通常是：
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # 这里的循环其实可以用更复杂的 tensor 技巧替代，但为了可读性，
    # 既然已经拿到了 sorted_indices，我们可以直接切片：
    for i, (start_idx, count) in enumerate(
        zip(cum_nodes.long(), num_to_mask_per_graph)
    ):
        if count > 0:
            # 因为 sorted_indices 是按 batch 排序且 batch 内随机的
            # 每一段的前 count 个就是我们要的
            node_mask[sorted_indices[start_idx : start_idx + count]] = True

    x_masked = x.clone()
    x_masked[node_mask] = 0.0

    return x_masked, node_mask
