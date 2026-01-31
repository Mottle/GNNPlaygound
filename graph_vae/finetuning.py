import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import track
from utils.scaffold import scaffold_split
from sklearn.metrics import roc_auc_score
from utils.seed_manual import seed_everything

from graph_vae.virtual_graph_vae import GNN
from graph_vae.virtual_node_pre_transform import AddVirtualNode


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        # 收集真实标签
        y_true.append(data.y.cpu().numpy())
        # 收集预测概率 (Sigmoid)
        y_scores.append(torch.sigmoid(out).cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)

    roc_list = []
    for i in range(y_true.shape[1]):
        # 过滤掉 NaN 标签
        is_labeled = ~np.isnan(y_true[:, i])
        # 确保该任务既有正例也有负例
        if np.sum(is_labeled) > 0 and len(np.unique(y_true[is_labeled, i])) == 2:
            roc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_scores[is_labeled, i])
            )

    return sum(roc_list) / len(roc_list) if len(roc_list) > 0 else 0


class FineTuningModel(nn.Module):
    def __init__(
        self,
        num_atom_types: int,
        num_node_features: int,
        num_bond_types: int,
        num_bond_features: int,
        hidden_channels: int,
        num_gnn_layers: int,
        backbone: str,
        num_tasks: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.atom_embedding = nn.Embedding(num_atom_types, hidden_channels)
        self.extra_feature_proj = nn.Linear(num_node_features - 1, hidden_channels)

        self.bond_embedding = nn.Embedding(num_bond_types, hidden_channels)
        self.extra_bond_proj = nn.Linear(num_bond_features - 1, hidden_channels)

        self.feature_fusion = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_channels, hidden_channels)
        )

        self.attr_fusion = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_channels, hidden_channels)
        )

        self.virtual_embedding = nn.Embedding(1, hidden_channels)

        self.encoder = GNN(
            hidden_channels, num_gnn_layers, backbone=backbone, dropout=dropout
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_tasks),
        )
        self.dropout = dropout

    def load_backbone(self, path: str, device):
        self.encoder.load(path=path, device=device)

    def forward(self, data):
        x = data.x
        edge_index_enc = data.edge_index_enc
        edge_attr_enc = data.edge_attr_enc
        mask = data.virtual_node_mask

        # === 步骤 A: 分割特征 ===
        # col 0: 原子类型 (需要转成 Long 用于查表)
        x_atom_type = x[:, 0].long()

        # col 1~8: 额外特征 (保持 Float 用于 Linear 计算)
        # 如果 x 本身是 long, 需要转 float
        x_extra_feats = x[:, 1:].float()

        bond_type = edge_attr_enc[:, 0].long()
        bond_extra_feats = edge_attr_enc[:, 1:].float()

        # === 步骤 B: 分别映射 ===
        # 1. Embedding 查表 -> [N, 64]
        h_atom = self.atom_embedding(x_atom_type)

        # 2. Linear 投影 -> [N, 64]
        h_extra = self.extra_feature_proj(x_extra_feats)

        # === 步骤 C: 空间融合 ===
        # 将基础语义(原子类型)与状态信息(电荷/度数)相加
        # 这种加法融合保留了所有信息，且形状依然是 [N, 64]
        # h = h_atom + h_extra
        h = torch.cat([h_atom, h_extra], dim=1)
        h = self.feature_fusion(h)
        h = F.leaky_relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h_bond = self.bond_embedding(bond_type)
        h_bond_extra = self.extra_bond_proj(bond_extra_feats)

        # edge_attr = h_bond + h_bond_extra
        edge_attr = torch.cat([h_bond, h_bond_extra], dim=1)
        edge_attr = self.attr_fusion(edge_attr)
        edge_attr = F.leaky_relu(edge_attr)
        edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        # === 步骤 D: 虚拟节点注入 ===
        num_virtual = mask.sum().item()
        if num_virtual > 0:
            h[mask] = self.virtual_embedding.weight.repeat(num_virtual, 1)

        # === 步骤 E: 编码与分类 ===
        z_all = self.encoder(h, edge_index_enc, edge_attr)
        z = z_all[mask]
        return self.classifier_head(z)


def main():
    DATA_PATH = "./graph_vae/dataset/MoleculeNet"
    PRETRAINED_PATH = "./graph_vae/saved/encoder.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    DATASET_NAME = "Tox21"
    # DATASET_NAME = 'BBBP'

    dataset = MoleculeNet(
        DATA_PATH, name=DATASET_NAME, pre_transform=AddVirtualNode(mode="moleculenet")
    )
    print(
        f" {DATASET_NAME}数据集加载完毕: {len(dataset)} graphs, Features: {dataset.num_features}"
    )

    train_dataset, val_dataset, test_dataset = scaffold_split(
        dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    max_atom_idx = 0
    max_bond_idx = 0
    for data in dataset:
        # 1. 扫描原子最大索引 (用于 num_atom_types)
        # 假设第 0 列是原子序号
        current_max_atom = data.x[:, 0].max().item()
        max_atom_idx = max(max_atom_idx, current_max_atom)

        # 2. 扫描边最大索引 (用于 num_bond_types)
        if data.edge_attr_enc is not None:
            # 假设第 0 列是键类型索引
            current_max_bond = data.edge_attr_enc[:, 0].max().item()
            max_bond_idx = max(max_bond_idx, current_max_bond)

        # 计算最终词表大小 (索引是从0开始的，所以大小是 max_idx + 1)
    calculated_num_atom_types = int(max_atom_idx + 1)
    calculated_num_bond_types = int(max_bond_idx + 1)

    model = FineTuningModel(
        num_atom_types=calculated_num_atom_types,
        num_node_features=dataset.num_features,
        num_bond_types=calculated_num_bond_types,
        num_bond_features=dataset[0].edge_attr_enc.size(1),
        hidden_channels=512,
        num_gnn_layers=5,
        backbone="gated_gcn",
        num_tasks=dataset.y.shape[1],
        dropout=0.5
    ).to(DEVICE)

    if os.path.exists(PRETRAINED_PATH):
        model.load_backbone(PRETRAINED_PATH, device=DEVICE)
    else:
        print("⚠️  警告: 未找到预训练权重，将从头开始训练！")

    # ==========================================
    # Phase 1: Warmup (冻结 Backbone)
    # ==========================================
    print("\n❄️  Phase 1: Warmup (Freezing Backbone)...")

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.01
    )

    for epoch in range(1, 21):
        total_loss = 0
        model.train()
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)

            # Loss 计算 (Mask NaN)
            target = data.y.float()
            is_labeled = ~torch.isnan(target)
            loss = F.binary_cross_entropy_with_logits(
                out[is_labeled], target[is_labeled]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Warmup Epoch {epoch}: Loss {total_loss/len(loader):.4f}")

    # ==========================================
    # Phase 2: Fine-tuning (全量微调)
    # ==========================================
    print("\n🔥 Phase 2: Full Fine-tuning (Unfreezing)...")

    for param in model.parameters():
        param.requires_grad = True

    # 1. 获取 Encoder 的参数 ID
    encoder_params_ids = list(map(id, model.encoder.parameters()))

    # 2. 筛选出“除 Encoder 以外的所有参数” (包括 Embedding, Head, Linear 等)
    base_params = filter(lambda p: id(p) not in encoder_params_ids, model.parameters())

    # 3. 定义优化器
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 0.0001},  # 骨干网络慢学
            {
                "params": base_params,
                "lr": 0.001,
            },  # 其他所有层快学 (自动包含所有没漏掉的层)
        ]
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    for epoch in track(range(1, 101), description="Fine-tuning"):
        # --- Train ---
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)

            target = data.y.float()
            is_labeled = ~torch.isnan(target)
            loss = F.binary_cross_entropy_with_logits(
                out[is_labeled], target[is_labeled]
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(loader)

        # --- Validation (基于 Loss 调整 LR) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)
                target = data.y.float()
                is_labeled = ~torch.isnan(target)
                if is_labeled.sum() > 0:
                    val_loss += F.binary_cross_entropy_with_logits(
                        out[is_labeled], target[is_labeled]
                    ).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # 可选：每 5 个 Epoch 打印一次 AUC 看看效果
        if epoch % 10 == 0:
            val_auc = eval_model(model, val_loader, DEVICE)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}"
            )

    print("\nTraining Finished. Starting Final Evaluation...")

    train_auc = eval_model(model, loader, DEVICE)
    val_auc = eval_model(model, val_loader, DEVICE)
    test_auc = eval_model(model, test_loader, DEVICE)

    print("-" * 30)
    print(f"🏆 Final Results:")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val   AUC: {val_auc:.4f}")
    print(f"Test  AUC: {test_auc:.4f} (最终指标)")
    print("-" * 30)

    # 保存微调后的模型
    if not os.path.exists("./graph_vae/saved/"):
        os.makedirs("./graph_vae/saved/")
    torch.save(
        model.state_dict(), f"./graph_vae/saved/{str.lower(DATASET_NAME)}_finetuned.pth"
    )


if __name__ == "__main__":
    seed_everything(42)
    main()
