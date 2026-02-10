import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from yacs.config import CfgNode as CN

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import track
from sklearn.metrics import roc_auc_score

from utils.scaffold import scaffold_split
from utils.seed_manual import seed_everything
from graph_ae.virtual_graph_ae import GNN
from graph_ae.virtual_node_pre_transform import AddVirtualNode


def get_cfg_defaults():
    cfg = CN()
    cfg.out_dir = "./results/finetune"

    # WandB
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.project = "gnn_finetune"
    cfg.wandb.entity = ""

    # Dataset
    cfg.dataset = CN()
    cfg.dataset.name = "Tox21"
    cfg.dataset.root = "./graph_ae/dataset/MoleculeNet"
    cfg.dataset.batch_size = 32
    cfg.dataset.split_ratios = [0.8, 0.1, 0.1]

    # Pretrained
    cfg.pretrained = CN()
    cfg.pretrained.path = "./graph_ae/saved/encoder.pth"

    # Model
    cfg.model = CN()
    cfg.model.hidden_dim = 512
    cfg.model.backbone = "gated_gcn"
    cfg.model.num_gnn_layers = 5
    cfg.model.dropout = 0.5
    cfg.model.norm = None

    # Train Strategies
    cfg.train = CN()
    cfg.train.warmup_epochs = 20
    cfg.train.warmup_lr = 0.01
    cfg.train.finetune_epochs = 100
    cfg.train.finetune_lr_base = 0.001
    cfg.train.finetune_lr_encoder = 0.0001
    cfg.train.finetune_lr_head = 0.0001
    cfg.train.factor = 0.5
    cfg.train.patience = 5
    cfg.train.min_lr = 1e-6

    return cfg


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    y_true = []
    y_scores = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        y_true.append(data.y.cpu().numpy())
        y_scores.append(torch.sigmoid(out).cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)

    roc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = ~np.isnan(y_true[:, i])
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
        norm: str,
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
            hidden_channels,
            num_gnn_layers,
            backbone=backbone,
            dropout=dropout,
            norm=norm,
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


# ----------------------------------------------------------------------
# 4. 主函数 (重构重点)
# ----------------------------------------------------------------------
def main(cfg):
    # --- A. WandB Init ---
    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            config=cfg,
            name=f"{cfg.dataset.name}_{cfg.model.backbone}_FT",
        )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- B. Data Loading ---
    dataset = MoleculeNet(
        cfg.dataset.root,
        name=cfg.dataset.name,
        pre_transform=AddVirtualNode(mode="moleculenet"),
    )
    print(f"📚 {cfg.dataset.name} Loaded: {len(dataset)} graphs")

    train_dataset, val_dataset, test_dataset = scaffold_split(
        dataset,
        train_ratio=cfg.dataset.split_ratios[0],
        val_ratio=cfg.dataset.split_ratios[1],
        test_ratio=cfg.dataset.split_ratios[2],
    )

    loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False
    )

    # --- C. Dynamic Vocab Calculation ---
    max_atom_idx = 0
    max_bond_idx = 0
    for data in dataset:
        max_atom_idx = max(max_atom_idx, data.x[:, 0].max().item())
        if data.edge_attr_enc is not None:
            max_bond_idx = max(max_bond_idx, data.edge_attr_enc[:, 0].max().item())

    # --- D. Model Init ---
    model = FineTuningModel(
        num_atom_types=int(max_atom_idx + 1),
        num_node_features=dataset.num_features,
        num_bond_types=int(max_bond_idx + 1),
        num_bond_features=dataset[0].edge_attr_enc.size(1),
        hidden_channels=cfg.model.hidden_dim,
        num_gnn_layers=cfg.model.num_gnn_layers,
        backbone=cfg.model.backbone,
        norm = cfg.model.norm,
        num_tasks=dataset.y.shape[1],
        dropout=cfg.model.dropout,
    ).to(DEVICE)

    # --- E. Load Pretrained ---
    if os.path.exists(cfg.pretrained.path):
        model.load_backbone(cfg.pretrained.path, device=DEVICE)
    else:
        print(f"⚠️ Warning: Pretrained model not found at {cfg.pretrained.path}")

    # ==========================================
    # Phase 1: Warmup (Freeze Backbone)
    # ==========================================
    print(f"\n❄️ Phase 1: Warmup ({cfg.train.warmup_epochs} epochs)...")

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.warmup_lr
    )

    for epoch in range(1, cfg.train.warmup_epochs + 1):
        total_loss = 0
        model.train()
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

        avg_loss = total_loss / len(loader)
        print(f"  Warmup Epoch {epoch}: Loss {avg_loss:.4f}")

        # Log Warmup
        if cfg.wandb.use:
            wandb.log(
                {
                    "phase": "warmup",
                    "epoch": epoch,
                    "warmup/train_loss": avg_loss,
                    "warmup/lr": cfg.train.warmup_lr,
                }
            )

    # ==========================================
    # Phase 2: Fine-tuning (Full Unfreeze)
    # ==========================================
    print(f"\n🔥 Phase 2: Full Fine-tuning ({cfg.train.finetune_epochs} epochs)...")

    for param in model.parameters():
        param.requires_grad = True

    # Differential Learning Rates
    encoder_params_ids = list(map(id, model.encoder.parameters()))
    head_params_ids = list(map(id, model.classifier_head.parameters()))
    base_params = filter(
        lambda p: id(p) not in encoder_params_ids and id(p) not in head_params_ids,
        model.parameters(),
    )

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": cfg.train.finetune_lr_encoder},
            {"params": base_params, "lr": cfg.train.finetune_lr_base},
            {
                "params": model.classifier_head.parameters(),
                "lr": cfg.train.finetune_lr_head,
            },
        ]
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.factor,
        patience=cfg.train.patience,
        min_lr=cfg.train.min_lr,
    )

    for epoch in track(
        range(1, cfg.train.finetune_epochs + 1), description="Fine-tuning"
    ):
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

        # --- Validation ---
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

        # 注意：ReduceLROnPlateau 只能监控一个指标，这里监控 val_loss
        scheduler.step(avg_val_loss)

        # 获取当前的两个 LR
        current_lr_encoder = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        # Log Metrics
        log_dict = {
            "phase": "finetune",
            "epoch": epoch,
            "finetune/train_loss": train_loss,
            "finetune/val_loss": avg_val_loss,
            "finetune/lr_encoder": current_lr_encoder,
            "finetune/lr_head": current_lr_head,
        }

        # 间隔计算 AUC (比较耗时，所以每 5 个 epoch 算一次)
        if epoch % 5 == 0 or epoch == cfg.train.finetune_epochs:
            val_auc = eval_model(model, val_loader, DEVICE)
            print(
                f"Epoch {epoch:02d} | val auc: {val_auc:.4f} | lr(encoder): {current_lr_encoder:.6f}"
            )
            log_dict["finetune/val_auc"] = val_auc

        if cfg.wandb.use:
            wandb.log(log_dict)

    # --- Final Evaluation ---
    print("\nTraining Finished. Evaluating...")
    train_auc = eval_model(model, loader, DEVICE)
    val_auc = eval_model(model, val_loader, DEVICE)
    test_auc = eval_model(model, test_loader, DEVICE)

    print("-" * 30)
    print(f"🏆 Final Results ({cfg.dataset.name}):")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val   AUC: {val_auc:.4f}")
    print(f"Test  AUC: {test_auc:.4f}")
    print("-" * 30)

    if cfg.wandb.use:
        wandb.log({"final/test_auc": test_auc, "final/val_auc": val_auc})
        wandb.finish()

    # Save
    if not os.path.exists("./graph_ae/saved/"):
        os.makedirs("./graph_ae/saved/")
    torch.save(
        model.state_dict(),
        f"./graph_ae/saved/{str.lower(cfg.dataset.name)}_finetuned.pth",
    )


if __name__ == "__main__":
    import sys

    seed_everything(42)

    cfg = get_cfg_defaults()

    # 命令行参数覆盖
    if "--cfg" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--cfg") + 1]
        cfg.merge_from_file(cfg_path)

    if "--opts" in sys.argv:
        opts = sys.argv[sys.argv.index("--opts") + 1 :]
        cfg.merge_from_list(opts)

    cfg.freeze()
    main(cfg)
