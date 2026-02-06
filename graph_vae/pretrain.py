import os
import shutil
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from rich.progress import track
from yacs.config import CfgNode as CN

from graph_vae.virtual_node_pre_transform import AddVirtualNode
from graph_vae.virtual_graph_vae import VirtualNodeGraphVAE
from utils.perf_counter import measure_time
from utils.seed_manual import seed_everything


def get_cfg_defaults():
    cfg = CN()
    cfg.seed = 42
    cfg.out_dir = "./results"
    # Dataset
    cfg.dataset = CN()
    cfg.dataset.name = "ZINC_full"
    cfg.dataset.dir = "./graph_vae/dataset/TUDataset"
    cfg.dataset.batch_size = 128
    cfg.dataset.num_workers = 0
    # Model
    cfg.model = CN()
    cfg.model.hidden_dim = 256
    cfg.model.backbone = "gated_gcn"
    cfg.model.norm = None
    cfg.model.num_gnn_layers = 5
    cfg.model.dropout = 0.5
    cfg.model.masked_ratio = 0.15
    # Train
    cfg.train = CN()
    cfg.train.max_epoch = 50
    cfg.train.lr = 0.001
    cfg.train.factor = 0.5
    cfg.train.patience = 5
    cfg.train.min_lr = 1e-6
    # WandB
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.project = "graph_vae"
    cfg.wandb.entity = ""  # 选填

    return cfg


# class Classifier(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(in_channels, out_channels),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(out_channels, out_channels),
#             torch.nn.Dropout(p=0.3),
#         )

#     def forward(self, x):
#         return self.mlp(x)


# class MultiTaskLoss(nn.Module):
#     def __init__(self, num_tasks=2):
#         super(MultiTaskLoss, self).__init__()
#         # 初始化 log_vars (对应公式中的 s)
#         # 初始化为 0 相当于初始权重 a=1, b=1
#         self.log_vars = nn.Parameter(torch.zeros(num_tasks))

#     def forward(self, input_losses):
#         """
#         input_losses: 一个包含多个 loss 值的列表 [loss_rec, loss_reg]
#         """
#         # 确保输入 loss 和参数在同一个设备上
#         total_loss = 0
#         for i, loss in enumerate(input_losses):
#             # 获取对应的 log_var
#             log_var = self.log_vars[i]

#             # 执行公式: 1/(2a^2) * loss + log(a)
#             # 等价于: 0.5 * exp(-s) * loss + 0.5 * s
#             weighted_loss = 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
#             total_loss += weighted_loss

#         return total_loss

#     def par(self):
#         return self.log_vars


# @torch.compile (Debug时建议先注释掉，稳定后再开)
def train_epoch(loader, optimizer, model, device):
    total_loss = 0
    # total_rec_loss = 0
    # total_reg_loss = 0
    model.train()
    # classifier.train()

    reg_criterion = torch.nn.L1Loss()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward
        z, pred, target = model(data)
        # y_hat = classifier(z)
        # y_target = data.y.view(-1, 1).float()

        # rec_loss = sce_loss(pred, target)
        # reg_loss = reg_criterion(y_hat, y_target)

        # loss = mtl_loss([rec_loss, reg_loss])
        # loss = rec_loss + reg_loss

        loss = F.cross_entropy(pred, target)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # total_rec_loss += rec_loss.item()
        # total_reg_loss += reg_loss.item()
        total_loss += loss.item()

    return total_loss  # , total_rec_loss, total_reg_loss


def main(cfg):
    # -------------------------------------------------------
    # 4. WandB 初始化 (核心)
    # -------------------------------------------------------
    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            config=cfg,  # 直接把 YACS 配置传上去，网页上能看到所有参数
            name=f"{cfg.model.backbone}_dim{cfg.model.hidden_dim}",
        )

    # 路径处理
    # processed_dir = os.path.join(cfg.dataset.dir, cfg.dataset.name, "processed")
    # 这里加个简单的逻辑：如果是为了调试，可以不强制删
    # if os.path.exists(processed_dir):
    #     print(f"Deleting old processed data at {processed_dir}...")
    #     shutil.rmtree(processed_dir)

    dataset = TUDataset(
        cfg.dataset.dir,
        name=cfg.dataset.name,
        pre_transform=AddVirtualNode(mode="zinc"),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 cfg 参数初始化模型
    model = VirtualNodeGraphVAE(
        in_channels=dataset.num_features,  # 假设第一个参数是 input_channels
        hidden_channels=cfg.model.hidden_dim,
        backbone=cfg.model.backbone,
        norm=cfg.model.norm,
        num_gnn_layers=cfg.model.num_gnn_layers,
        num_atom_features=dataset.num_node_features,
        num_bond_features=dataset.edge_attr.size(1) + 1,
        dropout=cfg.model.dropout,
        mask_ratio=cfg.model.masked_ratio,
    ).to(device)

    # classifier = Classifier(cfg.model.hidden_dim, 1).to(device)

    # mtl_loss = MultiTaskLoss(num_tasks=2).to(device)

    if cfg.wandb.use:
        wandb.watch(model, log="all")  # 监控梯度和参数分布

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        # + list(classifier.parameters())
        # + list(mtl_loss.parameters()),
        lr=cfg.train.lr,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.factor,
        patience=cfg.train.patience,
        min_lr=cfg.train.min_lr,
    )

    measured_train_epoch = measure_time(train_epoch)

    # -------------------------------------------------------
    # i 5. 训练循环 (加入 WandB Logging)
    # -------------------------------------------------------
    for epoch in track(
        range(1, cfg.train.max_epoch + 1), description=f"Train {dataset.name}:"
    ):

        epoch_loss, spend_time = measured_train_epoch(loader, optimizer, model, device)
        avg_loss = epoch_loss / len(loader)
        # avg_rec_loss = rec_loss / len(loader)
        # avg_reg_loss = reg_loss / len(loader)

        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}, loss: {avg_loss:.12f}, lr: {current_lr:.6f}, time: {spend_time:.2f}"
        )

        # ---> WandB Logging <---
        if cfg.wandb.use:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_loss,
                    # "train/rec_loss": avg_rec_loss,
                    # "train/reg_loss": avg_reg_loss,
                    # 'train/sigma1': abs(mtl_loss.par()[0].item()),
                    # 'train/sigma2': abs(mtl_loss.par()[1].item()),
                    "train/lr": current_lr,
                    "train/time_per_epoch": spend_time,
                }
            )

    model.save_encoder_and_decoder(path="./graph_vae/saved/")

    if cfg.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    import sys

    # 6. 配置加载逻辑
    cfg = get_cfg_defaults()

    # 简单解析命令行：python pretrain.py --cfg config.yaml
    if "--cfg" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--cfg") + 1]
        cfg.merge_from_file(cfg_path)

    # 支持命令行覆盖：python pretrain.py --opts train.lr 0.005
    if "--opts" in sys.argv:
        opts = sys.argv[sys.argv.index("--opts") + 1 :]
        cfg.merge_from_list(opts)

    cfg.freeze()

    seed_everything(cfg.seed)
    main(cfg)
