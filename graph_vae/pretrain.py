import shutil
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from graph_vae.virtual_node_pre_transform import AddVirtualNode
from graph_vae.virtual_graph_vae import VirtualNodeGraphVAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rich.progress import track
from utils.perf_counter import measure_time
from utils.seed_manual import seed_everything

# @torch.compile
def train_epoch(loader,optimizer, model, device):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # Forward
        pred, target = model(data)
        # Loss (MSE)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss
    

def main():
    path = "./graph_vae/dataset/TUDataset"
    # dataset_name = 'NCI1'
    dataset_name = "ZINC_full"
    BATCH_SIZE = 128

    processed_dir = os.path.join(path, dataset_name, "processed")
    if os.path.exists(processed_dir):
        print(
            f"Deleting old processed data at {processed_dir} to apply new transform..."
        )
        shutil.rmtree(processed_dir)

    dataset = TUDataset(
        path, name=dataset_name, pre_transform=AddVirtualNode(mode="zinc")
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VirtualNodeGraphVAE(
        dataset.num_features,
        hidden_channels=512,
        backbone="gated_gcn",
        num_gnn_layers=5,
        # num_features=dataset.num_features,
        num_bond_features=dataset.edge_attr.size(1) + 1,
        dropout=0.5
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,  # 衰减系数: new_lr = old_lr * 0.5
        patience=5,  # 忍耐度: 连续 5 个 epoch 指标没改善才衰减
        min_lr=1e-6,  # 学习率下限
    )

    measured_train_epoch = measure_time(train_epoch)

    model.train()
    for epoch in track(range(1, 50), description=f"Train {dataset}:"):
        total_loss = 0
        total_loss, spend_time = measured_train_epoch(loader, optimizer, model, device)
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}, loss: {avg_loss:.12f}, lr: {current_lr:.6f}, spend time: {spend_time}")
    model.save_encoder_and_decoder(path="./graph_vae/saved/")


if __name__ == "__main__":
    seed_everything(42)
    main()
