import shutil
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from virtual_node_pre_transform import AddVirtualNode
from virtual_graph_vae import VirtualNodeGraphVAE

def main():
    path = './graph_vae/dataset/TUDataset'
    dataset_name = 'NCI1'
    
    # ⚠️ 重要提示：
    # 如果你之前运行过代码，./data/TUDataset/NCI1/processed 目录已经存在。
    # PyG 会检测到已处理的数据，从而**忽略**新的 pre_transform。
    # 必须先删除 processed 目录！
    processed_dir = os.path.join(path, dataset_name, 'processed')
    if os.path.exists(processed_dir):
        print(f"Deleting old processed data at {processed_dir} to apply new transform...")
        shutil.rmtree(processed_dir)

    # 加载数据集并应用 Transform
    dataset = TUDataset(
        path, 
        name=dataset_name, 
        pre_transform=AddVirtualNode() # <--- 注入我们的逻辑
    )
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VirtualNodeGraphVAE(dataset.num_features, hidden_channels=64, backbone='gcn').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(1, 100):
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
            
        print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.6f}")

if __name__ == '__main__':
    main()