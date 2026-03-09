import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, ResGatedGraphConv, GINEConv
from torch_geometric.nn import BatchNorm, GraphNorm, LayerNorm


class GNN(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        backbone: str = ["gcn", "gin", "gat"],
        norm: str = ["layer_norm", "batch_norm", "graph_norm"],
        dropout: float = 0.2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.backbone = backbone
        self.norm = norm

        self.layers = nn.ModuleList([self.build_conv() for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([self.build_norm() for _ in range(self.num_layers)])

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
        elif self.backbone == "gated_gcn":
            return ResGatedGraphConv(
                self.channels, self.channels, edge_dim=self.channels
            )
        elif self.backbone == "gine":
            return GINEConv(
                nn.Sequential(
                    nn.Linear(self.channels, self.channels),
                    nn.LeakyReLU(),
                    nn.Linear(self.channels, self.channels),
                )
            )
        else:
            raise Exception()

    def build_norm(self):
        if self.norm == None:
            return None
        elif self.norm == "batch_norm":
            return BatchNorm(self.channels)
        elif self.norm == "layer_norm":
            return LayerNorm(self.channels)
        elif self.norm == "graph_norm":
            return GraphNorm(self.channels)
        else:
            raise NotImplementedError()

    def save(self, path: str):
        print(f"Saving GNN to {path}...")
        torch.save(self.state_dict(), path)

    def load(self, path: str, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading GNN from {path}...")
        loaded_state_dict = torch.load(path, map_location=device)
        model_state_dict = self.state_dict()

        total_layers = len(model_state_dict)
        matched_layers = 0

        for key, param in model_state_dict.items():
            # 检查 key 是否存在 且 形状是否一致
            if key in loaded_state_dict and loaded_state_dict[key].shape == param.shape:
                matched_layers += 1

        ratio = (matched_layers / total_layers) * 100 if total_layers > 0 else 0

        print(f"📊 Parameter Loading Report:")
        print(f"   - Total Layers in Model: {total_layers}")
        print(f"   - Matched Layers found:  {matched_layers}")
        print(f"   - Success Rate:          {ratio:.2f}%")

        # 建议改为 strict=False，这样即使不是 100% 也能加载成功，而不是直接报错
        missing_keys, unexpected_keys = self.load_state_dict(
            loaded_state_dict, strict=True
        )

        if len(missing_keys) > 0:
            print(
                f"⚠️  Warning: {len(missing_keys)} layers were missing and not loaded."
            )
            print(f"Missing: {missing_keys}")  # 需要详细信息时取消注释

        if len(unexpected_keys) > 0:
            print(
                f"⚠️  Warning: {len(unexpected_keys)} extra layers in file were ignored."
            )

        print("GNN weights load process completed.")

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        zipped = zip(self.layers, self.norms)
        for layer, norm in zipped:
            if edge_attr is not None:
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x, edge_index)

            if self.norm is not None:
                x = norm(x)

            x = F.leaky_relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
