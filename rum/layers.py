import math
import torch
from .random_walk import uniform_random_walk, uniqueness
from .rnn import GRU


class RUMLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        original_features: int,
        num_samples: int,
        length: int,
        dropout: float = 0.2,
        rnn: torch.nn.Module = GRU,
        random_walk: callable = uniform_random_walk,
        activation: callable = torch.nn.Identity(),
        edge_features: int = 0,
        binary: bool = True,
        directed: bool = False,
        degrees: bool = True,
        self_supervise: bool = True,
        **kwargs
    ):
        super().__init__()
        # out_features = out_features // 2
        # self.fc = torch.nn.Linear(in_features + 2 * out_features + 1, out_features, bias=False)
        self.rnn = rnn(
            in_features + 2 * out_features + int(degrees), out_features, **kwargs
        )
        self.rnn_walk = rnn(2, out_features, bidirectional=True, **kwargs)
        if edge_features > 0:
            self.fc_edge = torch.nn.Linear(
                edge_features, int(degrees) + in_features + 2 * out_features, bias=False
            )
        self.in_features = in_features
        self.out_features = out_features
        self.random_walk = random_walk
        self.num_samples = num_samples
        self.length = length
        self.dropout = torch.nn.Dropout(dropout)
        if self_supervise:
            self.self_supervise = SelfSupervise(
                out_features, original_features, binary=binary
            )
        else:
            self.self_supervise = None
        self.activation = activation
        self.directed = directed
        self.degrees = degrees

    def forward(self, data, h, y0, e=None, subsample=None):
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The graph data.

        h : Tensor
            The input features.

        Returns
        -------
        h : Tensor
            The output features.
        """
        walks, eids = self.random_walk(
            data,
            num_samples=self.num_samples,
            length=self.length,
            subsample=subsample,
        )
        if self.directed:
            walks = torch.where(
                walks == -1,
                walks[..., 0:1],
                walks,
            )

        # walks = torch.zeros(1, 5000, 4).int().cuda()
        # eids = None

        uniqueness_walk = uniqueness(walks)
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = uniqueness_walk / uniqueness_walk.shape[-1]
        uniqueness_walk = uniqueness_walk * math.pi * 2.0
        uniqueness_walk = torch.cat(
            [
                uniqueness_walk.sin().unsqueeze(-1),
                uniqueness_walk.cos().unsqueeze(-1),
            ],
            dim=-1,
        )

        # h = h[walks]

        # === 修复: 节点特征的 -1 索引处理 ===
        # 1. 创建一个全 0 的 Dummy 节点特征
        dummy_node = torch.zeros(1, h.size(-1), device=h.device, dtype=h.dtype)
        # 2. 将 Dummy 节点拼接到原节点特征矩阵的最后
        h_padded = torch.cat([h, dummy_node], dim=0)

        # 3. 原本的节点总数刚好是 Dummy 节点在 h_padded 中的索引
        dummy_node_idx = h.size(0)

        # 4. 将 walks 中的 -1 替换为 dummy_node_idx
        walks_safe = torch.where(walks == -1, dummy_node_idx, walks)

        # 5. 安全提取特征 (遇到原 -1 时将提取到全 0 向量)
        h = h_padded[walks_safe]

        num_directions = 2 if self.rnn_walk.bidirectional else 1
        h0 = torch.zeros(
            self.rnn_walk.num_layers * num_directions,
            *h.shape[:-2],
            self.out_features,
            device=h.device,
        )
        y_walk, h_walk = self.rnn_walk(uniqueness_walk, h0)
        h_walk = h_walk.mean(0, keepdim=True)

        if self.rnn.num_layers > 1:
            h_walk = h_walk.repeat(self.rnn.num_layers, 1, 1, 1)

        if self.degrees:
            # PyG: 计算入度
            edge_index = data.edge_index
            num_nodes = data.num_nodes
            node_degrees = torch.bincount(edge_index[1], minlength=num_nodes)

            dummy_degree = torch.zeros(
                1, device=node_degrees.device, dtype=node_degrees.dtype
            )
            node_degrees_padded = torch.cat([node_degrees, dummy_degree], dim=0)

            # degrees = (
            #     node_degrees[walks.flatten()]
            #     .float()
            #     .reshape(*walks.shape)
            #     .unsqueeze(-1)
            # )

            degrees = (
                node_degrees_padded[walks_safe.flatten()]
                .float()
                .reshape(*walks_safe.shape)
                .unsqueeze(-1)
            )

            # degrees = degrees / degrees.max()
            degrees = degrees / (
                degrees.max() + 1e-9
            )  # 添加 epsilon 防止图中所有节点度数均为 0 时发生除零错误

            h = torch.cat([h, y_walk, degrees], dim=-1)
        else:
            h = torch.cat([h, y_walk], dim=-1)
        # h = self.fc(h)
        # h = self.activation(h)

        # if e is not None:
        #     _h = torch.empty(
        #         *h.shape[:-2],
        #         2 * h.shape[-2] - 1,
        #         h.shape[-1],
        #         device=h.device,
        #         dtype=h.dtype,
        #     )
        #     _h[..., ::2, :] = h
        #     _h[..., 1::2, :] = self.fc_edge(e)[eids]
        #     h = _h

        if e is not None:
            _h = torch.empty(
                *h.shape[:-2],
                2 * h.shape[-2] - 1,
                h.shape[-1],
                device=h.device,
                dtype=h.dtype,
            )
            _h[..., ::2, :] = h

            # === 修复: 边特征的 -1 索引处理 ===
            edge_feats = self.fc_edge(e)

            # 1. 创建一个全 0 的 Dummy 边特征
            dummy_edge = torch.zeros(
                1, edge_feats.size(-1), device=edge_feats.device, dtype=edge_feats.dtype
            )
            # 2. 拼接到特征矩阵最后
            edge_padded = torch.cat([edge_feats, dummy_edge], dim=0)

            # 3. Dummy 边的索引即为原边数
            dummy_edge_idx = edge_feats.size(0)

            # 4. 将 eids 中的 -1 替换为 dummy_edge_idx
            eids_safe = torch.where(eids == -1, dummy_edge_idx, eids)

            # 5. 安全提取特征
            _h[..., 1::2, :] = edge_padded[eids_safe]
            h = _h

        y, h = self.rnn(h, h_walk)

        if self.training and self.self_supervise:
            if e is not None:
                y = y[..., ::2, :]
            loss = self.self_supervise(y, y0[walks])
        else:
            loss = 0.0

        h = self.activation(h)
        h = h.mean(0)
        h = self.dropout(h)

        return h, loss


class Consistency(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, probs):
        avg_probs = probs.mean(0)
        sharpened_probs = avg_probs.pow(1 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdim=True)
        loss = (sharpened_probs - avg_probs).pow(2).sum(-1).mean()
        return loss


class SelfSupervise(torch.nn.Module):
    def __init__(self, in_features, out_features, subsample=100, binary=True):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.subsample = subsample
        self.binary = binary

    def forward(self, y_hat, y):
        idxs = torch.randint(
            high=y_hat.shape[-3], size=(self.subsample,), device=y.device
        )
        y, y_hat = y.flatten(0, -3), y_hat.flatten(0, -3)
        y = y[..., idxs, 1:, :].contiguous()
        y_hat = y_hat[..., idxs, :-1, :].contiguous()
        y_hat = self.fc(y_hat)
        if self.binary:
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=y.detach().mean().pow(-1))(
                y_hat, y
            )
        else:
            # loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            loss = torch.nn.MSELoss()(y_hat, y)
        return loss
