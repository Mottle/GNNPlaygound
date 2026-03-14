from re import sub
from typing import Callable
import torch
from torch_geometric.nn import global_mean_pool
from .layers import RUMLayer, Consistency

# from utils.perf_counter import perf_counter


class RUMModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        depth: int,
        activation: Callable = torch.nn.ELU(),
        temperature=0.1,
        self_supervise_weight=0.05,
        consistency_weight=0.01,
        **kwargs,
    ):
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features, hidden_features, bias=True)
        self.fc_out = torch.nn.Linear(hidden_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.depth = depth
        self.layers = torch.nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                RUMLayer(hidden_features, hidden_features, in_features, **kwargs)
            )
        self.activation = activation
        self.consistency = Consistency(temperature=temperature)
        self.self_supervise_weight = self_supervise_weight
        self.consistency_weight = consistency_weight

    def forward(self, data, h, e=None, consistency_weight=None, subsample=None):
        # t_s = perf_counter()

        if consistency_weight is None:
            consistency_weight = self.consistency_weight

        h0 = h
        h = self.fc_in(h)
        loss = 0.0

        # t_layers_s = perf_counter()
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                h = h.mean(0)
            h, _loss = layer(data, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        # t_layers_e = perf_counter()

        h = self.fc_out(h).softmax(-1)

        # t_ss_s = perf_counter()
        if self.training:
            _loss = self.consistency(h)
            _loss = _loss * consistency_weight
            loss = loss + _loss
        # t_ss_e = perf_counter()

        # t_e = perf_counter()
        # print(f"Total forward time: {t_e - t_s:.4f} seconds")
        # print(f"  Layers time: {t_layers_e - t_layers_s:.4f} seconds, percentage: {(t_layers_e - t_layers_s) / (t_e - t_s) * 100:.2f}%")
        # print(f"  Consistency time: {t_ss_e - t_ss_s:.4f} seconds, percentage: {(t_ss_e - t_ss_s) / (t_e - t_s) * 100:.2f}%")
        return h, loss


class RUMGraphRegressionModel(RUMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_out = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.hidden_features),
            self.activation,
            torch.nn.Linear(self.hidden_features, self.hidden_features),
            self.activation,
            torch.nn.Dropout(kwargs["dropout"]),
            torch.nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, data, h, e=None, subsample=None):
        h0 = h
        h = self.fc_in(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # h = torch.nn.functional.tanh(h)
                h = torch.nn.SiLU()(h)
                h = h.mean(0)
            h, _loss = layer(data, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        # h = self.activation(h)
        h = h.mean(0)
        # PyG: batch 必须存在
        if hasattr(data, "batch"):
            h = global_mean_pool(h, data.batch)
        h = self.fc_out(h)
        return h, loss
