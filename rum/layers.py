from re import sub
from typing import Callable
import torch
import torch.nn.functional as F
# 引入 PyG 的全局池化层
from torch_geometric.nn import global_mean_pool
from .layers import RUMLayer, Consistency

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
        # 注意：这里的 RUMLayer 内部也需要适配 PyG (接收 edge_index)
        for _ in range(depth):
            self.layers.append(RUMLayer(hidden_features, hidden_features, in_features, **kwargs))
        self.activation = activation
        self.consistency = Consistency(temperature=temperature)
        self.self_supervise_weight = self_supervise_weight
        self.consistency_weight = consistency_weight

    def forward(self, x, edge_index, batch=None, e=None, consistency_weight=None, subsample=None):
        """
        参数:
        x: 节点特征 [num_nodes, in_features]
        edge_index: 边索引 [2, num_edges]
        batch: (可选) 批次向量 [num_nodes], 用于图分类/回归任务
        """
        if consistency_weight is None:
            consistency_weight = self.consistency_weight
        
        h0 = x # 保存原始特征
        h = self.fc_in(x)
        loss = 0.0
        
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # 假设 RUMLayer 返回 [samples, nodes, dim]，这里进行平均
                h = h.mean(0)
            
            # RUMLayer 接口需要改为接收 edge_index 而不是 g
            h, _loss = layer(edge_index, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
            
        h = self.fc_out(h).softmax(-1)
        
        if self.training:
            _loss = self.consistency(h)
            _loss = _loss * consistency_weight
            loss = loss + _loss
            
        return h, loss

class RUMGraphRegressionModel(RUMModel):
    def __init__(self, *args, **kwargs):
        # 确保 kwargs 中包含 dropout，或者设置默认值
        dropout = kwargs.get("dropout", 0.0) 
        super().__init__(*args, **kwargs)

        self.fc_out = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.hidden_features),
            self.activation,
            torch.nn.Linear(self.hidden_features, self.hidden_features),
            self.activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_features, self.out_features),
        )

    def forward(self, x, edge_index, batch=None, e=None, subsample=None):
        h0 = x
        h = self.fc_in(x)
        loss = 0.0
        
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # 对应原代码中的 h = torch.nn.SiLU()(h)
                h = F.silu(h) 
                h = h.mean(0)
            
            # 同样，layer 需要适配接收 edge_index
            h, _loss = layer(edge_index, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        
        # 处理最后一层的输出
        h = h.mean(0)
        
        # --- 核心改动点 ---
        # DGL: dgl.mean_nodes(g, "h")
        # PyG: global_mean_pool(x, batch)
        # 如果 batch 为 None (单图)，PyG 会默认处理为所有节点属于同一个图
        h = global_mean_pool(h, batch)
        
        h = self.fc_out(h)
        return h, loss