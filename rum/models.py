from typing import Callable
import torch
import torch.nn.functional as F
# 引入 PyG 的全局池化函数
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
        # 注意：这里的 RUMLayer 内部也需要适配 PyG (接收 edge_index 而不是 g)
        for _ in range(depth):
            self.layers.append(RUMLayer(hidden_features, hidden_features, in_features, **kwargs))
        self.activation = activation
        self.consistency = Consistency(temperature=temperature)
        self.self_supervise_weight = self_supervise_weight
        self.consistency_weight = consistency_weight

    def forward(self, x, edge_index, batch=None, e=None, consistency_weight=None, subsample=None):
        """
        参数:
            x (Tensor): 节点特征矩阵 [num_nodes, in_features]
            edge_index (LongTensor): 边索引 [2, num_edges]
            batch (LongTensor, optional): 批次向量 [num_nodes]，用于指示节点属于哪个图
            e (Tensor, optional): 边特征
        """
        if consistency_weight is None:
            consistency_weight = self.consistency_weight
        
        # PyG 中不需要 g.local_var()，因为操作通常是 out-of-place 的
        h0 = x 
        h = self.fc_in(x)
        loss = 0.0
        
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # 假设层输出形状包含采样维度，取平均
                h = h.mean(0)
            
            # 关键修改：传递 edge_index 而不是 g
            # 请确保 RUMLayer 的 forward 函数也同步修改了签名
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
        # 增加 get 默认值防止 kwargs 中缺少 dropout 报错
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
        """
        前向传播逻辑
        注意：batch 参数对于图回归任务是必须的
        """
        h0 = x
        h = self.fc_in(x)
        loss = 0.0
        
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # 对应原代码: h = torch.nn.SiLU()(h)
                h = F.silu(h) 
                h = h.mean(0)
            
            # 同样传递 edge_index
            h, _loss = layer(edge_index, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        
        # 处理最后一层输出
        h = h.mean(0)
        
        # --- 核心改动点: 图读出 (Readout) ---
        # DGL 原代码: 
        # g.ndata["h"] = h
        # h = dgl.mean_nodes(g, "h")
        
        # PyG 代码:
        # 使用 global_mean_pool 根据 batch 向量对属于同一图的节点特征求平均
        if batch is None:
             # 如果没有提供 batch，假设所有节点属于同一个图（全图平均）
            h = h.mean(dim=0, keepdim=True)
        else:
            h = global_mean_pool(h, batch)
        
        h = self.fc_out(h)
        return h, loss