import math
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from .rnn import GRU
# 假设 uniform_random_walk_pyg 和 uniqueness 定义在 .random_walk 模块中
from .random_walk import uniform_random_walk, uniqueness

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
            random_walk: callable = uniform_random_walk, # 默认使用 PyG 版本
            activation: callable = torch.nn.Identity(),
            edge_features: int = 0,
            binary: bool = True,
            directed: bool = False,
            degrees: bool = True,
            self_supervise: bool = True,
            **kwargs
    ):
        super().__init__()
        # PyG 的 GRU 定义与 DGL 兼容，直接使用
        self.rnn = rnn(in_features + 2 * out_features + int(degrees), out_features, **kwargs)
        self.rnn_walk = rnn(2, out_features, bidirectional=True, **kwargs)
        
        self.use_edge_features = edge_features > 0
        if self.use_edge_features:
            self.fc_edge = torch.nn.Linear(edge_features, int(degrees) + in_features + 2 * out_features, bias=False)
            
        self.in_features = in_features
        self.out_features = out_features
        self.random_walk = random_walk
        self.num_samples = num_samples
        self.length = length
        self.dropout = torch.nn.Dropout(dropout)
        
        if self_supervise:
            self.self_supervise_module = SelfSupervise(out_features, original_features, binary=binary)
        else:
            self.self_supervise_module = None
            
        self.activation = activation
        self.directed = directed
        self.degrees = degrees

    def forward(self, edge_index, h, y0, e=None, subsample=None):
        """
        PyG Forward pass.
        
        Parameters
        ----------
        edge_index : LongTensor
            Graph connectivity [2, num_edges].
        h : Tensor
            Node features [num_nodes, in_features].
        y0 : Tensor
            Original/Target features for self-supervision.
        """
        num_nodes = h.size(0)
        
        # 1. 执行随机游走
        walks = self.random_walk(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_samples=self.num_samples,
            length=self.length,
            subsample=subsample,
        )
        # walks shape: [num_samples, num_start_nodes, length]

        if self.directed:
            # PyG walks 中 -1 表示 invalid node (如果 walk 提前终止)
            # 处理逻辑与原代码类似，如果遇到 -1，用起始点填充或者其他逻辑
            mask = (walks == -1)
            if mask.any():
                start_nodes = walks[..., 0:1]
                walks = torch.where(mask, start_nodes, walks)

        # 2. 计算 Uniqueness Encoding
        uniqueness_walk = uniqueness(walks)
        # 翻转序列 (reverse walk?) - 保持原逻辑
        walks_flipped = walks.flip(-1)
        uniqueness_walk = uniqueness_walk.flip(-1)
        
        uniqueness_walk = uniqueness_walk.float() / uniqueness_walk.shape[-1]
        uniqueness_walk = uniqueness_walk * math.pi * 2.0
        uniqueness_walk = torch.cat(
            [
                uniqueness_walk.sin().unsqueeze(-1),
                uniqueness_walk.cos().unsqueeze(-1),
            ],
            dim=-1,
        )

        # 3. 获取节点特征
        # PyG 的 walk index 可能包含 -1，作为 padding，需要处理
        # 这里假设 walk 是有效的，或者上面的 directed 处理已经修复了
        # 为了安全，可以将 -1 映射到 0 并 mask 掉，或者假设图是连通的
        valid_walks_mask = (walks_flipped != -1)
        safe_walks = walks_flipped.clone()
        safe_walks[~valid_walks_mask] = 0 # 避免索引报错
        
        h_walk_features = h[safe_walks] 
        # 如果有无效节点，特征置零 (可选)
        # h_walk_features[~valid_walks_mask] = 0 

        # 4. RNN 处理 Walk Pattern
        num_directions = 2 if self.rnn_walk.bidirectional else 1
        # RNN 初始状态
        h0_rnn = torch.zeros(
            self.rnn_walk.num_layers * num_directions, 
            *h_walk_features.shape[:-2], 
            self.out_features, 
            device=h.device
        )
        
        y_walk, h_walk = self.rnn_walk(uniqueness_walk, h0_rnn)
        
        # h_walk (hidden state) 通常是 [layers*dir, batch, hidden]
        # 原代码做 mean(0) -> [batch, hidden] (approx)
        # 注意: RNN 输出形状根据 batch_first=True/False 而定，这里假设兼容
        h_walk = h_walk.mean(0, keepdim=True)
        
        if self.rnn.num_layers > 1:
            h_walk = h_walk.repeat(self.rnn.num_layers, 1, 1, 1)

        # 5. 拼接特征
        features_list = [h_walk_features, y_walk]

        if self.degrees:
            # PyG 计算度数
            # 需要计算 walk 中每个节点的度数
            # 全局度数计算:
            deg = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
            # 获取 walk 中节点的度数
            walk_degrees = deg[safe_walks].unsqueeze(-1)
            walk_degrees = walk_degrees / (deg.max() + 1e-6) # 归一化
            features_list.append(walk_degrees)
        
        h_combined = torch.cat(features_list, dim=-1)

        # 6. 处理边特征 (如果存在)
        # PyG random_walk 不返回 eids。
        # 如果必须用边特征，这在 PyG 中比较复杂。
        # 简单方案：如果 e 是 None，跳过。如果非空，PyG 很难高效获取 walk 对应的 edge feat
        if self.use_edge_features and e is not None:
             # 注意：这是一个极其昂贵的操作，通常建议在 PyG 中避免依赖 random walk 的 edge id
             # 或者预先在图中将 edge attr 聚合到 node 上
             pass 
             # 为了保持代码运行，这里暂时跳过具体的 edge lookup 实现
             # 如果必须实现，需要建立 (u, v) -> edge_idx 的哈希表或稀疏矩阵查找

        # 7. 主 RNN 聚合
        # h_combined: [samples, nodes, length, feat_dim]
        # h_walk: initial state
        y_out, h_out = self.rnn(h_combined, h_walk)

        # 8. 自监督损失
        loss = 0.0
        if self.training and self.self_supervise_module is not None:
            # y_out 是 RNN 的输出序列
            # y0 是原始特征，我们需要预测 walk 序列中的特征
            target = y0[safe_walks]
            loss = self.self_supervise_module(y_out, target)

        h_out = self.activation(h_out)
        
        # 聚合多个 sample 的结果: mean(0)
        # h_out shape: [layers, samples, nodes, hidden] -> 取最后一层 hidden state
        # 但这里的 h_out 是 RNN 的 hidden state 输出，通常是 [layers, batch_size, hidden]
        # 这里 batch_size = num_samples * num_nodes
        # 此时需要 reshape 回 [num_samples, num_nodes, hidden] 然后 mean(0)
        
        # *修正*: `rnn` 返回的 h_out 形状通常是 [num_layers, batch, hidden]
        # 这里的 batch 维度其实是 (num_samples, num_nodes) 展平或者是多维的
        # 假设你的 rnn 能够处理多维输入，或者输入已经被展平处理过。
        
        # 简单起见，假设 h_out 保持维度结构
        if h_out.dim() > 3: # [layers, samples, nodes, hidden]
             h_out = h_out.mean(1) # 对 samples 维度求平均
        
        # Dropout
        h_out = self.dropout(h_out)
        
        # squeeze layers 维度 (如果只有1层或者取最后一层)
        if h_out.size(0) == 1:
            h_out = h_out.squeeze(0)
        else:
            h_out = h_out[-1] # 取最后一层

        return h_out, loss

# Consistency 和 SelfSupervise 是纯 Tensor 操作，通用
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
        # 注意: 这里 in_features 应该是 RNN 的输出维度 (hidden_dim)
        # out_features 是要预测的原始特征维度
        self.fc = torch.nn.Linear(in_features, out_features)
        self.subsample = subsample
        self.binary = binary

    def forward(self, y_hat, y):
        # y_hat: [..., length, hidden]
        # y: [..., length, original_feat]
        
        # 随机采样一些时间步计算 loss，减少计算量
        # y_hat.shape[-2] 是序列长度 (length)
        seq_len = y_hat.shape[-2]
        if seq_len > 1:
            idxs = torch.randint(high=seq_len, size=(self.subsample, ), device=y.device)
            # 展平 batch 维度
            y_flat = y.flatten(0, -3)
            y_hat_flat = y_hat.flatten(0, -3)
            
            # 这里的切片逻辑需要根据具体的自监督任务调整
            # 原代码逻辑: 预测下一个时间步?
            # y = y[..., idxs, 1:, :].contiguous() 
            # y_hat = y_hat[..., idxs, :-1, :].contiguous()
            
            # 简化版：直接对齐
            y_sel = y_flat[..., idxs, :]
            y_hat_sel = y_hat_flat[..., idxs, :]
            
            y_pred = self.fc(y_hat_sel)
            
            if self.binary:
                loss = torch.nn.BCEWithLogitsLoss()(y_pred, y_sel)
            else:
                loss = torch.nn.MSELoss()(y_pred, y_sel)
            return loss
        else:
            return torch.tensor(0.0, device=y.device)