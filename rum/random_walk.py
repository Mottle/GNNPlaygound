import torch
from torch_cluster import random_walk as pyg_random_walk

def uniform_random_walk_pyg(edge_index, num_nodes, num_samples, length, subsample=None):
    """
    Random walk on a graph using PyG (torch_cluster).

    Parameters
    ----------
    edge_index : Tensor
        The edge index of the graph (2, E).
    num_nodes : int
        Total number of nodes in the graph.
    num_samples : int
        Number of random walks per node.
    length : int
        Length of each random walk (number of nodes).
    subsample : Tensor, optional
        Subset of nodes to start walks from.

    Returns
    -------
    walks : Tensor
        The random walks. Shape: (num_samples, num_start_nodes, length)
    eids : None
        PyG efficient implementation does not return edge IDs by default.
    """
    device = edge_index.device
    
    # 1. 确定起始节点
    if subsample is None:
        start_nodes = torch.arange(num_nodes, device=device)
        num_start_nodes = num_nodes
    else:
        start_nodes = subsample
        num_start_nodes = subsample.size(0)

    # 2. 重复起始节点以匹配 num_samples
    # DGL逻辑: repeat(num_samples) 生成 [0, 1, 2, 0, 1, 2...]
    start_nodes = start_nodes.repeat(num_samples)

    # 3. 执行随机游走
    # walk_length 是步数 (边的数量)，所以是 节点数(length) - 1
    # pyg_random_walk 需要 (row, col) 格式
    row, col = edge_index[0], edge_index[1]
    
    walks = pyg_random_walk(row, col, start_nodes, walk_length=length-1)

    # 4. Reshape
    # torch_cluster 返回形状为 [total_walks, length]
    # 我们将其 reshape 为 [num_samples, num_nodes, length] 以匹配原代码逻辑
    walks = walks.view(num_samples, num_start_nodes, length)
    
    # PyG 的 random_walk 不直接返回 edge_ids，通常返回 None 或者是为了保持接口一致
    eids = None 
    
    return walks, eids

# uniqueness 函数保持完全不变，因为它是纯 Tensor 操作
def uniqueness(walk):
    """
    Compute the uniqueness of a random walk.
    (This function is framework-agnostic)
    """
    # [B, N, L, 1] == [B, N, 1, L] -> [B, N, L, L]
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    # argmax returns the first index where value is 1 (True)
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal