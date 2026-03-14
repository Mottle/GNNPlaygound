import torch
from torch_cluster import random_walk

# def uniform_random_walk(data, num_samples, length, subsample=None):
#     device = data.edge_index.device
#     num_total_nodes = data.num_nodes

#     if subsample is None:
#         nodes = torch.arange(num_total_nodes, device=device)
#         num_nodes = num_total_nodes
#     else:
#         nodes = subsample
#         num_nodes = subsample.size(0)

#     nodes = nodes.repeat(num_samples)
#     row, col = data.edge_index

#     # ==========================================
#     # 极致提速：构建 O(1) 稠密查找表并缓存在 data 中
#     # ==========================================
#     if not hasattr(data, '_fast_edge_cache'):
#         # 1. 确保边索引按源节点排序
#         sorted_idx = torch.argsort(row)
#         sorted_row = row[sorted_idx]
#         sorted_col = col[sorted_idx]

#         # 2. 计算节点度数与行偏移
#         degree = torch.bincount(sorted_row, minlength=num_total_nodes)
#         max_degree = degree.max().item()

#         row_ptr = torch.zeros(num_total_nodes + 1, dtype=torch.long, device=device)
#         row_ptr[1:] = degree.cumsum(0)

#         # 3. 向量化计算每条边在源节点邻居列表中的局部索引 (0 到 degree[u]-1)
#         local_idx = torch.arange(sorted_row.size(0), device=device) - row_ptr[sorted_row]

#         # 4. 初始化形状为 (N, max_degree) 的查找表，用 -1 填充
#         adj_nodes = torch.full((num_total_nodes, max_degree), -1, device=device, dtype=torch.long)
#         adj_eids = torch.full((num_total_nodes, max_degree), -1, device=device, dtype=torch.long)

#         # 5. 填入目标节点与原始边 ID
#         adj_nodes[sorted_row, local_idx] = sorted_col
#         adj_eids[sorted_row, local_idx] = sorted_idx

#         data._fast_edge_cache = (adj_nodes, adj_eids)
#     else:
#         adj_nodes, adj_eids = data._fast_edge_cache

#     # ==========================================
#     # 执行随机游走
#     # ==========================================
#     walks = random_walk(
#         row, col, nodes, walk_length=length - 1, num_nodes=num_total_nodes
#     )

#     u = walks[:, :-1].flatten()
#     v = walks[:, 1:].flatten()

#     eids_flat = torch.full_like(u, -1)
#     mask = (u != -1) & (v != -1)

#     if mask.any():
#         u_valid = u[mask]
#         v_valid = v[mask]

#         # ==========================================
#         # O(1) 极速匹配边 ID
#         # ==========================================
#         # 提取起点的所有邻居，形状: (有效游走步数, max_degree)
#         target_nodes = adj_nodes[u_valid]

#         # 广播比较找到目标节点 v 的位置
#         match = target_nodes == v_valid.unsqueeze(-1)

#         # 获取匹配成功的掩码和局部索引
#         valid_match_mask = match.any(dim=-1)
#         match_local_idx = match.long().argmax(dim=-1)

#         # 仅提取真实存在的边 ID
#         u_final = u_valid[valid_match_mask]
#         idx_final = match_local_idx[valid_match_mask]

#         matched_eids = adj_eids[u_final, idx_final]

#         # 映射回原展平数组
#         valid_positions = mask.nonzero(as_tuple=True)[0]
#         actual_match_positions = valid_positions[valid_match_mask]
#         eids_flat[actual_match_positions] = matched_eids

#     walks = walks.view(num_samples, num_nodes, length)
#     eids = eids_flat.view(num_samples, num_nodes, length - 1)

#     return walks, eids


def uniform_random_walk(data, num_samples, length, subsample=None):
    """
    Random walk on a PyG graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The graph.
    num_samples : int
        Number of random walks per node.
    length : int
        Length of each random walk (number of nodes).
    subsample : Tensor, optional
        Nodes to subsample.

    Returns
    -------
    walks : Tensor
        The random walks, shape (num_samples, num_nodes, length).
    eids : Tensor
        The edge IDs traversed, shape (num_samples, num_nodes, length-1).
    """
    device = data.edge_index.device
    num_total_nodes = data.num_nodes

    if subsample is None:
        nodes = torch.arange(num_total_nodes, device=device)
        num_nodes = num_total_nodes
    else:
        nodes = subsample
        num_nodes = subsample.size(0)

    # 将起始节点复制 num_samples 次
    nodes = nodes.repeat(num_samples)

    row, col = data.edge_index

    # 执行随机游走。PyG的 walk_length 指的是游走的步数，即 length - 1
    # 返回的 walks 形状为 (num_samples * num_nodes, length)
    walks = random_walk(
        row, col, nodes, walk_length=length - 1, num_nodes=num_total_nodes
    )

    # -----------------------------------------
    # 提取经过的边 ID (eids) 以对齐 DGL 的行为
    # -----------------------------------------
    u = walks[:, :-1].flatten()
    v = walks[:, 1:].flatten()

    # 构造边哈希映射 (u * N + v) 以便快速查找原图中的边索引
    edge_hash = row * num_total_nodes + col
    sorted_idx = torch.argsort(edge_hash)
    sorted_hash = edge_hash[sorted_idx]

    # 初始化边 ID 为 -1（与 DGL 保持一致，表示截断或无效游走）
    eids_flat = torch.full_like(u, -1)

    # 过滤掉填充部分 (-1)
    mask = (u != -1) & (v != -1)

    if mask.any():
        u_valid = u[mask]
        v_valid = v[mask]
        walk_hash = u_valid * num_total_nodes + v_valid

        # 使用二分查找寻找对应的边 ID
        found_idx = torch.searchsorted(sorted_hash, walk_hash)
        # 防止越界
        found_idx = found_idx.clamp(max=sorted_hash.size(0) - 1)

        # 验证是否精确匹配（处理可能存在的孤立节点游走异常）
        match = sorted_hash[found_idx] == walk_hash
        matched_eids = sorted_idx[found_idx[match]]

        # 将匹配成功的边 ID 映射回平铺的 eids_flat 张量中
        valid_positions = mask.nonzero(as_tuple=True)[0]
        actual_match_positions = valid_positions[match]
        eids_flat[actual_match_positions] = matched_eids

    # 将形状重塑为 (num_samples, num_nodes, ...)
    walks = walks.view(num_samples, num_nodes, length)
    eids = eids_flat.view(num_samples, num_nodes, length - 1)

    return walks, eids


# uniqueness 函数无需任何修改，保持原有逻辑即可
def uniqueness(walk):
    walk_equal = walk.unsqueeze(-1) == walk.unsqueeze(-2)
    walk_equal = (1 * walk_equal).argmax(dim=-1)
    return walk_equal
