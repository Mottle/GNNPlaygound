import torch
from torch_geometric.data import Data

class AddVirtualNode(object):
    def __call__(self, data):
        num_nodes = data.num_nodes
        feature_dim = data.x.size(1)
        
        # 1. 扩展节点特征 (末尾追加虚拟节点)
        # 初始化为0，模型里会替换为 Embedding
        virtual_node_x = torch.zeros((1, feature_dim), dtype=data.x.dtype)
        data.x = torch.cat([data.x, virtual_node_x], dim=0)
        
        # 2. 生成虚拟边连接
        sources = torch.arange(num_nodes, dtype=torch.long)
        targets = torch.full((num_nodes,), num_nodes, dtype=torch.long)
        
        # V -> S (Encoder 用)
        v_to_s = torch.stack([sources, targets], dim=0)
        # S -> V (Decoder 用)
        s_to_v = torch.stack([targets, sources], dim=0)
        
        # 3. 合并边 (这是你要求的关键：作为完整的图)
        # edge_index_enc 包含: 原图内部边 + (V->S)
        data.edge_index_enc = torch.cat([data.edge_index, v_to_s], dim=1)
        
        # edge_index_dec 包含: 原图内部边 + (S->V)
        data.edge_index_dec = torch.cat([data.edge_index, s_to_v], dim=1)
        
        # 4. 辅助掩码
        mask = torch.zeros(num_nodes + 1, dtype=torch.bool)
        mask[num_nodes] = True
        data.virtual_node_mask = mask
        
        data.num_nodes = num_nodes + 1
        
        # 清理掉原始的 edge_index，防止误用 (可选)
        # data.edge_index = None 
        
        return data