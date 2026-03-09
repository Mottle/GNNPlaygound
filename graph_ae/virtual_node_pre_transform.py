import torch
import torch.nn.functional as F


class AddVirtualNode(object):
    def __init__(self, mode="zinc"):
        """
        Args:
            mode (str):
                - "zinc": 适用于 One-Hot 编码 (Float Tensor)。
                          处理逻辑 Pad 0, 末尾置 1.0。
                - "moleculenet": 适用于 Integer Index 编码 (Long Tensor)。
                          处理逻辑 Pad 0, 末尾置 1 (作为类别的索引)。
        """
        assert mode in ["zinc", "moleculenet"], "Mode must be 'zinc' or 'moleculenet'"
        self.mode = mode

    def __call__(self, data):
        num_nodes = data.num_nodes

        # ---------------------------------------------------------
        # 1. 扩展节点特征 (x)
        # ---------------------------------------------------------
        if data.x is not None:
            feature_dim = data.x.size(1)
            # 初始化虚拟节点特征 (全0)
            virtual_node_x = torch.zeros((1, feature_dim), dtype=data.x.dtype)
            data.x = torch.cat([data.x, virtual_node_x], dim=0)

        # ---------------------------------------------------------
        # 2. 生成虚拟边索引 (edge_index)
        # ---------------------------------------------------------
        # 生成源节点和目标节点索引
        sources = torch.arange(num_nodes, dtype=torch.long)
        targets = torch.full((num_nodes,), num_nodes, dtype=torch.long)

        # V -> S (原节点 -> 虚拟节点)
        v_to_s_index = torch.stack([sources, targets], dim=0)
        # S -> V (虚拟节点 -> 原节点)
        s_to_v_index = torch.stack([targets, sources], dim=0)

        # ---------------------------------------------------------
        # 3. 处理边特征 (edge_attr) - 【核心修改】
        # ---------------------------------------------------------
        if data.edge_attr is not None:
            current_dim = data.edge_attr.size(1)
            original_dtype = data.edge_attr.dtype

            # --- 分支 A: ZINC 模式 (Float/One-Hot) ---
            if self.mode == "zinc":
                # 确保数据是 Float (ZINC通常是Float)
                if original_dtype != torch.float:
                    data.edge_attr = data.edge_attr.float()

                # 扩展原始边: [E, D] -> [E, D+1] (末尾补0)
                original_edge_attr = F.pad(data.edge_attr, (0, 1), value=0)

                # 生成虚拟边: [N, D+1] -> 末尾置 1.0
                virtual_edge_attr = torch.zeros(
                    (num_nodes, current_dim + 1), dtype=torch.float
                )
                virtual_edge_attr[:, -1] = 1.0

            # --- 分支 B: MoleculeNet 模式 (Long/Integer) ---
            else:  # mode == "moleculenet"
                # 确保数据是 Long (Embedding需要Long)
                if original_dtype != torch.long:
                    data.edge_attr = data.edge_attr.long()

                # 扩展原始边: [E, D] -> [E, D+1] (末尾补0)
                # 这里的0意味着在新增的"Is_Virtual"列中，它是"False"
                original_edge_attr = F.pad(data.edge_attr, (0, 1), value=0)

                # 生成虚拟边: [N, D+1] -> 末尾置 1
                # 这里的1意味着在新增的"Is_Virtual"列中，它是"True"
                virtual_edge_attr = torch.zeros(
                    (num_nodes, current_dim + 1), dtype=torch.long
                )
                virtual_edge_attr[:, -1] = 1

            # --- 合并边特征 ---
            # Encoder: 原边 + (V->S)
            data.edge_attr_enc = torch.cat(
                [original_edge_attr, virtual_edge_attr], dim=0
            )
            # Decoder: 原边 + (S->V)
            data.edge_attr_dec = torch.cat(
                [original_edge_attr, virtual_edge_attr], dim=0
            )

            # 记录新的维度 (供模型初始化 Linear 或 Embedding 使用)
            data.edge_attr_dim = current_dim + 1

        else:
            # 如果原数据没有边特征，设为 None
            data.edge_attr_enc = None
            data.edge_attr_dec = None
            data.edge_attr_dim = 0

        # ---------------------------------------------------------
        # 4. 合并边索引
        # ---------------------------------------------------------
        data.edge_index_enc = torch.cat([data.edge_index, v_to_s_index], dim=1)
        data.edge_index_dec = torch.cat([data.edge_index, s_to_v_index], dim=1)

        # ---------------------------------------------------------
        # 5. 元数据更新
        # ---------------------------------------------------------
        mask = torch.zeros(num_nodes + 1, dtype=torch.bool)
        mask[num_nodes] = True
        data.virtual_node_mask = mask

        data.num_nodes = num_nodes + 1

        return data
