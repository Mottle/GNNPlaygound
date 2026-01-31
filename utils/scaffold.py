from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch

def generate_scaffold(smiles, include_chirality=True):
    """
    与 HiMol 保持一致，默认开启手性 (include_chirality=True)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold

def scaffold_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    完全复刻 DeepChem/HiMol 的确定性划分逻辑
    """
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    # 1. 生成骨架映射
    scaffolds = defaultdict(list)
    print("Generating scaffolds (Deterministic)...")
    
    for idx, data in enumerate(dataset):
        # 兼容性处理：有些数据集 smiles 在 data.smiles，有些在 data.raw_smiles
        smiles = data.smiles if hasattr(data, 'smiles') else data.raw_smiles
        scaffold = generate_scaffold(smiles, include_chirality=True) # ⚠️ 开启手性
        if scaffold is not None:
            scaffolds[scaffold].append(idx)
        else:
            scaffolds["INVALID"].append(idx)

    # 2. 排序 (DeepChem 风格)
    # 先把 dict 转成 list
    scaffold_sets = list(scaffolds.values())
    
    # ⚠️ 关键修改：为了和 HiMol 一致，不使用随机 shuffle
    # 而是使用 (长度, 第一个索引) 进行排序
    # 这样可以保证：由大到小排，且相同大小的组，索引小的在前 -> 100% 确定性
    scaffold_sets.sort(key=lambda x: (len(x), x[0]), reverse=True)

    # 3. 贪婪分配
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = len(dataset) * train_ratio
    val_cutoff = len(dataset) * (train_ratio + val_ratio)

    for group in scaffold_sets:
        # HiMol 的逻辑是:
        # 如果加入后不超过训练集界限 -> 进训练集
        # 否则，如果加入后不超过验证集界限 -> 进验证集
        # 否则 -> 进测试集
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    print(f"Scaffold Split Result: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
    
    return (
        dataset[torch.tensor(train_idx)],
        dataset[torch.tensor(val_idx)],
        dataset[torch.tensor(test_idx)]
    )



# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from collections import defaultdict
# import numpy as np
# import torch

# def generate_scaffold(smiles, include_chirality=False):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None # 处理无效分子
#     scaffold = MurckoScaffold.MurckoScaffoldSmiles(
#         mol=mol, includeChirality=include_chirality
#     )
#     return scaffold

# def scaffold_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
#     assert train_ratio + val_ratio + test_ratio == 1.0
    
#     # 1. 按照骨架将分子索引分组
#     scaffolds = defaultdict(list)
#     print("Generating scaffolds...")
    
#     for idx, data in enumerate(dataset):
#         # MoleculeNet 数据集通常会在 data.smiles 中存储 SMILES 字符串
#         smiles = data.smiles
#         scaffold = generate_scaffold(smiles)
#         if scaffold is not None:
#             scaffolds[scaffold].append(idx)
#         else:
#             # 极少数情况下解析失败，放入单独的一组
#             scaffolds["INVALID"].append(idx)

#     # 2. 按照每个骨架组包含的分子数量降序排列
#     # 这样可以保证训练集包含多样性较高的骨架
#     scaffold_sets = list(scaffolds.values())
#     # 引入随机性：同等大小的骨架组顺序打乱，避免每次结果完全死板
#     np.random.seed(seed)
#     np.random.shuffle(scaffold_sets)
#     # 按大小排序 (大组在前)
#     scaffold_sets.sort(key=lambda x: len(x), reverse=True)

#     # 3. 分配索引
#     train_idx, val_idx, test_idx = [], [], []
#     train_cutoff = len(dataset) * train_ratio
#     val_cutoff = len(dataset) * (train_ratio + val_ratio)

#     for group in scaffold_sets:
#         if len(train_idx) + len(group) < train_cutoff:
#             train_idx.extend(group)
#         elif len(train_idx) + len(val_idx) + len(group) < val_cutoff:
#             val_idx.extend(group)
#         else:
#             test_idx.extend(group)

#     print(f"Scaffold Split Result: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
    
#     # 4. 返回 Subset
#     return (
#         dataset[torch.tensor(train_idx)],
#         dataset[torch.tensor(val_idx)],
#         dataset[torch.tensor(test_idx)]
#     )