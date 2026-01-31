import random
import os
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    设置全局随机种子以保证结果可复现
    """
    # 1. Python built-in random
    random.seed(seed)
    
    # 2. Environment variable (Hash seed)
    # 禁止哈希随机化，保证实验的可复现性（例如字典的迭代顺序）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    
    # 5. Deterministic algorithms (强制使用确定性算法)
    # 注意：这可能会降低训练速度，但能保证严格的一致性
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
