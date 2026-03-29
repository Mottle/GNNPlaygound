# PAG Training YACS Configuration

## Overview
将 PAG 模型的所有超参数配置迁移到 YACS 配置系统，实现配置文件的集中管理。

## 设计原则
1. **配置分层**: 模型配置、训练配置、数据集配置分离
2. **类型安全**: 使用 YACS 的类型系统确保配置正确性
3. **默认值**: 提供合理的默认值，允许快速启动
4. **可扩展**: 便于添加新的模型或配置项
5. **向后兼容**: 保持现有训练脚本的兼容性

## 配置结构

```python
# config/pag_model.py
from yacs.config import CfgNode as CN

@build_config_parser
def get_cfg_defaults():
    parser = CN()
    # 模型配置
    parser.MODEL = CN()
    parser.MODEL.NAME = 'PathAttentionGraphormer'
    parser.MODEL.LAYERS = 4
    parser.MODEL.HIDDEN_DIM = 128
    parser.MODEL.DROPOUT = 0.3
    
    # Path Attention 配置
    parser.MODEL.PA_LAYERS = 1
    parser.MODEL.PA_DROPOUT = 0.4
    parser.MODEL.NUM_ME_LAYERS = 1
    parser.MODEL.NUM_LE_DEPTH = 3
    parser.MODEL.NUM_LE_SAMPLES = 4
    parser.MODEL.LE_RW_LENGTH = 2
    
    # 编码器配置
    parser.MODEL.ENCODER = 'type_dict'
    parser.MODEL.RRWP_MAX_LENGTH = 20
    
    # 训练配置
    parser.TRAIN = CN()
    parser.TRAIN.BATCH_SIZE = 256
    parser.TRAIN.LR = 0.0005
    parser.TRAIN.WEIGHT_DECAY = 0.0001
    parser.TRAIN.WEARMUP_STEPS = 0
    parser.TRAIN.WEARMUP_RATIO = 0.1
    parser.TRAIN.EARLY_STOP = 30
    
    # 数据增强配置
    parser.TRAIN.AUGMENTATION = False
    
    # AMP 混合精度训练
    parser.TRAIN.USE_AMP = True
    
    # K-Fold 交叉验证
    parser.TRAIN.NUM_FOLDS = 10
    parser.TRAIN.SHUFFLE = True
    
    return parser
```

## 实现步骤

### Step 1: 创建配置模块
```bash
# 创建配置目录
mkdir -p config

# 创建基础配置文件
touch config/pag_model.py
touch config/training_config.py
touch config/dataset_config.py
```

### Step 2: 实现配置加载器
```python
# config/pag_model.py
from yacs.config import CfgNode as CN
import argparse

@build_config_parser
def get_cfg_defaults():
    parser = CN()
    # ... 配置定义 ...
    return parser

def get_cfg_from_file(config_file):
    """
    从 YAML 文件加载配置
    
    Args:
        config_file: YAML 配置文件路径
        
    Returns:
        配置对象
    """
    parser = get_cfg_defaults()
    parser.merge_from_file(config_file)
    return parser

def get_cfg_from_args(args):
    """
    从命令行参数加载配置
    
    Args:
        args: 命令行参数
        
    Returns:
        配置对象
    """
    parser = get_cfg_defaults()
    parser.merge_from_file(args.config_file)
    parser.merge_from_list(args.opts)
    return parser
```

### Step 3: 修改模型构建函数
```python
# pag/model.py
from config.pag_model import get_cfg_from_args

def build_model(config):
    """
    从配置构建 PAG 模型
    
    Args:
        config: YACS 配置对象
        
    Returns:
        模型和分类器
    """
    # 使用配置参数而非硬编码
    model = PathAttentionGraphormer(
        in_channels=config.MODEL.IN_CHANNELS,
        hidden_dim=config.MODEL.HIDDEN_DIM,
        num_layers=config.MODEL.LAYERS,
        pa_layers=[
            PathAttentionBlock(
                channels=config.MODEL.HIDDEN_DIM,
                num_me_layers=config.MODEL.NUM_ME_LAYERS,
                num_le_depth=config.MODEL.NUM_LE_DEPTH,
                num_le_samples=config.MODEL.NUM_LE_SAMPLES,
                le_rw_length=config.MODEL.LE_RW_LENGTH,
                me_dropout=getattr(config, 'me_dropout', config.MODEL.PA_DROPOUT),
                le_dropout=getattr(config, 'le_dropout', config.MODEL.PA_DROPOUT),
                pa_dropout=getattr(config, 'pa_dropout', config.MODEL.PA_DROPOUT),
                pa_temp=getattr(config, 'pa_temp', 1.0),
            ) for _ in range(config.MODEL.PA_LAYERS)
        ],
        encoder_type=config.MODEL.ENCODER,
        use_node_attr=getattr(config, 'use_node_attr', False),
        edge_in_channels=getattr(config, 'edge_in_channels', None),
        pre_transform=config.MODEL.RRWP_MAX_LENGTH,
    )
    
    classifier = Classifier(
        config.MODEL.HIDDEN_DIM, 
        config.MODEL.HIDDEN_DIM, 
        num_classes
    )
    
    return model, classifier
```

### Step 4: 修改训练脚本
```python
# pag/train/tu.py
from config.pag_model import get_cfg_from_args, get_cfg_defaults

def main(args):
    # 从配置加载参数
    config = get_cfg_from_args(args)
    
    # 记录配置
    logger = setup_logger("pag_training")
    logger.info(f"配置文件: {args.config_file}")
    logger.info(f"模型: {config.MODEL.NAME}")
    logger.info(f"隐藏层维度: {config.MODEL.HIDDEN_DIM}")
    logger.info(f"Dropout: {config.MODEL.DROPOUT}")
    logger.info(f"PA Dropout: {config.MODEL.PA_DROPOUT}")
    
    # 构建模型
    model, classifier = build_model(config)
    
    # 训练循环
    for epoch in range(config.TRAIN.MAX_EPOCHS):
        train_loss, train_acc = train_model(
            model, classifier, train_loader, optimizer, 
            scheduler, criterion, device, 
            use_amp=config.TRAIN.USE_AMP
        )
        # ...
```

## 配置文件示例

### 基础配置 (config/pag_base.yaml)
```yaml
MODEL:
  NAME: PathAttentionGraphormer
  HIDDEN_DIM: 128
  LAYERS: 4
  DROPOUT: 0.3

TRAIN:
  BATCH_SIZE: 256
  LR: 0.0005
  WEIGHT_DECAY: 0.0001
  MAX_EPOCHS: 200
  EARLY_STOP: 30
  USE_AMP: True
  NUM_FOLDS: 10
```

### 数据集特定配置 (config/tu_mutagenicity.yaml)
```yaml
MODEL:
  NAME: PathAttentionGraphormer
  HIDDEN_DIM: 128
  LAYERS: 4
  DROPOUT: 0.3
  IN_CHANNELS: 10  # MUTAGENICITY 数据集特征维度

TRAIN:
  BATCH_SIZE: 256
  LR: 0.0005
  MAX_EPOCHS: 200
  NUM_FOLDS: 10

DATASET:
  NAME: MUTAGENICITY
  NUM_CLASSES: 2
```

### 高性能配置 (config/pag_large.yaml)
```yaml
MODEL:
  NAME: PathAttentionGraphormer
  HIDDEN_DIM: 256
  LAYERS: 6
  DROPOUT: 0.4

TRAIN:
  BATCH_SIZE: 512
  LR: 0.0003
  WEIGHT_DECAY: 0.0001
  MAX_EPOCHS: 500
  EARLY_STOP: 50
```

## 迁移指南

### 从 BenchmarkConfig 迁移
```python
# 创建适配器
# config/legacy_adapter.py
from tu.benchmark_config import BenchmarkConfig
from yacs.config import CfgNode as CN

def adapt_benchmark_config(benchmark_config):
    """
    将 BenchmarkConfig 转换为 YACS 配置
    
    Args:
        benchmark_config: BenchmarkConfig 实例
        
    Returns:
        YACS 配置对象
    """
    yacs_cfg = get_cfg_defaults()
    
    # 复制属性
    yacs_cfg.MODEL.NAME = benchmark_config.model
    yacs_cfg.MODEL.HIDDEN_DIM = benchmark_config.hidden_channels
    yacs_cfg.MODEL.LAYERS = benchmark_config.num_layers
    yacs_cfg.MODEL.DROPOUT = benchmark_config.dropout
    yacs_cfg.MODEL.PA_DROPOUT = getattr(benchmark_config, 'pa_dropout', 0.4)
    yacs_cfg.TRAIN.BATCH_SIZE = benchmark_config.batch_size
    yacs_cfg.TRAIN.LR = benchmark_config.lr
    yacs_cfg.TRAIN.NUM_FOLDS = getattr(benchmark_config, 'k_fold', 10)
    
    return yacs_cfg
```

## 使用示例

### 命令行使用
```bash
# 使用默认配置
python -m pag.train.tu

# 使用指定配置文件
python -m pag.train.tu --config-file config/pag_base.yaml

# 覆盖特定参数
python -m pag.train.tu --config-file config/pag_base.yaml \
    MODEL.LAYERS 6 \
    MODEL.HIDDEN_DIM 256

# 从命令行覆盖所有参数
python -m pag.train.tu \
    MODEL.HIDDEN_DIM 256 \
    MODEL.LAYERS 6 \
    TRAIN.BATCH_SIZE 512 \
    TRAIN.LR 0.0003
```

### Python 脚本中使用
```python
# 加载配置
from config.pag_model import get_cfg_from_file

# 方式1: 从文件加载
config = get_cfg_from_file("config/pag_base.yaml")

# 方式2: 使用默认配置
from config.pag_model import get_cfg_defaults
config = get_cfg_defaults()

# 方式3: 编程式配置
config = get_cfg_defaults()
config.MODEL.HIDDEN_DIM = 256
config.TRAIN.BATCH_SIZE = 512

# 构建模型
model, classifier = build_model(config)
```

## 优势

1. **集中管理**: 所有超参数在一个文件中
2. **版本控制**: 配置文件可以版本控制
3. **实验追踪**: 不同配置对应不同实验
4. **团队协作**: 配置文件可以共享
5. **类型安全**: YACS 验证配置类型
6. **灵活覆盖**: 命令行、文件、编程式三种方式
7. **文档化**: YAML 格式便于阅读
8. **调试友好**: 配置问题易于定位

## 注意事项

1. **保持兼容**: 保留 BenchmarkConfig 支持
2. **渐进迁移后**: 先迁移新功能，保留旧接口
3. **验证配置**: 添加配置验证逻辑
4. **文档完善**: 为每个配置项添加注释
5. **测试充分**: 测试配置加载和模型构建
6. **性能考虑**: 配置加载不应成为瓶颈

## 下一步

1. [ ] 创建 config 目录和配置文件
2. [ ] 实现 YACS 配置模块
3. [ ] 修改 pag/model.py 使用配置
4. [ ] 修改 pag/train/tu.py 使用配置
5. [ ] 创建配置文件示例
6. [ ] 添加配置验证
7. [ ] 编写单元测试
8. [ ] 更新文档
