# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-29
**Commit:** e4de698 (main)
**Stack:** Python 3.13, PyTorch 2.8, PyG 2.6.0, CUDA 12.9

## OVERVIEW
GNN research framework implementing Path Attention Graphormer (PAG) for graph classification benchmarks on TUDataset. Includes RUM, GRIT, GraphAE modules.

## STRUCTURE
```
GNNPlayground/
├── configs/      # YAML configuration files ONLY (organized by project)
│   ├── pag/      # PAG model configs
│   │   └── pag_base.yaml
│   └── graph_ae/ # GraphAE configs
├── pag/          # PAG model implementation
│   ├── train/    # Training scripts (tu.py)
│   ├── encoder/  # Feature, TypeDict, RRWP encoders
│   ├── layer/    # PathAttentionBlock, GRIT layers
│   └── model.py # PathAttentionGraphormer
├── tu/           # Benchmark framework
├── config/       # Python config loaders ONLY (YACS)
├── utils/        # Logger, training_utils, losses
├── grit/         # GRIT layer implementation
├── rum/          # RUM (Random Walk) modules
└── graph_ae/     # Graph AutoEncoder
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| PAG model architecture | `pag/model.py`, `pag/path_attention.py` | Core model definition |
| Training pipeline | `pag/train/tu.py` | 668 LOC, needs refactoring |
| YACS config loaders | `config/__init__.py`, `config/pag_model.py` | Python config system |
| Config files | `configs/pag/pag_base.yaml` | YAML config presets (NEW location) |
| Legacy configs | `tu/benchmark_config.py` | BenchmarkConfig class (backward compat) |
| TUDataset loading | `tu/dataset/` | Graph dataset adapters |
| Utility functions | `utils/training_utils.py`, `utils/logger.py` | New logging system |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| PathAttentionGraphormer | Class | `pag/model.py` | Main PAG model |
| PathAttentionBlock | Class | `pag/layer/block.py` | Attention layer with path-aware pooling |
| PathAttentionAttention | Class | `pag/path_attention.py` | Path attention mechanism |
| train_model | Function | `pag/train/tu.py:50` | Training loop (FIXED) |
| test_model | Function | `pag/train/tu.py:141` | Evaluation loop |
| run_k_fold4dataset | Function | `pag/train/tu.py:364` | K-fold cross-validation |
| BenchmarkConfig | Class | `tu/benchmark_config.py` | Legacy configuration management |
| get_cfg_defaults | Function | `config/pag_model.py` | Default YACS configuration |
| get_cfg_from_file | Function | `config/pag_model.py` | Load YACS config from YAML file |
| get_cfg_from_args | Function | `config/pag_model.py` | Load YACS config from command args |
| validate_config | Function | `config/pag_model.py` | Validate YACS config values |
| setup_logger | Function | `utils/logger.py` | Logging infrastructure |

## CONVENTIONS
- Environment: `pixi` package manager (pixi.toml)
- Python version: 3.13 (strict pin)
- PyTorch: 2.8 with CUDA 12.9
- PyG dependencies: Custom wheel URLs for PT 2.8/CUDA 12.9
- Logging: Use `utils.logger` (new), NOT print statements
- GPU: Auto-detect CUDA, fallback to CPU
- Training: Run via `pixi run python -m pag.train.tu`
- Configuration: Use YACS config system (preferred) or BenchmarkConfig (legacy)

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER use `print()` for training output** - Use logger instead
- **NEVER hardcode dropout values** - Use config attributes (YACS or BenchmarkConfig)
- **NEVER duplicate train/test logic** - Extract to training_utils
- **Training loop**: Uses pooler/model correctly, no NameError
- **Global variables**: Avoid `run_device` - pass as parameter
- **Unused code**: Remove `set_random_seed` function (line 551-557)
- **Config system**: Use YACS config for new code, BenchmarkConfig for legacy

## CONFIGURATION SYSTEMS

### New YACS Config System (Preferred)

The YACS configuration system provides a structured, hierarchical config management:

```python
from config import get_cfg_defaults, get_cfg_from_file, get_cfg_from_args, validate_config

# Get default configuration
cfg = get_cfg_defaults()

# Load from YAML file
cfg = get_cfg_from_file("configs/pag/pag_base.yaml")

# Load from command line args (supports --config_file and --opts overrides)
cfg = get_cfg_from_args(args)

# Validate configuration values
validate_config(cfg)

# Access values hierarchically
hidden_dim = cfg.MODEL.HIDDEN_DIM
dropout = cfg.MODEL.DROPOUT
batch_size = cfg.TRAIN.BATCH_SIZE
```

YAML config files are located in `configs/`:
- `configs/pag/pag_base.yaml` - Base configuration for most datasets
- `configs/graph_ae/` - GraphAE configuration presets

### Model Configuration Options

Key model parameters in `configs/pag/pag_base.yaml`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 128 | Model hidden dimension |
| `num_classes` | int | - | Output classes (dataset-specific) |
| `dropout` | float | 0.5 | General dropout rate |
| `me_dropout` | float | 0.5 | Message encoding dropout |
| `le_dropout` | float | 0.5 | Link encoding dropout |
| `pa_dropout` | float | 0.5 | Path attention dropout |
| `grit_num_heads` | int | 4 | Number of attention heads in GRIT layer |
| `grit_deg_scaler` | bool | true | Enable degree scaling in GRIT |
| `grit_signed_sqrt` | bool | true | Enable signed sqrt normalization in GRIT attention |

### Legacy BenchmarkConfig System

The BenchmarkConfig class provides backward compatibility for existing code:

```python
from tu.benchmark_config import BenchmarkConfig

# Create config with custom values
config = BenchmarkConfig(
    model="PathAttentionGraphormer",
    dataset="MUTAGENICITY",
    dropout=0.3,
    me_dropout=0.3,
    le_dropout=0.3,
    pa_dropout=0.4
)
```

### Migration from BenchmarkConfig to YACS

The `build_models()` function in `pag/train/tu.py` supports both systems:

```python
# Old BenchmarkConfig style (still works)
config = BenchmarkConfig(model="pag", dropout=0.3)
pooler, model = build_models(config, dataset, run_device)

# New YACS style (preferred)
from config import get_cfg_from_file
cfg = get_cfg_from_file("configs/pag/pag_base.yaml")
pooler, model = build_models(cfg, dataset, run_device)
```

## UNIQUE STYLES
- Modular model construction: `build_models()` with config-driven layer instantiation
- K-fold validation built-in: `run_k_fold4dataset()` handles cross-validation
- Rich.progress for training: Visual progress bars in training loop
- Wandb integration: Automatic logging to Weights & Biases
- Backup-first refactoring: Original files saved as `.before_refactor`

## COMMANDS
```bash
# Training
pixi run python -m pag.train.tu

# Environment
pixi shell
pixi install

# Linting
pixi run black .

# Screen sessions
screen -S gnn-training
screen -r gnn-training
```

## RECENT CHANGES (2026-03-29)
- **Directory reorganization**: YAML configs moved from `config/` to `configs/` (Python code stays in `config/`)
- **Config system refactored**: All keys now lowercase (model.hidden_dim not model.MODEL.HIDDEN_DIM)
- **pa_blocks configurable**: PathAttentionBlock params (depth, samples, rw_length, temp) now in configs/pag/pag_base.yaml
- **GRIT params configurable**: `grit_num_heads`, `grit_deg_scaler`, `grit_signed_sqrt` now in configs/pag/pag_base.yaml
- **Training stability fixed**: dropout=0.5, lr=0.0001, weight_decay=0.005, gradient_clip=0.5, early_stop=50
- **Logging improved**: RichHandler for console, logs saved to logs/training_<timestamp>.log
- **AttributeError fixed**: Format string error after training completion resolved

## NOTES
- **VRAM heavy**: Training consumes >20GB VRAM on large datasets
- **Training now stable**: Overfitting issue resolved with regularization adjustments
- **Screen session**: Use `screen -S gnn-training` to run training in background
- **Test coverage**: Minimal (rum/tests/ has some tests)
- **Skills installed**: 10 Claude Code skills for code quality, testing, PyTorch, logging
