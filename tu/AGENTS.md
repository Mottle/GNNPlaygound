# TU (TUDataset Benchmark) Module

**Purpose**: Benchmark framework for TUDataset graph classification with k-fold cross-validation

## OVERVIEW
Training pipeline and benchmark utilities for graph classification tasks.

## STRUCTURE
```
tu/
├── train.py            # Main training script (23053 LOC, OLD)
├── tu.py               # Legacy training entry
├── benchmark_config.py   # BenchmarkConfig class (53 LOC)
├── benchmark_result.py   # BenchmarkResult class
├── classifier.py        # Classifier wrapper
└── dataset/            # TUDataset loaders
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Config management | `benchmark_config.py` | Hyperparameters, datasets |
| Result tracking | `benchmark_result.py` | K-fold results aggregation |
| Dataset loading | `dataset/` | TUDataset adapters |
| Training logic | `train.py` | 23053 LOC (legacy) |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| BenchmarkConfig | Class | `benchmark_config.py` | Configuration class |
| BenchmarkResult | Class | `benchmark_result.py` | Result storage |
| Classifier | Class | `classifier.py` | Model wrapper |

## CONVENTIONS
- Datasets: Use predefined dataset names in `BenchmarkConfig`
- Results: Aggregate across k-folds, report mean/std
- Config: YACS-based configuration system

## ANTI-PATTERNS
- **Avoid `train.py`** - Use `pag/train/tu.py` for PAG model
- **Don't modify datasets** - Use immutable TUDataset objects
