# PAG (Path Attention Graphormer) Module

**Purpose**: Core implementation of Path Attention Graphormer model for graph classification

## OVERVIEW
PAG model with path-aware attention mechanism, multiple encoder types, and GRIT integration.

## STRUCTURE
```
pag/
├── model.py              # PathAttentionGraphormer (110 LOC)
├── path_attention.py      # PathAttention, LocalPathAttention (190 LOC)
├── train.py              # Training entry point
├── connection.py         # HyperConnection module
├── fusion.py             # AFF, IAFF, GAFF fusion modules
├── layer/               # Attention and GRIT layers
│   ├── block.py         # PathAttentionBlock (94 LOC)
│   └── grit_layer.py    # MultiHeadAttentionLayerGritSparse, GritTransformerLayer
├── encoder/             # Feature encoders
│   ├── feature_encoder.py
│   ├── type_dict_encoder.py
│   └── rrwp_encoder.py
├── train/               # Training scripts
│   └── tu.py           # TUDataset training (668 LOC, REFACTORING NEEDED)
└── dataset/             # Dataset adapters
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model definition | `model.py` | Main model class |
| Path attention | `path_attention.py` | Core attention mechanism |
| Training loop | `train/tu.py` | **668 LOC, needs refactoring** |
| Layer implementations | `layer/` | PathAttentionBlock, GRIT |
| Encoders | `encoder/` | Feature, TypeDict, RRWP |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| PathAttentionGraphormer | Class | `model.py:35` | Main model class |
| PathAttention | Class | `path_attention.py:8` | Global path attention |
| LocalPathAttention | Class | `path_attention.py:101` | Local path attention |
| PathAttentionBlock | Class | `layer/block.py:12` | Block with path pooling |
| MultiHeadAttentionLayerGritSparse | Class | `layer/grit_layer.py:42` | GRIT attention layer |
| GritTransformerLayer | Class | `layer/grit_layer.py:165` | GRIT transformer |
| TypeDictNodeEncoder | Class | `encoder/type_dict_encoder.py:85` | Node feature encoding |
| FeatureEncoder | Class | `encoder/feature_encoder.py:14` | Base feature encoder |
| RRWPLinearNodeEncoder | Class | `encoder/rrwp_encoder.py:56` | RRWP encoding |

## CONVENTIONS
- Model config: Use `BenchmarkConfig` for hyperparameters
- Encoders: Modular design, swapable via config
- Attention: Path-aware pooling with configurable k-hop

## ANTI-PATTERNS
- **NEVER hardcode dropout** in `build_models()` - use `getattr(config, 'pa_dropout', 0.4)`
- **Avoid global `run_device`** - pass device as parameter
- **Train script bugs**: Line 122 in `train/tu.py` uses undefined `model`

## KNOWN ISSUES
- `train/tu.py` has 668 LOC with severe code duplication
- Training loop duplicates logic (lines 75-77 vs 157-159)
- Variable naming bug: `pooler` vs `model` inconsistency
- Dead code: `set_random_seed()` unused (lines 551-557)

## REFACTORING STATUS
- **utils/logger.py**: ✅ Created (84 LOC)
- **utils/training_utils.py**: ✅ Created (172 LOC)
- **train/tu.py**: ⚠️ Needs rewrite (currently backed up as `train/tu.py.before_refactor`)
