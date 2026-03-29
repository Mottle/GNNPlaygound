# PAG Training Module

**Purpose**: Training scripts for PAG model on TUDataset benchmarks

## OVERVIEW
Training pipeline with k-fold cross-validation for PAG model evaluation.

## STRUCTURE
```
pag/train/
└── tu.py    # Main training script (668 LOC, REFACTORING NEEDED)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Training loop | `tu.py:50-140` | `train_model()` function |
| Evaluation | `tu.py:141-182` | `test_model()` function |
| K-fold runner | `tu.py:364-414` | `run_k_fold4dataset()` |
| Model building | `tu.py:287-357` | `build_models()` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| train_model | Function | `tu.py:50` | Single epoch training (FIXED) |
| test_model | Function | `tu.py:141` | Evaluation loop |
| run_fold | Function | `tu.py:184` | Run single fold |
| build_models | Function | `tu.py:287` | Build PAG model |
| build_optimizer | Function | `tu.py:358` | Build optimizer/scheduler |
| run_k_fold4dataset | Function | `tu.py:364` | K-fold cross-validation |

## CURRENT STATE
- Training loop is functional and stable
- Configuration uses lowercase keys (YACS system)
- pa_blocks configurable via config file

## REMAINING ISSUES
- Some code duplication remains (could be refactored)
- Lines 75-77 vs 157-159: Duplicate data preprocessing
- Lines 85-93 vs 166-174: Duplicate forward propagation
- Lines 133-136 vs 176-179: Duplicate statistics update
- Lines 71 vs 150: Duplicate device string extraction

## ANTI-PATTERNS
- **NEVER use `print()`** - Use `utils.logger` (new)
- **NEVER duplicate logic**` - Extract to `utils.training_utils`
- **NEVER hardcode dropout** - Use `getattr(config, 'pa_dropout', 0.4)`
- **NEVER use global `run_device`** - Pass as parameter

## REFACTORING PLAN
1. ✅ Create `utils/logger.py` (84 LOC, DONE)
2. ✅ Create `utils/training_utils.py` (172 LOC, DONE)
3. ✅ Fix model/pooler variable bug (DONE)
4. ✅ Replace print() with logger (DONE)
5. ⚠️ Remove remaining code duplication (low priority)
6. ⚠️ Full tu.py rewrite (low priority - works as-is)

## NEW UTILITIES AVAILABLE
```python
# From utils/training_utils.py
prepare_data(data, device)
forward_model(pooler, classifier, data, criterion, use_amp)
update_progress(stats, loss, y, pred)
get_device_type(device)
log_epoch_progress(logger, epoch, num_epochs, stats)
log_fold_results(logger, fold, results)
log_dataset_results(logger, dataset_name, results)

# From utils/logger.py
setup_logger(name, log_file, level)
get_logger(name)
```

## BACKUP
Original file: `pag/train/tu.py.before_refactor`
