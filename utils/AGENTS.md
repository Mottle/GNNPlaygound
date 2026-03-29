# Utils Module

**Purpose**: Shared utilities for logging, training helpers, and loss functions

## OVERVIEW
Common functions used across training, logging, and model evaluation.

## STRUCTURE
```
utils/
├── logger.py               # Logging system (84 LOC, NEW)
├── training_utils.py       # Training helpers (172 LOC, NEW)
├── loss/                  # Loss functions
│   ├── loss.py            # Base losses
│   └── multitask_loss.py  # Multi-task loss
├── perf_counter.py        # Performance timing
├── seed_manual.py         # Seed setting
└── scaffold.py           # Code scaffolding
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Logging setup | `logger.py` | File + console handlers |
| Training helpers | `training_utils.py` | Data prep, forward, logging |
| Loss functions | `loss/` | Multi-task support |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| setup_logger | Function | `logger.py` | Create logger with file/console |
| get_logger | Function | `logger.py` | Retrieve existing logger |
| prepare_data | Function | `training_utils.py` | Data preprocessing |
| forward_model | Function | `training_utils.py` | Forward propagation |
| update_progress | Function | `training_utils.py` | Statistics update |
| log_epoch_progress | Function | `training_utils.py` | Epoch logging |
| log_fold_results | Function | `training_utils.py` | Fold results logging |

## CONVENTIONS
- **Logging**: ALWAYS use `get_logger(__name__)`, never print()
- **File logs**: Saved to `utils/logs/` directory
- **Timestamp**: Logs include timestamp and function name

## LOGGING USAGE
```python
from utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.debug(f"Shape: {tensor.shape}")
logger.warning("High memory usage")
logger.error("Training failed", exc_info=True)
```

## TRAINING UTILS USAGE
```python
from utils.training_utils import (
    prepare_data, forward_model, update_progress,
    log_epoch_progress, log_fold_results
)

# Prepare data
data = prepare_data(data, device)

# Forward pass
loss, pred = forward_model(pooler, classifier, data, criterion, use_amp)

# Update stats
stats = update_progress(stats, loss.item(), y, pred)

# Log progress
log_epoch_progress(logger, epoch, num_epochs, stats)
```

## ANTI-PATTERNS
- **NEVER use print()** - Use logger instead
- **NEVER duplicate logic** - Extract to training_utils before creating new code
