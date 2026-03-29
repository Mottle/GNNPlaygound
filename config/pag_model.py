import argparse
from yacs.config import CfgNode as CN


def get_cfg_defaults() -> CN:
    parser = CN()

    # Model configuration
    parser.model = CN()
    parser.model.name = "PathAttentionGraphormer"
    parser.model.layers = 4
    parser.model.hidden_dim = 128
    parser.model.dropout = 0.5
    parser.model.num_me_layers = 1
    parser.model.me_dropout = 0.5
    parser.model.le_dropout = 0.5
    parser.model.encoder = "type_dict"
    parser.model.rrwp_max_length = 20
    parser.model.use_node_attr = False
    parser.model.edge_in_channels = None

    # GRIT (macro_encoder) configuration
    parser.model.grit_num_heads = 4
    parser.model.grit_deg_scaler = True
    parser.model.grit_signed_sqrt = True

    # Path Attention Block configurations
    # Each block: (num_le_depth, num_le_samples, le_rw_length, pa_temp)
    parser.model.pa_blocks = [
        {"depth": 3, "samples": 4, "rw_length": 2, "temp": 1.0},
        {"depth": 2, "samples": 4, "rw_length": 4, "temp": 0.8},
        {"depth": 2, "samples": 2, "rw_length": 8, "temp": 0.6},
        {"depth": 2, "samples": 2, "rw_length": 16, "temp": 0.4},
    ]
    parser.model.pa_dropout = 0.4

    # Training configuration
    parser.train = CN()
    parser.train.batch_size = 64
    parser.train.weight_decay = 0.005
    parser.train.warmup_steps = 0
    parser.train.warmup_ratio = 0.1
    parser.train.early_stop = 50
    parser.train.max_epochs = 200
    parser.train.use_amp = True
    parser.train.num_folds = 10
    parser.train.shuffle = True
    parser.train.gradient_clip = 0.5
    parser.train.label_smoothing = 0.1

    # Grouped learning rates for different model components
    parser.train.lr_macro = 0.0001
    parser.train.lr_pa = 0.0001
    parser.train.lr_other = 0.0001

    # Dataset configuration
    parser.dataset = CN()
    parser.dataset.name = "nci1"
    parser.dataset.sets = "common"
    parser.dataset.catch_error = True

    parser.seed = None

    return parser


def get_cfg_from_file(config_file: str) -> CN:
    parser = get_cfg_defaults()
    parser.merge_from_file(config_file)
    return parser


def get_cfg_from_args(args: argparse.Namespace) -> CN:
    parser = get_cfg_defaults()
    if hasattr(args, "config_file") and args.config_file:
        parser.merge_from_file(args.config_file)
    if hasattr(args, "opts") and args.opts:
        parser.merge_from_list(args.opts)
    return parser


def validate_config(cfg: CN) -> None:
    """
    Validate configuration values to ensure they are within acceptable ranges.

    Raises:
        ValueError: If any configuration value is invalid
        TypeError: If any configuration value has wrong type

    Args:
        cfg: Configuration node to validate
    """
    # Valid model names
    valid_model_names = ["PathAttentionGraphormer", "pag"]

    # Validate model.name - enum validation
    if not isinstance(cfg.model.name, str):
        raise TypeError(
            f"model.name must be a string, got {type(cfg.model.name).__name__}"
        )
    if cfg.model.name not in valid_model_names:
        raise ValueError(
            f"model.name must be one of {valid_model_names}, got '{cfg.model.name}'"
        )

    # Validate model.layers - positive integer
    if not isinstance(cfg.model.layers, int):
        raise TypeError(
            f"model.layers must be an integer, got {type(cfg.model.layers).__name__}"
        )
    if cfg.model.layers <= 0:
        raise ValueError(
            f"model.layers must be a positive integer, got {cfg.model.layers}"
        )

    # Validate model.hidden_dim - positive integer
    if not isinstance(cfg.model.hidden_dim, int):
        raise TypeError(
            f"model.hidden_dim must be an integer, got {type(cfg.model.hidden_dim).__name__}"
        )
    if cfg.model.hidden_dim <= 0:
        raise ValueError(
            f"model.hidden_dim must be a positive integer, got {cfg.model.hidden_dim}"
        )

    # Validate dropout values - range validation [0.0, 1.0]
    dropout_params = {
        "model.dropout": cfg.model.dropout,
        "model.pa_dropout": cfg.model.pa_dropout,
    }

    for param_name, param_value in dropout_params.items():
        if not isinstance(param_value, (int, float)):
            raise TypeError(
                f"{param_name} must be a number, got {type(param_value).__name__}"
            )
        if param_value < 0.0 or param_value > 1.0:
            raise ValueError(
                f"{param_name} must be between 0.0 and 1.0, got {param_value}"
            )

    # Validate train.batch_size - positive integer
    if not isinstance(cfg.train.batch_size, int):
        raise TypeError(
            f"train.batch_size must be an integer, got {type(cfg.train.batch_size).__name__}"
        )
    if cfg.train.batch_size <= 0:
        raise ValueError(
            f"train.batch_size must be a positive integer, got {cfg.train.batch_size}"
        )

    # Validate train.num_folds - positive integer
    if not isinstance(cfg.train.num_folds, int):
        raise TypeError(
            f"train.num_folds must be an integer, got {type(cfg.train.num_folds).__name__}"
        )
    if cfg.train.num_folds <= 0:
        raise ValueError(
            f"train.num_folds must be a positive integer, got {cfg.train.num_folds}"
        )

    # Validate model.pa_blocks - must be a list of dicts with required keys
    if not isinstance(cfg.model.pa_blocks, list):
        raise TypeError(
            f"model.pa_blocks must be a list, got {type(cfg.model.pa_blocks).__name__}"
        )
    required_block_keys = {"depth", "samples", "rw_length", "temp"}
    for i, block in enumerate(cfg.model.pa_blocks):
        if not isinstance(block, dict):
            raise TypeError(
                f"model.pa_blocks[{i}] must be a dict, got {type(block).__name__}"
            )
        for key in required_block_keys:
            if key not in block:
                raise ValueError(f"model.pa_blocks[{i}] missing required key '{key}'")

    # Validate GRIT parameters
    if not isinstance(cfg.model.grit_num_heads, int):
        raise TypeError(
            f"model.grit_num_heads must be an integer, got {type(cfg.model.grit_num_heads).__name__}"
        )
    if cfg.model.grit_num_heads <= 0:
        raise ValueError(
            f"model.grit_num_heads must be a positive integer, got {cfg.model.grit_num_heads}"
        )
    if cfg.model.hidden_dim % cfg.model.grit_num_heads != 0:
        raise ValueError(
            f"model.hidden_dim ({cfg.model.hidden_dim}) must be divisible by "
            f"model.grit_num_heads ({cfg.model.grit_num_heads})"
        )

    if not isinstance(cfg.model.grit_deg_scaler, bool):
        raise TypeError(
            f"model.grit_deg_scaler must be a boolean, got {type(cfg.model.grit_deg_scaler).__name__}"
        )

    if not isinstance(cfg.model.grit_signed_sqrt, bool):
        raise TypeError(
            f"model.grit_signed_sqrt must be a boolean, got {type(cfg.model.grit_signed_sqrt).__name__}"
        )
