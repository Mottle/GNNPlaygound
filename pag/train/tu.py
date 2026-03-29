import torch
import numpy as np
from torch import nn

from torch_geometric.loader import DataLoader
from rich.progress import track
from utils.perf_counter import get_time_sync
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from tu.benchmark_config import BenchmarkConfig
from tu.benchmark_result import BenchmarkResult
from torch.optim import Adam
from tu.classifier import Classifier
from torch_geometric.datasets import TUDataset
from pag.model import PathAttentionGraphormer
from torch.optim.lr_scheduler import LambdaLR
from utils.training_utils import (
    prepare_data,
    forward_model,
    update_progress,
    get_device_type,
    run_epoch,
)
from utils.logger import setup_logger, get_logger

try:
    from config import get_cfg_defaults

    YACS_AVAILABLE = True
except ImportError:
    YACS_AVAILABLE = False

logger = setup_logger("training", log_dir="logs", log_to_file=True)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create LR scheduler with linear warmup and decay."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(loss1, loss2):
    return loss1 + loss2 / (loss1 + loss2 + 1e-6).detach()


def run_fold(dataset, loader, current_fold: int, config, run_device):
    is_yacs = hasattr(config, "model")
    log_prefix = f"fold: {current_fold+1}"
    logger.info(f"{log_prefix} {dataset} 开始训练...")
    train_loader, val_loader, test_loader = loader
    stamp_start = get_time_sync()

    if is_yacs:
        epochs = config.train.max_epochs
        early_stop = config.train.early_stop
        early_stop_epochs = config.train.early_stop
        use_amp = config.train.use_amp
        gradient_clip = config.train.gradient_clip
        label_smoothing = config.train.label_smoothing
    else:
        epochs = config.epochs
        early_stop = config.early_stop
        early_stop_epochs = config.early_stop_epochs
        use_amp = config.amp
        gradient_clip = 1.0
        label_smoothing = getattr(config, "label_smoothing", 0.1)

    model, classifier = build_models(
        dataset.num_node_features, dataset.num_classes, config, run_device
    )
    optimizer = build_optimizer(model, classifier, config)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=epochs / 10 * len(train_loader),
        num_training_steps=epochs * len(train_loader),
    )

    metrics = {
        "train_loss": [],
        "train_acc": BenchmarkResult(),
        "val_loss": [],
        "val_acc": BenchmarkResult(),
        "test_loss": [],
        "test_acc": BenchmarkResult(),
    }
    no_increased_times = 0
    no_record_epoch_num = 0

    model_name = config.model.name if is_yacs else config.model
    for epoch in track(range(epochs), description=f"{log_prefix} 训练 {dataset}"):
        train_loss, train_acc = run_epoch(
            model,
            classifier,
            train_loader,
            criterion,
            run_device,
            model_name,
            optimizer,
            scheduler,
            use_amp,
            "train",
            "Train epoch",
            gradient_clip,
        )
        val_loss, val_acc = run_epoch(
            model,
            classifier,
            val_loader,
            criterion,
            run_device,
            model_name,
            use_amp=use_amp,
            mode="eval",
            progress_desc="Val epoch",
            gradient_clip=gradient_clip,
        )
        test_loss, test_acc = run_epoch(
            model,
            classifier,
            test_loader,
            criterion,
            run_device,
            model_name,
            use_amp=use_amp,
            mode="eval",
            progress_desc="Test epoch",
            gradient_clip=gradient_clip,
        )

        if epoch > no_record_epoch_num:
            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(train_acc)
            metrics["val_loss"].append(val_loss)
            metrics["val_acc"].append(val_acc)
            metrics["test_loss"].append(test_loss)
            metrics["test_acc"].append(test_acc)

        if early_stop and epoch > no_record_epoch_num:
            if val_acc < metrics["val_acc"].get_max():
                no_increased_times += 1
            else:
                no_increased_times = 0
            if no_increased_times >= early_stop_epochs:
                logger.info(f"Early stop at epoch {epoch+1}")
                break

        logger.info(
            f"{log_prefix}, Epoch {epoch+1:03d} "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, lr: [macro:{optimizer.param_groups[0]["lr"]:.6f}, pa:{optimizer.param_groups[1]["lr"]:.6f}, other:{optimizer.param_groups[2]["lr"]:.6f}]'
        )

    stamp_end = get_time_sync()
    test_acc_res = metrics["test_acc"]
    logger.info(
        f"{log_prefix} {test_acc_res.format()}\n"
        + f"{log_prefix} {test_acc_res.format(last=1)}\n"
        + f"{log_prefix} {test_acc_res.format(last=10)}\n"
        + f"{log_prefix} {test_acc_res.format(last=50)}\n"
        + f"{log_prefix} 总运行时间: {(stamp_end - stamp_start) / 60:.4f}min"
    )
    return metrics["train_acc"], test_acc_res


def build_models(num_node_features, num_classes, config, run_device):
    """Build PAG model and classifier from config.

    Args:
        num_node_features: Input node features count
        num_classes: Output classes count
        config: BenchmarkConfig or YACS CfgNode
        run_device: Device for models

    Returns: Tuple of (model, classifier)
    """
    input_dim = max(num_node_features, 1)
    is_yacs = hasattr(config, "model")
    cfg = config.model if is_yacs else config

    params = [
        ("hidden_dim", "hidden_channels", None),
        ("layers", "num_layers", None),
        ("name", "model", None),
        ("dropout", "dropout", None),
        ("me_dropout", "me_dropout", 0.3),
        ("le_dropout", "le_dropout", 0.3),
        ("pa_dropout", "pa_dropout", 0.4 if not is_yacs else None),
    ]

    hidden_dim, num_layers, model_type, dropout, me_dropout, le_dropout, pa_dropout = [
        getattr(cfg, yacs_attr if is_yacs else legacy_attr, default)
        for yacs_attr, legacy_attr, default in params
    ]

    model = None
    if model_type in ("pag", "PathAttentionGraphormer"):
        from pag.layer.block import PathAttentionBlock

        pa_configs = (
            cfg.pa_blocks
            if is_yacs
            else [(3, 4, 2, 1.0), (2, 4, 4, 0.8), (2, 2, 8, 0.6), (2, 2, 16, 0.4)]
        )
        if is_yacs:
            pa_layers = [
                PathAttentionBlock(
                    channels=hidden_dim,
                    num_me_layers=cfg.num_me_layers,
                    num_le_depth=b["depth"],
                    num_le_samples=b["samples"],
                    le_rw_length=b["rw_length"],
                    me_dropout=me_dropout,
                    le_dropout=le_dropout,
                    pa_dropout=pa_dropout,
                    pa_temp=b["temp"],
                    grit_num_heads=cfg.grit_num_heads,
                    grit_deg_scaler=cfg.grit_deg_scaler,
                    grit_signed_sqrt=cfg.grit_signed_sqrt,
                    grit_bn_momentum=cfg.grit_bn_momentum,
                )
                for b in pa_configs
            ]
        else:
            pa_layers = [
                PathAttentionBlock(
                    channels=hidden_dim,
                    num_me_layers=1,
                    num_le_depth=d,
                    num_le_samples=s,
                    le_rw_length=r,
                    me_dropout=me_dropout,
                    le_dropout=le_dropout,
                    pa_dropout=pa_dropout,
                    pa_temp=t,
                )
                for d, s, r, t in pa_configs
            ]
        model = PathAttentionGraphormer(
            node_in_channels=input_dim,
            edge_in_channels=None,
            channels=hidden_dim,
            pa_layers=pa_layers,
        ).to(run_device)

    classifier = Classifier(hidden_dim, hidden_dim, num_classes).to(run_device)
    return model, classifier


def build_optimizer(model, classifier, config):
    is_yacs = hasattr(config, "model")
    if is_yacs:
        weight_decay = config.train.weight_decay
        macro_encoder_prefixes = [
            "feature_encoder",
            "node_rrwp_encoder",
            "edge_rrwp_encoder",
        ]
        pa_prefixes = ["pa_blocks"]
        other_prefixes = ["node_hc", "feature_hc", "node_layer_norms", "feature_norms"]

        macro_params = []
        pa_params = []
        other_params = []

        for name, param in model.named_parameters():
            if any(prefix in name for prefix in macro_encoder_prefixes):
                macro_params.append(param)
            elif any(prefix in name for prefix in pa_prefixes):
                pa_params.append(param)
            elif any(prefix in name for prefix in other_prefixes):
                other_params.append(param)
            else:
                other_params.append(param)

        for name, param in classifier.named_parameters():
            other_params.append(param)

        param_groups = [
            {
                "params": macro_params,
                "lr": config.train.lr_macro,
                "weight_decay": weight_decay,
            },
            {
                "params": pa_params,
                "lr": config.train.lr_pa,
                "weight_decay": weight_decay,
            },
            {
                "params": other_params,
                "lr": config.train.lr_other,
                "weight_decay": weight_decay,
            },
        ]
        return Adam(param_groups)
    else:
        lr = config.lr
        weight_decay = getattr(config, "weight_decay", 0.0)
        return Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )


def run_k_fold4dataset(dataset, config, run_device):
    is_yacs = hasattr(config, "model")
    if is_yacs:
        kfold = config.train.num_folds
        seed = config.seed if config.seed is not None else 0
        model_name = config.model.name
        batch_size = config.train.batch_size
        catch_error = config.dataset.catch_error
    else:
        kfold = config.kfold
        seed = config.seed
        model_name = config.model
        batch_size = config.batch_size
        catch_error = config.catch_error

    kfold_dataset = kfold_split(dataset, kfold, seed)
    results = []
    all_start = get_time_sync()

    for fold, (train_idx, test_idx) in track(
        enumerate(kfold_dataset), total=kfold, description="k-fold"
    ):
        logger.info(f"Run model: {model_name}")
        train_idx, val_idx = split_train_val(train_idx, kfold)
        train_loader, val_loader, test_loader = process_dataset(
            dataset, train_idx, val_idx, test_idx, batch_size
        )
        train_result, test_result = safe_execute(
            run_fold,
            f"fold-{fold} 运行 {dataset} 时出错",
            catch_error,
            (None, None),
            dataset,
            (train_loader, val_loader, test_loader),
            current_fold=fold,
            config=config,
            run_device=run_device,
        )
        if test_result is not None:
            results.append(test_result)
    all_end = get_time_sync()

    all, last, last_10, last_50 = process_results(results)
    logger.info(
        "----------RESULTS----------\n",
        f"spend time: {(all_end - all_start) / 60:.2f} min\n",
        f"{dataset} for {kfold} fold:\n",
        f"all     : {format_result(all[0], all[1])}\n",
        f"last-50 : {format_result(last_50[0], last_50[1])}\n",
        f"last-10 : {format_result(last_10[0], last_10[1])}\n",
        f"last    : {format_result(last[0], last[1])}\n",
        "---------------------------\n",
    )

    return all, last, last_10, last_50


def safe_execute(func, error_msg, catch_error: bool, error_value=None, *args, **kwargs):
    """Safe execution with error handling. Returns result of func or error_value if error occurs."""
    if catch_error:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{error_msg}: {e}", exc_info=True)
            return error_value
    return func(*args, **kwargs)


def format_result(mean, std):
    """Format mean and std as percentage string (e.g., "73.21% ± 4.50%")."""
    return f"{mean * 100:.2f}% ± {std * 100:.2f}%"


def process_results(results: list[BenchmarkResult]):
    """Compute aggregate stats from BenchmarkResult list. Returns (all, last, last_10, last_50) as (mean, std)."""
    metrics = [(1, "last"), (10, "last_10"), (50, "last_50"), (None, "all")]
    results_dict = {}
    for window, key in metrics:
        values = (
            [result.get_mean(window) for result in results]
            if window
            else [result.get_mean() for result in results]
        )
        results_dict[key] = (np.mean(values), np.std(values))
    return (
        results_dict["all"],
        results_dict["last"],
        results_dict["last_10"],
        results_dict["last_50"],
    )


def split_train_val(train_idx, kfold):
    """Split training indices into train and validation sets. Returns (new_train_idx, val_idx)."""
    val_ratio = 1.0 / float(kfold - 1)
    permuted = np.random.permutation(train_idx)
    val_size = int(len(permuted) * val_ratio)
    val_idx = permuted[:val_size].tolist()
    new_train_idx = permuted[val_size:].tolist()
    return new_train_idx, val_idx


def process_dataset(dataset, train_ids, val_ids, test_ids, batch_size):
    """Create DataLoader for dataset splits. Returns (train_loader, val_loader, test_loader)."""
    train_ids = np.random.permutation(train_ids).tolist()
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    return train_loader, val_loader, test_loader


def kfold_split(source_dataset, k, seed):
    """Generate k-fold splits for a dataset. Returns KFold splitter."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    return kfold.split(source_dataset)


DATASET_SETS = {
    "simple": ["NCI1", "COX2", "IMDB-BINARY"],
    "common": [
        "DD",
        "PROTEINS",
        "NCI1",
        "NCI109",
        "COX2",
        "IMDB-BINARY",
        "IMDB-MULTI",
        "FRANKENSTEIN",
    ],
    "com": ["COLLAB"],
    "bio&chem": ["NCI1", "NCI109", "COX2"],
    "dense": [
        "mit_ct1",
        "mit_ct2",
        "highschool_ct1",
        "highschool_ct2",
        "infectious_ct1",
        "infectious_ct2",
    ],
}


def datasets(sets="common", name=None):
    """Generator for TUDataset objects. Yields TUDataset for each dataset in the set.

    Args:
        sets: Dataset set name (key from DATASET_SETS). Ignored if name is provided.
        name: Single dataset name to use. If provided, overrides sets.
    """
    from pag.utils.rrwp import add_full_rrwp

    if name:
        dataset_list = [name]
    else:
        dataset_list = DATASET_SETS.get(sets, [])

    for dataset_name in dataset_list:
        use_node_attr = dataset_name == "FRANKENSTEIN"
        yield TUDataset(
            root="./pag/dataset/",
            name=dataset_name,
            use_node_attr=use_node_attr,
            pre_transform=add_full_rrwp,
        )


def save_result(results, filename, spent_time, config, config_disp=False):
    """Save results to file."""
    if not results:
        logger.warning("save_result empty")
        return
    with open(filename, "a", encoding="utf-8") as f:
        if config_disp:
            if hasattr(config, "model"):
                f.write(
                    f"model: {config.model.name}, hidden_dim: {config.model.hidden_dim}, dropout: {config.model.dropout}\n"
                )
                f.write(
                    f"train: lr=[macro:{config.train.lr_macro}, pa:{config.train.lr_pa}, other:{config.train.lr_other}], "
                    f"batch_size={config.train.batch_size}, epochs={config.train.max_epochs}\n"
                )
            else:
                f.write(f"{config.format()}")
        for name, all, last, last10, last50 in results:
            f.write(f"{name} all   : {format_result(all[0], all[1])}\n")
            f.write(f"{name} last  : {format_result(last[0], last[1])}\n")
            f.write(f"{name} last10: {format_result(last10[0], last10[1])}\n")
            f.write(f"{name} last50: {format_result(last50[0], last50[1])}\n")
        f.write(f"总运行时间: {spent_time / 60:.2f} min\n")


def run(config, run_device):
    is_yacs = hasattr(config, "model")

    if is_yacs:
        import random

        seed = config.seed if config.seed is not None else random.randint(0, 2**31 - 1)
        random.seed(seed)
        import numpy as np

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        dataset_name = config.dataset.name
        dataset_sets = config.dataset.sets
        model_name = config.model.name
        catch_error = config.dataset.catch_error
        logger.info("=" * 60)
        logger.info("训练参数配置:")
        logger.info(f"  模型: {config.model.name}")
        logger.info(f"  hidden_dim: {config.model.hidden_dim}")
        logger.info(f"  layers: {config.model.layers}")
        logger.info(f"  dropout: {config.model.dropout}")
        logger.info(f"  me_dropout: {config.model.me_dropout}")
        logger.info(f"  le_dropout: {config.model.le_dropout}")
        logger.info(f"  pa_dropout: {config.model.pa_dropout}")
        logger.info(f"  num_me_layers: {config.model.num_me_layers}")
        logger.info(f"  encoder: {config.model.encoder}")
        logger.info(f"  pa_blocks:")
        for i, b in enumerate(config.model.pa_blocks):
            logger.info(
                f"    block{i}: depth={b['depth']}, samples={b['samples']}, rw_length={b['rw_length']}, temp={b['temp']}"
            )
        logger.info(
            f"  学习率 (分组): macro={config.train.lr_macro}, pa={config.train.lr_pa}, other={config.train.lr_other}"
        )
        logger.info(f"  batch_size: {config.train.batch_size}")
        logger.info(f"  max_epochs: {config.train.max_epochs}")
        logger.info(f"  num_folds: {config.train.num_folds}")
        logger.info(f"  label_smoothing: {config.train.label_smoothing}")
        logger.info(f"  数据集: {dataset_name or dataset_sets}")
        logger.info(f"  seed: {seed}")
        logger.info("=" * 60)
    else:
        config.apply_random_seed()
        dataset_name = None
        dataset_sets = config.sets
        model_name = config.model
        catch_error = config.catch_error

    all_start = get_time_sync()
    results = []

    for dataset_id, dataset in enumerate(
        track(
            datasets(sets=dataset_sets, name=dataset_name), description="All Datasets"
        )
    ):
        dataset_start = get_time_sync()
        all, last, last10, last50 = safe_execute(
            run_k_fold4dataset,
            f"运行 {dataset} 时出错",
            catch_error,
            (
                BenchmarkResult(),
                BenchmarkResult(),
                BenchmarkResult(),
                BenchmarkResult(),
            ),
            dataset,
            config,
            run_device,
        )
        results.append((f"{dataset}", all, last, last10, last50))
        save_result(
            [(f"{dataset}", all, last, last10, last50)],
            f"./stgnn/result_fin/{model_name}.txt",
            get_time_sync() - dataset_start,
            config,
            dataset_id == 0,
        )

    if is_yacs:
        logger.info(
            f"model: {config.model.name}, layers: {config.model.layers}, "
            f"hidden_dim: {config.model.hidden_dim}, dropout: {config.model.dropout}\n"
        )
        logger.info(
            f"train: lr=[macro:{config.train.lr_macro}, pa:{config.train.lr_pa}, other:{config.train.lr_other}], "
            f"batch_size={config.train.batch_size}, "
            f"max_epochs={config.train.max_epochs}, num_folds={config.train.num_folds}\n"
        )
    else:
        logger.info(f"{config.format()}\n")
    logger.info(f"总运行时间: {(get_time_sync() - all_start) / 60:.2f} min")


def get_default_config():
    """Get default BenchmarkConfig for backward compatibility."""
    config = BenchmarkConfig()
    config.hidden_channels = 128
    config.num_layers = 3
    config.graph_norm = True
    config.batch_size = 256
    config.epochs = 500
    config.dropout = getattr(config, "dropout", 0.3)
    config.sets = "bio&chem"
    config.catch_error = False
    config.early_stop = True
    config.early_stop_epochs = 30
    config.seed = None
    config.kfold = 10
    config.lr = 0.0005
    return config


def create_arg_parser():
    """Create argument parser for training configuration."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PAG model on TUDataset")
    parser.add_argument(
        "--config_file", type=str, default="", help="Path to YAML config file"
    )
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="Override config options (e.g., train.lr_macro=0.001 train.lr_pa=0.0005 model.layers=4)",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (overrides config)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset set to use (overrides config)",
    )
    return parser


if __name__ == "__main__":
    import sys

    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    # Try to use YACS config if available
    if YACS_AVAILABLE and args.config_file:
        from config import get_cfg_from_args, validate_config

        config = get_cfg_from_args(args)
        try:
            validate_config(config)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid config: {e}")
            sys.exit(1)
        logger.info(f"Using YACS config from {args.config_file}")
    else:
        # Fall back to legacy BenchmarkConfig
        config = get_default_config()
        if args.model:
            config.model = args.model
        if args.dataset:
            config.sets = args.dataset
        logger.info("Using legacy BenchmarkConfig (no config file specified)")

    for model_name in [config.model.name if hasattr(config, "model") else config.model]:
        if hasattr(config, "model"):
            config.model.name = model_name
        else:
            config.model = model_name
        if hasattr(config, "seed"):
            config.seed = 0
        else:
            config.seed = 0
        safe_execute(
            run,
            f"运行 {model_name} 时出错",
            (
                getattr(config, "dataset.catch_error", True)
                if hasattr(config, "dataset")
                else config.catch_error
            ),
            None,
            config,
            run_device,
        )
