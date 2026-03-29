"""
训练工具模块
提取通用的训练逻辑，减少代码冗余
"""

import torch
import logging
from contextlib import nullcontext
from typing import Optional, Tuple
from rich.progress import track

logger = logging.getLogger("training.utils")


def prepare_data(data, device: torch.device) -> torch.Tensor:
    """
    预处理图数据

    Args:
        data: 图数据对象
        device: 目标设备

    Returns:
        移动到设备后的数据对象
    """
    if data.x is None or data.x.size(1) <= 0:
        data.x = torch.ones((data.num_nodes, 1))
    return data.to(device)


def compute_loss(loss1: torch.Tensor, loss2: torch.Tensor) -> torch.Tensor:
    """
    计算组合损失

    Args:
        loss1: 主损失
        loss2: 附加损失

    Returns:
        组合后的损失
    """
    return loss1 + loss2 / (loss1 + loss2 + 1e-6).detach()


def forward_model(
    pooler, classifier, data, criterion, model_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    模型前向传播

    Args:
        pooler: 图神经网络模型
        classifier: 分类器
        data: 输入数据
        criterion: 损失函数
        model_name: 模型名称

    Returns:
        (pooled, out, loss)
    """
    if model_name == "rum":
        pooled, additional_loss = pooler(data)
    elif model_name in ("pag", "PathAttentionGraphormer"):
        pooled, h, attn_w, additional_loss = pooler(data)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    out = classifier(pooled)
    loss = compute_loss(criterion(out, data.y), additional_loss)

    return pooled, out, loss


def update_progress(
    total_loss: float,
    correct: int,
    total: int,
    loss: float,
    pred: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[float, int, int]:
    """
    更新训练进度统计

    Args:
        total_loss: 累计损失
        correct: 正确预测数
        total: 总样本数
        loss: 当前批次损失
        pred: 预测结果
        y: 真实标签

    Returns:
        (total_loss, correct, total)
    """
    total_loss += loss.item()
    correct += (pred == y).sum().item()
    total += y.size(0)
    return total_loss, correct, total


def get_device_type(device: torch.device) -> str:
    """
    获取设备类型字符串

    Args:
        device: PyTorch 设备对象

    Returns:
        "cuda" 或 "cpu"
    """
    return "cuda" if "cuda" in str(device) else "cpu"


def log_epoch_progress(
    fold: int,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    test_loss: float,
    test_acc: float,
    lr: float,
):
    """
    记录单个epoch的进度

    Args:
        fold: 当前fold
        epoch: 当前epoch
        train_loss: 训练损失
        train_acc: 训练准确率
        val_loss: 验证损失
        val_acc: 验证准确率
        test_loss: 测试损失
        test_acc: 测试准确率
        lr: 当前学习率
    """
    log_msg = (
        f"fold:{fold+1}, Epoch {epoch+1:03d} "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
        f"lr: {lr:.6f}"
    )
    logger.info(log_msg)


def log_fold_results(fold: int, test_acc_res, duration_minutes: float):
    """
    记录单个fold的结果

    Args:
        fold: 当前fold
        test_acc_res: 测试准确率结果对象
        duration_minutes: 训练时长（分钟）
    """
    log_msg = (
        f"fold:{fold+1} {test_acc_res.format('all')}\n"
        f"fold:{fold+1} {test_acc_res.format('last', last=1)}\n"
        f"fold:{fold+1} {test_acc_res.format('last10', last=10)}\n"
        f"fold:{fold+1} {test_acc_res.format('last50', last=50)}\n"
        f"fold:{fold+1} 总运行时间: {duration_minutes:.2f}min"
    )
    logger.info(log_msg)


def log_dataset_results(dataset_name: str, results: list, duration_minutes: float):
    """
    记录整个数据集的结果

    Args:
        dataset_name: 数据集名称
        results: 结果列表 [(all, last, last10, last50), ...]
        duration_minutes: 总时长（分钟）
    """
    format_result = lambda mean, std: f"{mean * 100:.2f}% ± {std * 100:.2f}%"

    for name, all_res, last_res, last10_res, last50_res in results:
        log_msg = (
            f"---------- RESULTS ----------\n"
            f"Dataset: {name}\n"
            f"Time: {duration_minutes:.2f} min\n"
            f"All:   {format_result(all_res[0], all_res[1])}\n"
            f"Last:  {format_result(last_res[0], last_res[1])}\n"
            f"Last10:{format_result(last10_res[0], last10_res[1])}\n"
            f"Last50:{format_result(last50_res[0], last50_res[1])}\n"
            f"----------------------------"
        )
        logger.info(log_msg)


def run_epoch(
    pooler,
    classifier,
    data_loader,
    criterion,
    device: torch.device,
    model_name: str,
    optimizer=None,
    scheduler=None,
    use_amp: bool = False,
    mode: str = "train",
    progress_desc: str = "Run epoch",
    gradient_clip: float = 1.0,
) -> Tuple[float, float]:
    """
    运行单个训练或评估epoch

    Args:
        pooler: 图神经网络模型
        classifier: 分类器
        data_loader: 数据加载器
        criterion: 损失函数
        device: 目标设备
        model_name: 模型名称
        optimizer: 优化器 (仅训练模式需要)
        scheduler: 学习率调度器 (仅训练模式需要)
        use_amp: 是否使用混合精度训练
        mode: 运行模式 ('train' 或 'eval')
        progress_desc: 进度条描述

    Returns:
        (avg_loss, accuracy)
    """
    # 设置模型模式
    if mode == "train":
        pooler.train()
        classifier.train()
    else:
        pooler.eval()
        classifier.eval()

    # 初始化统计
    total_loss = 0.0
    correct = 0
    total = 0

    # 获取设备类型
    device_type = get_device_type(device)

    # 初始化 AMP Scaler (仅训练模式)
    scaler = torch.amp.GradScaler(enabled=use_amp) if mode == "train" else None

    # 设置梯度上下文
    grad_context = torch.no_grad() if mode == "eval" else nullcontext()

    with grad_context:
        for data in track(data_loader, description=progress_desc, disable=True):
            # 训练模式: 清零梯度
            if mode == "train":
                optimizer.zero_grad()

            # 预处理数据
            data = prepare_data(data, device)

            # 前向传播
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pooled, out, loss = forward_model(
                    pooler, classifier, data, criterion, model_name
                )

            # 训练模式: 反向传播与优化器更新
            if mode == "train":
                max_grad_norm = gradient_clip

                if use_amp:
                    # 缩放损失并反向传播
                    scaler.scale(loss).backward()
                    # 在裁剪前，必须显式取消梯度的缩放
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        pooler.parameters(), max_norm=max_grad_norm
                    )

                    scale_before = scaler.get_scale()

                    # scaler.step() 会先取消缩放梯度，如果梯度无 inf/NaN，则调用
                    scaler.step(optimizer)
                    # 更新缩放
                    scaler.update()
                    # 判断是否跳过了 optimizer.step()
                    skip_lr_sched = scale_before > scaler.get_scale()

                    # 只有在没有跳过的情况下，才更新学习率
                    if not skip_lr_sched:
                        if scheduler is not None:
                            scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        pooler.parameters(), max_norm=max_grad_norm
                    )
                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

            # 更新统计
            total_loss, correct, total = update_progress(
                total_loss, correct, total, loss, out.argmax(dim=1), data.y
            )

    total_batches = len(data_loader)
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc
