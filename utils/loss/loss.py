import torch.nn.functional as F


def sce_loss(h_hat, h_target, alpha=2.0):
    """
    Scaled Cosine Error: 针对隐向量重建的鲁棒损失
    """
    h_hat = F.normalize(h_hat, p=2, dim=-1)
    h_target = F.normalize(h_target, p=2, dim=-1)
    # 计算余弦距离的幂次
    loss = (1 - (h_hat * h_target).sum(dim=-1)).pow(alpha)
    return loss.mean()
