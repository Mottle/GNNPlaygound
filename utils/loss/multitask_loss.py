import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskLoss, self).__init__()
        # 初始化 log_vars (对应公式中的 s)
        # 初始化为 0 相当于初始权重 a=1, b=1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, input_losses):
        """
        input_losses: 一个包含多个 loss 值的列表 [loss_rec, loss_reg]
        """
        # 确保输入 loss 和参数在同一个设备上
        total_loss = 0
        for i, loss in enumerate(input_losses):
            # 获取对应的 log_var
            log_var = self.log_vars[i]

            # 执行公式: 1/(2a^2) * loss + log(a)
            # 等价于: 0.5 * exp(-s) * loss + 0.5 * s
            weighted_loss = 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
            total_loss += weighted_loss

        return total_loss

    def get_vars(self):
        return self.log_vars
