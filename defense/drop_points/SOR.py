"""SOR defense proposed by ICCV'19 paper DUP-Net"""
import torch
import torch.nn as nn


class SORDefense(nn.Module):
    """Statistical outlier removal as defense.
    """

    def __init__(self, k=2, alpha=1.1):
        """SOR defense.

        Args:
            k (int, optional): kNN. Defaults to 2.
            alpha (float, optional): \miu + \alpha * std. Defaults to 1.1.
        """
        super(SORDefense, self).__init__()

        self.k = k
        self.alpha = alpha

    def outlier_removal(self, x):
        """Removes large kNN distance points.

        Args:
            x (torch.FloatTensor): batch input pc, [B, K, 3]

        Returns:
            torch.FloatTensor: pc after outlier removal, [B, N, 3]
        """
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K] 计算点云之间的内积
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K] 计算点云的平方和
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]  计算点云之间的欧氏距离
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]  对-dist进行排序，获取每行中最大的k+1个值
        value = -(neg_value[..., 1:])  # [B, K, k]  除第一个值以外的所有值取相反数，并赋值给value。这是为了得到每个点云与其最近的k个点之间的距离
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B] 标准差
        threshold = mean + self.alpha * std  # [B] 阈值
        bool_mask = (value <= threshold[:, None])  # [B, K]  True表示该点云被保留，False表示该点云被丢弃
        sel_pc = [x[i][bool_mask[i]] for i in range(B)]
        return sel_pc

    def forward(self, x):
        with torch.no_grad():
            x = self.outlier_removal(x)
        return x
