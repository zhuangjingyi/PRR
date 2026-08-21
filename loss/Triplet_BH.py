import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    计算两个嵌入矩阵之间的欧几里得距离
    Args:
        x: (B1, D) 嵌入向量
        y: (B2, D) 嵌入向量
    Returns:
        dist: (B1, B2) 成对距离矩阵
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = torch.sqrt(torch.abs(dist) + 1e-12)  # 加 epsilon 防止 sqrt(0) 梯度爆炸
    return dist


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss
    参考论文: In Defense of the Triplet Loss for Person Re-Identification
    arXiv: https://arxiv.org/abs/1703.07737
    code: https://github.com/AsuradaYuci/tripletreid-zhushi/tree/master 
    
    Args:
        margin: float → hard margin (max(diff + margin, 0))
                'soft' → softplus(diff)
                'none' → 不加 margin
        metric: 'euclidean' | 'sqeuclidean' | 'cityblock'
        normalize: 是否对嵌入做 L2 归一化
    """
    def __init__(self, margin='soft', metric='euclidean', normalize=True):
        super().__init__()
        self.margin = margin
        self.metric = metric
        self.normalize = normalize

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) 嵌入向量
            labels:     (B,)   行人 ID 标签
        Returns:
            loss: 标量，平均 batch hard triplet loss
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算距离矩阵 (B, B)
        if self.metric == 'euclidean':
            dists = euclidean_dist(embeddings, embeddings)
        elif self.metric == 'sqeuclidean':
            dists = torch.pow(euclidean_dist(embeddings, embeddings), 2)
        else:
            raise NotImplementedError(f"Metric {self.metric} not implemented")

        # 构建正/负样本 mask
        labels = labels.unsqueeze(0)  # (1, B)
        same_identity = labels == labels.t()  # (B, B), 同行人=True
        eye = torch.eye(labels.size(1), dtype=torch.bool, device=labels.device)
        positive_mask = same_identity ^ eye       # 同行人 且 不是自己
        negative_mask = ~same_identity            # 不同行人

        # 挖掘最远正样本和最近负样本
        # 正样本距离 -> 取每行最大值（最远正样本）
        pos_dists = dists * positive_mask.float()
        pos_dists[~positive_mask] = -1e9  # 无效位置设为极小值
        furthest_positive, _ = pos_dists.max(dim=1)

        # 负样本距离 -> 取每行最小值（最近负样本）
        neg_dists = dists * negative_mask.float()
        neg_dists[~negative_mask] = 1e9   # 无效位置设为极大值
        closest_negative, _ = neg_dists.min(dim=1)

        # 计算 loss
        diff = furthest_positive - closest_negative

        if isinstance(self.margin, (int, float)):
            loss = torch.clamp(diff + self.margin, min=0.0)
        elif self.margin == 'soft':
            loss = F.softplus(diff)
        elif self.margin.lower() == 'none':
            loss = diff
        else:
            raise NotImplementedError(f"Margin '{self.margin}' not supported")

        # 只对 active 的样本求平均（loss > 1e-5）
        active = loss > 1e-5
        if active.sum() == 0:
            return loss.mean()
        return loss[active].mean()


# ========== 使用示例 ==========
if __name__ == '__main__':
    # 模拟 PK 采样: P=4 个人, 每人 K=4 张图, batch = 16
    criterion = BatchHardTripletLoss(margin='soft', metric='euclidean')

    emb = torch.randn(16, 128)  # (16, 128) 嵌入向量
    # 4 个人, 每人 4 张图
    lbl = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])

    loss = criterion(emb, lbl)
    print(f"Batch Hard Triplet Loss: {loss.item():.4f}")
