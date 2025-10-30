"""
TCM-RX 损失函数模块
InfoNCE对比损失实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def info_nce(similarities: torch.Tensor,
             temperature: float = 0.1,
             labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    InfoNCE对比损失（对称in-batch negatives）

    Args:
        similarities: 相似度矩阵 [batch_size, batch_size]
        temperature: 温度参数
        labels: 标签张量 [batch_size]（可选，默认为对角线）

    Returns:
        损失值
    """
    if temperature <= 0:
        raise ValueError(f"温度参数必须大于0，当前值: {temperature}")

    batch_size = similarities.size(0)

    # 温度缩放
    scaled_similarities = similarities / temperature

    # 如果没有提供标签，使用对角线（即每个样本与自己匹配）
    if labels is None:
        labels = torch.arange(batch_size, device=similarities.device)

    # 计算交叉熵损失
    # 对于每个样本i，将其与所有样本的相似度视为类别概率
    # 正样本是对角线位置
    loss = F.cross_entropy(scaled_similarities, labels)

    return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比损失模块
    """

    def __init__(self, temperature: float = 0.1):
        """
        初始化InfoNCE损失

        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.loss_fn = info_nce

    def forward(self,
                disease_embeddings: torch.Tensor,
                formula_embeddings: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            disease_embeddings: 疾病嵌入 [batch_size, embedding_dim]
            formula_embeddings: 方剂嵌入 [batch_size, embedding_dim]
            labels: 标签 [batch_size]（可选）

        Returns:
            损失值
        """
        # 计算相似度矩阵
        # disease_embeddings @ formula_embeddings.T 得到 [batch_size, batch_size]
        similarities = torch.mm(disease_embeddings, formula_embeddings.t())

        # 应用InfoNCE损失
        loss = self.loss_fn(similarities, self.temperature, labels)

        return loss

    def compute_similarity_matrix(self,
                                disease_embeddings: torch.Tensor,
                                formula_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算相似度矩阵

        Args:
            disease_embeddings: 疾病嵌入 [batch_size, embedding_dim]
            formula_embeddings: 方剂嵌入 [batch_size, embedding_dim]

        Returns:
            相似度矩阵 [batch_size, batch_size]
        """
        # L2归一化
        disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1)
        formula_embeddings = F.normalize(formula_embeddings, p=2, dim=1)

        # 计算余弦相似度
        similarities = torch.mm(disease_embeddings, formula_embeddings.t())

        return similarities


class ContrastiveAccuracy(nn.Module):
    """
    对比学习准确率指标
    """

    def __init__(self, temperature: float = 0.1):
        """
        初始化对比准确率

        Args:
            temperature: 温度参数（用于与损失函数一致）
        """
        super().__init__()
        self.temperature = temperature

    def forward(self,
                disease_embeddings: torch.Tensor,
                formula_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算对比准确率

        Args:
            disease_embeddings: 疾病嵌入 [batch_size, embedding_dim]
            formula_embeddings: 方剂嵌入 [batch_size, embedding_dim]

        Returns:
            准确率（标量）
        """
        with torch.no_grad():
            # 计算相似度矩阵
            similarities = torch.mm(disease_embeddings, formula_embeddings.t())

            # 对于每个疾病样本，预测最匹配的方剂
            predicted = similarities.argmax(dim=1)

            # 正确答案是对角线
            correct = (predicted == torch.arange(len(predicted), device=predicted.device)).float()

            accuracy = correct.mean()

        return accuracy


def compute_contrastive_metrics(disease_embeddings: torch.Tensor,
                               formula_embeddings: torch.Tensor,
                               temperature: float = 0.1) -> dict:
    """
    计算对比学习相关指标

    Args:
        disease_embeddings: 疾病嵌入 [batch_size, embedding_dim]
        formula_embeddings: 方剂嵌入 [batch_size, embedding_dim]
        temperature: 温度参数

    Returns:
        指标字典
    """
    with torch.no_grad():
        # L2归一化
        disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1)
        formula_embeddings = F.normalize(formula_embeddings, p=2, dim=1)

        # 计算相似度矩阵
        similarities = torch.mm(disease_embeddings, formula_embeddings.t())

        # InfoNCE损失
        labels = torch.arange(len(similarities), device=similarities.device)
        loss = info_nce(similarities, temperature, labels)

        # 准确率
        predicted = similarities.argmax(dim=1)
        accuracy = (predicted == labels).float().mean()

        # 平均相似度（正样本）
        positive_similarities = similarities[torch.arange(len(similarities))]
        mean_positive_sim = positive_similarities.mean()

        # 平均相似度（负样本）
        mask = torch.eye(len(similarities), device=similarities.device).bool()
        negative_similarities = similarities[~mask]
        mean_negative_sim = negative_similarities.mean()

    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'mean_positive_similarity': mean_positive_sim.item(),
        'mean_negative_similarity': mean_negative_sim.item(),
        'similarity_gap': (mean_positive_sim - mean_negative_sim).item()
    }