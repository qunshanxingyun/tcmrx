"""
TCM-RX 聚合器模块
WeightedSum / Attention 聚合实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WeightedSumAggregator(nn.Module):
    """
    加权求和聚合器
    先验权重求和→L2归一化
    """

    def __init__(self, embedding_dim: int):
        """
        初始化加权求和聚合器

        Args:
            embedding_dim: 嵌入维度
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            embeddings: 输入嵌入 [batch_size, num_items, embedding_dim]
            mask: 掩码 [batch_size, num_items] (可选)

        Returns:
            聚合后的嵌入 [batch_size, embedding_dim]
        """
        batch_size, num_items, embedding_dim = embeddings.size()

        if mask is not None:
            # 应用掩码
            mask = mask.unsqueeze(-1)  # [batch_size, num_items, 1]
            embeddings = embeddings * mask

        # 沿着num_items维度求和（已经应用了权重）
        summed = torch.sum(embeddings, dim=1)  # [batch_size, embedding_dim]

        # L2归一化
        normalized = F.normalize(summed, p=2, dim=1)

        return normalized


class AttentionAggregator(nn.Module):
    """
    注意力聚合器
    单头注意力微调先验权重
    """

    def __init__(self, embedding_dim: int, dropout_rate: float = 0.1):
        """
        初始化注意力聚合器

        Args:
            embedding_dim: 嵌入维度
            dropout_rate: Dropout率
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # 注意力投影层
        self.attention_proj = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

        # 缩放因子（类似Transformer中的注意力）
        self.scale = 1.0 / math.sqrt(embedding_dim)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            embeddings: 输入嵌入 [batch_size, num_items, embedding_dim]
                       （已经应用了先验权重）
            mask: 掩码 [batch_size, num_items] (可选)

        Returns:
            聚合后的嵌入 [batch_size, embedding_dim]
        """
        batch_size, num_items, embedding_dim = embeddings.size()

        # 计算注意力分数
        attention_scores = self.attention_proj(embeddings)  # [batch_size, num_items, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, num_items]
        attention_scores = attention_scores * self.scale

        if mask is not None:
            # 应用掩码（将掩码位置设为负无穷）
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_items]
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)  # [batch_size, num_items, embedding_dim]

        # 求和
        summed = torch.sum(weighted_embeddings, dim=1)  # [batch_size, embedding_dim]

        # L2归一化
        normalized = F.normalize(summed, p=2, dim=1)

        return normalized


class MultiHeadAttentionAggregator(nn.Module):
    """
    多头注意力聚合器（可选的高级版本）
    """

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout_rate: float = 0.1):
        """
        初始化多头注意力聚合器

        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            dropout_rate: Dropout率
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "嵌入维度必须能被头数整除"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # 线性投影层
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            embeddings: 输入嵌入 [batch_size, num_items, embedding_dim]
            mask: 掩码 [batch_size, num_items] (可选)

        Returns:
            聚合后的嵌入 [batch_size, embedding_dim]
        """
        batch_size, num_items, embedding_dim = embeddings.size()

        # 计算Q, K, V
        Q = self.q_proj(embeddings)  # [batch_size, num_items, embedding_dim]
        K = self.k_proj(embeddings)  # [batch_size, num_items, embedding_dim]
        V = self.v_proj(embeddings)  # [batch_size, num_items, embedding_dim]

        # 重塑为多头格式
        Q = Q.view(batch_size, num_items, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_items, head_dim]
        K = K.view(batch_size, num_items, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_items, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, num_items, num_items]

        if mask is not None:
            # 扩展掩码以适应多头
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, num_items]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)  # [batch_size, num_heads, num_items, head_dim]

        # 重塑回原始格式
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_items, embedding_dim)

        # 输出投影
        output = self.out_proj(attended)

        # 全局平均池化（因为我们需要一个向量表示）
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch_size, num_items, 1]
            output = output * mask

        # 求和并归一化
        summed = torch.sum(output, dim=1)  # [batch_size, embedding_dim]
        normalized = F.normalize(summed, p=2, dim=1)

        return normalized


class AggregatorFactory:
    """
    聚合器工厂
    """

    @staticmethod
    def create_aggregator(aggregator_type: str, embedding_dim: int,
                          num_heads: Optional[int] = None,
                          dropout_rate: float = 0.1) -> nn.Module:
        """
        创建聚合器

        Args:
            aggregator_type: 聚合器类型 ('weighted_sum', 'attention', 'multihead_attention')
            embedding_dim: 嵌入维度
            num_heads: 多头注意力的头数（仅用于multihead_attention）
            dropout_rate: Dropout率

        Returns:
            聚合器实例
        """
        if aggregator_type == "weighted_sum":
            return WeightedSumAggregator(embedding_dim)
        elif aggregator_type == "attention":
            return AttentionAggregator(embedding_dim, dropout_rate)
        elif aggregator_type == "multihead_attention":
            if num_heads is None:
                num_heads = 8  # 默认值
            return MultiHeadAttentionAggregator(embedding_dim, num_heads, dropout_rate)
        else:
            raise ValueError(f"未知的聚合器类型: {aggregator_type}")

    @staticmethod
    def get_available_aggregators() -> list:
        """
        获取可用的聚合器类型

        Returns:
            聚合器类型列表
        """
        return ["weighted_sum", "attention", "multihead_attention"]