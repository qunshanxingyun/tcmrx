"""
TCM-RX 编码器模块
轻量编码器：Embedding + 简单聚合
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TargetEncoder(nn.Module):
    """
    靶点编码器
    可学习的嵌入向量
    """

    def __init__(self, num_targets: int, embedding_dim: int, dropout_rate: float = 0.1):
        """
        初始化靶点编码器

        Args:
            num_targets: 靶点数量
            embedding_dim: 嵌入维度
            dropout_rate: Dropout率
        """
        super().__init__()
        self.embedding = nn.Embedding(num_targets, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim

        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, target_indices: torch.Tensor, target_weights: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            target_indices: 靶点索引 [batch_size, num_targets]
            target_weights: 靶点权重 [batch_size, num_targets]

        Returns:
            靶点嵌入 [batch_size, num_targets, embedding_dim]
        """
        # 获取嵌入
        embeddings = self.embedding(target_indices)  # [batch_size, num_targets, embedding_dim]
        embeddings = self.dropout(embeddings)

        # 应用权重
        weights = target_weights.unsqueeze(-1)  # [batch_size, num_targets, 1]
        weighted_embeddings = embeddings * weights

        return weighted_embeddings


class DiseaseEncoder(nn.Module):
    """
    疾病编码器
    基于靶点聚合的疾病表示
    """

    def __init__(self, target_encoder: TargetEncoder, aggregator: nn.Module):
        """
        初始化疾病编码器

        Args:
            target_encoder: 靶点编码器
            aggregator: 聚合器
        """
        super().__init__()
        self.target_encoder = target_encoder
        self.aggregator = aggregator

    def forward(self, target_indices: torch.Tensor, target_weights: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            target_indices: 靶点索引 [batch_size, num_targets]
            target_weights: 靶点权重 [batch_size, num_targets]
            mask: 掩码 [batch_size, num_targets] (可选)

        Returns:
            疾病嵌入 [batch_size, embedding_dim]
        """
        # 获取靶点嵌入
        target_embeddings = self.target_encoder(target_indices, target_weights)

        # 聚合为疾病表示
        disease_embedding = self.aggregator(target_embeddings, mask)

        return disease_embedding


class FormulaEncoder(nn.Module):
    """
    方剂编码器
    基于靶点聚合的方剂表示
    """

    def __init__(self, target_encoder: TargetEncoder, aggregator: nn.Module):
        """
        初始化方剂编码器

        Args:
            target_encoder: 靶点编码器
            aggregator: 聚合器
        """
        super().__init__()
        self.target_encoder = target_encoder
        self.aggregator = aggregator

    def forward(self, target_indices: torch.Tensor, target_weights: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            target_indices: 靶点索引 [batch_size, num_targets]
            target_weights: 靶点权重 [batch_size, num_targets]
            mask: 掩码 [batch_size, num_targets] (可选)

        Returns:
            方剂嵌入 [batch_size, embedding_dim]
        """
        # 获取靶点嵌入
        target_embeddings = self.target_encoder(target_indices, target_weights)

        # 聚合为方剂表示
        formula_embedding = self.aggregator(target_embeddings, mask)

        return formula_embedding


class EncoderFactory:
    """
    编码器工厂
    """

    @staticmethod
    def create_target_encoder(num_targets: int, embedding_dim: int,
                            dropout_rate: float = 0.1) -> TargetEncoder:
        """
        创建靶点编码器

        Args:
            num_targets: 靶点数量
            embedding_dim: 嵌入维度
            dropout_rate: Dropout率

        Returns:
            靶点编码器实例
        """
        return TargetEncoder(num_targets, embedding_dim, dropout_rate)

    @staticmethod
    def create_disease_encoder(num_targets: int, embedding_dim: int,
                             aggregator: nn.Module, dropout_rate: float = 0.1) -> DiseaseEncoder:
        """
        创建疾病编码器

        Args:
            num_targets: 靶点数量
            embedding_dim: 嵌入维度
            aggregator: 聚合器
            dropout_rate: Dropout率

        Returns:
            疾病编码器实例
        """
        target_encoder = TargetEncoder(num_targets, embedding_dim, dropout_rate)
        return DiseaseEncoder(target_encoder, aggregator)

    @staticmethod
    def create_formula_encoder(num_targets: int, embedding_dim: int,
                              aggregator: nn.Module, dropout_rate: float = 0.1) -> FormulaEncoder:
        """
        创建方剂编码器

        Args:
            num_targets: 靶点数量
            embedding_dim: 嵌入维度
            aggregator: 聚合器
            dropout_rate: Dropout率

        Returns:
            方剂编码器实例
        """
        target_encoder = TargetEncoder(num_targets, embedding_dim, dropout_rate)
        return FormulaEncoder(target_encoder, aggregator)

    @staticmethod
    def create_shared_encoder_system(num_targets: int, embedding_dim: int,
                                   disease_aggregator: nn.Module,
                                   formula_aggregator: nn.Module,
                                   dropout_rate: float = 0.1) -> Tuple[DiseaseEncoder, FormulaEncoder]:
        """
        创建共享靶点编码器的双塔系统

        Args:
            num_targets: 靶点数量
            embedding_dim: 嵌入维度
            disease_aggregator: 疾病侧聚合器
            formula_aggregator: 方剂侧聚合器
            dropout_rate: Dropout率

        Returns:
            (疾病编码器, 方剂编码器)
        """
        # 共享靶点编码器
        target_encoder = TargetEncoder(num_targets, embedding_dim, dropout_rate)

        # 创建双塔编码器
        disease_encoder = DiseaseEncoder(target_encoder, disease_aggregator)
        formula_encoder = FormulaEncoder(target_encoder, formula_aggregator)

        return disease_encoder, formula_encoder