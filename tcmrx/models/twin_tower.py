"""
TCM-RX 双塔模型模块
双塔主模型（相似度=余弦）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from .encoders import DiseaseEncoder, FormulaEncoder
from .aggregators import AggregatorFactory

logger = logging.getLogger(__name__)


ACTIVATIONS = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}


class TowerProjection(nn.Module):
    """可选的塔顶 MLP 投影头."""

    def __init__(self,
                 embedding_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 activation: str = 'gelu',
                 dropout: float = 0.1,
                 layer_norm: bool = True,
                 residual: bool = True):
        super().__init__()

        hidden_dims = hidden_dims or []
        if activation not in ACTIVATIONS:
            raise ValueError(f"未知激活函数: {activation}")

        layers: List[nn.Module] = []
        in_dim = embedding_dim
        act_layer = ACTIVATIONS[activation]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embedding_dim))
        self.projection = nn.Sequential(*layers)
        self.use_residual = residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.projection(inputs)
        if self.use_residual and projected.shape == inputs.shape:
            projected = projected + inputs
        return projected


class DualTowerModel(nn.Module):
    """
    双塔对比学习模型
    """

    def __init__(self, config: Dict):
        """
        初始化双塔模型

        Args:
            config: 模型配置字典
        """
        super().__init__()
        self.config = config

        # 模型参数
        self.embedding_dim = config['embedding_dim']
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.temperature = config.get('temperature', 0.1)
        self.aggregator_type = config.get('aggregator_type', 'attention')

        # 实体数量（这些会在数据构建阶段设置）
        self.num_diseases = 0
        self.num_formulas = 0
        self.num_targets = 0

        tower_head_config = config.get('tower_head') or {}
        enabled = tower_head_config.get('enabled', False)
        if enabled:
            shared = tower_head_config.get('share', True)
            head_kwargs = {
                'embedding_dim': self.embedding_dim,
                'hidden_dims': tower_head_config.get('hidden_dims'),
                'activation': tower_head_config.get('activation', 'gelu'),
                'dropout': tower_head_config.get('dropout', 0.1),
                'layer_norm': tower_head_config.get('layer_norm', True),
                'residual': tower_head_config.get('residual', True),
            }
            head_module = TowerProjection(**head_kwargs)
            if shared:
                self.disease_head = head_module
                self.formula_head = head_module
            else:
                self.disease_head = TowerProjection(**head_kwargs)
                self.formula_head = TowerProjection(**head_kwargs)
        else:
            self.disease_head = nn.Identity()
            self.formula_head = nn.Identity()

        # 创建聚合器
        disease_aggregator = AggregatorFactory.create_aggregator(
            self.aggregator_type, self.embedding_dim, dropout_rate=self.dropout_rate
        )
        formula_aggregator = AggregatorFactory.create_aggregator(
            self.aggregator_type, self.embedding_dim, dropout_rate=self.dropout_rate
        )

        # 编码器（先创建空的，稍后设置数量）
        self.disease_encoder = None
        self.formula_encoder = None
        self.target_encoder = None

        # 冻结温度参数（可学习）
        if isinstance(self.temperature, (float, int)):
            self.temperature = nn.Parameter(torch.tensor(self.temperature, dtype=torch.float))
            logger.info(f"使用可学习温度参数，初始值: {self.temperature.item()}")

    def set_entity_counts(self, num_diseases: int, num_formulas: int, num_targets: int) -> None:
        """
        设置实体数量（用于初始化嵌入层）

        Args:
            num_diseases: 疾病数量
            num_formulas: 方剂数量
            num_targets: 靶点数量
        """
        self.num_diseases = num_diseases
        self.num_formulas = num_formulas
        self.num_targets = num_targets

        # 创建聚合器
        disease_aggregator = AggregatorFactory.create_aggregator(
            self.aggregator_type, self.embedding_dim, dropout_rate=self.dropout_rate
        )
        formula_aggregator = AggregatorFactory.create_aggregator(
            self.aggregator_type, self.embedding_dim, dropout_rate=self.dropout_rate
        )

        # 重新创建编码器
        from .encoders import EncoderFactory
        self.disease_encoder, self.formula_encoder = EncoderFactory.create_shared_encoder_system(
            num_targets, self.embedding_dim, disease_aggregator, formula_aggregator, self.dropout_rate
        )

        self.target_encoder = self.disease_encoder.target_encoder  # 共享的靶点编码器

        logger.info(f"设置实体数量: diseases={num_diseases}, formulas={num_formulas}, targets={num_targets}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            batch: 批次数据字典，包含：
                - disease_target_indices: [batch_size, max_disease_targets]
                - disease_target_weights: [batch_size, max_disease_targets]
                - disease_mask: [batch_size, max_disease_targets]
                - formula_target_indices: [batch_size, max_formula_targets]
                - formula_target_weights: [batch_size, max_formula_targets]
                - formula_mask: [batch_size, max_formula_targets]

        Returns:
            包含相似度和嵌入的字典
        """
        if self.disease_encoder is None or self.formula_encoder is None:
            raise ValueError("请先调用 set_entity_counts() 设置实体数量")

        # 编码疾病侧
        disease_embeddings = self.disease_encoder(
            batch['disease_target_indices'],
            batch['disease_target_weights'],
            batch['disease_mask']
        )  # [batch_size, embedding_dim]
        disease_embeddings = self.disease_head(disease_embeddings)
        disease_embeddings = F.normalize(disease_embeddings, p=2, dim=1)

        # 编码方剂侧
        formula_embeddings = self.formula_encoder(
            batch['formula_target_indices'],
            batch['formula_target_weights'],
            batch['formula_mask']
        )  # [batch_size, embedding_dim]
        formula_embeddings = self.formula_head(formula_embeddings)
        formula_embeddings = F.normalize(formula_embeddings, p=2, dim=1)

        # 计算相似度矩阵
        similarities = self.compute_similarity_matrix(disease_embeddings, formula_embeddings)

        # 应用温度缩放
        if isinstance(self.temperature, nn.Parameter):
            scaled_similarities = similarities / self.temperature.clamp(min=0.01)
        else:
            scaled_similarities = similarities / max(self.temperature, 0.01)

        return {
            'disease_embeddings': disease_embeddings,
            'formula_embeddings': formula_embeddings,
            'similarities': similarities,
            'scaled_similarities': scaled_similarities
        }

    def compute_similarity_matrix(self, disease_embeddings: torch.Tensor,
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
        disease_embeddings = torch.nn.functional.normalize(disease_embeddings, p=2, dim=1)
        formula_embeddings = torch.nn.functional.normalize(formula_embeddings, p=2, dim=1)

        # 计算余弦相似度
        similarities = torch.mm(disease_embeddings, formula_embeddings.t())

        return similarities

    def encode_diseases(self, disease_target_indices: torch.Tensor,
                        disease_target_weights: torch.Tensor,
                        disease_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码疾病

        Args:
            disease_target_indices: 疾病靶点索引 [batch_size, max_targets]
            disease_target_weights: 疾病靶点权重 [batch_size, max_targets]
            disease_mask: 疾病掩码 [batch_size, max_targets]

        Returns:
            疾病嵌入 [batch_size, embedding_dim]
        """
        if self.disease_encoder is None:
            raise ValueError("请先调用 set_entity_counts() 设置实体数量")

        return self.disease_encoder(disease_target_indices, disease_target_weights, disease_mask)

    def encode_formulas(self, formula_target_indices: torch.Tensor,
                       formula_target_weights: torch.Tensor,
                       formula_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码方剂

        Args:
            formula_target_indices: 方剂靶点索引 [batch_size, max_targets]
            formula_target_weights: 方剂靶点权重 [batch_size, max_targets]
            formula_mask: 方剂掩码 [batch_size, max_targets]

        Returns:
            方剂嵌入 [batch_size, embedding_dim]
        """
        if self.formula_encoder is None:
            raise ValueError("请先调用 set_entity_counts() 设置实体数量")

        return self.formula_encoder(formula_target_indices, formula_target_weights, formula_mask)

    def get_target_embeddings(self) -> torch.Tensor:
        """
        获取所有靶点的嵌入

        Returns:
            靶点嵌入 [num_targets, embedding_dim]
        """
        if self.target_encoder is None:
            raise ValueError("请先调用 set_entity_counts() 设置实体数量")

        # 创建所有靶点的索引
        device = next(self.parameters()).device
        all_target_indices = torch.arange(self.num_targets, device=device, dtype=torch.long)
        all_target_weights = torch.ones(self.num_targets, device=device, dtype=torch.float)

        was_training = self.target_encoder.training
        self.target_encoder.eval()
        with torch.no_grad():
            embeddings = self.target_encoder(
                all_target_indices.unsqueeze(0),
                all_target_weights.unsqueeze(0)
            ).squeeze(0)
        if was_training:
            self.target_encoder.train()

        return embeddings

    def get_temperature(self) -> float:
        """
        获取当前温度参数

        Returns:
            温度值
        """
        if isinstance(self.temperature, nn.Parameter):
            return self.temperature.item()
        else:
            return self.temperature

    def print_model_info(self) -> None:
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info("双塔模型信息:")
        logger.info(f"嵌入维度: {self.embedding_dim}")
        logger.info(f"聚合器类型: {self.aggregator_type}")
        logger.info(f"温度参数: {self.get_temperature()}")
        logger.info(f"实体数量: diseases={self.num_diseases}, formulas={self.num_formulas}, targets={self.num_targets}")
        logger.info(f"总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"模型大小: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        logger.info("=" * 60)