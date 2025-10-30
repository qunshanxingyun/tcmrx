"""
TCM-RX 数据集构建模块
组装 (Disease, Formula) 训练/验证样本
"""

import pandas as pd
from typing import Dict, List, Tuple, Set
import numpy as np
import logging

from .id_maps import IDMapper
from .filters import apply_topk_to_set, compute_inverse_freq, apply_inverse_freq_weights, normalize_weights

logger = logging.getLogger(__name__)


class TrainingSample:
    """训练样本数据结构"""

    def __init__(self, disease_id: str, formula_id: str, label: int = 1):
        self.disease_id = disease_id
        self.formula_id = formula_id
        self.label = label

    def __repr__(self):
        return f"TrainingSample(disease={self.disease_id}, formula={self.formula_id}, label={self.label})"


class TCMRXDataset:
    """
    TCM-RX训练数据集

    负责：
    1. 构建ID映射
    2. 处理疾病/方剂靶点集合
    3. 组装训练样本
    4. 应用过滤和权重
    """

    def __init__(self, config: Dict):
        """
        初始化数据集构建器

        Args:
            config: 配置字典
        """
        self.config = config
        self.id_mapper = IDMapper()

        # 原始数据
        self.disease_targets = {}  # {disease_id: [(target_id, weight), ...]}
        self.formula_targets = {}  # {formula_id: [(target_id, weight), ...]}
        self.positive_pairs = []   # [(disease_id, formula_id), ...]

        # 处理后的数据
        self.processed_disease_targets = {}
        self.processed_formula_targets = {}
        self.training_samples = []

    def build_from_raw_data(self,
                          disease_targets_raw: Dict[str, List[Tuple[str, float]]],
                          formula_targets_raw: Dict[str, List[Tuple[str, float]]],
                          positive_pairs_raw: List[Tuple[str, str]]) -> None:
        """
        从原始数据构建数据集

        Args:
            disease_targets_raw: 疾病靶点集合
            formula_targets_raw: 方剂靶点集合
            positive_pairs_raw: 正样本对
        """
        logger.info("开始构建TCM-RX数据集...")

        # 存储原始数据
        self.disease_targets = disease_targets_raw
        self.formula_targets = formula_targets_raw
        self.positive_pairs = positive_pairs_raw

        # 构建ID映射
        self._build_id_mappings()

        # 处理靶点集合
        self._process_target_sets()

        # 组装训练样本
        self._build_training_samples()

        logger.info(f"数据集构建完成: {len(self.training_samples)} 个训练样本")

    def _build_id_mappings(self) -> None:
        """构建稳定的ID映射"""
        logger.info("构建ID映射...")

        # 收集所有实体ID
        all_diseases = set(self.disease_targets.keys())
        all_formulas = set(self.formula_targets.keys())
        all_targets = set()

        for targets in self.disease_targets.values():
            all_targets.update(tid for tid, _ in targets)

        for targets in self.formula_targets.values():
            all_targets.update(tid for tid, _ in targets)

        # 添加到映射器
        self.id_mapper.add_ids('disease', all_diseases)
        self.id_mapper.add_ids('formula', all_formulas)
        self.id_mapper.add_ids('target', all_targets)

        # 打印统计信息
        self.id_mapper.print_stats()

    def _process_target_sets(self) -> None:
        """处理靶点集合：过滤、加权、截断"""
        logger.info("处理靶点集合...")

        # 配置参数
        topk_d = self.config.get('filtering', {}).get('topk_d')
        topk_f = self.config.get('filtering', {}).get('topk_f')
        inverse_freq_weight = self.config.get('filtering', {}).get('inverse_freq_weight', True)

        # 处理疾病侧
        disease_targets_processed = self.disease_targets.copy()

        # 逆频重加权
        if inverse_freq_weight:
            logger.info("计算疾病侧逆频权重...")
            inverse_freq_weights = compute_inverse_freq(disease_targets_processed)
            disease_targets_processed = apply_inverse_freq_weights(
                disease_targets_processed, inverse_freq_weights)

        # Top-K截断
        if topk_d:
            disease_targets_processed = apply_topk_to_set(
                disease_targets_processed, topk_d, sort_by='weight')

        # 权重归一化
        disease_targets_processed = normalize_weights(disease_targets_processed)

        self.processed_disease_targets = disease_targets_processed

        # 处理方剂侧
        formula_targets_processed = self.formula_targets.copy()

        # Top-K截断
        if topk_f:
            formula_targets_processed = apply_topk_to_set(
                formula_targets_processed, topk_f, sort_by='weight')

        # 权重归一化
        formula_targets_processed = normalize_weights(formula_targets_processed)

        self.processed_formula_targets = formula_targets_processed

    def _build_training_samples(self) -> None:
        """构建训练样本（仅正样本，负采样在batch中生成）"""
        logger.info("构建训练样本...")

        valid_pairs = []

        for disease_id, formula_id in self.positive_pairs:
            # 确保两侧都有靶点信息
            if (disease_id in self.processed_disease_targets and
                formula_id in self.processed_formula_targets):

                # 确保靶点集合非空
                if (self.processed_disease_targets[disease_id] and
                    self.processed_formula_targets[formula_id]):

                    valid_pairs.append(TrainingSample(disease_id, formula_id, 1))

        self.training_samples = valid_pairs
        logger.info(f"有效正样本: {len(valid_pairs)} / {len(self.positive_pairs)} "
                   f"({len(valid_pairs)/len(self.positive_pairs)*100:.1f}%)")

    def get_disease_targets(self, disease_id: str) -> List[Tuple[int, float]]:
        """
        获取疾病的靶点列表（整数索引）

        Args:
            disease_id: 疾病ID

        Returns:
            [(target_index, weight), ...]
        """
        if disease_id not in self.processed_disease_targets:
            return []

        targets = self.processed_disease_targets[disease_id]
        target_indices = [self.id_mapper.get_index('target', tid) for tid, _ in targets]
        weights = [weight for _, weight in targets]

        return list(zip(target_indices, weights))

    def get_formula_targets(self, formula_id: str) -> List[Tuple[int, float]]:
        """
        获取方剂的靶点列表（整数索引）

        Args:
            formula_id: 方剂ID

        Returns:
            [(target_index, weight), ...]
        """
        if formula_id not in self.processed_formula_targets:
            return []

        targets = self.processed_formula_targets[formula_id]
        target_indices = [self.id_mapper.get_index('target', tid) for tid, _ in targets]
        weights = [weight for _, weight in targets]

        return list(zip(target_indices, weights))

    def get_entity_indices(self) -> Tuple[List[int], List[int], int]:
        """
        获取所有实体的整数索引

        Returns:
            (disease_indices, formula_indices, num_targets)
        """
        disease_indices = self.id_mapper.get_all_indices('disease')
        formula_indices = self.id_mapper.get_all_indices('formula')
        num_targets = self.id_mapper.get_count('target')

        return disease_indices, formula_indices, num_targets

    def get_sample_by_index(self, idx: int) -> Tuple[int, int, int]:
        """
        根据索引获取样本

        Args:
            idx: 样本索引

        Returns:
            (disease_index, formula_index, label)
        """
        if idx >= len(self.training_samples):
            raise IndexError(f"样本索引超出范围: {idx} >= {len(self.training_samples)}")

        sample = self.training_samples[idx]
        disease_idx = self.id_mapper.get_index('disease', sample.disease_id)
        formula_idx = self.id_mapper.get_index('formula', sample.formula_id)

        return disease_idx, formula_idx, sample.label

    def get_disease_targets_by_index(self, disease_idx: int) -> List[Tuple[int, float]]:
        """
        根据整数索引获取疾病靶点

        Args:
            disease_idx: 疾病整数索引

        Returns:
            [(target_index, weight), ...]
        """
        disease_id = self.id_mapper.get_string_id('disease', disease_idx)
        return self.get_disease_targets(disease_id)

    def get_formula_targets_by_index(self, formula_idx: int) -> List[Tuple[int, float]]:
        """
        根据整数索引获取方剂靶点

        Args:
            formula_idx: 方剂整数索引

        Returns:
            [(target_index, weight), ...]
        """
        formula_id = self.id_mapper.get_string_id('formula', formula_idx)
        return self.get_formula_targets(formula_id)

    def __len__(self) -> int:
        return len(self.training_samples)

    def __repr__(self) -> str:
        return (f"TCMRXDataset("
                f"diseases={self.id_mapper.get_count('disease')}, "
                f"formulas={self.id_mapper.get_count('formula')}, "
                f"targets={self.id_mapper.get_count('target')}, "
                f"samples={len(self.training_samples)})")