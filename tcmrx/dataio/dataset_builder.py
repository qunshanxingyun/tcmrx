"""
TCM-RX 数据集构建模块
组装 (Disease, Formula) 训练/验证样本
"""

from typing import Dict, List, Tuple, Set
import numpy as np
import logging
import math
from collections import defaultdict

from .id_maps import IDMapper
from .filters import (
    apply_topk_to_set,
    compute_inverse_freq,
    apply_inverse_freq_weights,
    normalize_weights,
    trim_target_sets,
    compute_frequency_weights,
    apply_frequency_weights,
    penalize_common_targets,
)

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
        self.raw_positive_pairs: List[Tuple[str, str]] = []
        self.positive_pairs: List[Tuple[str, str]] = []

        # 处理后的数据
        self.processed_disease_targets = {}
        self.processed_formula_targets = {}
        self.training_samples = []

        seed = config.get('training', {}).get('seed', 42)
        self.rng = np.random.default_rng(seed)
        self.split_name = 'train'

    def build_from_raw_data(self,
                          disease_targets_raw: Dict[str, List[Tuple[str, float]]],
                          formula_targets_raw: Dict[str, List[Tuple[str, float]]],
                          positive_pairs_raw: List[Tuple[str, str]],
                          split_name: str = 'train') -> None:
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
        self.raw_positive_pairs = list(positive_pairs_raw)
        self.split_name = split_name
        self.positive_pairs = self._rebalance_positive_pairs(self.raw_positive_pairs, split_name)

        if len(self.positive_pairs) != len(self.raw_positive_pairs):
            logger.info(
                "正样本重平衡(%s): %d -> %d",
                split_name,
                len(self.raw_positive_pairs),
                len(self.positive_pairs),
            )

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

        filtering_cfg = self.config.get('filtering', {})
        topk_d = filtering_cfg.get('topk_d')
        topk_f = filtering_cfg.get('topk_f')
        inverse_freq_weight = filtering_cfg.get('inverse_freq_weight', True)

        disease_trim_cfg = self._resolve_side_config(
            filtering_cfg.get('disease_target_trimming'),
            'disease'
        )
        formula_trim_cfg = self._resolve_side_config(
            filtering_cfg.get('formula_target_trimming'),
            'formula'
        )

        penalty_cfg = filtering_cfg.get('common_target_penalty')
        disease_penalty_cfg = self._resolve_side_config(penalty_cfg, 'disease')
        formula_penalty_cfg = self._resolve_side_config(penalty_cfg, 'formula')

        freq_cfg = filtering_cfg.get('frequency_reweighting')
        disease_freq_cfg = self._resolve_side_config(freq_cfg, 'disease')
        formula_freq_cfg = self._resolve_side_config(freq_cfg, 'formula')

        # 处理疾病侧
        disease_targets_processed = self.disease_targets.copy()

        # 逆频重加权
        if inverse_freq_weight:
            logger.info("计算疾病侧逆频权重...")
            inverse_freq_weights = compute_inverse_freq(disease_targets_processed)
            disease_targets_processed = apply_inverse_freq_weights(
                disease_targets_processed, inverse_freq_weights)

        if disease_freq_cfg:
            disease_targets_processed = self._apply_frequency_weighting(
                disease_targets_processed,
                disease_freq_cfg,
                '疾病侧',
            )

        if disease_penalty_cfg:
            disease_targets_processed = self._apply_common_target_penalty(
                disease_targets_processed,
                disease_penalty_cfg,
                '疾病侧',
            )

        if disease_trim_cfg:
            disease_targets_processed = self._apply_trimming(
                disease_targets_processed,
                disease_trim_cfg,
                '疾病侧',
            )
        elif topk_d:
            disease_targets_processed = apply_topk_to_set(
                disease_targets_processed, topk_d, sort_by='weight')

        # 权重归一化
        disease_targets_processed = normalize_weights(disease_targets_processed)

        self.processed_disease_targets = disease_targets_processed

        # 处理方剂侧
        formula_targets_processed = self.formula_targets.copy()

        if formula_freq_cfg:
            formula_targets_processed = self._apply_frequency_weighting(
                formula_targets_processed,
                formula_freq_cfg,
                '方剂侧',
            )

        if formula_penalty_cfg:
            formula_targets_processed = self._apply_common_target_penalty(
                formula_targets_processed,
                formula_penalty_cfg,
                '方剂侧',
            )

        if formula_trim_cfg:
            formula_targets_processed = self._apply_trimming(
                formula_targets_processed,
                formula_trim_cfg,
                '方剂侧',
            )
        elif topk_f:
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

    @staticmethod
    def _resolve_side_config(config_value, side: str):
        if config_value is None:
            return None

        if isinstance(config_value, bool):
            return {} if config_value else None

        if not isinstance(config_value, dict):
            return config_value

        if side in config_value:
            return config_value.get(side)

        has_side_keys = any(k in config_value for k in ('disease', 'formula'))
        if has_side_keys:
            return None

        return config_value

    def _apply_frequency_weighting(self, target_sets, freq_cfg, prefix: str):
        if not freq_cfg:
            return target_sets

        if isinstance(freq_cfg, bool):
            if not freq_cfg:
                return target_sets
            freq_cfg = {}

        if freq_cfg.get('enabled', True) is False:
            logger.info("%s跳过频率重加权（配置禁用）", prefix)
            return target_sets

        freq_weights = compute_frequency_weights(
            target_sets,
            method=freq_cfg.get('method', 'idf'),
            smooth=freq_cfg.get('smooth', 1.0),
            power=freq_cfg.get('power', 1.0),
            min_weight=freq_cfg.get('min_weight'),
            max_weight=freq_cfg.get('max_weight'),
            base=freq_cfg.get('base', math.e),
        )

        blend = freq_cfg.get('blend', 1.0)
        normalize = freq_cfg.get('normalize', False)

        logger.info("%s应用频率重加权: method=%s, blend=%.2f", prefix, freq_cfg.get('method', 'idf'), blend)
        return apply_frequency_weights(target_sets, freq_weights, blend=blend, normalize=normalize)

    def _apply_trimming(self, target_sets, trim_cfg, prefix: str):
        if not trim_cfg:
            return target_sets

        if isinstance(trim_cfg, bool):
            return target_sets if not trim_cfg else trim_target_sets(target_sets)

        if trim_cfg.get('enabled', True) is False:
            logger.info("%s跳过靶点裁剪（配置禁用）", prefix)
            return target_sets

        params = {
            'max_items': trim_cfg.get('max_items'),
            'min_items': trim_cfg.get('min_items', 0),
            'mass_threshold': trim_cfg.get('mass_threshold'),
            'weight_floor': trim_cfg.get('weight_floor'),
            'sort_by': trim_cfg.get('sort_by', 'weight'),
            'log_prefix': f"{prefix}",
        }

        logger.info("%s靶点裁剪: %s", prefix, {k: v for k, v in params.items() if v is not None})
        return trim_target_sets(target_sets, **params)

    def _apply_common_target_penalty(self, target_sets, penalty_cfg, prefix: str):
        if not penalty_cfg:
            return target_sets

        if isinstance(penalty_cfg, bool):
            return penalize_common_targets(target_sets) if penalty_cfg else target_sets

        if penalty_cfg.get('enabled', True) is False:
            logger.info("%s跳过高频靶点惩罚（配置禁用）", prefix)
            return target_sets

        params = {
            'top_n': penalty_cfg.get('top_n', 100),
            'multiplier': penalty_cfg.get('multiplier', 0.1),
            'min_frequency': penalty_cfg.get('min_frequency', 1),
            'log_prefix': f"{prefix}高频惩罚",
        }
        return penalize_common_targets(target_sets, **params)

    def _rebalance_positive_pairs(self, positive_pairs, split_name: str):
        sampling_cfg = self.config.get('sampling', {}).get('disease_pair_balancing', {})
        if not sampling_cfg or sampling_cfg.get('enabled', True) is False:
            return list(positive_pairs)

        apply_to = sampling_cfg.get('apply_to', ['train'])
        if split_name not in apply_to:
            return list(positive_pairs)

        positive_pairs = list(positive_pairs)
        if not positive_pairs:
            return []

        high_threshold = sampling_cfg.get('high_freq_threshold', 200)
        low_threshold = sampling_cfg.get('low_freq_threshold', 20)
        max_high_pairs = sampling_cfg.get('max_pairs_high', 100)
        low_multiplier = max(int(sampling_cfg.get('upsample_low_multiplier', 3)), 1)
        medium_multiplier = max(int(sampling_cfg.get('upsample_medium_multiplier', 1)), 1)

        disease_to_formulas: Dict[str, List[str]] = defaultdict(list)
        for disease_id, formula_id in positive_pairs:
            disease_to_formulas[disease_id].append(formula_id)

        rebalanced_pairs: List[Tuple[str, str]] = []
        high_freq_diseases = 0
        low_freq_diseases = 0

        for disease_id, formulas in disease_to_formulas.items():
            count = len(formulas)
            formulas_array = list(formulas)
            self.rng.shuffle(formulas_array)

            if count >= high_threshold:
                take = min(count, max_high_pairs)
                selected = formulas_array[:take]
                rebalanced_pairs.extend((disease_id, fid) for fid in selected)
                high_freq_diseases += 1
            elif count < low_threshold:
                for _ in range(low_multiplier):
                    rebalanced_pairs.extend((disease_id, fid) for fid in formulas_array)
                low_freq_diseases += 1
            else:
                for _ in range(medium_multiplier):
                    rebalanced_pairs.extend((disease_id, fid) for fid in formulas_array)

        logger.info(
            "重平衡统计(%s): 高频疾病=%d, 低频疾病=%d, 原样本=%d, 新样本=%d",
            split_name,
            high_freq_diseases,
            low_freq_diseases,
            len(positive_pairs),
            len(rebalanced_pairs),
        )

        return rebalanced_pairs

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