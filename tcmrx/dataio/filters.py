"""
TCM-RX 数据过滤模块
可选过滤/加权功能，默认可不启用
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)


def filter_sd1_by_pki(sd1_df: pd.DataFrame, threshold: float = 6.0) -> pd.DataFrame:
    """
    按预测亲和力阈值过滤SD1

    Args:
        sd1_df: SD1预测表
        threshold: pKi阈值

    Returns:
        过滤后的DataFrame
    """
    if threshold is None:
        logger.info("跳过pKi阈值过滤")
        return sd1_df

    # 确保predicted_binding_affinity是数值类型
    sd1_df = sd1_df.copy()
    sd1_df['predicted_binding_affinity'] = pd.to_numeric(
        sd1_df['predicted_binding_affinity'],
        errors='coerce'
    ).fillna(threshold - 1)  # 无效数据设为小于阈值的值

    before_count = len(sd1_df)
    sd1_df = sd1_df[sd1_df['predicted_binding_affinity'] >= threshold].copy()
    after_count = len(sd1_df)

    logger.info(f"pKi阈值过滤 ({threshold}): {before_count} -> {after_count} "
                f"({after_count/before_count*100:.1f}% 保留)")
    return sd1_df


def per_chemical_topk(chemical_to_targets: Dict[str, List[Tuple[str, float]]],
                      k: int = 15) -> Dict[str, List[Tuple[str, float]]]:
    """
    每个化合物选择前k个靶点

    Args:
        chemical_to_targets: 化合物->靶点映射
        k: 每个化合物保留的靶点数量

    Returns:
        过滤后的映射
    """
    if k is None:
        logger.info("跳过每化合物TopK过滤")
        return chemical_to_targets

    result = {}
    total_before = 0
    total_after = 0

    for chemical, targets in chemical_to_targets.items():
        total_before += len(targets)
        # 按亲和力降序排序，取前k个
        sorted_targets = sorted(targets, key=lambda x: x[1], reverse=True)[:k]
        result[chemical] = sorted_targets
        total_after += len(sorted_targets)

    logger.info(f"每化合物Top-{k}: {total_before} -> {total_after} "
                f"({total_after/total_before*100:.1f}% 保留)")
    return result


def apply_topk_to_set(target_sets: Dict[str, List[Tuple[str, float]]],
                    top_k: Optional[int] = None,
                    sort_by: str = 'weight') -> Dict[str, List[Tuple[str, float]]]:
    """
    对疾病/方剂靶点集合进行Top-K截断

    Args:
        target_sets: {entity: [(target_id, weight), ...]}
        top_k: 保留的靶点数量
        sort_by: 排序依据 ('weight' 或 'random')

    Returns:
        截断后的映射
    """
    if top_k is None:
        logger.info("跳过集合TopK截断")
        return target_sets

    result = {}
    total_before = 0
    total_after = 0

    for entity, targets in target_sets.items():
        total_before += len(targets)

        if sort_by == 'weight':
            # 按权重降序排序
            sorted_targets = sorted(targets, key=lambda x: x[1], reverse=True)
        else:
            # 随机排序
            sorted_targets = targets.copy()
            np.random.shuffle(sorted_targets)

        result[entity] = sorted_targets[:top_k]
        total_after += len(sorted_targets)

    logger.info(f"集合Top-{top_k}: {total_before} -> {total_after} "
                f"({total_after/total_before*100:.1f}% 保留)")
    return result


def compute_inverse_freq(target_sets: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
    """
    计算靶点逆频权重

    Args:
        target_sets: {entity: [(target_id, weight), ...]}

    Returns:
        {target_id: inverse_frequency_weight}
    """
    # 统计每个靶点出现的频率
    target_counter = Counter()
    for targets in target_sets.values():
        for target_id, _ in targets:
            target_counter[target_id] += 1

    # 计算逆频权重: w = 1 / log(1 + freq)
    inverse_freq_weights = {}
    for target_id, freq in target_counter.items():
        inverse_freq_weights[target_id] = 1.0 / math.log(1 + freq)

    logger.info(f"计算了 {len(inverse_freq_weights)} 个靶点的逆频权重 "
                f"(min: {min(inverse_freq_weights.values()):.4f}, "
                f"max: {max(inverse_freq_weights.values()):.4f})")
    return inverse_freq_weights


def apply_inverse_freq_weights(target_sets: Dict[str, List[Tuple[str, float]]],
                            inverse_freq_weights: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
    """
    应用逆频重加权

    Args:
        target_sets: {entity: [(target_id, weight), ...]}
        inverse_freq_weights: {target_id: weight}

    Returns:
        重加权后的映射
    """
    result = {}
    for entity, targets in target_sets.items():
        weighted_targets = []
        for target_id, original_weight in targets:
            inv_freq_weight = inverse_freq_weights.get(target_id, 1.0)
            # 最终权重 = 原始权重 * 逆频权重
            final_weight = original_weight * inv_freq_weight
            weighted_targets.append((target_id, final_weight))
        result[entity] = weighted_targets

    logger.info("应用逆频重加权完成")
    return result


def dosage_softmax(dosages: List[float], temperature: float = 1.0) -> List[float]:
    """
    对剂量比例进行softmax

    Args:
        dosages: 剂量比例列表
        temperature: softmax温度

    Returns:
        softmax后的权重列表
    """
    if temperature is None or temperature <= 0:
        # 归一化到[0,1]
        total = sum(dosages)
        if total > 0:
            return [d / total for d in dosages]
        else:
            return [1.0 / len(dosages)] * len(dosages)

    # 计算softmax
    exp_dosages = [np.exp(d / temperature) for d in dosages]
    total = sum(exp_dosages)
    if total > 0:
        return [d / total for d in exp_dosages]
    else:
        return [1.0 / len(dosages)] * len(dosages)


def normalize_weights(target_sets: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    对每个实体内的权重进行L1归一化

    Args:
        target_sets: {entity: [(target_id, weight), ...]}

    Returns:
        归一化后的映射
    """
    result = {}
    for entity, targets in target_sets.items():
        if not targets:
            result[entity] = targets
            continue

        weights = [weight for _, weight in targets]
        total_weight = sum(weights)

        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            result[entity] = list(zip([tid for tid, _ in targets], normalized_weights))
        else:
            # 如果权重全为0，赋予均匀权重
            uniform_weight = 1.0 / len(targets)
            result[entity] = [(tid, uniform_weight) for tid, _ in targets]

    logger.info("权重归一化完成")
    return result