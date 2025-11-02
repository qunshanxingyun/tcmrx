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

        trimmed = sorted_targets[:top_k]
        result[entity] = trimmed
        total_after += len(trimmed)

    logger.info(f"集合Top-{top_k}: {total_before} -> {total_after} "
                f"({(total_after/total_before*100) if total_before else 0:.1f}% 保留)")
    return result


def _maybe_shuffle(targets: List[Tuple[str, float]], sort_by: str) -> List[Tuple[str, float]]:
    if sort_by == 'weight':
        return sorted(targets, key=lambda x: x[1], reverse=True)

    shuffled = targets.copy()
    np.random.shuffle(shuffled)
    return shuffled


def trim_target_sets(
    target_sets: Dict[str, List[Tuple[str, float]]],
    max_items: Optional[int] = None,
    min_items: int = 0,
    mass_threshold: Optional[float] = None,
    weight_floor: Optional[float] = None,
    sort_by: str = 'weight',
    log_prefix: str = 'trim'
) -> Dict[str, List[Tuple[str, float]]]:
    """根据多种条件裁剪实体的靶点集合。

    该方法支持以下裁剪策略：

    * ``max_items``：上限保留的靶点数量；
    * ``min_items``：保证至少保留的数量；
    * ``mass_threshold``：按照权重降序累积达到指定质量阈值后停止；
    * ``weight_floor``：过滤掉权重低于阈值的靶点；
    * ``sort_by``：控制裁剪前的排序方式（``weight`` / ``random``）。

    参数之间按 ``weight_floor`` -> ``mass_threshold`` -> ``max_items`` 顺序依次应用，
    并确保最终结果不少于 ``min_items``。
    """

    if not any([max_items, min_items, mass_threshold, weight_floor]) and sort_by == 'weight':
        logger.info("跳过靶点裁剪（未启用任何约束）")
        return target_sets

    result: Dict[str, List[Tuple[str, float]]] = {}
    total_before = 0
    total_after = 0

    for entity, targets in target_sets.items():
        total_before += len(targets)

        if not targets:
            result[entity] = targets
            continue

        ordered = _maybe_shuffle(targets, sort_by)

        if weight_floor is not None:
            filtered = [pair for pair in ordered if pair[1] >= weight_floor]
            if min_items and len(filtered) < min_items:
                filtered = ordered[:min_items]
        else:
            filtered = ordered

        if mass_threshold is not None and filtered:
            total_weight = sum(max(weight, 0.0) for _, weight in filtered)
            if total_weight > 0:
                cumulative = 0.0
                mass_cut = []
                for idx, (target_id, weight) in enumerate(filtered):
                    mass_cut.append((target_id, weight))
                    cumulative += max(weight, 0.0)
                    if idx + 1 < min_items:
                        continue
                    if cumulative / total_weight >= mass_threshold:
                        break
                filtered = mass_cut

        if max_items is not None and len(filtered) > max_items:
            filtered = filtered[:max_items]

        if min_items and len(filtered) < min_items:
            filtered = ordered[:min(min_items, len(ordered))]

        result[entity] = filtered
        total_after += len(filtered)

    logger.info(
        f"{log_prefix}裁剪: {total_before} -> {total_after} "
        f"({(total_after/total_before*100) if total_before else 0:.1f}% 保留)"
    )
    return result


def compute_inverse_freq(target_sets: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
    """
    保留旧接口，调用 :func:`compute_frequency_weights` 获取 ``1/log`` 逆频权重。
    """

    return compute_frequency_weights(
        target_sets,
        method='log_reciprocal',
        smooth=1.0,
        power=1.0,
    )


def compute_frequency_weights(
    target_sets: Dict[str, List[Tuple[str, float]]],
    method: str = 'idf',
    smooth: float = 1.0,
    power: float = 1.0,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    base: float = math.e,
) -> Dict[str, float]:
    """计算基于频率的重加权系数。

    Parameters
    ----------
    method
        ``'idf'`` -> ``((N + smooth) / (freq + smooth)) ** power``;
        ``'log_idf'`` -> ``log((N + smooth)/(freq + smooth), base) ** power``;
        ``'log_reciprocal'`` -> ``1 / log(freq + smooth) ** power``；
        ``'reciprocal'`` -> ``1 / (freq + smooth) ** power``。
    smooth
        防止除零/取对数时出现极端值。
    power
        控制放大程度。
    min_weight/max_weight
        对结果进行裁剪以避免极端比率。
    base
        ``log_idf`` 模式下使用的对数底。
    """

    if not target_sets:
        return {}

    target_counter = Counter()
    for targets in target_sets.values():
        for target_id, _ in targets:
            target_counter[target_id] += 1

    total_entities = max(len(target_sets), 1)
    weights = {}

    for target_id, freq in target_counter.items():
        freq = max(freq, 1)
        if method == 'idf':
            raw = ((total_entities + smooth) / (freq + smooth)) ** power
        elif method == 'log_idf':
            ratio = max((total_entities + smooth) / (freq + smooth), 1.0)
            raw = math.log(ratio, base) ** power if ratio > 1.0 else 1.0
        elif method == 'log_reciprocal':
            raw = 1.0 / (math.log(freq + smooth, base) ** power)
        elif method == 'reciprocal':
            raw = 1.0 / ((freq + smooth) ** power)
        else:
            raise ValueError(f"未知的频率加权方法: {method}")

        if min_weight is not None:
            raw = max(min_weight, raw)
        if max_weight is not None:
            raw = min(max_weight, raw)

        weights[target_id] = raw

    if weights:
        logger.info(
            "频率权重: %d 个靶点 (min=%.4f, max=%.4f)",
            len(weights),
            min(weights.values()),
            max(weights.values()),
        )
    return weights


def apply_inverse_freq_weights(target_sets: Dict[str, List[Tuple[str, float]]],
                            inverse_freq_weights: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
    """向后兼容的包装器。"""

    return apply_frequency_weights(target_sets, inverse_freq_weights)


def apply_frequency_weights(
    target_sets: Dict[str, List[Tuple[str, float]]],
    freq_weights: Dict[str, float],
    blend: float = 1.0,
    normalize: bool = False,
) -> Dict[str, List[Tuple[str, float]]]:
    """应用频率重加权，同时支持幂次混合与可选归一化。"""

    if not freq_weights:
        logger.info("跳过频率重加权（未提供权重表）")
        return target_sets

    result: Dict[str, List[Tuple[str, float]]] = {}

    for entity, targets in target_sets.items():
        weighted_targets = []
        for target_id, original_weight in targets:
            freq_weight = freq_weights.get(target_id, 1.0)
            adjusted = original_weight * (freq_weight ** blend)
            weighted_targets.append((target_id, adjusted))

        if normalize and weighted_targets:
            weights = [w for _, w in weighted_targets]
            total = sum(weights)
            if total > 0:
                weighted_targets = [
                    (tid, w / total) for tid, w in weighted_targets
                ]

        result[entity] = weighted_targets

    logger.info("频率重加权应用完成")
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