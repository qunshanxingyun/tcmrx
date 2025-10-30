"""
TCM-RX 数据划分模块
按"疾病"分层切分，保留冷启动评估
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def stratified_disease_split(disease_formula_pairs: List[Tuple[str, str]],
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           seed: int = 42) -> Dict[str, List[Tuple[str, str]]]:
    """
    按疾病分层切分数据

    Args:
        disease_formula_pairs: 疾病-方剂对列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        划分结果字典
    """
    # 检查比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为1.0，当前为{total_ratio}")

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 按疾病分组
    disease_to_pairs = {}
    for disease_id, formula_id in disease_formula_pairs:
        if disease_id not in disease_to_pairs:
            disease_to_pairs[disease_id] = []
        disease_to_pairs[disease_id].append((disease_id, formula_id))

    logger.info(f"总疾病数: {len(disease_to_pairs)}")
    logger.info(f"总样本数: {len(disease_formula_pairs)}")

    # 疾病列表
    diseases = list(disease_to_pairs.keys())
    random.shuffle(diseases)

    # 计算每个集合的疾病数量
    num_diseases = len(diseases)
    num_train_diseases = int(num_diseases * train_ratio)
    num_val_diseases = int(num_diseases * val_ratio)

    # 划分疾病
    train_diseases = set(diseases[:num_train_diseases])
    val_diseases = set(diseases[num_train_diseases:num_train_diseases + num_val_diseases])
    test_diseases = set(diseases[num_train_diseases + num_val_diseases:])

    # 根据疾病划分样本
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for disease_id, pairs in disease_to_pairs.items():
        if disease_id in train_diseases:
            train_pairs.extend(pairs)
        elif disease_id in val_diseases:
            val_pairs.extend(pairs)
        elif disease_id in test_diseases:
            test_pairs.extend(pairs)

    logger.info(f"疾病划分: 训练 {len(train_diseases)}, 验证 {len(val_diseases)}, 测试 {len(test_diseases)}")
    logger.info(f"样本划分: 训练 {len(train_pairs)}, 验证 {len(val_pairs)}, 测试 {len(test_pairs)}")

    return {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs,
        'train_diseases': train_diseases,
        'val_diseases': val_diseases,
        'test_diseases': test_diseases
    }


def identify_cold_start_diseases(disease_formula_pairs: List[Tuple[str, str]],
                                 disease_targets: Dict[str, List[Tuple[str, float]]],
                                 min_targets: int = 1) -> Tuple[Set[str], Set[str]]:
    """
    识别冷启动疾病（无分子信息的疾病）

    Args:
        disease_formula_pairs: 疾病-方剂对列表
        disease_targets: 疾病靶点字典
        min_targets: 最小靶点数量阈值

    Returns:
        (有分子信息的疾病集合, 冷启动疾病集合)
    """
    all_diseases = set(disease for disease, _ in disease_formula_pairs)
    diseases_with_targets = set(disease for disease in disease_targets.keys()
                              if len(disease_targets[disease]) >= min_targets)

    cold_start_diseases = all_diseases - diseases_with_targets

    logger.info(f"总疾病数: {len(all_diseases)}")
    logger.info(f"有靶点信息的疾病数: {len(diseases_with_targets)}")
    logger.info(f"冷启动疾病数: {len(cold_start_diseases)} "
               f"({len(cold_start_diseases)/len(all_diseases)*100:.1f}%)")

    return diseases_with_targets, cold_start_diseases


def create_cold_start_split(disease_formula_pairs: List[Tuple[str, str]],
                           cold_start_diseases: Set[str],
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           seed: int = 42) -> Dict[str, List[Tuple[str, str]]]:
    """
    为冷启动疾病创建特殊的划分

    Args:
        disease_formula_pairs: 疾病-方剂对列表
        cold_start_diseases: 冷启动疾病集合
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        划分结果字典
    """
    # 过滤出冷启动疾病的配对
    cold_start_pairs = [(d, f) for d, f in disease_formula_pairs if d in cold_start_diseases]

    if not cold_start_pairs:
        logger.warning("没有冷启动疾病配对")
        return {'cold_start_train': [], 'cold_start_val': [], 'cold_start_test': []}

    # 按疾病分组
    disease_to_pairs = {}
    for disease_id, formula_id in cold_start_pairs:
        if disease_id not in disease_to_pairs:
            disease_to_pairs[disease_id] = []
        disease_to_pairs[disease_id].append((disease_id, formula_id))

    # 随机划分疾病
    diseases = list(disease_to_pairs.keys())
    random.seed(seed)
    random.shuffle(diseases)

    num_diseases = len(diseases)
    num_train_diseases = int(num_diseases * train_ratio)
    num_val_diseases = int(num_diseases * val_ratio)

    train_diseases = set(diseases[:num_train_diseases])
    val_diseases = set(diseases[num_train_diseases:num_train_diseases + num_val_diseases])
    test_diseases = set(diseases[num_train_diseases + num_val_diseases:])

    # 划分样本
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for disease_id, pairs in disease_to_pairs.items():
        if disease_id in train_diseases:
            train_pairs.extend(pairs)
        elif disease_id in val_diseases:
            val_pairs.extend(pairs)
        elif disease_id in test_diseases:
            test_pairs.extend(pairs)

    logger.info(f"冷启动疾病划分: 训练 {len(train_diseases)}, 验证 {len(val_diseases)}, 测试 {len(test_diseases)}")
    logger.info(f"冷启动样本划分: 训练 {len(train_pairs)}, 验证 {len(val_pairs)}, 测试 {len(test_pairs)}")

    return {
        'cold_start_train': train_pairs,
        'cold_start_val': val_pairs,
        'cold_start_test': test_pairs
    }


def get_split_statistics(split_result: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Dict[str, int]]:
    """
    获取划分统计信息

    Args:
        split_result: 划分结果字典

    Returns:
        统计信息字典
    """
    stats = {}

    for split_name, pairs in split_result.items():
        if not pairs:
            continue

        diseases = set(d for d, _ in pairs)
        formulas = set(f for _, f in pairs)

        stats[split_name] = {
            'num_pairs': len(pairs),
            'num_diseases': len(diseases),
            'num_formulas': len(formulas)
        }

    return stats


def validate_split(split_result: Dict[str, List[Tuple[str, str]]],
                  min_samples_per_split: int = 10) -> bool:
    """
    验证划分结果的有效性

    Args:
        split_result: 划分结果字典
        min_samples_per_split: 每个划分的最小样本数

    Returns:
        是否有效
    """
    for split_name, pairs in split_result.items():
        if len(pairs) < min_samples_per_split:
            logger.error(f"划分 {split_name} 样本数过少: {len(pairs)} < {min_samples_per_split}")
            return False

        # 检查是否有重复
        if len(pairs) != len(set(pairs)):
            logger.error(f"划分 {split_name} 存在重复样本")
            return False

    # 检查划分之间是否有重叠
    main_splits = ['train', 'val', 'test']
    main_pairs = []
    for split_name in main_splits:
        if split_name in split_result:
            main_pairs.extend(split_result[split_name])

    if len(main_pairs) != len(set(main_pairs)):
        logger.error("主要划分之间存在重叠样本")
        return False

    logger.info("划分验证通过")
    return True