#!/usr/bin/env python3
"""
测试对比学习机制是否正常工作
"""

import sys
import os
from pathlib import Path
import logging
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataio.readers import TSVReader
from dataio.joins import *
from dataio.filters import *
from dataio.dataset_builder import TCMRXDataset
from models.twin_tower import DualTowerModel
from core.batching import create_data_loaders
from core.utils import load_config, get_device, setup_logging, set_random_seed
from core.losses import InfoNCELoss, compute_contrastive_metrics

logger = logging.getLogger(__name__)


def test_contrastive_learning():
    """测试对比学习机制"""
    setup_logging(level="INFO")
    logger.info("开始测试对比学习机制...")

    # 加载配置
    config = load_config("config/default.yaml")
    paths_config = load_config("config/paths.yaml")

    # 设置随机种子
    set_random_seed(config['training']['seed'])

    # 读取少量数据进行测试
    logger.info("读取数据...")
    reader = TSVReader(paths_config)

    formula_tables = reader.read_formula_tables()
    disease_tables = reader.read_disease_tables()
    prediction_tables = reader.read_prediction_tables()

    # 数据连接
    cpms_to_chp_map = cpms_to_chp(formula_tables['D4_CPM_CHP'])
    chp_to_chemicals_map = chp_to_chemicals(formula_tables['D9_CHP_InChIKey'], formula_tables.get('D12_InChIKey'))

    # 限制化学数据量
    sd1_df = prediction_tables['SD1_predicted'].sample(n=50000, random_state=42)
    chemical_to_targets_map = chemicals_to_targets(sd1_df)

    icd11_to_targets_map = icd11_to_targets(
        disease_tables['D19_ICD11_CUI'],
        disease_tables['D20_ICD11_MeSH'],
        disease_tables['D22_CUI_targets'],
        disease_tables['D23_MeSH_targets']
    )

    # 构建靶点集合
    formula_targets_raw = formulas_to_targets(cpms_to_chp_map, chp_to_chemicals_map, chemical_to_targets_map)
    disease_targets_raw = diseases_to_targets(icd11_to_targets_map)

    # 获取监督对并限制数量
    cpms_to_icd11_map = cpms_to_icd11(formula_tables['D5_CPM_ICD11'])
    positive_pairs_raw = [(icd11, cpm) for cpm, icd11_list in cpms_to_icd11_map.items() for icd11 in icd11_list]

    # 只取100个样本进行快速测试
    import random
    random.seed(42)
    positive_pairs_raw = random.sample(positive_pairs_raw, min(100, len(positive_pairs_raw)))

    logger.info(f"数据准备完成: {len(formula_targets_raw)} 方剂, {len(disease_targets_raw)} 疾病, {len(positive_pairs_raw)} 监督对")

    # 构建数据集
    dataset = TCMRXDataset(config)
    dataset.build_from_raw_data(disease_targets_raw, formula_targets_raw, positive_pairs_raw)

    logger.info(f"数据集构建完成: {dataset}")

    # 创建模型
    device = get_device("auto")
    disease_indices, formula_indices, num_targets = dataset.get_entity_indices()

    model_config = {
        'embedding_dim': 256,
        'dropout_rate': 0.1,
        'temperature': 0.1,
        'aggregator_type': 'attention'
    }
    model = DualTowerModel(model_config)
    model.set_entity_counts(len(disease_indices), len(formula_indices), num_targets)
    model = model.to(device)

    # 创建数据加载器
    train_loader, _ = create_data_loaders(
        dataset, None,
        batch_size=8,  # 更小的批次，确保有数据
        num_workers=0,
        pin_memory=True
    )

    logger.info(f"数据加载器创建完成: {len(train_loader.dataset)} 样本")

    # 测试前向传播和损失计算
    model.eval()
    loss_fn = InfoNCELoss(temperature=0.1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            # 移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(batch)
            disease_embeddings = outputs['disease_embeddings']
            formula_embeddings = outputs['formula_embeddings']
            similarities = outputs['scaled_similarities']

            logger.info(f"批次 {batch_idx + 1}:")
            logger.info(f"  疾病嵌入形状: {disease_embeddings.shape}")
            logger.info(f"  方剂嵌入形状: {formula_embeddings.shape}")
            logger.info(f"  相似度矩阵形状: {similarities.shape}")

            # 计算损失（使用嵌入而不是预计算的相似度）
            loss = loss_fn(disease_embeddings, formula_embeddings)

            # 计算准确率
            labels = torch.arange(len(similarities), device=similarities.device)
            predicted = similarities.argmax(dim=1)
            accuracy = (predicted == labels).float().mean()

            # 计算对比学习指标
            metrics = compute_contrastive_metrics(
                outputs['disease_embeddings'],
                outputs['formula_embeddings'],
                temperature=0.1
            )

            logger.info(f"  相似度范围: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
            logger.info(f"  对角线相似度: {torch.diag(similarities).mean().item():.4f}")
            logger.info(f"  非对角线相似度: {similarities[~torch.eye(len(similarities), dtype=bool)].mean().item():.4f}")
            logger.info(f"  损失: {loss.item():.4f}")
            logger.info(f"  准确率: {accuracy.item():.4f}")
            logger.info(f"  对比学习准确率: {metrics['accuracy']:.4f}")

            # 只测试第一个批次
            break

    logger.info("对比学习机制测试完成!")


if __name__ == "__main__":
    test_contrastive_learning()