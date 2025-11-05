#!/usr/bin/env python3
"""
TCM-RX 完整性检查脚本
读少量TSV→跑一小批前向，验证"能跑通"
"""

import sys
import os
from pathlib import Path
import logging
import time
import torch
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataio.readers import TSVReader
from dataio.joins import *
from dataio.filters import *
from dataio.dataset_builder import TCMRXDataset
from models.twin_tower import DualTowerModel
from core.batching import create_data_loaders, collate_fn
from core.utils import load_config, get_device, setup_logging, set_random_seed
from training.splits import stratified_disease_split

logger = logging.getLogger(__name__)


def run_sanity_check(config_path: str = "config/default.yaml",
                     paths_path: str = "config/paths.yaml",
                     max_samples: int = 200,  # 增加样本数确保有足够数据
                     max_cpms: int = 1000,    # 增加CPM数量
                     max_diseases: int = 500):  # 增加疾病数量
    """
    运行完整性检查

    Args:
        config_path: 配置文件路径
        paths_path: 路径配置文件路径
        max_samples: 最大样本数（用于快速测试）
        max_cpms: 最大CPM数量（内存限制）
        max_diseases: 最大疾病数量（内存限制）
    """
    # 设置日志
    setup_logging(level="INFO", experiment_name="sanity_check")
    logger.info("开始TCM-RX完整性检查...")

    try:
        # 1. 加载配置
        logger.info("1. 加载配置...")
        config = load_config(config_path)
        paths_config = load_config(paths_path)

        # 设置随机种子
        set_random_seed(config['training']['seed'])

        # 2. 读取数据
        logger.info("2. 读取数据...")
        reader = TSVReader(paths_config)

        # 读取各表
        formula_tables = reader.read_formula_tables()
        disease_tables = reader.read_disease_tables()
        prediction_tables = reader.read_prediction_tables()

        logger.info(f"成功读取: 方剂表 {len(formula_tables)}, 疾病表 {len(disease_tables)}, 预测表 {len(prediction_tables)}")

        # 3. 数据连接
        logger.info("3. 数据连接...")

        # 方剂侧连接
        cpms_to_chp_map = cpms_to_chp(formula_tables['D4_CPM_CHP'])
        d12_table = formula_tables.get('D12_InChIKey')
        chp_to_chemicals_map = chp_to_chemicals(formula_tables['D9_CHP_InChIKey'], d12_table)
        chemical_to_pathways_map = chemicals_to_pathways(d12_table)

        # 限制chemical-to-targets的数据量（避免内存溢出）
        sd1_df = prediction_tables['SD1_predicted']
        max_chemical_rows = min(100000, len(sd1_df))  # 最多10万行预测数据
        if len(sd1_df) > max_chemical_rows:
            logger.info(f"限制化学-靶点预测数据：{len(sd1_df)} -> {max_chemical_rows} 行")
            sd1_df = sd1_df.sample(n=max_chemical_rows, random_state=42)

        chemical_to_targets_map = chemicals_to_targets(
            sd1_df,
            prediction_tables.get('D13_InChIKey_EntrezID')
        )

        pathway_config = config.get('pathways', {})
        target_to_pathways_map = build_target_to_pathways(
            chemical_to_targets_map,
            chemical_to_pathways_map,
            prefix=pathway_config.get('prefix', 'pathway:'),
            max_pathways_per_target=pathway_config.get('bridge', {}).get('max_pathways_per_target', 32),
            min_weight=pathway_config.get('bridge', {}).get('min_weight', 1e-4),
        )

        # 疾病侧连接
        icd11_to_targets_map = icd11_to_targets(
            disease_tables['D19_ICD11_CUI'],
            disease_tables['D20_ICD11_MeSH'],
            disease_tables['D22_CUI_targets'],
            disease_tables['D23_MeSH_targets']
        )

        # 构建方剂靶点集合
        logger.info("构建方剂靶点集合...")
        formula_targets_raw = formulas_to_targets(
            cpms_to_chp_map,
            chp_to_chemicals_map,
            chemical_to_targets_map,
            chemical_to_pathways_map=chemical_to_pathways_map,
            pathway_config=pathway_config,
        )

        # 构建疾病靶点集合
        logger.info("构建疾病靶点集合...")
        disease_targets_raw = diseases_to_targets(
            icd11_to_targets_map,
            target_to_pathways_map=target_to_pathways_map,
            pathway_config=pathway_config,
        )

        # 获取监督对
        cpms_to_icd11_map = cpms_to_icd11(formula_tables['D5_CPM_ICD11'])
        positive_pairs_raw = [(icd11, cpm) for cpm, icd11_list in cpms_to_icd11_map.items() for icd11 in icd11_list]

        logger.info(f"连接完成: {len(formula_targets_raw)} 方剂, {len(disease_targets_raw)} 疾病, {len(positive_pairs_raw)} 正样本对")

        # 先对监督对进行采样，确保数据一致性
        if max_samples and len(positive_pairs_raw) > max_samples:
            logger.info(f"限制监督对数量: {len(positive_pairs_raw)} -> {max_samples}")
            import random
            random.seed(config['training']['seed'])
            positive_pairs_raw = random.sample(positive_pairs_raw, max_samples)

        # 根据采样的监督对确定需要保留的CPM和疾病
        sampled_cpms = set(cpm for _, cpm in positive_pairs_raw)
        sampled_diseases = set(icd11 for icd11, _ in positive_pairs_raw)

        # 过滤数据以匹配采样的监督对
        logger.info(f"根据监督对过滤数据: CPM {len(cpms_to_chp_map)} -> {len(sampled_cpms)}")
        cpms_to_chp_map = {cpm: data for cpm, data in cpms_to_chp_map.items() if cpm in sampled_cpms}

        logger.info(f"根据监督对过滤数据: 疾病 {len(icd11_to_targets_map)} -> {len(sampled_diseases)}")
        icd11_to_targets_map = {icd11: data for icd11, data in icd11_to_targets_map.items() if icd11 in sampled_diseases}

        # 4. 过滤处理（如果启用）
        logger.info("4. 应用过滤...")
        filtering_config = config.get('filtering', {})

        # SD1过滤（使用已采样的sd1_df）
        if filtering_config.get('pki_threshold'):
            sd1_df = filter_sd1_by_pki(sd1_df, filtering_config['pki_threshold'])
            # 重新构建化学-靶点映射
            chemical_to_targets_map = chemicals_to_targets(
                sd1_df,
                prediction_tables.get('D13_InChIKey_EntrezID')
            )
            target_to_pathways_map = build_target_to_pathways(
                chemical_to_targets_map,
                chemical_to_pathways_map,
                prefix=pathway_config.get('prefix', 'pathway:'),
                max_pathways_per_target=pathway_config.get('bridge', {}).get('max_pathways_per_target', 32),
                min_weight=pathway_config.get('bridge', {}).get('min_weight', 1e-4),
            )
            formula_targets_raw = formulas_to_targets(
                cpms_to_chp_map,
                chp_to_chemicals_map,
                chemical_to_targets_map,
                chemical_to_pathways_map=chemical_to_pathways_map,
                pathway_config=pathway_config,
            )
            disease_targets_raw = diseases_to_targets(
                icd11_to_targets_map,
                target_to_pathways_map=target_to_pathways_map,
                pathway_config=pathway_config,
            )

        # 6. 构建数据集
        logger.info("6. 构建数据集...")
        logger.info(f"输入数据统计: 方剂靶点 {len(formula_targets_raw)}, 疾病靶点 {len(disease_targets_raw)}, 监督对 {len(positive_pairs_raw)}")

        dataset = TCMRXDataset(config)
        dataset.build_from_raw_data(disease_targets_raw, formula_targets_raw, positive_pairs_raw, split_name='train')

        logger.info(f"数据集构建完成: {dataset}")
        logger.info(f"训练样本数量: {len(dataset.training_samples) if hasattr(dataset, 'training_samples') else '未知'}")

        # 7. 创建模型
        logger.info("7. 创建模型...")
        device = get_device(config['training']['device'])

        # 设置实体数量
        disease_indices, formula_indices, num_targets = dataset.get_entity_indices()
        model_config = {
            'embedding_dim': config['model']['embedding_dim'],
            'dropout_rate': config['model']['dropout_rate'],
            'temperature': config['model']['temperature'],
            'aggregator_type': config['model']['aggregator_type']
        }
        model = DualTowerModel(model_config)
        model.set_entity_counts(len(disease_indices), len(formula_indices), num_targets)
        model = model.to(device)

        logger.info(f"模型创建完成: 参数量 {sum(p.numel() for p in model.parameters()):,}")

        # 8. 创建数据加载器
        logger.info("8. 创建数据加载器...")

        # 简单划分（仅用于测试）
        split_ratio = 0.8
        split_idx = int(len(dataset) * split_ratio)
        train_samples = dataset.training_samples[:split_idx]
        val_samples = dataset.training_samples[split_idx:]

        # 临时修改数据集的训练样本
        dataset.training_samples = train_samples

        train_loader, _ = create_data_loaders(
            dataset, None,
            batch_size=min(config['training']['batch_size'], 16),  # 小批次
            num_workers=0  # 避免多进程问题
        )

        logger.info(f"数据加载器创建完成: 训练集 {len(train_loader.dataset)} 样本")

        # 9. 前向传播测试
        logger.info("9. 前向传播测试...")
        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                # 移动到设备
                batch = {k: v.to(device) for k, v in batch.items()}

                # 前向传播
                outputs = model(batch)

                # 检查输出
                similarities = outputs['similarities']
                logger.info(f"批次 {batch_idx + 1}: 相似度矩阵形状 {similarities.shape}")
                logger.info(f"批次 {batch_idx + 1}: 相似度范围 [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")

                # 只测试第一个批次
                break

        # 10. 成功验证
        logger.info("=" * 60)
        logger.info("🎉 TCM-RX完整性检查通过！")
        logger.info("所有核心组件都能正常工作:")
        logger.info("  ✅ 数据读取和连接")
        logger.info("  ✅ 数据集构建")
        logger.info("  ✅ 模型创建和前向传播")
        logger.info("  ✅ 批处理流水线")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ 完整性检查失败!")
        logger.error(f"错误信息: {str(e)}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_sanity_check()
    sys.exit(0 if success else 1)