#!/usr/bin/env python3
"""
TCM-RX 主训练脚本
读配置→训练→保存模型
"""

import sys
import os
from pathlib import Path
import logging
import time
import argparse
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
from core.utils import load_config, get_device, setup_logging, set_random_seed, log_model_info
from core.losses import InfoNCELoss
from training.train_loop import TrainingLoop
from training.splits import stratified_disease_split, identify_cold_start_diseases, validate_split, create_cold_start_split
from training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TCM-RX 训练脚本")

    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--paths", type=str, default="config/paths.yaml",
                        help="数据路径配置文件路径")
    parser.add_argument("--experiment", type=str, default=None,
                        help="实验名称（自动生成如果为空）")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（覆盖配置文件）")
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda"],
                        help="计算设备（覆盖配置文件）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置文件）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批大小（覆盖配置文件）")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率（覆盖配置文件）")

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_arguments()

    # 加载配置
    config = load_config(args.config)
    paths_config = load_config(args.paths)

    # 命令行参数覆盖
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.device is not None:
        config['training']['device'] = args.device
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr

    # 设置实验名称
    if args.experiment is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment = f"tcmrx_train_{timestamp}"

    # 设置日志
    setup_logging(
        log_dir=config['logging']['log_dir'],
        level="INFO",
        experiment_name=args.experiment
    )

    logger.info("开始TCM-RX训练")
    logger.info(f"实验名称: {args.experiment}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"路径配置: {args.paths}")

    start_time = time.time()

    try:
        # 1. 设置随机种子和设备
        set_random_seed(config['training']['seed'])
        device = get_device(config['training']['device'])
        logger.info(f"使用设备: {device}")

        # 2. 读取数据
        logger.info("读取数据...")
        reader = TSVReader(paths_config)

        formula_tables = reader.read_formula_tables()
        disease_tables = reader.read_disease_tables()
        prediction_tables = reader.read_prediction_tables()

        logger.info(f"数据读取完成: {len(formula_tables)} 方剂表, {len(disease_tables)} 疾病表, {len(prediction_tables)} 预测表")

        # 3. 数据连接和过滤
        logger.info("数据连接和过滤...")
        filtering_config = config.get('filtering', {})

        # 方剂侧连接
        cpms_to_chp_map = cpms_to_chp(formula_tables['D4_CPM_CHP'])
        chp_to_chemicals_map = chp_to_chemicals(formula_tables['D9_CHP_InChIKey'], formula_tables.get('D12_InChIKey'))

        # 处理化学-靶点预测
        sd1_df = prediction_tables['SD1_predicted']
        sd1_df = filter_sd1_by_pki(sd1_df, filtering_config.get('pki_threshold'))
        chemical_to_targets_map = per_chemical_topk(
            chemicals_to_targets(sd1_df), filtering_config.get('topk_c')
        )

        # 疾病侧连接
        icd11_to_targets_map = icd11_to_targets(
            disease_tables['D19_ICD11_CUI'],
            disease_tables['D20_ICD11_MeSH'],
            disease_tables['D22_CUI_targets'],
            disease_tables['D23_MeSH_targets']
        )

        # 构建靶点集合
        formula_targets_raw = formulas_to_targets(cpms_to_chp_map, chp_to_chemicals_map, chemical_to_targets_map)
        disease_targets_raw = diseases_to_targets(icd11_to_targets_map)

        # 获取监督对
        cpms_to_icd11_map = cpms_to_icd11(formula_tables['D5_CPM_ICD11'])
        positive_pairs_raw = [(icd11, cpm) for cpm, icd11_list in cpms_to_icd11_map.items() for icd11 in icd11_list]

        logger.info(f"数据处理完成: {len(formula_targets_raw)} 方剂, {len(disease_targets_raw)} 疾病, {len(positive_pairs_raw)} 正样本对")

        # 4. 数据划分
        logger.info("数据划分...")
        split_config = config['split']
        split_result = stratified_disease_split(
            positive_pairs_raw,
            train_ratio=split_config['train_ratio'],
            val_ratio=split_config['val_ratio'],
            test_ratio=split_config['test_ratio'],
            seed=config['training']['seed']
        )

        # 验证划分
        if not validate_split(split_result):
            raise ValueError("数据划分验证失败")

        # 冷启动识别（如果启用）
        cold_start_splits = {}
        if split_config.get('cold_start_eval'):
            diseases_with_targets, cold_start_diseases = identify_cold_start_diseases(
                positive_pairs_raw, disease_targets_raw
            )
            if cold_start_diseases:
                cold_start_splits = create_cold_start_split(
                    positive_pairs_raw, cold_start_diseases,
                    train_ratio=split_config['train_ratio'],
                    val_ratio=split_config['val_ratio'],
                    test_ratio=split_config['test_ratio'],
                    seed=config['training']['seed']
                )

        # 5. 构建数据集
        logger.info("构建数据集...")
        dataset = TCMRXDataset(config)
        dataset.build_from_raw_data(disease_targets_raw, formula_targets_raw, split_result['train'])

        logger.info(f"训练数据集: {dataset}")

        # 6. 创建模型
        logger.info("创建模型...")
        model_config = config['model']
        model = DualTowerModel(model_config)

        # 设置实体数量
        disease_indices, formula_indices, num_targets = dataset.get_entity_indices()
        model.set_entity_counts(len(disease_indices), len(formula_indices), num_targets)
        model = model.to(device)

        log_model_info(model)

        # 7. 创建数据加载器
        logger.info("创建数据加载器...")

        # 构建验证数据集
        if split_result['val']:
            val_dataset = TCMRXDataset(config)
            val_dataset.build_from_raw_data(disease_targets_raw, formula_targets_raw, split_result['val'])
        else:
            val_dataset = None

        train_loader, val_loader = create_data_loaders(
            dataset, val_dataset,
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=True
        )

        # 8. 创建优化器和损失函数
        logger.info("创建优化器和损失函数...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )

        loss_fn = InfoNCELoss(temperature=model.get_temperature())

        # 9. 创建训练循环
        logger.info("开始训练...")
        training_loop = TrainingLoop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            mixed_precision=config['training']['mixed_precision']
        )

        # 10. 恢复训练（如果指定）
        if args.resume:
            logger.info(f"恢复训练: {args.resume}")
            from core.utils import load_checkpoint
            checkpoint = load_checkpoint(args.resume, device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            training_loop.current_epoch = checkpoint['epoch']

        # 11. 开始训练
        training_loop.train(
            num_epochs=config['training']['epochs'],
            save_every=config['training']['eval_every'],
            validate_every=config['training']['eval_every'],
            checkpoint_dir=config['logging']['checkpoint_dir'],
            experiment_name=args.experiment
        )

        # 12. 最终评估
        logger.info("最终评估...")
        if val_loader:
            evaluator = ModelEvaluator(model, device)
            final_metrics = evaluator.evaluate_dataset(val_loader)

            # 保存评估结果
            evaluator.save_evaluation_results(
                final_metrics,
                f"{config['logging']['checkpoint_dir']}/{args.experiment}_final_metrics.json"
            )

            logger.info("最终评估指标:")
            for k, v in final_metrics.items():
                if any(metric in k for metric in ['recall', 'precision', 'ndcg', 'mrr']):
                    logger.info(f"  {k}: {v:.4f}")

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🎉 训练完成!")
        logger.info(f"总耗时: {total_time/3600:.2f} 小时")
        logger.info(f"实验名称: {args.experiment}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ 训练失败!")
        logger.error(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()