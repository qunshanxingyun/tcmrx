#!/usr/bin/env python3
"""
TCM-RX 评估脚本
载入模型→评估→输出指标
"""

import sys
import os
from pathlib import Path
import logging
import time
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataio.readers import TSVReader
from dataio.joins import *
from dataio.filters import *
from dataio.dataset_builder import TCMRXDataset
from models.twin_tower import DualTowerModel
from core.batching import create_data_loaders
from core.utils import load_config, get_device, setup_logging, load_checkpoint
from training.splits import stratified_disease_split
from training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TCM-RX 评估脚本")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--paths", type=str, default="config/paths.yaml",
                        help="数据路径配置文件路径")
    parser.add_argument("--data-split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="评估数据集")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="评估批大小")
    parser.add_argument("--k-values", type=int, nargs='+', default=[1, 5, 10, 20],
                        help="评估K值列表")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出文件路径")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="是否保存嵌入向量")

    return parser.parse_args()


def main():
    """主评估函数"""
    args = parse_arguments()

    # 加载配置
    config = load_config(args.config)
    paths_config = load_config(args.paths)

    # 设置日志
    setup_logging(level="INFO")
    logger.info("开始TCM-RX模型评估")
    logger.info(f"检查点: {args.checkpoint}")
    logger.info(f"数据划分: {args.data_split}")

    start_time = time.time()

    try:
        # 1. 设置设备
        device = get_device("auto")
        logger.info(f"使用设备: {device}")

        # 2. 加载模型
        logger.info("加载模型...")
        checkpoint = load_checkpoint(args.checkpoint, device)

        # 从检查点中恢复配置（如果存在）
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            model_config = config['model']

        model = DualTowerModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        logger.info(f"模型加载完成，epoch: {checkpoint.get('epoch', 'unknown')}")

        # 3. 读取数据
        logger.info("读取数据...")
        reader = TSVReader(paths_config)

        formula_tables = reader.read_formula_tables()
        disease_tables = reader.read_disease_tables()
        prediction_tables = reader.read_prediction_tables()

        # 4. 数据连接和过滤
        logger.info("数据处理...")
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

        # 5. 数据划分
        logger.info("数据划分...")
        split_config = config['split']
        split_result = stratified_disease_split(
            positive_pairs_raw,
            train_ratio=split_config['train_ratio'],
            val_ratio=split_config['val_ratio'],
            test_ratio=split_config['test_ratio'],
            seed=config['training']['seed']
        )

        # 选择要评估的数据集
        if args.data_split in split_result:
            eval_pairs = split_result[args.data_split]
        else:
            raise ValueError(f"未知的数据划分: {args.data_split}")

        # 6. 构建数据集
        logger.info(f"构建{args.data_split}数据集...")
        dataset = TCMRXDataset(config)
        dataset.build_from_raw_data(disease_targets_raw, formula_targets_raw, eval_pairs)

        # 7. 创建数据加载器
        logger.info("创建数据加载器...")
        _, val_loader = create_data_loaders(
            None, dataset,  # 作为验证集
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True
        )

        # 8. 创建评估器
        logger.info("开始评估...")
        evaluator = ModelEvaluator(model, device, args.k_values)

        # 9. 执行评估
        evaluation_results = evaluator.evaluate_dataset(
            val_loader, return_embeddings=args.save_embeddings
        )

        if args.save_embeddings:
            # 分离嵌入结果
            metrics = evaluation_results
            embeddings_data = {
                'disease_embeddings': evaluation_results['disease_embeddings'].numpy(),
                'formula_embeddings': evaluation_results['formula_embeddings'].numpy(),
                'similarities': evaluation_results['similarities'].numpy(),
                'disease_indices': evaluation_results['disease_indices'].numpy(),
                'formula_indices': evaluation_results['formula_indices'].numpy()
            }
        else:
            metrics = evaluation_results
            embeddings_data = None

        # 10. 输出结果
        logger.info("=" * 60)
        logger.info("评估结果:")
        logger.info("-" * 30)

        # 打印主要指标
        for k in args.k_values:
            if f'recall@{k}' in metrics:
                logger.info(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            if f'precision@{k}' in metrics:
                logger.info(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            if f'ndcg@{k}' in metrics:
                logger.info(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
            if f'hit_rate@{k}' in metrics:
                logger.info(f"Hit Rate@{k}: {metrics[f'hit_rate@{k}']:.4f}")

        if 'mrr' in metrics:
            logger.info(f"MRR: {metrics['mrr']:.4f}")

        # 相似度统计
        logger.info("-" * 30)
        logger.info(f"对角线相似度统计:")
        logger.info(f"  平均值: {metrics.get('mean_diagonal_similarity', 0):.4f}")
        logger.info(f"  标准差: {metrics.get('std_diagonal_similarity', 0):.4f}")
        logger.info(f"  最大值: {metrics.get('max_diagonal_similarity', 0):.4f}")
        logger.info(f"  最小值: {metrics.get('min_diagonal_similarity', 0):.4f}")

        # 保存结果
        if args.output:
            import json
            output_path = args.output

            # 保存指标
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            logger.info(f"评估指标已保存: {output_path}")

            # 保存嵌入（如果需要）
            if args.save_embeddings and embeddings_data:
                import numpy as np
                embeddings_path = output_path.replace('.json', '_embeddings.npz')
                np.savez_compressed(embeddings_path, **embeddings_data)
                logger.info(f"嵌入向量已保存: {embeddings_path}")

        # 生成报告
        report = evaluator.generate_evaluation_report(
            test_metrics=metrics
        )

        # 保存报告
        report_path = args.output.replace('.json', '_report.txt') if args.output else "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"评估报告已保存: {report_path}")
        logger.info("=" * 60)

        total_time = time.time() - start_time
        logger.info(f"评估完成，耗时: {total_time:.2f} 秒")

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ 评估失败!")
        logger.error(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()