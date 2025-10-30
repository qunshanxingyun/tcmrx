"""
TCM-RX 评估器模块
统一评估入口
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from core.metrics import RankingMetrics, create_similarity_matrix
from core.utils import format_time
import time

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    模型评估器
    """

    def __init__(self, model, device: torch.device, k_values: List[int] = [1, 5, 10, 20]):
        """
        初始化评估器

        Args:
            model: 要评估的模型
            device: 计算设备
            k_values: K值列表
        """
        self.model = model
        self.device = device
        self.k_values = k_values
        self.metrics_calculator = RankingMetrics(k_values)

        # 将模型移动到设备并设置为评估模式
        self.model = self.model.to(device)
        self.model.eval()

    def evaluate_dataset(self, dataset_loader: DataLoader,
                        return_embeddings: bool = False) -> Dict[str, float]:
        """
        评估数据集

        Args:
            dataset_loader: 数据加载器
            return_embeddings: 是否返回嵌入向量

        Returns:
            评估指标字典
        """
        logger.info("开始评估数据集...")

        all_disease_embeddings = []
        all_formula_embeddings = []
        all_disease_indices = []
        all_formula_indices = []

        start_time = time.time()

        # 收集所有嵌入
        with torch.no_grad():
            for batch in tqdm(dataset_loader, desc="收集嵌入"):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(batch)

                # 收集嵌入
                all_disease_embeddings.append(outputs['disease_embeddings'].cpu())
                all_formula_embeddings.append(outputs['formula_embeddings'].cpu())
                all_disease_indices.append(batch['disease_indices'].cpu())
                all_formula_indices.append(batch['formula_indices'].cpu())

        # 合并所有嵌入
        disease_embeddings = torch.cat(all_disease_embeddings, dim=0)
        formula_embeddings = torch.cat(all_formula_embeddings, dim=0)
        disease_indices = torch.cat(all_disease_indices, dim=0)
        formula_indices = torch.cat(all_formula_indices, dim=0)

        # 计算相似度矩阵
        similarities = create_similarity_matrix(
            disease_embeddings, formula_embeddings, self.model.num_formulas
        )

        # 创建真实标签矩阵（基于批内配对）
        true_pairs = self._create_true_pairs_matrix(disease_indices, formula_indices)

        # 计算评估指标
        metrics = self.metrics_calculator.compute_all_metrics(similarities, true_pairs)

        # 添加相似度统计
        with torch.no_grad():
            diag_similarities = torch.diag(similarities)
            metrics.update({
                'mean_diagonal_similarity': diag_similarities.mean().item(),
                'std_diagonal_similarity': diag_similarities.std().item(),
                'max_diagonal_similarity': diag_similarities.max().item(),
                'min_diagonal_similarity': diag_similarities.min().item()
            })

        eval_time = time.time() - start_time
        logger.info(f"评估完成，耗时: {format_time(eval_time)}")

        # 打印主要指标
        for k in self.k_values:
            if f'recall@{k}' in metrics:
                logger.info(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            if f'ndcg@{k}' in metrics:
                logger.info(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        if 'mrr' in metrics:
            logger.info(f"MRR: {metrics['mrr']:.4f}")

        if return_embeddings:
            return metrics, {
                'disease_embeddings': disease_embeddings,
                'formula_embeddings': formula_embeddings,
                'similarities': similarities,
                'disease_indices': disease_indices,
                'formula_indices': formula_indices
            }
        else:
            return metrics

    def _create_true_pairs_matrix(self, disease_indices: torch.Tensor,
                                formula_indices: torch.Tensor) -> torch.Tensor:
        """
        创建真实配对矩阵

        Args:
            disease_indices: 疾病索引 [batch_size]
            formula_indices: 方剂索引 [batch_size]

        Returns:
            真实配对矩阵 [batch_size, num_formulas]
        """
        batch_size = len(disease_indices)
        num_formulas = self.model.num_formulas

        # 创建零矩阵
        true_pairs = torch.zeros(batch_size, num_formulas, dtype=torch.float)

        # 标记对角线位置为正样本
        for i in range(batch_size):
            true_pairs[i, formula_indices[i]] = 1.0

        return true_pairs

    def evaluate_cold_start(self, cold_start_loader: DataLoader) -> Dict[str, float]:
        """
        评估冷启动性能

        Args:
            cold_start_loader: 冷启动数据加载器

        Returns:
            冷启动评估指标
        """
        logger.info("开始冷启动评估...")

        # 使用标准评估流程
        metrics = self.evaluate_dataset(cold_start_loader)

        # 添加冷启动特定指标
        cold_start_metrics = {f'cold_start_{k}': v for k, v in metrics.items()}

        logger.info("冷启动评估完成")
        return cold_start_metrics

    def compute_embeddings_for_all_entities(self) -> Dict[str, torch.Tensor]:
        """
        计算所有实体的嵌入

        Returns:
            包含所有实体嵌入的字典
        """
        logger.info("计算所有实体嵌入...")

        # 获取靶点嵌入
        target_embeddings = self.model.get_target_embeddings()

        # 创建所有疾病和方剂的占位符嵌入（全零）
        # 在实际应用中，这些可能需要额外的计算
        disease_embeddings = torch.zeros(self.model.num_diseases, self.model.embedding_dim)
        formula_embeddings = torch.zeros(self.model.num_formulas, self.model.embedding_dim)

        logger.info("实体嵌入计算完成")

        return {
            'target_embeddings': target_embeddings,
            'disease_embeddings': disease_embeddings,
            'formula_embeddings': formula_embeddings
        }

    def save_evaluation_results(self, metrics: Dict[str, float],
                              filepath: str) -> None:
        """
        保存评估结果

        Args:
            metrics: 评估指标字典
            filepath: 保存路径
        """
        import json
        import os

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存结果
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"评估结果已保存: {filepath}")

    def generate_evaluation_report(self, train_metrics: Optional[Dict[str, float]] = None,
                                 val_metrics: Optional[Dict[str, float]] = None,
                                 test_metrics: Optional[Dict[str, float]] = None,
                                 cold_start_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        生成评估报告

        Args:
            train_metrics: 训练集指标
            val_metrics: 验证集指标
            test_metrics: 测试集指标
            cold_start_metrics: 冷启动指标

        Returns:
            评估报告字符串
        """
        report = []
        report.append("=" * 60)
        report.append("TCM-RX 模型评估报告")
        report.append("=" * 60)

        def _format_metrics(metrics_dict: Dict[str, float], name: str):
            if not metrics_dict:
                return
            report.append(f"\n{name}:")
            report.append("-" * len(name))
            for k in sorted(metrics_dict.keys()):
                if any(metric in k for metric in ['recall', 'precision', 'ndcg', 'hit_rate', 'mrr']):
                    report.append(f"  {k}: {metrics_dict[k]:.4f}")

        _format_metrics(train_metrics, "训练集指标")
        _format_metrics(val_metrics, "验证集指标")
        _format_metrics(test_metrics, "测试集指标")
        _format_metrics(cold_start_metrics, "冷启动指标")

        # 模型信息
        report.append(f"\n模型信息:")
        report.append("-" * 6)
        report.append(f"  嵌入维度: {self.model.embedding_dim}")
        report.append(f"  温度参数: {self.model.get_temperature():.4f}")
        report.append(f"  疾病数量: {self.model.num_diseases}")
        report.append(f"  方剂数量: {self.model.num_formulas}")
        report.append(f"  靶点数量: {self.model.num_targets}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)