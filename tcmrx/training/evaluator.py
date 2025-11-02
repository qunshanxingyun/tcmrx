"""
TCM-RX 评估器模块
统一评估入口
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
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
        positive_pairs: Dict[int, set] = {}

        start_time = time.time()

        # 收集所有嵌入
        with torch.no_grad():
            for batch in tqdm(dataset_loader, desc="收集嵌入"):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(batch)

                # 收集嵌入
                disease_batch_indices = batch['disease_indices'].cpu()
                formula_batch_indices = batch['formula_indices'].cpu()

                all_disease_embeddings.append(outputs['disease_embeddings'].cpu())
                all_formula_embeddings.append(outputs['formula_embeddings'].cpu())
                all_disease_indices.append(disease_batch_indices)
                all_formula_indices.append(formula_batch_indices)

                for disease_idx, formula_idx in zip(disease_batch_indices.tolist(),
                                                   formula_batch_indices.tolist()):
                    if disease_idx not in positive_pairs:
                        positive_pairs[disease_idx] = set()
                    positive_pairs[disease_idx].add(formula_idx)

        # 合并并聚合所有嵌入
        disease_embeddings = torch.cat(all_disease_embeddings, dim=0)
        formula_embeddings = torch.cat(all_formula_embeddings, dim=0)
        disease_indices = torch.cat(all_disease_indices, dim=0)
        formula_indices = torch.cat(all_formula_indices, dim=0)

        aggregated = self._aggregate_embeddings(
            disease_embeddings,
            disease_indices,
            formula_embeddings,
            formula_indices
        )

        disease_matrix = aggregated['disease_embeddings']
        formula_matrix = aggregated['formula_embeddings']
        eval_disease_indices = aggregated['disease_indices']
        eval_formula_indices = aggregated['formula_indices']

        similarities = create_similarity_matrix(
            disease_matrix,
            formula_matrix,
            formula_matrix.size(0)
        )

        true_pairs = self._create_true_pairs_matrix(
            eval_disease_indices,
            eval_formula_indices,
            positive_pairs
        )

        # 计算评估指标
        metrics = self.metrics_calculator.compute_all_metrics(similarities, true_pairs)

        # 添加正样本相似度统计
        with torch.no_grad():
            formula_lookup = {int(idx): pos for pos, idx in enumerate(eval_formula_indices.tolist())}
            positive_scores = []
            for row, disease_id in enumerate(eval_disease_indices.tolist()):
                for formula_id in positive_pairs.get(int(disease_id), set()):
                    col = formula_lookup.get(int(formula_id))
                    if col is not None:
                        positive_scores.append(similarities[row, col].item())

            if positive_scores:
                positive_scores_tensor = torch.tensor(positive_scores)
                metrics.update({
                    'mean_positive_similarity': positive_scores_tensor.mean().item(),
                    'std_positive_similarity': positive_scores_tensor.std().item(),
                    'max_positive_similarity': positive_scores_tensor.max().item(),
                    'min_positive_similarity': positive_scores_tensor.min().item()
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
                'disease_embeddings': disease_matrix,
                'formula_embeddings': formula_matrix,
                'similarities': similarities,
                'disease_indices': eval_disease_indices,
                'formula_indices': eval_formula_indices
            }
        else:
            return metrics

    def _aggregate_embeddings(self,
                               disease_embeddings: torch.Tensor,
                               disease_indices: torch.Tensor,
                               formula_embeddings: torch.Tensor,
                               formula_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将重复的疾病/方剂样本聚合为唯一索引的表示"""

        embedding_dim = disease_embeddings.size(1)
        device = disease_embeddings.device
        dtype = disease_embeddings.dtype

        disease_bank = torch.zeros(self.model.num_diseases, embedding_dim, device=device, dtype=dtype)
        disease_counts = torch.zeros(self.model.num_diseases, 1, device=device, dtype=dtype)
        disease_bank.index_add_(0, disease_indices, disease_embeddings)
        disease_counts.index_add_(0, disease_indices, torch.ones(disease_indices.size(0), 1, device=device, dtype=dtype))
        valid_disease_mask = disease_counts.squeeze(-1) > 0
        valid_disease_indices = valid_disease_mask.nonzero(as_tuple=False).squeeze(-1)
        disease_bank[valid_disease_mask] = disease_bank[valid_disease_mask] / disease_counts[valid_disease_mask]

        formula_bank = torch.zeros(self.model.num_formulas, embedding_dim, device=device, dtype=dtype)
        formula_counts = torch.zeros(self.model.num_formulas, 1, device=device, dtype=dtype)
        formula_bank.index_add_(0, formula_indices, formula_embeddings)
        formula_counts.index_add_(0, formula_indices, torch.ones(formula_indices.size(0), 1, device=device, dtype=dtype))
        valid_formula_mask = formula_counts.squeeze(-1) > 0
        valid_formula_indices = valid_formula_mask.nonzero(as_tuple=False).squeeze(-1)
        formula_bank[valid_formula_mask] = formula_bank[valid_formula_mask] / formula_counts[valid_formula_mask]

        return {
            'disease_embeddings': disease_bank[valid_disease_indices],
            'formula_embeddings': formula_bank[valid_formula_indices],
            'disease_indices': valid_disease_indices,
            'formula_indices': valid_formula_indices
        }

    def _create_true_pairs_matrix(self, disease_indices: torch.Tensor,
                                formula_indices: torch.Tensor,
                                positive_pairs: Dict[int, set]) -> torch.Tensor:
        """
        创建真实配对矩阵

        Args:
            disease_indices: 唯一疾病索引 [num_diseases]
            formula_indices: 唯一方剂索引 [num_formulas]
            positive_pairs: 疾病→正样本方剂集合

        Returns:
            真实配对矩阵 [num_diseases, num_formulas]
        """
        batch_size = len(disease_indices)
        num_formulas = len(formula_indices)

        true_pairs = torch.zeros(batch_size, num_formulas, dtype=torch.float)

        formula_lookup = {int(idx): pos for pos, idx in enumerate(formula_indices.tolist())}

        for row, disease_id in enumerate(disease_indices.tolist()):
            positives = positive_pairs.get(int(disease_id), set())
            for formula_id in positives:
                col = formula_lookup.get(int(formula_id))
                if col is not None:
                    true_pairs[row, col] = 1.0

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