"""
TCM-RX 评估指标模块
Recall@K、NDCG@K等排序指标实现
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def recall_at_k(predictions: torch.Tensor,
                targets: torch.Tensor,
                k: int) -> float:
    """
    计算Recall@K

    Args:
        predictions: 预测分数 [batch_size, num_items]
        targets: 真实标签 [batch_size, num_items] (1表示正样本)
        k: K值

    Returns:
        Recall@K值
    """
    batch_size = predictions.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    # 检查Top-K中是否有正样本
    recalls = []
    for i in range(batch_size):
        # 获取该样本的真实正样本位置
        true_positions = torch.where(targets[i] == 1)[0]

        if len(true_positions) == 0:
            # 如果没有真实正样本，跳过
            continue

        # 检查Top-K预测中是否包含真实正样本
        top_k_set = set(top_k_indices[i].cpu().numpy())
        true_set = set(true_positions.cpu().numpy())

        # 计算召回率
        intersection = len(top_k_set & true_set)
        recall = intersection / len(true_set)
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def precision_at_k(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   k: int) -> float:
    """
    计算Precision@K

    Args:
        predictions: 预测分数 [batch_size, num_items]
        targets: 真实标签 [batch_size, num_items] (1表示正样本)
        k: K值

    Returns:
        Precision@K值
    """
    batch_size = predictions.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    # 计算精确率
    precisions = []
    for i in range(batch_size):
        top_k_set = set(top_k_indices[i].cpu().numpy())
        true_set = set(torch.where(targets[i] == 1)[0].cpu().numpy())

        if len(top_k_set) == 0:
            continue

        # 计算Top-K中正样本的比例
        intersection = len(top_k_set & true_set)
        precision = intersection / len(top_k_set)
        precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def ndcg_at_k(predictions: torch.Tensor,
              targets: torch.Tensor,
              k: int) -> float:
    """
    计算NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        predictions: 预测分数 [batch_size, num_items]
        targets: 真实标签 [batch_size, num_items] (1表示正样本)
        k: K值

    Returns:
        NDCG@K值
    """
    batch_size = predictions.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    ndcg_scores = []
    for i in range(batch_size):
        # 获取真实标签
        true_labels = targets[i].cpu().numpy()

        # 计算DCG
        dcg = 0.0
        for rank, idx in enumerate(top_k_indices[i].cpu().numpy()):
            # DCG = sum(relevance / log2(rank + 2))
            # 这里relevance就是二值标签 (0或1)
            relevance = true_labels[idx]
            dcg += relevance / np.log2(rank + 2) if relevance > 0 else 0.0

        # 计算IDCG (理想DCG)
        # 按真实标签排序，计算理想情况下的DCG
        sorted_labels = np.sort(true_labels)[::-1]  # 降序
        ideal_k = min(k, len(sorted_labels))
        idcg = 0.0
        for rank in range(ideal_k):
            relevance = sorted_labels[rank]
            idcg += relevance / np.log2(rank + 2) if relevance > 0 else 0.0

        # 计算NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def mean_reciprocal_rank(predictions: torch.Tensor,
                       targets: torch.Tensor) -> float:
    """
    计算平均倒数排名 (MRR)

    Args:
        predictions: 预测分数 [batch_size, num_items]
        targets: 真实标签 [batch_size, num_items] (1表示正样本)

    Returns:
        MRR值
    """
    batch_size = predictions.size(0)

    # 按预测分数降序排列
    _, sorted_indices = torch.sort(predictions, dim=1, descending=True)

    reciprocal_ranks = []
    for i in range(batch_size):
        # 找到第一个正样本的位置
        true_positions = torch.where(targets[i] == 1)[0]

        if len(true_positions) == 0:
            continue

        # 在排序后的列表中查找第一个正样本
        for rank, idx in enumerate(sorted_indices[i].cpu().numpy()):
            if idx in true_positions.cpu().numpy():
                reciprocal_ranks.append(1.0 / (rank + 1))
                break

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def hit_rate_at_k(predictions: torch.Tensor,
                  targets: torch.Tensor,
                  k: int) -> float:
    """
    计算Hit Rate@K (至少有一个命中的比例)

    Args:
        predictions: 预测分数 [batch_size, num_items]
        targets: 真实标签 [batch_size, num_items] (1表示正样本)
        k: K值

    Returns:
        Hit Rate@K值
    """
    batch_size = predictions.size(0)

    # 获取Top-K预测
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    hits = 0
    for i in range(batch_size):
        true_positions = torch.where(targets[i] == 1)[0]

        if len(true_positions) == 0:
            continue

        # 检查Top-K中是否有正样本
        top_k_set = set(top_k_indices[i].cpu().numpy())
        true_set = set(true_positions.cpu().numpy())

        if len(top_k_set & true_set) > 0:
            hits += 1

    valid_samples = sum(1 for i in range(batch_size)
                       if len(torch.where(targets[i] == 1)[0]) > 0)

    return hits / valid_samples if valid_samples > 0 else 0.0


class RankingMetrics:
    """
    排序指标计算器
    """

    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        """
        初始化排序指标计算器

        Args:
            k_values: K值列表
        """
        self.k_values = k_values

    def compute_all_metrics(self,
                           predictions: torch.Tensor,
                           targets: torch.Tensor) -> Dict[str, float]:
        """
        计算所有排序指标

        Args:
            predictions: 预测分数 [batch_size, num_items]
            targets: 真实标签 [batch_size, num_items]

        Returns:
            指标字典
        """
        metrics = {}

        # 计算不同K值的指标
        for k in self.k_values:
            if k <= predictions.size(1):
                metrics[f'recall@{k}'] = recall_at_k(predictions, targets, k)
                metrics[f'precision@{k}'] = precision_at_k(predictions, targets, k)
                metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, targets, k)
                metrics[f'hit_rate@{k}'] = hit_rate_at_k(predictions, targets, k)

        # 计算MRR（不需要K值）
        metrics['mrr'] = mean_reciprocal_rank(predictions, targets)

        return metrics

    def compute_single_k_metrics(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                k: int) -> Dict[str, float]:
        """
        计算特定K值的指标

        Args:
            predictions: 预测分数 [batch_size, num_items]
            targets: 真实标签 [batch_size, num_items]
            k: K值

        Returns:
            指标字典
        """
        if k > predictions.size(1):
            raise ValueError(f"K值 {k} 超出项目数量 {predictions.size(1)}")

        return {
            f'recall@{k}': recall_at_k(predictions, targets, k),
            f'precision@{k}': precision_at_k(predictions, targets, k),
            f'ndcg@{k}': ndcg_at_k(predictions, targets, k),
            f'hit_rate@{k}': hit_rate_at_k(predictions, targets, k)
        }


def create_similarity_matrix(disease_embeddings: torch.Tensor,
                            formula_embeddings: torch.Tensor,
                            num_formulas: int) -> torch.Tensor:
    """
    为评估创建相似度矩阵

    Args:
        disease_embeddings: 疾病嵌入 [num_diseases, embedding_dim]
        formula_embeddings: 方剂嵌入 [num_formulas, embedding_dim]
        num_formulas: 方剂总数

    Returns:
        相似度矩阵 [num_diseases, num_formulas]
    """
    # L2归一化
    disease_embeddings = torch.nn.functional.normalize(disease_embeddings, p=2, dim=1)
    formula_embeddings = torch.nn.functional.normalize(formula_embeddings, p=2, dim=1)

    # 计算余弦相似度
    similarities = torch.mm(disease_embeddings, formula_embeddings.t())

    return similarities


def evaluate_recommendations(similarities: torch.Tensor,
                           true_pairs: torch.Tensor,
                           k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    评估推荐结果

    Args:
        similarities: 相似度矩阵 [num_diseases, num_formulas]
        true_pairs: 真实配对矩阵 [num_diseases, num_formulas] (1表示正样本)
        k_values: K值列表

    Returns:
        评估指标字典
    """
    metrics_calculator = RankingMetrics(k_values)
    return metrics_calculator.compute_all_metrics(similarities, true_pairs)