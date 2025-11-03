"""
TCM-RX 批处理模块
PyTorch Dataset/DataLoader实现
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class TCMRXDataset(Dataset):
    """
    TCM-RX PyTorch数据集

    将集合型输入（若干target_id + 权重）打包为批，供模型聚合
    """

    def __init__(self, dataset_builder, mode: str = 'train'):
        """
        初始化数据集

        Args:
            dataset_builder: TCMRXDatasetBuilder实例
            mode: 模式 ('train', 'val', 'test')
        """
        self.dataset_builder = dataset_builder
        self.mode = mode
        self.num_targets = dataset_builder.id_mapper.get_count('target')

    def __len__(self) -> int:
        return len(self.dataset_builder)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含疾病和方剂信息的字典
        """
        disease_idx, formula_idx, label = self.dataset_builder.get_sample_by_index(idx)

        # 获取疾病靶点
        disease_targets = self.dataset_builder.get_disease_targets_by_index(disease_idx)
        # 获取方剂靶点
        formula_targets = self.dataset_builder.get_formula_targets_by_index(formula_idx)

        return {
            'disease_idx': disease_idx,
            'formula_idx': formula_idx,
            'disease_target_indices': [t[0] for t in disease_targets],
            'disease_target_weights': [t[1] for t in disease_targets],
            'formula_target_indices': [t[0] for t in formula_targets],
            'formula_target_weights': [t[1] for t in formula_targets],
            'label': label
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    将批次的样本整理为张量

    Args:
        batch: 样本列表

    Returns:
        批次张量字典
    """
    batch_size = len(batch)

    # 收集所有数据
    disease_indices = []
    formula_indices = []
    labels = []

    # 处理变长序列
    disease_target_lists = []
    disease_weight_lists = []
    formula_target_lists = []
    formula_weight_lists = []

    max_disease_targets = 0
    max_formula_targets = 0

    for item in batch:
        disease_indices.append(item['disease_idx'])
        formula_indices.append(item['formula_idx'])
        labels.append(item['label'])

        disease_targets = item['disease_target_indices']
        disease_weights = item['disease_target_weights']
        formula_targets = item['formula_target_indices']
        formula_weights = item['formula_target_weights']

        disease_target_lists.append(disease_targets)
        disease_weight_lists.append(disease_weights)
        formula_target_lists.append(formula_targets)
        formula_weight_lists.append(formula_weights)

        max_disease_targets = max(max_disease_targets, len(disease_targets))
        max_formula_targets = max(max_formula_targets, len(formula_targets))

    # 创建填充后的张量
    disease_target_indices = torch.zeros(batch_size, max_disease_targets, dtype=torch.long)
    disease_target_weights = torch.zeros(batch_size, max_disease_targets, dtype=torch.float)
    formula_target_indices = torch.zeros(batch_size, max_formula_targets, dtype=torch.long)
    formula_target_weights = torch.zeros(batch_size, max_formula_targets, dtype=torch.float)

    disease_mask = torch.zeros(batch_size, max_disease_targets, dtype=torch.float)
    formula_mask = torch.zeros(batch_size, max_formula_targets, dtype=torch.float)

    # 填充数据
    for i in range(batch_size):
        # 疾病侧
        disease_targets = disease_target_lists[i]
        disease_weights = disease_weight_lists[i]
        if disease_targets:
            valid_len = len(disease_targets)
            disease_target_indices[i, :valid_len] = torch.tensor(disease_targets)
            disease_target_weights[i, :valid_len] = torch.tensor(disease_weights)
            disease_mask[i, :valid_len] = 1.0

        # 方剂侧
        formula_targets = formula_target_lists[i]
        formula_weights = formula_weight_lists[i]
        if formula_targets:
            valid_len = len(formula_targets)
            formula_target_indices[i, :valid_len] = torch.tensor(formula_targets)
            formula_target_weights[i, :valid_len] = torch.tensor(formula_weights)
            formula_mask[i, :valid_len] = 1.0

    return {
        'disease_indices': torch.tensor(disease_indices, dtype=torch.long),
        'formula_indices': torch.tensor(formula_indices, dtype=torch.long),
        'disease_target_indices': disease_target_indices,
        'disease_target_weights': disease_target_weights,
        'disease_mask': disease_mask,
        'formula_target_indices': formula_target_indices,
        'formula_target_weights': formula_target_weights,
        'formula_mask': formula_mask,
        'labels': torch.tensor(labels, dtype=torch.float)
    }


def create_data_loaders(train_dataset: 'TCMRXDatasetBuilder',
                       val_dataset: 'TCMRXDatasetBuilder' = None,
                       batch_size: int = 256,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_dataset: 训练数据集构建器
        val_dataset: 验证数据集构建器
        batch_size: 批大小
        num_workers: 工作进程数
        pin_memory: 是否固定内存

    Returns:
        (train_loader, val_loader)
    """
    # 创建PyTorch数据集
    train_pytorch_dataset = TCMRXDataset(train_dataset, mode='train')
    val_pytorch_dataset = TCMRXDataset(val_dataset, mode='val') if val_dataset else None

    # 创建数据加载器
    train_loader = DataLoader(
        train_pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,  # 恢复shuffle，每个epoch重新随机排列
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 确保批大小一致
    )

    val_loader = None
    if val_pytorch_dataset is not None:
        val_loader = DataLoader(
            val_pytorch_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

    logger.info(f"创建数据加载器: 训练集 {len(train_pytorch_dataset)} 样本, "
                f"验证集 {len(val_pytorch_dataset) if val_pytorch_dataset else 0} 样本")

    return train_loader, val_loader


class BatchSampler:
    """
    批次采样器，用于负采样
    """

    def __init__(self, num_formulas: int, negative_ratio: float = 1.0):
        """
        初始化批次采样器

        Args:
            num_formulas: 方剂总数
            negative_ratio: 负样本比例
        """
        self.num_formulas = num_formulas
        self.negative_ratio = negative_ratio

    def sample_negatives(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        采样负样本

        Args:
            batch_size: 批大小
            device: 设备

        Returns:
            负样本方剂索引 [batch_size, num_negatives]
        """
        num_negatives = max(1, int(batch_size * self.negative_ratio))
        # 随机采样负样本
        negatives = torch.randint(0, self.num_formulas, (batch_size, num_negatives), device=device)
        return negatives