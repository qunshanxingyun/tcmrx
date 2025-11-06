"""
TCM-RX 训练循环模块
标准训练循环（前向→loss→反传→日志）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
from typing import Dict, Optional
from tqdm import tqdm

from core.losses import compute_contrastive_metrics, multi_positive_info_nce
from core.utils import save_checkpoint, format_time

logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    训练循环管理器
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = torch.device('cpu'),
                 mixed_precision: bool = False):
        """
        初始化训练循环

        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            mixed_precision: 是否使用混合精度
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # 训练状态
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.training_history = []

        # 将模型移动到设备
        self.model = self.model.to(device)

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            训练指标字典
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")

        # 使用多正样本感知的InfoNCE，避免同一疾病在同批出现时将真阳性当作负样本
        ce_loss_fn = nn.CrossEntropyLoss()

        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    # 原始相似度（未缩放）用于自定义InfoNCE；也保留scaled用于参照
                    similarities = outputs['similarities']
                    # 多正样本：行标签为疾病索引
                    row_labels = batch['disease_indices']
                    loss = multi_positive_info_nce(similarities, row_labels, temperature=self.model.get_temperature())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                similarities = outputs['similarities']
                row_labels = batch['disease_indices']
                loss = multi_positive_info_nce(similarities, row_labels, temperature=self.model.get_temperature())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            with torch.no_grad():
                metrics = compute_contrastive_metrics(
                    outputs['disease_embeddings'],
                    outputs['formula_embeddings'],
                    temperature=self.model.get_temperature(),
                    row_labels=batch['disease_indices']
                )

            # 累计损失
            epoch_loss += loss.item()

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.3f}',
                'LR': f'{current_lr:.2e}'
            })

            # 定期记录日志
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {self.current_epoch + 1}, Batch {batch_idx}/{num_batches}, "
                           f"Loss: {loss.item():.4f}, Acc: {metrics['accuracy']:.3f}")

        # 平均损失
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {self.current_epoch + 1} 训练完成, 平均损失: {avg_loss:.4f}")

        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': avg_loss}

    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch

        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            logger.warning("没有验证数据加载器，跳过验证")
            return {}

        self.model.eval()
        val_loss = 0.0
        val_metrics_sum = {}
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(batch)
                similarities = outputs['similarities']
                row_labels = batch['disease_indices']
                loss = multi_positive_info_nce(similarities, row_labels, temperature=self.model.get_temperature())

                val_loss += loss.item()

                # 计算指标
                metrics = compute_contrastive_metrics(
                    outputs['disease_embeddings'],
                    outputs['formula_embeddings'],
                    temperature=self.model.get_temperature(),
                    row_labels=batch['disease_indices']
                )

                # 累积指标
                for key, value in metrics.items():
                    if key not in val_metrics_sum:
                        val_metrics_sum[key] = 0
                    val_metrics_sum[key] += value

        # 计算平均值
        avg_val_loss = val_loss / num_batches
        avg_val_metrics = {k: v / num_batches for k, v in val_metrics_sum.items()}

        logger.info(f"Epoch {self.current_epoch + 1} 验证完成, "
                   f"验证损失: {avg_val_loss:.4f}, "
                   f"验证准确率: {avg_val_metrics.get('accuracy', 0):.3f}")

        return {'loss': avg_val_loss, **avg_val_metrics}

    def train(self, num_epochs: int, save_every: int = 5,
              validate_every: int = 1, checkpoint_dir: str = "checkpoints",
              experiment_name: str = "experiment") -> None:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数
            save_every: 每隔多少轮保存检查点
            validate_every: 每隔多少轮验证
            checkpoint_dir: 检查点目录
            experiment_name: 实验名称
        """
        logger.info(f"开始训练，共 {num_epochs} 轮")
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = {}
            if self.val_loader is not None and epoch % validate_every == 0:
                val_metrics = self.validate_epoch()

            # 记录训练历史
            epoch_time = time.time() - epoch_start_time
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics.get('accuracy', 0),
                'val_loss': val_metrics.get('loss', 0),
                'val_accuracy': val_metrics.get('accuracy', 0),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            self.training_history.append(epoch_info)

            # 保存检查点
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1, checkpoint_dir, experiment_name)

            # 保存最佳模型
            if val_metrics and val_metrics.get('accuracy', 0) > self.best_val_metric:
                self.best_val_metric = val_metrics.get('accuracy', 0)
                self._save_checkpoint(epoch + 1, checkpoint_dir, experiment_name, is_best=True)

            # 打印epoch总结
            logger.info(f"Epoch {epoch + 1}/{num_epochs} 完成 "
                       f"({format_time(epoch_time)}) - "
                       f"训练损失: {train_metrics['loss']:.4f}, "
                       f"训练准确率: {train_metrics.get('accuracy', 0):.3f}")
            if val_metrics:
                logger.info(f"验证损失: {val_metrics.get('loss', 0):.4f}, "
                           f"验证准确率: {val_metrics.get('accuracy', 0):.3f} "
                           f"{'[最佳]' if val_metrics.get('accuracy', 0) == self.best_val_metric else ''}")

        total_time = time.time() - start_time
        logger.info(f"训练完成！总时间: {format_time(total_time)}")
        logger.info(f"最佳验证准确率: {self.best_val_metric:.3f}")

        # 保存最终模型
        self._save_checkpoint(num_epochs, checkpoint_dir, experiment_name, is_final=True)

    def _save_checkpoint(self, epoch: int, checkpoint_dir: str, experiment_name: str,
                       is_best: bool = False, is_final: bool = False) -> None:
        """
        保存检查点

        Args:
            epoch: 当前epoch
            checkpoint_dir: 检查点目录
            experiment_name: 实验名称
            is_best: 是否为最佳模型
            is_final: 是否为最终模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 确定文件名
        if is_final:
            filename = f"{experiment_name}_final.pt"
        elif is_best:
            filename = f"{experiment_name}_best.pt"
        else:
            filename = f"{experiment_name}_epoch_{epoch}.pt"

        filepath = f"{checkpoint_dir}/{filename}"
        save_checkpoint(checkpoint, filepath)

        if is_best:
            logger.info(f"保存最佳模型: {filepath}")
        elif is_final:
            logger.info(f"保存最终模型: {filepath}")
        else:
            logger.info(f"保存检查点: {filepath}")
