"""
TCM-RX 核心工具模块
日志、随机种、断言等工具函数
"""

import random
import numpy as np
import torch
import logging
import os
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(log_dir: str = "logs",
                 level: str = "INFO",
                 experiment_name: Optional[str] = None) -> None:
    """
    设置日志系统

    Args:
        log_dir: 日志目录
        level: 日志级别
        experiment_name: 实验名称
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 生成日志文件名
    if experiment_name:
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
    else:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")


def set_random_seed(seed: int) -> None:
    """
    设置随机种子以确保可复现性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的可确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger = logging.getLogger(__name__)
    logger.info(f"设置随机种子: {seed}")


def get_device(device: str = "auto") -> torch.device:
    """
    获取计算设备

    Args:
        device: 设备类型 ("auto", "cpu", "cuda")

    Returns:
        torch设备对象
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger = logging.getLogger(__name__)
            logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger = logging.getLogger(__name__)
            logger.info("CUDA不可用，使用CPU")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            logger = logging.getLogger(__name__)
            logger.warning("请求使用CUDA但不可用，回退到CPU")
            device = "cpu"

    return torch.device(device)


def assert_non_empty(collection, name: str) -> None:
    """
    断言集合非空

    Args:
        collection: 集合对象
        name: 集合名称（用于错误信息）

    Raises:
        AssertionError: 如果集合为空
    """
    if not collection:
        raise AssertionError(f"{name} 不能为空")


def assert_valid_index(idx: int, max_idx: int, name: str) -> None:
    """
    断言索引有效

    Args:
        idx: 索引值
        max_idx: 最大索引值
        name: 索引名称（用于错误信息）

    Raises:
        AssertionError: 如果索引无效
    """
    if idx < 0 or idx >= max_idx:
        raise AssertionError(f"{name} 索引 {idx} 超出范围 [0, {max_idx})")


def assert_positive(value: float, name: str) -> None:
    """
    断言数值为正

    Args:
        value: 数值
        name: 数值名称（用于错误信息）

    Raises:
        AssertionError: 如果数值非正
    """
    if value <= 0:
        raise AssertionError(f"{name} 必须为正数，当前值: {value}")


def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # 简单的变量替换
    config_content = config_content.replace('${data_root}', 'TCM-MKG-data')

    config = yaml.safe_load(config_content)

    logger = logging.getLogger(__name__)
    logger.info(f"加载配置文件: {config_path}")
    return config


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    合并配置字典

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged_config = _deep_merge(base_config, override_config)

    logger = logging.getLogger(__name__)
    logger.info("配置合并完成")
    return merged_config


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def save_checkpoint(state: dict, filepath: str) -> None:
    """
    保存检查点

    Args:
        state: 检查点状态字典
        filepath: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(state, filepath)
    logger = logging.getLogger(__name__)
    logger.info(f"检查点已保存: {filepath}")


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """
    加载检查点

    Args:
        filepath: 检查点路径
        device: 设备

    Returns:
        检查点状态字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")

    state = torch.load(filepath, map_location=device)
    logger = logging.getLogger(__name__)
    logger.info(f"检查点已加载: {filepath}")
    return state


def log_model_info(model: torch.nn.Module) -> None:
    """
    记录模型信息

    Args:
        model: PyTorch模型
    """
    logger = logging.getLogger(__name__)
    param_count = count_parameters(model)

    logger.info("=" * 50)
    logger.info("模型信息:")
    logger.info(f"参数总数: {param_count:,}")
    logger.info(f"模型大小: ~{param_count * 4 / 1024 / 1024:.2f} MB (FP32)")

    # 记录模型结构
    logger.info("模型结构:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只记录叶子模块
            logger.info(f"  {name}: {module.__class__.__name__}")

    logger.info("=" * 50)