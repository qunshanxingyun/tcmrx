"""
TCM-RX 数据读取模块
使用pandas读取TSV文件并进行基本校验
"""

import pandas as pd
import os
from pathlib import Path
from typing import Dict, Optional
import logging

from .schema_map import validate_table_columns, get_required_columns

logger = logging.getLogger(__name__)


def read_tsv(table_name: str, file_path: str, required_only: bool = True) -> pd.DataFrame:
    """
    读取TSV文件并校验必需列

    Args:
        table_name: 表名，用于schema验证
        file_path: 文件路径
        required_only: 是否只保留必需列

    Returns:
        pandas DataFrame

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 缺少必需列或文件为空
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取文件
    try:
        df = pd.read_csv(file_path, sep='\t', dtype=str)
        logger.info(f"成功读取 {table_name}: {len(df)} 行, {len(df.columns)} 列")
    except Exception as e:
        raise ValueError(f"读取文件 {file_path} 失败: {e}")

    # 检查文件是否为空
    if df.empty:
        raise ValueError(f"文件 {file_path} 为空")

    # 校验必需列
    validate_table_columns(df, table_name)

    # 如果只需要必需列，进行过滤
    if required_only:
        required_cols = get_required_columns(table_name)
        if required_cols:
            existing_cols = [col for col in required_cols if col in df.columns]
            df = df[existing_cols]
            logger.info(f"过滤后保留列: {existing_cols}")

    # 去除空值行（基于关键列）
    key_cols = get_required_columns(table_name)
    if key_cols:
        before_count = len(df)
        # 只要关键列不全为空就保留
        df = df.dropna(subset=key_cols, how='all')
        after_count = len(df)
        if before_count != after_count:
            logger.warning(f"{table_name} 去除空值行: {before_count} -> {after_count}")

    return df


class TSVReader:
    """TSV文件读取器，支持批量读取和缓存"""

    def __init__(self, paths_config: Dict[str, str]):
        """
        初始化读取器

        Args:
            paths_config: 路径配置字典
        """
        self.paths_config = paths_config
        self.cache = {}  # 简单缓存

    def read_formula_tables(self) -> Dict[str, pd.DataFrame]:
        """
        读取所有方剂相关表

        Returns:
            表名到DataFrame的映射
        """
        formula_paths = {
            'D4_CPM_CHP': self.paths_config['formulas']['D4_CPM_CHP'],
            'D5_CPM_ICD11': self.paths_config['formulas']['D5_CPM_ICD11'],
            'D6_CHP': self.paths_config['formulas']['D6_CHP'],
            'D9_CHP_InChIKey': self.paths_config['formulas']['D9_CHP_InChIKey'],
            'D12_InChIKey': self.paths_config['formulas']['D12_InChIKey'],
        }

        return self._read_tables(formula_paths, "方剂侧")

    def read_disease_tables(self) -> Dict[str, pd.DataFrame]:
        """
        读取所有疾病相关表

        Returns:
            表名到DataFrame的映射
        """
        disease_paths = {
            'D19_ICD11_CUI': self.paths_config['diseases']['D19_ICD11_CUI'],
            'D20_ICD11_MeSH': self.paths_config['diseases']['D20_ICD11_MeSH'],
            'D22_CUI_targets': self.paths_config['targets']['D22_CUI_targets'],
            'D23_MeSH_targets': self.paths_config['targets']['D23_MeSH_targets'],
        }

        return self._read_tables(disease_paths, "疾病侧")

    def read_prediction_tables(self) -> Dict[str, pd.DataFrame]:
        """
        读取预测相关表

        Returns:
            表名到DataFrame的映射
        """
        prediction_paths = {
            'SD1_predicted': self.paths_config['predictions']['SD1_predicted'],
        }

        # 可选的实验边
        d13_path = self.paths_config['predictions'].get('D13_InChIKey_EntrezID')
        if d13_path and os.path.exists(d13_path):
            prediction_paths['D13_InChIKey_EntrezID'] = d13_path

        return self._read_tables(prediction_paths, "预测")

    def _read_tables(self, path_dict: Dict[str, str], category: str) -> Dict[str, pd.DataFrame]:
        """
        批量读取表

        Args:
            path_dict: 表名到路径的映射
            category: 类别名称（用于日志）

        Returns:
            表名到DataFrame的映射
        """
        logger.info(f"开始读取{category}表...")
        results = {}

        for table_name, file_path in path_dict.items():
            if table_name in self.cache:
                logger.info(f"从缓存读取 {table_name}")
                results[table_name] = self.cache[table_name]
                continue

            try:
                df = read_tsv(table_name, file_path)
                results[table_name] = df
                self.cache[table_name] = df  # 缓存
            except Exception as e:
                logger.error(f"读取{category}表 {table_name} 失败: {e}")
                raise

        logger.info(f"{category}表读取完成: {list(results.keys())}")
        return results