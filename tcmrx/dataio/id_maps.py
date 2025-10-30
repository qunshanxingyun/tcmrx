"""
TCM-RX ID映射模块
建立稳定的 string_id ↔ int 索引映射
"""

from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class IDMapper:
    """
    稳定的ID映射器
    保证同一ID每次运行得到同一索引
    """

    def __init__(self):
        self.mappings = {}  # {entity_type: {string_id: int_index}}
        self.reverse_mappings = {}  # {entity_type: {int_index: string_id}}
        self.next_indices = {}  # {entity_type: next_index}

    def add_entity_type(self, entity_type: str) -> None:
        """
        添加新的实体类型

        Args:
            entity_type: 实体类型名称
        """
        if entity_type not in self.mappings:
            self.mappings[entity_type] = {}
            self.reverse_mappings[entity_type] = {}
            self.next_indices[entity_type] = 0
            logger.info(f"添加实体类型: {entity_type}")

    def add_ids(self, entity_type: str, ids: Set[str]) -> None:
        """
        批量添加ID

        Args:
            entity_type: 实体类型
            ids: ID集合
        """
        if entity_type not in self.mappings:
            self.add_entity_type(entity_type)

        # 对ID进行排序，确保稳定性
        sorted_ids = sorted(ids)
        added_count = 0

        for string_id in sorted_ids:
            if string_id not in self.mappings[entity_type]:
                int_index = self.next_indices[entity_type]
                self.mappings[entity_type][string_id] = int_index
                self.reverse_mappings[entity_type][int_index] = string_id
                self.next_indices[entity_type] += 1
                added_count += 1

        logger.info(f"{entity_type}: 添加 {added_count}/{len(sorted_ids)} 个新ID, "
                    f"总计 {len(self.mappings[entity_type])} 个")

    def get_index(self, entity_type: str, string_id: str) -> int:
        """
        获取字符串ID对应的整数索引

        Args:
            entity_type: 实体类型
            string_id: 字符串ID

        Returns:
            整数索引
        """
        if entity_type not in self.mappings:
            raise ValueError(f"未知实体类型: {entity_type}")

        if string_id not in self.mappings[entity_type]:
            raise ValueError(f"{entity_type}中未知ID: {string_id}")

        return self.mappings[entity_type][string_id]

    def get_string_id(self, entity_type: str, int_index: int) -> str:
        """
        获取整数索引对应的字符串ID

        Args:
            entity_type: 实体类型
            int_index: 整数索引

        Returns:
            字符串ID
        """
        if entity_type not in self.reverse_mappings:
            raise ValueError(f"未知实体类型: {entity_type}")

        if int_index not in self.reverse_mappings[entity_type]:
            raise ValueError(f"{entity_type}中未知索引: {int_index}")

        return self.reverse_mappings[entity_type][int_index]

    def get_all_indices(self, entity_type: str) -> List[int]:
        """
        获取某个实体类型的所有整数索引

        Args:
            entity_type: 实体类型

        Returns:
            整数索引列表
        """
        if entity_type not in self.mappings:
            return []

        return list(self.mappings[entity_type].values())

    def get_count(self, entity_type: str) -> int:
        """
        获取某个实体类型的数量

        Args:
            entity_type: 实体类型

        Returns:
            数量
        """
        if entity_type not in self.mappings:
            return 0

        return len(self.mappings[entity_type])

    def convert_to_indices(self, entity_type: str, string_ids: List[str]) -> List[int]:
        """
        批量转换为整数索引

        Args:
            entity_type: 实体类型
            string_ids: 字符串ID列表

        Returns:
            整数索引列表
        """
        return [self.get_index(entity_type, sid) for sid in string_ids]

    def convert_to_string_ids(self, entity_type: str, indices: List[int]) -> List[str]:
        """
        批量转换为字符串ID

        Args:
            entity_type: 实体类型
            indices: 整数索引列表

        Returns:
            字符串ID列表
        """
        return [self.get_string_id(entity_type, idx) for idx in indices]

    def get_stats(self) -> Dict[str, int]:
        """
        获取统计信息

        Returns:
            各实体类型的数量统计
        """
        return {entity_type: len(mapping)
                for entity_type, mapping in self.mappings.items()}

    def print_stats(self) -> None:
        """打印统计信息"""
        stats = self.get_stats()
        logger.info("ID映射统计:")
        for entity_type, count in stats.items():
            logger.info(f"  {entity_type}: {count} 个")