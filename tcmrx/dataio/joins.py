"""
TCM-RX 数据连接模块
实现业务连接：疾病靶点集、方剂靶点集
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


def cpms_to_icd11(d5_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    CPM与ICD11一对多映射

    Args:
        d5_df: D5_CPM_ICD11表

    Returns:
        {CPM_ID: [ICD11_code_list]}
    """
    # 按CPM分组，收集所有ICD11代码
    result = d5_df.groupby('CPM_ID')['ICD11_code'].agg(list).to_dict()

    logger.info(f"CPM->ICD11映射: {len(result)}个CPM, "
                f"平均每个CPM对应{np.mean([len(v) for v in result.values()]):.1f}个疾病")
    return result


def icd11_to_targets(d19_df: pd.DataFrame, d20_df: pd.DataFrame,
                    d22_df: pd.DataFrame, d23_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    ICD11经CUI/MeSH汇总到EntrezID集合

    Args:
        d19_df: ICD11->CUI映射
        d20_df: ICD11->MeSH映射
        d22_df: CUI->EntrezID映射
        d23_df: MeSH->EntrezID映射

    Returns:
        {ICD11_code: {EntrezID_set}}
    """
    # 构建映射字典
    icd11_to_cuis = d19_df.groupby('ICD11_code')['CUI'].apply(set).to_dict()
    icd11_to_meshes = d20_df.groupby('ICD11_code')['MeSH'].apply(set).to_dict()
    cui_to_targets = d22_df.groupby('CUI')['EntrezID'].apply(set).to_dict()
    mesh_to_targets = d23_df.groupby('MeSH')['EntrezID'].apply(set).to_dict()

    # 合并所有ICD11
    all_icd11 = set(icd11_to_cuis.keys()) | set(icd11_to_meshes.keys())
    result = {}

    for icd11 in all_icd11:
        targets = set()

        # 通过CUI获取靶点
        cuis = icd11_to_cuis.get(icd11, set())
        for cui in cuis:
            targets.update(cui_to_targets.get(cui, set()))

        # 通过MeSH获取靶点
        meshes = icd11_to_meshes.get(icd11, set())
        for mesh in meshes:
            targets.update(mesh_to_targets.get(mesh, set()))

        if targets:
            result[icd11] = targets

    logger.info(f"ICD11->Target映射: {len(result)}个疾病有靶点信息, "
                f"平均每个疾病{np.mean([len(v) for v in result.values()]):.1f}个靶点")
    return result


def cpms_to_chp(d4_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
    """
    CPM到CHP映射，保留剂量比例

    Args:
        d4_df: D4_CPM_CHP表

    Returns:
        {CPM_ID: [(CHP_ID, dosage_ratio), ...]}
    """
    # 填充缺失的剂量比例为1.0
    d4_df = d4_df.copy()
    d4_df['Dosage_ratio'] = pd.to_numeric(d4_df['Dosage_ratio'], errors='coerce').fillna(1.0)

    # 确保所有dosage_ratio都是float类型
    d4_df['Dosage_ratio'] = d4_df['Dosage_ratio'].astype(float)

    # 按CPM分组
    result = {}
    for cpm_id, group in d4_df.groupby('CPM_ID'):
        result[cpm_id] = [(row['CHP_ID'], float(row['Dosage_ratio']))
                         for _, row in group.iterrows()]

    logger.info(f"CPM->CHP映射: {len(result)}个CPM, "
                f"平均每个CPM包含{np.mean([len(v) for v in result.values()]):.1f}个药材")
    return result


def _normalize_pathway_name(name: str) -> str:
    return name.strip().lower().replace(' ', '_') if isinstance(name, str) else ''


def chp_to_chemicals(d9_df: pd.DataFrame, d12_df: pd.DataFrame = None) -> Dict[str, Set[str]]:
    """
    CHP到化合物映射

    Args:
        d9_df: D9_CHP_InChIKey表
        d12_df: D12_InChIKey表（可选，用于过滤）

    Returns:
        {CHP_ID: {InChIKey_set}}
    """
    # 如果提供了D12，过滤出存在的化合物
    if d12_df is not None:
        valid_chemicals = set(d12_df['InChIKey'].unique())
        d9_df = d9_df[d9_df['InChIKey'].isin(valid_chemicals)]
        logger.info(f"过滤后保留 {len(d9_df)} 条CHP->化合物记录（基于D12）")

    # 按CHP分组
    result = d9_df.groupby('CHP_ID')['InChIKey'].apply(set).to_dict()

    logger.info(f"CHP->Chemical映射: {len(result)}个CHP, "
                f"平均每个CHP对应{np.mean([len(v) for v in result.values()]):.1f}个化合物")
    return result


def chemicals_to_pathways(d12_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Extract pathway annotations per chemical from D12."""

    if d12_df is None or d12_df.empty:
        logger.info("D12缺失或为空，跳过通路信息构建")
        return {}

    pathway_columns = [
        col for col in d12_df.columns
        if isinstance(col, str) and col.lower().startswith('pathway')
    ]

    if not pathway_columns:
        logger.info("D12中未找到Pathway列，跳过通路信息构建")
        return {}

    mapping: Dict[str, Set[str]] = {}

    for inchikey, group in d12_df.groupby('InChIKey'):
        pathways: Set[str] = set()
        for col in pathway_columns:
            for value in group[col].dropna().unique():
                normalized = _normalize_pathway_name(str(value))
                if normalized:
                    pathways.add(normalized)
        if pathways:
            mapping[inchikey] = pathways

    logger.info("化合物->通路映射: %d 个化合物包含通路注释", len(mapping))
    return mapping


def build_target_to_pathways(
    chemical_to_targets_map: Dict[str, List[Tuple[str, float]]],
    chemical_to_pathways_map: Dict[str, Set[str]],
    prefix: str = 'pathway:',
    max_pathways_per_target: int = 32,
    min_weight: float = 1e-4,
) -> Dict[str, List[Tuple[str, float]]]:
    """Aggregate pathway distributions for each target via shared chemicals."""

    if not chemical_to_pathways_map:
        return {}

    pathway_counter: Dict[str, Counter] = defaultdict(Counter)

    for chemical, target_list in chemical_to_targets_map.items():
        pathways = chemical_to_pathways_map.get(chemical)
        if not pathways:
            continue

        for target_id, affinity in target_list:
            try:
                affinity_float = max(float(affinity), 0.0)
            except (TypeError, ValueError):
                affinity_float = 0.0

            if affinity_float <= 0:
                continue

            for pathway in pathways:
                pathway_id = f"{prefix}{pathway}"
                pathway_counter[target_id][pathway_id] += affinity_float

    target_to_pathways: Dict[str, List[Tuple[str, float]]] = {}

    for target_id, counter in pathway_counter.items():
        if not counter:
            continue
        total = sum(counter.values())
        if total <= 0:
            continue
        normalized = [
            (pathway_id, weight / total)
            for pathway_id, weight in counter.most_common(max_pathways_per_target)
            if weight / total >= min_weight
        ]
        if normalized:
            target_to_pathways[target_id] = normalized

    if target_to_pathways:
        logger.info(
            "目标->通路映射: %d 个靶点具备通路投影 (max=%d)",
            len(target_to_pathways),
            max(len(v) for v in target_to_pathways.values()),
        )

    return target_to_pathways


def chemicals_to_targets(sd1_df: pd.DataFrame, experimental_df: pd.DataFrame = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    化合物到靶点映射

    Args:
        sd1_df: SD1预测表
        experimental_df: 实验边表（可选）

    Returns:
        {InChIKey: [(EntrezID, weight), ...]}
    """
    result = {}

    # 处理实验边（优先级更高）
    if experimental_df is not None:
        for inchikey, group in experimental_df.groupby('InChIKey'):
            result[inchikey] = [(entrez_id, 1.0) for entrez_id in group['EntrezID']]
        logger.info(f"实验边: {len(result)}个化合物")

    # 处理预测边
    for inchikey, group in sd1_df.groupby('InChIKey'):
        targets = [(row['EntrezID'], row['predicted_binding_affinity'])
                  for _, row in group.iterrows()]

        if inchikey in result:
            # 合并实验边和预测边
            result[inchikey].extend(targets)
        else:
            result[inchikey] = targets

    logger.info(f"化合物->Target映射: {len(result)}个化合物, "
                f"平均每个化合物{np.mean([len(v) for v in result.values()]):.1f}个靶点")
    return result


def formulas_to_targets(
    cpms_to_chp_map: Dict[str, List[Tuple[str, float]]],
    chp_to_chemicals_map: Dict[str, Set[str]],
    chemical_to_targets_map: Dict[str, List[Tuple[str, float]]],
    *,
    chemical_to_pathways_map: Dict[str, Set[str]] = None,
    pathway_config: Dict = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    CPM→CHP→Chemical→Target聚合得到方剂靶点集合

    Args:
        cpms_to_chp_map: CPM->CHP映射（含剂量）
        chp_to_chemicals_map: CHP->Chemical映射
        chemical_to_targets_map: Chemical->Target映射（含权重）

    Returns:
        {CPM_ID: [(EntrezID, weight), ...]}
    """
    result = {}
    pathway_enabled = False
    formula_pathway_cfg = {}

    if pathway_config:
        pathway_enabled = pathway_config.get('enabled', True)
        formula_pathway_cfg = pathway_config.get('formula', {}) if pathway_enabled else {}

    prefix = (pathway_config or {}).get('prefix', 'pathway:')

    for cpm_id, chp_list in cpms_to_chp_map.items():
        target_weights: Dict[str, float] = defaultdict(float)
        pathway_scores: Counter = Counter()

        for chp_id, dosage_ratio in chp_list:
            # 安全检查：确保dosage_ratio是单个数值
            try:
                dosage_float = float(dosage_ratio)
            except (TypeError, ValueError):
                logger.warning(f"无效的剂量比例 {dosage_ratio} (CPM: {cpm_id}, CHP: {chp_id})，使用默认值1.0")
                dosage_float = 1.0

            # 获取CHP对应的化合物
            chemicals = chp_to_chemicals_map.get(chp_id, set())

            for chemical in chemicals:
                # 获取化合物对应的靶点
                targets = chemical_to_targets_map.get(chemical, [])

                # 传播剂量权重
                chem_targets = targets
                for target_id, affinity in chem_targets:
                    try:
                        affinity_float = float(affinity)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"无效的亲和力值 {affinity} (Chemical: {chemical}, Target: {target_id})，使用默认值1.0"
                        )
                        affinity_float = 1.0

                    weight = dosage_float * affinity_float
                    target_weights[target_id] += weight

                if pathway_enabled and formula_pathway_cfg.get('enabled', True):
                    pathways = chemical_to_pathways_map.get(chemical) if chemical_to_pathways_map else None
                    if pathways:
                        chem_strength = max(
                            (max(float(a), 0.0) for _, a in chem_targets if a is not None),
                            default=1.0,
                        )
                        pathway_weight = dosage_float * chem_strength
                        for pathway in pathways:
                            normalized = _normalize_pathway_name(pathway)
                            if normalized:
                                pathway_scores[f"{prefix}{normalized}"] += pathway_weight

        aggregated_targets = sorted(target_weights.items(), key=lambda x: x[1], reverse=True)

        if aggregated_targets:
            result[cpm_id] = aggregated_targets

        if pathway_scores:
            max_items = formula_pathway_cfg.get('max_items', 500)
            min_weight = formula_pathway_cfg.get('min_weight', 1e-4)
            sorted_pathways = [
                (pathway_id, weight)
                for pathway_id, weight in pathway_scores.most_common(max_items)
                if weight >= min_weight
            ]
            if sorted_pathways:
                result.setdefault(cpm_id, [])
                result[cpm_id].extend(sorted_pathways)

    logger.info(f"方剂->Target聚合: {len(result)}个方剂有靶点信息, "
                f"平均每个方剂{np.mean([len(v) for v in result.values()]):.1f}个靶点")
    return result


def diseases_to_targets(
    icd11_to_targets_map: Dict[str, Set[str]],
    target_to_pathways_map: Dict[str, List[Tuple[str, float]]] = None,
    pathway_config: Dict = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    疾病到靶点映射（统一权重为1.0）

    Args:
        icd11_to_targets_map: ICD11->Target集合

    Returns:
        {ICD11_code: [(EntrezID, 1.0), ...]}
    """
    result = {}

    pathway_enabled = False
    disease_cfg = {}
    if pathway_config:
        pathway_enabled = pathway_config.get('enabled', True)
        disease_cfg = pathway_config.get('disease', {}) if pathway_enabled else {}

    prefix = (pathway_config or {}).get('prefix', 'pathway:')
    blend = disease_cfg.get('blend', 0.3)
    max_items = disease_cfg.get('max_items', 300)
    min_weight = disease_cfg.get('min_weight', 1e-4)

    for icd11_code, targets in icd11_to_targets_map.items():
        base_targets = [(target_id, 1.0) for target_id in targets]

        if (
            pathway_enabled
            and disease_cfg.get('enabled', True)
            and target_to_pathways_map
            and blend > 0
        ):
            pathway_scores: Counter = Counter()
            for target_id, weight in base_targets:
                pathway_list = target_to_pathways_map.get(target_id, [])
                for pathway_id, pathway_weight in pathway_list:
                    if not pathway_id.startswith(prefix):
                        pathway_id = f"{prefix}{_normalize_pathway_name(pathway_id)}"
                    contribution = blend * weight * pathway_weight
                    if contribution > 0:
                        pathway_scores[pathway_id] += contribution

            if pathway_scores:
                sorted_scores = [
                    (pathway_id, score)
                    for pathway_id, score in pathway_scores.most_common(max_items)
                    if score >= min_weight
                ]
                if sorted_scores:
                    base_targets = base_targets + sorted_scores

        result[icd11_code] = base_targets

    logger.info(f"疾病->Target映射: {len(result)}个疾病有靶点信息")
    return result