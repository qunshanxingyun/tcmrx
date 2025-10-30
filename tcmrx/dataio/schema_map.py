"""
TCM-RX 数据列名映射与定义
严格依据《TCM-MKG文件列名说明.md》，不自造字段
"""

# 方剂侧相关文件必需列
FORMULA_REQUIRED_COLS = {
    # 中成药↔中药饮片 (D4)
    "D4_CPM_CHP": [
        "CPM_ID",           # 中成药标识符
        "CHP_ID",           # 中药饮片标识符
        "Dosage_ratio"      # 剂量比例
    ],

    # 中成药↔ICD11 (D5) - 监督对
    "D5_CPM_ICD11": [
        "CPM_ID",           # 中成药标识符
        "ICD11_code"        # ICD-11疾病代码
    ],

    # 中药饮片主表 (D6)
    "D6_CHP": [
        "CHP_ID",                    # 中药饮片唯一标识符
        "Chinese_herbal_pieces"      # 中文名称
    ],

    # 中药饮片↔化合物 (D9)
    "D9_CHP_InChIKey": [
        "CHP_ID",           # 中药饮片标识符
        "InChIKey",         # InChI化学标识符
        "Source"            # 来源数据库
    ],

    # 化合物详细信息 (D12)
    "D12_InChIKey": [
        "InChIKey",         # InChI化学标识符
        "SMILES",           # SMILES分子式
        "MolWt",            # 分子量
        "QED",              # 药效学质量评估
        "TPSA",             # 极表面积
        "MolLogP",          # 脂水分配系数
    ],
}

# 疾病侧相关文件必需列
DISEASE_REQUIRED_COLS = {
    # ICD11↔CUI (D19)
    "D19_ICD11_CUI": [
        "ICD11_code",       # ICD-11疾病代码
        "CUI"               # UML概念唯一标识符
    ],

    # ICD11↔MeSH (D20)
    "D20_ICD11_MeSH": [
        "ICD11_code",       # ICD-11疾病代码
        "MeSH"              # MeSH ID
    ],

    # CUI↔靶点 (D22)
    "D22_CUI_targets": [
        "CUI",              # UML概念唯一标识符
        "EntrezID"          # Entrez基因ID
    ],

    # MeSH↔靶点 (D23)
    "D23_MeSH_targets": [
        "MeSH",             # MeSH ID
        "EntrezID"          # Entrez基因ID
    ],
}

# 靶点相关文件必需列
TARGET_REQUIRED_COLS = {
    # 靶点映射信息 (D17)
    "D17_Target_Symbol_Mapping": [
        "UniProtID",        # 蛋白质UniProt ID
        "GeneSymbol",       # 基因符号
        "EntrezID",         # Entrez基因ID
        "Sequence"          # 蛋白质序列
    ],
}

# 预测文件必需列
PREDICTION_REQUIRED_COLS = {
    # 预测亲和力 (SD1)
    "SD1_predicted": [
        "InChIKey",                      # InChI化学标识符
        "EntrezID",                      # Entrez基因ID
        "predicted_binding_affinity"     # 预测结合亲和力
    ],

    # 实验边 (D13) - 可选
    "D13_InChIKey_EntrezID": [
        "InChIKey",         # InChI化学标识符
        "EntrezID"          # Entrez基因ID
    ],
}

# 所有必需列的汇总
ALL_REQUIRED_COLS = {
    **FORMULA_REQUIRED_COLS,
    **DISEASE_REQUIRED_COLS,
    **TARGET_REQUIRED_COLS,
    **PREDICTION_REQUIRED_COLS,
}

# 列名注释（来自TCM-MKG文件列名说明.md）
COLUMN_DESCRIPTIONS = {
    # 核心标识符
    "CPM_ID": "中成药唯一标识符，格式：CPM + 5位数字",
    "CHP_ID": "中药饮片唯一标识符，格式：CHP + 5位数字",
    "ICD11_code": "ICD-11疾病代码，国际疾病分类第11版代码",
    "InChIKey": "InChI化学标识符，国际化学标识符",
    "EntrezID": "Entrez基因ID，NCBI基因数据库标识符",
    "CUI": "UML概念唯一标识符，Unified Medical Language System概念ID",
    "MeSH": "MeSH ID，医学主题词表标识符",

    # 关系和权重
    "Dosage_ratio": "中药饮片在中成药中的剂量比例，部分为空",
    "predicted_binding_affinity": "预测结合亲和力，数值型预测结果",

    # 化合物特征
    "SMILES": "SMILES分子式，化学结构简化线性输入规范",
    "MolWt": "分子量，分子的近似分子量",
    "QED": "药效学质量评估，数值型药效质量指标",
    "TPSA": "极表面积，拓扑极表面积",
    "MolLogP": "脂水分配系数，脂溶性指标",

    # 来源信息
    "Source": "来源数据库，如：HERB2.0、TCMID等",
}


def get_required_columns(table_name: str) -> list:
    """
    获取指定表的必需列列表

    Args:
        table_name: 表名，如 'D4_CPM_CHP'

    Returns:
        必需列名列表
    """
    return ALL_REQUIRED_COLS.get(table_name, [])


def validate_table_columns(df, table_name: str) -> None:
    """
    验证DataFrame是否包含所有必需列

    Args:
        df: pandas DataFrame
        table_name: 表名

    Raises:
        ValueError: 如果缺少必需列
    """
    required_cols = get_required_columns(table_name)
    if not required_cols:
        return  # 该表无需验证

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"表 {table_name} 缺少必需列: {missing_cols}. "
            f"实际列: {list(df.columns)}"
        )


def get_column_description(column_name: str) -> str:
    """
    获取列名的描述

    Args:
        column_name: 列名

    Returns:
        列描述文本
    """
    return COLUMN_DESCRIPTIONS.get(column_name, "无描述")