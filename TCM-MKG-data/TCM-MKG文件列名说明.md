## 文件结构概览

TCM-MKG（Traditional Chinese Medicine Knowledge Graph）中医药知识图谱包含25个数据文件，涵盖中医术语、中成药、中药饮片、化学成分、靶点、疾病等多维度数据。

---

## D1_TCM_terminology.tsv

**文件描述：** 中医药术语表，包含中医核心概念和治疗原则

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| TCMT_ID | 术语唯一标识符 | 格式：TCMT + 5位数字（如TCMT00001） |
| Chinese_group | 中文分类/分组 | 如：治则、治法等中医概念分类 |
| English_group | 英文分类/分组 | 如：Treatment principles、Treatment methods |
| Chinese_term | 中文术语 | 中医药专业术语，如：急则治标、缓则治本 |
| Pinyin_term | 拼音标注 | 中文术语的拼音读法，如：jí zé zhì biāo |
| Chinese_synonyms | 中文同义词 | 术语的其他中文表达，部分为空 |
| English_term | 英文术语 | 术语的标准英文翻译 |
| Synonyms | 英文同义词 | 术语的其他英文表达，部分为空 |
| English_definition_description | 英文定义描述 | 详细的英文定义和解释说明 |

---

## D2_Chinese_patent_medicine.tsv

**文件描述：** 中成药基本信息表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CPM_ID | 中成药唯一标识符 | 格式：CPM + 5位数字 |
| Chinese_patent_medicine | 中文名称 | 中成药的中文名称，如：阿归养血胶囊 |
| Pinyin_term | 拼音标注 | 中成药名称的拼音 |
| Routes_of_administration | 给药途径 | 如：Oral（口服）等给药方式 |

---

## D3_CPM_TCMT.tsv

**文件描述：** 中成药与中医药术语的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CPM_ID | 中成药标识符 | 关联D2的中成药ID |
| TCMT_ID | 术语标识符 | 关联D1的中医药术语ID |
| Chinese_group | 中文分类 | 中医治疗原则/方法的分类 |
| English_group | 英文分类 | 对应的英文分类 |
| Chinese_term | 中文术语 | 具体的中医药术语 |
| Pinyin_term | 拼音标注 | 术语的拼音 |
| English_term | 英文术语 | 术语的英文翻译 |
| Synonyms | 英文同义词 | 术语的其他英文表达 |

---

## D4_CPM_CHP.tsv

**文件描述：** 中成药与中药饮片的成分关联表（体现中成药包含哪些中药饮片）

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CPM_ID | 中成药标识符 | 关联D2的中成药ID |
| CHP_ID | 中药饮片标识符 | 关联D6的中药饮片ID |
| Dosage_ratio | 剂量比例 | 中药饮片在中成药中的剂量比例，部分为空 |

---

## D5_CPM_ICD11.tsv

**文件描述：** 中成药与ICD-11疾病分类的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CPM_ID | 中成药标识符 | 关联D2的中成药ID |
| ICD11_code | ICD-11疾病代码 | 国际疾病分类第11版代码 |

---

## D6_Chinese_herbal_pieces.tsv

**文件描述：** 中药饮片基础信息表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CHP_ID | 中药饮片唯一标识符 | 格式：CHP + 5位数字 |
| Chinese_herbal_pieces | 中文名称 | 中药饮片的中文名称 |
| Chinese_synonyms | 中文同义词 | 药材的其他中文名称 |
| Pinyin_term | 拼音标注 | 药材名称的拼音 |
| English_term | 英文名称 | 药材的英文名称 |
| Sources | 来源分类 | 如：Viridiplantae（绿色植物界） |

---

## D7_CHP_Medicinal_properties.tsv

**文件描述：** 中药饮片的药性信息表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CHP_ID | 中药饮片标识符 | 关联D6的中药饮片ID |
| Medicinal_properties | 药性特征 | 如：Sweet medicinal（甘味药） |
| Class | 药性类别 | 如：Medicinal flavor（药味） |
| x_rank | X轴排名 | 数值型，在二维空间中的X坐标 |
| y_rank | Y轴排名 | 数值型，在二维空间中的Y坐标 |

---

## D8_CHP_PO.tsv

**文件描述：** 中药饮片与药用生物来源的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CHP_ID | 中药饮片标识符 | 关联D6的中药饮片ID |
| Sources | 生物分类来源 | 如：Viridiplantae |
| species_name | 物种名称 | 拉丁文学名 |
| species_ID | 物种ID | 生物分类学中的物种标识符 |

---

## D9_CHP_InChIKey.tsv

**文件描述：** 中药饮片与InChIKey化学标识符的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CHP_ID | 中药饮片标识符 | 关联D6的中药饮片ID |
| InChIKey | InChI化学标识符 | 国际化学标识符，用于唯一标识化合物 |
| Source | 来源数据库 | 如：HERB2.0、TCMID等 |

---

## D10_Pharmacognostic_origin.tsv

**文件描述：** 生药学来源信息表，包含生物分类学层次结构

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| kingdom_Name | 界名称 | 如：Viridiplantae |
| kingdom_ID | 界ID | 生物分类学标识符 |
| phylum_Name | 门名称 | 如：Streptophyta |
| phylum_ID | 门ID | 生物分类学标识符 |
| class_Name | 纲名称 | 如：Magnoliopsida |
| class_ID | 纲ID | 生物分类学标识符 |
| order_Name | 目名称 | 如：Asterales |
| order_ID | 目ID | 生物分类学标识符 |
| family_Name | 科名称 | 如：Asteraceae |
| family_ID | 科ID | 生物分类学标识符 |
| genus_Name | 属名称 | 如：Doronicum |
| genus_ID | 属ID | 生物分类学标识符 |
| species_name | 物种名称 | 拉丁文学名 |
| species_ID | 物种ID | 生物分类学标识符 |

---

## D11_PO_InChIKey.tsv

**文件描述：** 药用生物来源与InChIKey的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| SpeciesID | 物种ID | 生物分类学物种标识符 |
| InChIKey | InChI化学标识符 | 国际化学标识符 |
| Source | 来源数据库 | 如：TCMID、TCMbank、HERB2.0 |

---

## D12_InChIKey.tsv

**文件描述：** 化合物详细信息表，包含分子结构和化学性质

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| InChIKey | InChI化学标识符 | 国际化学标识符 |
| SMILES | SMILES分子式 | 化学结构简化线性输入规范 |
| InChI | InChI国际化学标识符 | 国际化学标识符完整格式 |
| Molecular_Formula | 分子式 | 如：C30H50O8 |
| Chiral_Scaffold | 手性骨架结构 | 分子的三维结构描述 |
| Achiral_Scaffold | 非手性骨架结构 | 分子的二维结构描述 |
| QED | 药效学质量评估 | 数值型药效质量指标 |
| MolWt | 分子量 | 分子的近似分子量 |
| ExactMolWt | 精确分子量 | 分子的精确分子量 |
| TPSA | 极表面积 | 拓扑极表面积 |
| MolLogP | 脂水分配系数 | 脂溶性指标 |
| NumHAcceptors | 氢键受体数 | 分子中氢键受体数量 |
| NumHDonors | 氢键给体数 | 分子中氢键给体数量 |
| NumRotatableBonds | 可旋转键数 | 分子中可旋转化学键数量 |
| MolMR | 分子折射率 | 分子折射率 |
| FractionCSP3 | CSP3比例 | sp3杂化碳原子比例 |
| Pathway1 | 代谢通路1 | 化合物参与的代谢通路 |
| Pathway2 | 代谢通路2 | 化合物参与的代谢通路 |
| Pathway3 | 代谢通路3 | 化合物参与的代谢通路 |
| Superclass1 | 超级分类1 | 化合物化学分类 |
| Superclass2 | 超级分类2 | 化合物化学分类 |
| Superclass3 | 超级分类3 | 化合物化学分类 |
| Class1 | 化学分类1 | 化合物具体化学分类 |
| Class2 | 化学分类2 | 化合物具体化学分类 |
| Class3 | 化学分类3 | 化合物具体化学分类 |
| IsGlycoside | 是否为糖苷 | 布尔值，Yes/No |

---

## D13_InChIKey_EntrezID.tsv

**文件描述：** 化合物与基因数据库Entrez ID的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| InChIKey | InChI化学标识符 | 国际化学标识符 |
| EntrezID | Entrez基因ID | NCBI基因数据库标识符 |

---

## D14_InChIKey_SourceID.tsv

**文件描述：** 化合物与其他数据库源ID的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| InChIKey | InChI化学标识符 | 国际化学标识符 |
| SourceID | 源数据库ID | 如：TTD数据库中的T82668 |
| Source | 来源数据库 | 如：TTD |

---

## D15_InChIKey_distance.tsv

**文件描述：** 化合物结构相似性距离矩阵表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| InChIKey | InChI化学标识符 | 化合物1的标识符 |
| Descriptors1 | 描述符1 | 结构描述符数值特征 |
| Descriptors2 | 描述符2 | 结构描述符数值特征 |
| Fingerprints1 | 指纹1 | 化学结构指纹特征 |
| Fingerprints2 | 指纹2 | 化学结构指纹特征 |
| GAT1 | 图注意力网络1 | 基于图神经网络的相似性特征 |
| GAT2 | 图注意力网络2 | 基于图神经网络的相似性特征 |
| GAE1 | 图自编码器1 | 基于图自编码器的相似性特征 |
| GAE2 | 图自编码器2 | 基于图自编码器的相似性特征 |
| GCN1 | 图卷积网络1 | 基于图卷积网络的相似性特征 |
| GCN2 | 图卷积网络2 | 基于图卷积网络的相似性特征 |

---

## D16_Protein_protein_interactions.tsv

**文件描述：** 蛋白质相互作用关系表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| EntrezID1 | 蛋白质1的Entrez ID | NCBI基因数据库标识符 |
| EntrezID2 | 蛋白质2的Entrez ID | NCBI基因数据库标识符 |

---

## D17_Target_Symbol_Mapping.tsv

**文件描述：** 靶点基因符号映射表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| UniProtID | 蛋白质UniProt ID | 蛋白质数据库标识符 |
| GeneSymbol | 基因符号 | 基因的标准符号表示 |
| EntrezID | Entrez基因ID | NCBI基因数据库标识符 |
| ENSGID | Ensembl基因ID | Ensembl基因组数据库标识符 |
| Sequence | 蛋白质序列 | 氨基酸序列字符串 |

---

## D18_ICD11.tsv

**文件描述：** ICD-11国际疾病分类表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| ICD11_code | ICD-11疾病代码 | 国际疾病分类第11版代码 |
| BlockId | 块ID | 分类层级标识符 |
| English_term | 英文术语 | 疾病的英文名称 |
| Chinese_term | 中文术语 | 疾病的中文名称 |
| ClassKind | 分类类型 | 如：chapter、block、category |
| DepthInKind | 分类深度 | 在分类体系中的层级深度 |
| IsResidual | 是否为残余类 | 布尔值，表示是否为残余分类 |
| ChapterNo | 章节号 | 疾病分类章节编号 |
| BrowserLink | 浏览器链接 | 相关分类的浏览链接 |
| isLeaf | 是否为叶子节点 | 布尔值，是否为最细粒度分类 |
| Primary tabulation | 主要列表 | 主要的疾病分类列表 |
| Grouping1-5 | 分组1-5 | 多层级分组信息 |
| Version | 版本信息 | 数据版本时间戳 |

---

## D19_ICD11_CUI.tsv

**文件描述：** ICD-11与UMLS CUI的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| ICD11_code | ICD-11疾病代码 | 国际疾病分类第11版代码 |
| CUI | UML概念唯一标识符 | Unified Medical Language System概念ID |
| Semantic_type | 语义类型 | UMLS语义分类 |
| Term | 术语名称 | 具体的疾病或概念名称 |

---

## D20_ICD11_MeSH.tsv

**文件描述：** ICD-11与MeSH医学主题词的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| ICD11_code | ICD-11疾病代码 | 国际疾病分类第11版代码 |
| MeSH | MeSH ID | 医学主题词表标识符 |

---

## D21_ICD11_DOID.tsv

**文件描述：** ICD-11与人类疾病本体（DOID）的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| ICD11_code | ICD-11疾病代码 | 国际疾病分类第11版代码 |
| DOID | 疾病本体ID | Disease Ontology标识符 |

---

## D22_CUI_targets.tsv

**文件描述：** UML概念与靶点基因的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| CUI | UML概念唯一标识符 | Unified Medical Language System概念ID |
| EntrezID | Entrez基因ID | NCBI基因数据库标识符 |

---

## D23_MeSH_targets.tsv

**文件描述：** MeSH医学主题词与靶点基因的关联表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| MeSH | MeSH ID | 医学主题词表标识符 |
| EntrezID | Entrez基因ID | NCBI基因数据库标识符 |

---

## D24_DOID_targets.tsv

**文件描述：** 疾病本体（DOID）与靶点基因的关联表

| 列名       | 说明         | 数据特点                |
| -------- | ---------- | ------------------- |
| DOID     | 疾病本体ID     | Disease Ontology标识符 |
| EntrezID | Entrez基因ID | NCBI基因数据库标识符        |

---

## SD1_predicted_InChIKey_EntrezID.tsv

**文件描述：** 预测的化合物-靶点结合亲和力表

| 列名 | 说明 | 数据特点 |
|------|------|----------|
| InChIKey | InChI化学标识符 | 化合物标识符 |
| EntrezID | Entrez基因ID | 靶点基因ID |
| predicted_binding_affinity | 预测结合亲和力 | 数值型预测结果 |

---

## 关联文件关系总结

### 1. 中药成分层级关系
- **D6** → **D8** → **D10**：中药饮片 → 生物来源 → 生物分类学层次
- **D6** → **D9** → **D12**：中药饮片 → 化合物标识符 → 化合物详细信息

### 2. 中成药成分构成
- **D2** → **D4** → **D6**：中成药 → 中药饮片配比 → 中药饮片信息
- **D2** → **D3** → **D1**：中成药 → 治疗原则/方法 → 中医术语

### 3. 疾病分类关联
- **D18** → **D19** → **D20** → **D21**：ICD-11 → UML → MeSH → 疾病本体
- **D19** → **D22**：UML概念 → 靶点基因
- **D20** → **D23**：MeSH → 靶点基因
- **D21** → **D24**：疾病本体 → 靶点基因

### 4. 化合物-靶点关联
- **D12** → **D13** → **D17**：化合物 → 基因ID → 基因信息
- **D12** → **D16**：化合物 → 蛋白质相互作用
- **D13** → **D22/D23/D24**：基因ID → 多种疾病靶点关联
- **SD1**：化合物-靶点预测结合亲和力

### 5. 数据库关联
- **D11/D13/D14**：不同来源的InChIKey映射关系
- **D15**：化合物结构相似性计算

---

## 数据库缩写说明及案例

- **CPM**: Chinese Patent Medicine (中成药)
- **CHP**: Chinese Herbal Pieces (中药饮片)
- **PO**: Pharmacognostic Origins (生药学来源)
- **TCMT**: TCM Terminology (中医药术语)
- **InChIKey**: IUPAC International Chemical Identifier Key
- **ICD11**: International Classification of Diseases 11th Edition
- **MeSH**: Medical Subject Headings (医学主题词表)
- **DOID**: Disease Ontology ID (疾病本体ID)
- **CUI**: Concept Unique Identifier (UML概念唯一标识符)
- **EntrezID**: NCBI基因数据库标识符
- **UniProtID**: 蛋白质数据库标识符
- **SMILES**: Simplified Molecular Input Line Entry System
- **QED**: Quantitative Estimate of Drug-likeness
  具体例子说明

### 1. MeSH (医学主题词表) - 实际例子

场景： 一篇关于"黄连治疗糖尿病"的研究论文

MeSH词汇表应用：
原始论文关键词：
- "黄连" → MeSH: "Coptis" (D003056)
- "糖尿病" → MeSH: "Diabetes Mellitus" (D003926)
- "炎症" → MeSH: "Inflammation" (D007211)
- "血糖" → MeSH: "Blood Glucose" (D001920)

在TCM-MKG中的映射：
D20_ICD11_MeSH.tsv 可能包含：
ICD11_code    MeSH
1A80.0        D003056  # 黄连相关的某种感染性疾病
5A80.0        D003926  # 糖尿病

实际应用： 研究人员可以用"Diabetes Mellitus"这个标准词汇来搜索所有相关的中药研究，无论原文用的是"糖尿病"、"diabetes"还是"糖代谢异常"。

### 2. ICD-11 (国际疾病分类) - 实际例子

场景： 中成药"消渴丸"的适应症

ICD-11编码体系：
中成药：消渴丸 (D2中的某条记录)
适应症ICD-11编码：
- 5A80.0  # 糖尿病
- 5A80.1  # 胰岛素依赖型糖尿病
- 5A80.2  # 非胰岛素依赖型糖尿病
- EB81.0  # 妊娠期糖尿病

在D5_CPM_ICD11.tsv中的表现：
CPM_ID    ICD11_code
CPM12345  5A80.0     # 消渴丸可用于糖尿病
CPM12345  5A80.2     # 特别是非胰岛素依赖型糖尿病

临床意义： 医生可以在电子病历系统中输入糖尿病ICD代码，系统会推荐相关的中成药，实现中西医结合的智能诊疗。

### 3. DOID (疾病本体) - 实际例子

场景： "当归"治疗多种妇科疾病的机制研究

DOID层次结构：
DOID:0050959  # 生殖系统疾病
├── DOID:0080139  # 妇科疾病
│   ├── DOID:0080142  # 月经失调
│   │   ├── DOID:0060746  # 痛经
│   │   └── DOID:0080143  # 月经不调
│   └── DOID:0080145  # 更年期综合征
└── DOID:0050961  # 妊娠相关疾病

当归相关靶点映射 (D24_DOID_targets.tsv)：
DOID                 EntrezID    GeneSymbol
DOID:0060746         5728        PTGS2    # 痛经相关靶点
DOID:0080143         5594        ESR1     # 雌激素受体
DOID:0080145         3535        ESR2     # 雌激素受体β

研究价值： 研究人员可以发现当归通过调节ESR1/ESR2靶点对多种妇科疾病都有治疗作用，体现了中医"异病同治"的理论。

### 4. InChIKey (化学标识符) - 实际例子

场景： "黄芩"的主要化学成分研究

黄芩的化学成分：
化合物1: 黄芩苷 (Baicalin)
- SMILES: OC1=C(C=CC(=C1O)C2=CC(=C(C=C2)O[C@H]3[C@@H](O[C@H]4C[C@@H](O[C@@H](C4=O)C3=O)CO)O)O)O
- InChIKey: UYTPKODJCMYKBU-GASJCYHMSA-N
- 分子式: C21H18O11
- InChI: InChI=1S/C21H18O11/c22-7-15-16(28)18(30-15)20(32)14-6-10(24)12-4-3-9(23)5-13(12)26-14/h3-7,15,18,20,22,28,30H,1-2H2

在D12_InChIKey.tsv中的记录：
InChIKey                                SMILES                                                                
UYTPKODJCMYKBU-GASJCYHMSA-N    OC1=C(C=CC(=C1O)C2=CC(=C(C=C2)O[C@H]3[C@@H](O[C@H]4C[C@@H](O[C@@H](C4=O)C3=O)CO)O)O)

实际应用： 研究者可以精确地搜索黄芩苷的所有相关研究，不用担心名称变体（如"baicalin"、"黄芩苷"、"baicalein-7-glucuronide"等）。

### 5. CUI (概念唯一标识符) - 实际例子

场景： 跨数据库的中药"人参"研究整合

UMLS CUI整合：
``人参"在不同系统中的映射：
- 中文名: "人参" → CUI: C0030278
- 英文名: "Ginseng" → CUI: C0030278
- 拉丁名: "Panax ginseng" → CUI: C0030278
- 药理作用: "免疫调节" → CUI: C0012634

在D19_ICD11_CUI.tsv中的表现：
ICD11_code    CUI                 Semantic_type           Term
DA4A.0        C0030278           Pharmacologic Substance  Ginseng
DA4A.0        C0012634           Physiologic Process      Immune response modulation

在D22_CUI_targets.tsv中：
CUI                 EntrezID
C0030278           3576      # 人参相关靶点
C0012634           3552      # 免疫调节相关靶点

**数据整合价值：** 研究人员可以通过CUI C0030278，同时获取人参在中医、西医、药理、临床等各个维度的信息，实现真正意义上的知识整合。

## 中医药数据中的典型应用场景

### 场景1：临床用药推荐

患者诊断: 2型糖尿病 (ICD-11: 5A80.2)
↓
系统推荐:
- 中成药: 消渴丸 (CPM12345)
- 中药饮片: 黄芩 (CHP00891), 黄连 (CHP01562)
- 化学成分: 黄芩苷 (InChIKey: UYTPKODJCMYKBU-GASJCYHMSA-N)
- 作用靶点: PPARG (EntrezID: 5468)
- 疾病关联: 糖尿病 (DOID: 0080584)

### 场景2：中药药理机制研究

中药: 当归 (D6_Chinese_herbal_pieces)
↓
化学成分: 阿魏酸 (InChIKey: FWMNVHUIDOGXXF-UHFFFAOYSA-N)
↓
疾病关联: 痛经 (DOID: 0060746)、月经不调 (DOID: 0080143)
↓
作用靶点: ESR1 (EntrezID: 2099)、PTGS2 (EntrezID: 5728)
↓
临床效果: 通过调节雌激素受体和炎症因子改善妇科疾病

### 场景3：中成药配伍分析

中成药: 逍遥丸 (CPM23456)
↓
组成饮片:
- 柴胡 (CHP00123)
- 当归 (CHP00891)
- 白芍 (CHP01567)
- 白术 (CHP02234)
↓
对应ICD-11编码:
- 抑郁障碍 (6A70.0)
- 焦虑障碍 (6A70.1)
- 胃肠功能紊乱 (DB80.0)

### 场景4：中药化学成分网络分析

中药: 黄芪 (CHP00345)
↓
活性成分群:
- 毛蕊异黄酮 (InChIKey: GQYACZPWUKQKGZ-UHFFFAOYSA-N)
- 黄芪甲苷 (InChIKey: JXXVDIRWIVZJLO-REOHCLBHSA-N)
- 芒柄花黄素 (InChIKey: ZZZMOPACZSLIPG-UHFFFAOYSA-N)
↓
共同靶点网络:
- 免疫调节: TNF, IL6, IL1B
- 抗氧化: SOD, CAT, GPX
- 抗炎: COX2, NOS2
---

*最后更新：2025年10月23日*
*数据来源：TCM-MKG中医药知识图谱*