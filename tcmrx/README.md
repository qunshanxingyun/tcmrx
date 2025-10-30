# TCM-RX: 基于双塔对比学习的中医药推荐系统

## 项目概述

TCM-RX是一个基于双塔对比学习的中医药推荐系统，实现疾病与方剂的智能匹配。系统采用极简主义设计，专注于核心功能，避免过度工程化。

### 核心特性

- **双塔架构**: 疾病塔和方剂塔独立编码，支持高效推理
- **对比学习**: 使用InfoNCE损失进行端到端训练
- **内存优先**: 全流程在内存中处理，避免中间文件
- **模块化设计**: 清晰的数据流：读取→连接→过滤→聚合→训练
- **可配置性**: 所有关键参数可通过配置文件调整

## 项目结构

```
tcmrx/
├── config/                    # 配置文件
│   ├── default.yaml          # 超参数与开关
│   └── paths.yaml            # TSV文件路径
├── dataio/                  # 数据处理模块
│   ├── schema_map.py        # 列名映射与验证
│   ├── readers.py           # TSV读取与校验
│   ├── joins.py             # 业务连接逻辑
│   ├── filters.py           # 过滤与加权
│   ├── id_maps.py           # ID映射管理
│   └── dataset_builder.py   # 数据集构建
├── core/                    # 核心模块
│   ├── batching.py          # PyTorch数据处理
│   ├── losses.py            # InfoNCE损失函数
│   ├── metrics.py           # 评估指标
│   └── utils.py             # 工具函数
├── models/                  # 模型实现
│   ├── encoders.py          # 编码器（嵌入+聚合）
│   ├── aggregators.py       # 聚合策略
│   └── twin_tower.py        # 双塔主模型
├── training/                # 训练模块
│   ├── train_loop.py        # 训练循环
│   ├── evaluator.py         # 模型评估
│   └── splits.py            # 数据划分
├── scripts/                 # 执行脚本
│   ├── sanity_check.py      # 完整性检查
│   ├── run_train.py         # 主训练脚本
│   └── run_eval.py          # 评估脚本
├── tests/                   # 测试代码
├── TCM-MKG-data/            # 原始TSV数据
├── logs/                    # 日志目录
├── checkpoints/             # 模型检查点
├── pyproject.toml           # 项目配置
└── README.md                # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate dgl_env

# 安装依赖
pip install -e .
```

### 2. 完整性检查

```bash
# 验证所有组件能正常工作
python scripts/sanity_check.py
```

### 3. 模型训练

```bash
# 基础训练
python scripts/run_train.py

# 自定义参数训练
python scripts/run_train.py \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --experiment "tcmrx_v1"
```

### 4. 模型评估

```bash
# 评估模型
python scripts/run_eval.py \
    --checkpoint checkpoints/tcmrx_v1_best.pt \
    --data-split test \
    --k-values 1 5 10 20
```

## 核心算法

### 数据流程

1. **读取阶段**: 使用`readers.py`读取TSV文件，校验必需列
2. **连接阶段**: 使用`joins.py`实现业务连接：
   - 疾病侧：ICD11 → CUI/MeSH → EntrezID
   - 方剂侧：CPM → CHP → InChIKey → EntrezID
3. **过滤阶段**: 使用`filters.py`应用Top-K截断和逆频重加权
4. **聚合阶段**: 使用`aggregators.py`聚合为向量表示
5. **训练阶段**: 使用InfoNCE损失进行对比学习

### 模型架构

- **靶点编码器**: 可学习的嵌入向量
- **聚合器**: 支持加权求和、注意力、多头注意力
- **双塔模型**: 独立编码疾病和方剂，计算余弦相似度
- **损失函数**: InfoNCE对比损失，温度缩放

### 评估指标

- **Recall@K**: 召回率
- **NDCG@K**: 归一化折损累积增益
- **MRR**: 平均倒数排名
- **Hit Rate@K**: 命中率

## 配置说明

### 模型配置 (config/default.yaml)

```yaml
model:
  embedding_dim: 256        # 嵌入维度
  temperature: 0.1           # 对比学习温度
  dropout_rate: 0.1         # Dropout率
  aggregator_type: attention # 聚合器类型

training:
  batch_size: 256          # 批大小
  epochs: 50              # 训练轮数
  lr: 0.0001              # 学习率
  device: "auto"          # 计算设备

filtering:
  topk_d: 1000            # 疾病侧Top-K
  topk_f: 3000            # 方剂侧Top-K
  pki_threshold: 6.0       # 亲和力阈值
  inverse_freq_weight: true # 逆频重加权
```

### 路径配置 (config/paths.yaml)

```yaml
data_root: "TCM-MKG-data"

formulas:
  D4_CPM_CHP: "${data_root}/D4_CPM_CHP.tsv"
  D5_CPM_ICD11: "${data_root}/D5_CPM_ICD11.tsv"
  # ... 其他方剂相关文件

diseases:
  D19_ICD11_CUI: "${data_root}/D19_ICD11_CUI.tsv"
  D20_ICD11_MeSH: "${data_root}/D20_ICD11_MeSH.tsv"
  # ... 其他疾病相关文件
```

## 设计原则

### 极简主义
- **内存优先**: 不写中间文件，全在内存处理
- **单一职责**: 每个模块只做一件事
- **可配置性**: 所有关键参数都可配置

### 工程友好
- **严格校验**: 缺列立即报错，提供清晰错误信息
- **稳定复现**: ID映射保证每次运行结果一致
- **模块化设计**: 支持独立测试和扩展

### 性能优化
- **批量处理**: 优化数据加载和模型训练
- **内存管理**: Top-K截断控制内存使用
- **设备适配**: 自动检测CUDA/CPU设备

## 数据要求

项目依赖TCM-MKG数据集，包含以下核心文件：

### 必需文件
- **D4_CPM_CHP.tsv**: 中成药↔中药饮片（含剂量）
- **D5_CPM_ICD11.tsv**: 中成药↔ICD11疾病（监督对）
- **D9_CHP_InChIKey.tsv**: 中药饮片↔化合物
- **SD1_predicted_InChIKey_EntrezID.tsv**: 化合物↔靶点预测
- **D19_ICD11_CUI.tsv**: ICD11↔CUI映射
- **D20_ICD11_MeSH.tsv**: ICD11↔MeSH映射
- **D22_CUI_targets.tsv**: CUI↔靶点
- **D23_MeSH_targets.tsv**: MeSH↔靶点

### 可选文件
- **D12_InChIKey.tsv**: 化合物详细信息
- **D13_InChIKey_EntrezID.tsv**: 实验边（优先级更高）

## 故障排查

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 降低topk_d/topk_f参数
   - 使用混合精度训练

2. **训练不收敛**
   - 调整学习率（1e-5 到 1e-3）
   - 检查温度参数（0.05-0.2）
   - 验证数据质量

3. **数据读取错误**
   - 检查paths.yaml中的文件路径
   - 确认TSV文件格式正确
   - 运行sanity_check.py验证

### 日志分析

```bash
# 查看训练日志
tail -f logs/experiment_name.log

# 检查错误信息
grep "ERROR" logs/experiment_name.log
```

## 扩展开发

### 添加新的聚合器

1. 在`models/aggregators.py`中实现新聚合器类
2. 在`AggregatorFactory`中注册新聚合器
3. 更新配置文件添加新选项

### 添加新的评估指标

1. 在`core/metrics.py`中实现新指标函数
2. 在`RankingMetrics`中集成新指标
3. 更新评估脚本和报告生成

### 支持新的数据源

1. 在`dataio/schema_map.py`中添加列名定义
2. 在`dataio/readers.py`中添加读取逻辑
3. 在`dataio/joins.py`中实现连接逻辑

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request！请确保：

1. 代码符合项目风格
2. 添加必要的测试
3. 更新相关文档
4. 通过所有检查脚本

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者

---

**TCM-RX v0.1.0** - 极简主义的中医药推荐系统