# 训练与测试流程指南

本指南旨在帮助研究人员快速复现当前双塔模型的训练、评估与诊断流程，并提供常见可选参数说明。所有命令均需在项目根目录 `tcmrx/` 下执行（即包含 `config/`、`scripts/` 子目录的路径）。

## 1. 环境准备

1. 安装依赖：
   ```bash
   pip install -e .
   ```
2. 准备数据：确保 `config/paths.yaml` 中的路径指向 `TCM-MKG` 数据拆分文件夹。若使用仓库默认路径结构，无需额外修改。

## 2. 快速训练（轻量调试）

使用 `scripts/run_train.py` 执行快速训练，可覆盖关键超参数。常用于验证数据管道或模型变更后的基本可运行性。

```bash
python -m scripts.run_train \
  --config config/default.yaml \
  --paths config/paths.yaml \
  --epochs 2 \
  --batch-size 64 \
  --experiment debug_run
```

**常用参数说明**
- `--config`: 指定模型与预处理超参数配置。
- `--paths`: 指向数据路径配置文件。
- `--epochs`: 覆盖训练轮数，快速调试时建议 1–3。
- `--batch-size`: 覆盖批大小，受显存/内存影响。
- `--lr`: 覆盖学习率。
- `--device`: `auto`、`cpu` 或 `cuda`。
- `--resume`: 继续训练的检查点路径。

输出的日志、检查点与指标将写入 `config/default.yaml` 中 `logging.log_dir` 指定的目录。

## 3. 标准训练

在正式实验中建议直接使用配置文件内的默认轮数与批大小：

```bash
python -m scripts.run_train --config config/default.yaml --paths config/paths.yaml --experiment full_train
```

训练完成后，脚本会自动执行验证/测试评估，并在日志中记录指标。若启用了混合精度或余弦学习率等特性，可在配置文件中调整。

## 4. 推理与评估

### 4.1 使用保存的检查点评估

```bash
python -m scripts.run_eval \
  --config config/default.yaml \
  --paths config/paths.yaml \
  --checkpoint runs/<experiment>/checkpoints/best.pt
```

**可选参数**
- `--split`: 指定评估划分（`train`/`val`/`test`/`cold_start`）。
- `--topk`: 评估时截断预测候选数量。
- `--batch-size`: 覆盖评估批大小。
- `--metrics-only`: 仅输出指标而不生成详细相似度文件。

### 4.2 生成方剂/疾病嵌入

`run_eval.py` 支持通过 `--export-embeddings` 导出疾病与方剂的表示，生成的向量用于下游分析或可视化。

## 5. 数据与分布诊断

### 5.1 数据集统计

```bash
python -m scripts.analyze_dataset --config config/default.yaml --paths config/paths.yaml --output reports/dataset_stats.json
```

该脚本复用训练管道的预处理逻辑，输出疾病/方剂靶点统计、权重熵、重叠度以及高频靶点提示等诊断信息。

### 5.2 过滤与配对 sanity check

```bash
python -m scripts.sanity_check --config config/default.yaml --paths config/paths.yaml
```

用于确认方剂/疾病目标集合、正样本配对及重平衡逻辑是否符合预期。

## 6. 超参数搜索

当需要批量探索预处理或模型超参数时，可使用 `scripts/hparam_search.py`。

```bash
python -m scripts.hparam_search \
  --config config/default.yaml \
  --paths config/paths.yaml \
  --search-grid config/search_spaces.yaml \
  --max-trials 12 \
  --parallel 2
```

- `--dry-run`: 仅打印即将尝试的组合，不实际训练。
- `--resume`: 从已有搜索目录继续。
- `--max-trials`: 最大试验数量。
- `--parallel`: 并发运行任务数（依赖可用 GPU/CPU）。

## 7. 测试与回归校验

### 7.1 单元测试

```bash
pytest
```

覆盖聚合、采样、投影头等关键逻辑，建议在修改预处理或模型结构后执行。

### 7.2 对比学习端到端测试

```bash
python -m scripts.test_contrastive --config config/default.yaml --paths config/paths.yaml
```

该脚本使用小规模子集验证训练循环是否可在几步内收敛，并检查负采样、损失计算与指标接口。

## 8. 推荐工作流

1. **数据检查**：运行 `analyze_dataset` 获取稀疏化与频率统计；根据输出调整 `config/default.yaml` 中的 `filtering`、`sampling`、`pathways` 配置。
2. **快速训练**：使用短轮数验证修改是否可行；若日志或指标异常，先运行 `sanity_check`。
3. **完整训练+评估**：固定随机种子与实验名，运行标准训练与评估；使用 `run_eval` 导出嵌入并对关键疾病做人工复核。
4. **批量调参（可选）**：通过 `hparam_search` 探索不同截断阈值、逆频率权重、采样强度等组合。

如需进一步扩展通路特征或疾病重采样策略，请查阅 `docs/current_data_status.md` 与 `docs/data_improvement_plans.md` 中的详细建议。
