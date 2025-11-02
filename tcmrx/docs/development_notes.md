# 开发修复说明

本文档总结了针对双塔模型训练与评估流程的一系列修复，并提供了后续开发时需要注意的要点。

## 1. 批处理与掩码
- `core/batching.py` 现在根据真实的靶点数量构造 `disease_mask` / `formula_mask`，不会再把合法的靶点 `0` 误判为 padding。
- 任何新增的数据预处理逻辑必须确保在 `collate_fn` 中同步更新长度相关的掩码。

## 2. 训练循环
- `training/train_loop.py` 统一了标准精度与混合精度分支的反向传播流程，`loss.backward()` / `optimizer.step()` 只会执行一次。
- 若需要扩展新的损失函数，可在循环顶部替换 `loss_fn`，并确保混合精度分支使用 `autocast` + `GradScaler`。

## 3. 评估流程
- `training/evaluator.py` 在评估阶段会对重复出现的疾病和方剂索引做平均聚合，并基于全量候选方剂计算排名指标。
- 真实标签矩阵与相似度矩阵现在严格对齐，能够正确反映召回、精确率、NDCG 等指标。
- 若新增指标，请复用 `_aggregate_embeddings` 的输出，避免再次引入批内偏差。

## 4. 靶点导出
- `models/twin_tower.py` 的 `get_target_embeddings` 通过共享的 `TargetEncoder` 生成稳定的靶点向量，不再调用未定义的低层接口。

## 5. 建议的回归测试
- 常规回归测试：`pytest`（覆盖单元测试）
- 端到端烟雾测试：执行一轮 `python scripts/run_train.py --config <cfg>` 并确认评估指标不再全部接近 0。

如需进一步扩展功能（例如加入新的聚合器或对比损失），建议在上述改动基础上编写针对性的单元测试，以避免评估或掩码逻辑再次回退。

## 6. 性能诊断与塔顶扩展
- `config/default.yaml` 新增 `model.tower_head` 配置，可在聚合后堆叠 MLP + 残差投影，以增强疾病/方剂表示能力。默认关闭，开启后建议同步更新回归测试覆盖投影头路径。
- `scripts/analyze_dataset.py` 复用训练前处理逻辑输出靶点分布、链接多重度与正样本靶点重叠度，推荐在调参前先运行 `python -m tcmrx.scripts.analyze_dataset --output data/analysis.json` 获取定量诊断。该脚本的结构便于后续添加新的统计项或导出硬负样本候选列表。

## 7. 自适应靶点筛选与频率调节
- `filtering.disease_target_trimming` / `filtering.formula_target_trimming` 提供 `max_items`、`mass_threshold`、`min_items` 等选项，可按靶点权重累积质量自动截断集合，避免所有实体都被硬性裁到固定上限。
- `filtering.frequency_reweighting` 分别为疾病侧与方剂侧引入 IDF 类似的频率调节，默认将高频靶点下调、低频靶点上调，缓解“热门靶点统治表示”的问题。
- 新增 `tcmrx.tests.test_target_processing` 回归用例，验证裁剪逻辑对最低保留数量、逆频削弱等行为的正确性。

## 8. 网格搜索工具
- `python -m tcmrx.scripts.hparam_search` 会针对常见的截断/频率参数建立网格，并顺序调用 `run_train.py`。可通过 `--grid` 指定 YAML 文件自定义网格，或使用 `--dry-run` 查看组合列表后再执行。
- 建议配合 `scripts/analyze_dataset.py` 的诊断结果筛选参数范围，并在搜索结果中保留指标日志以便对比。
