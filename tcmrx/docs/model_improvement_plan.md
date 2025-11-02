# 模型性能改进分析与方案

## 背景
在最新版代码修复评估管线后，默认配置训练 50 轮得到的召回与 NDCG 指标仍偏低，说明虽然训练能够稳定收敛，但塔结构提取到的信号有限，需要从**模型表达能力**与**数据可区分度**两方面进一步优化。

## 现状分析
1. **正样本重叠度有限**：`scripts/analyze_dataset.py` 提供的 `pair_overlap` 统计可以量化疾病靶点集合与方剂靶点集合的交集比例，初步样本显示平均 Jaccard 值普遍低于 0.2，这意味着仅依靠靶点集合的交集来区分正负样本信号较弱。【F:scripts/analyze_dataset.py†L162-L211】
2. **疾病/方剂关联高度不均衡**：同一疾病往往对应多条处方，且靶点覆盖高度重叠，导致在对比学习中产生大量“几乎等价”的负样本，削弱模型梯度。【F:scripts/analyze_dataset.py†L120-L160】【F:scripts/analyze_dataset.py†L213-L236】
3. **塔头表达能力有限**：原始双塔直接对聚合结果做 L2 归一化后参与相似度计算，没有额外的非线性投影，难以在低重叠场景中放大细微差异。【F:models/twin_tower.py†L17-L107】

## 改进方案
### 1. 塔顶 MLP 投影
新增的 `tower_head` 配置允许在聚合输出后叠加可配置的残差 MLP 投影，再进行归一化，显式提升特征变换能力。默认保持关闭以兼容旧模型，启用后可通过 `hidden_dims`、`activation` 等参数调节深度与非线性。【F:models/twin_tower.py†L24-L95】【F:config/default.yaml†L16-L27】

建议尝试：
- 将 `hidden_dims` 设为 `[512, 256]` 并启用 `residual=true`，观察召回是否提升；
- 在验证集上做网格搜索，组合不同激活函数（`gelu`、`silu`）和 dropout。

### 2. 数据分布洞察
`scripts/analyze_dataset.py` 复用了训练前处理逻辑，输出靶点数量分布、靶点流行度、疾病/方剂多重配对情况以及正样本靶点重叠度，为定量诊断提供依据。【F:tcmrx/scripts/analyze_dataset.py†L1-L244】
`scripts/analyze_dataset.py` 复用了训练前处理逻辑，输出靶点数量分布、靶点流行度、疾病/方剂多重配对情况以及正样本靶点重叠度，为定量诊断提供依据。【F:scripts/analyze_dataset.py†L1-L244】

建议流程：
1. 运行 `python -m scripts.analyze_dataset --output data/analysis.json`；
2. 检查 `target_frequency`：若前 10 个靶点覆盖度过高（>40%），考虑提高 `filtering.topk_*` 或使用逆频权重；
3. 根据 `pair_overlap` 的 coverage 分布，挑选覆盖度极低的疾病，分析是否缺乏正样本或缺失靶点。

### 3. 自适应靶点筛选
- 默认配置的 `filtering.disease_target_trimming` / `filtering.formula_target_trimming` 会依据权重累积质量与最大保留数协同裁剪靶点集合，使多数疾病不再卡在硬性上限 1000 / 3000。可通过 `mass_threshold` 调节保留的权重比例，或通过 `weight_floor` 排除极小权重靶点。【F:tcmrx/config/default.yaml†L28-L60】【F:tcmrx/dataio/filters.py†L70-L160】
- `filtering.frequency_reweighting` 默认启用 IDF 型重加权，缓和超高频靶点造成的偏置。调大 `power` 会进一步压低热门靶点的贡献。【F:tcmrx/config/default.yaml†L62-L82】【F:tcmrx/dataio/filters.py†L162-L230】

推荐先结合 `analyze_dataset` 的 `target_frequency` 输出确定靶点长尾程度，再在 `mass_threshold` / `power` 范围内做网格搜索：

```
python -m tcmrx.scripts.hparam_search --grid configs/hparam_grid.yaml --extra-args --epochs 20
```

其中 `hparam_search` 会顺序调用 `run_train.py`，生成的临时配置保存在 `hparam_runs/` 下，便于对比不同参数带来的评估指标差异。【F:tcmrx/scripts/hparam_search.py†L1-L143】

### 3. 硬负样本采样（后续）
结合分析结果，为每个疾病构造与其靶点集合高度重合但未被标记为正样本的方剂，作为显式硬负样本加入损失；可通过扩展 `TrainingLoop` 在每个 batch 内追加来自分析脚本输出的“难例候选”。

### 4. 多模态增强（后续）
若分析发现靶点信号仍不足，可考虑在塔顶拼接疾病描述、方剂功效等文本嵌入，再通过新增的 MLP 投影头统一对齐向量空间，为召回引入额外判别特征。

## 预期收益
- **短期**：塔顶投影 + 数据再加权可在无需额外数据的情况下提升疾病-方剂区分度；
- **中期**：通过数据分析脚本的反馈实现靶点筛选、难例挖掘，减少“虚假负样本”；
- **长期**：多模态/多任务扩展与硬负样本训练结合，可逐步逼近临床检索的真实需求。

