# douke_genome

豆科基因组预训练大模型项目。

## 当前目标

构建一个面向豆科（Fabaceae/Leguminosae）的 DNA foundation model，用于学习豆科跨属、跨物种的基因组序列模式，并支持基因结构识别、功能区域表征、TE/重复序列识别、启动子/调控区域建模、基因功能预测和大豆育种相关下游任务。

训练资源策略：优先让正式方案能在 2 张 A100 40G GPU 范围内启动和推进；如果正式 token budget、上下文长度或吞吐要求超出该配置，则扩展到更多 GPU，而不是降低为非正式小模型。

## 当前数据状态

更新时间：2026-06-07 22:56:09 CST

- 全局去冗余 assembly group：526 个。
- 可用于训练的去冗余基因组：493 个。
- 覆盖属数：69 个。
- genome QC：493/493 为 `ok`。
- 结构注释覆盖：251 个基因组有结构注释。
- 功能注释覆盖：128 个基因组有功能注释。
- TE/repeat 注释覆盖：47 个基因组有 repeat 注释。
- 三类注释均齐全：20 个基因组。

关键索引：

- `data/metadata/legume_family_nonredundant_assemblies.tsv`
- `data/metadata/legume_family_nonredundant_files.tsv`
- `data/metadata/legume_family_nonredundant_duplicate_groups.tsv`

## 正式训练路线

本项目不做小测试模型作为最终路线。正式路线为：

1. 用 493 个去冗余豆科基因组训练 DNA-only foundation model。
2. 训练时保留全部数据，但采用属级平衡采样，避免 Glycine 和 Medicago 等大属支配模型。
3. 使用双向、反向互补等变的长程 DNA 模型作为主干。
4. 在 DNA-only 预训练后，使用结构注释、功能注释、TE/repeat 注释进行 annotation-aware 多任务继续预训练。
5. 最终面向豆科和大豆育种任务进行微调与评估。

详细方案见：

- `PLAN.md`
- `PROGRESS.md`

## GitHub 更新规则

每完成一个小阶段，更新 `PROGRESS.md`，每条进展必须带具体时间点。项目方案、模型结构、数据状态和训练结果同步提交到 GitHub。
