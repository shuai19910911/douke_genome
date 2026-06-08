# douke_genome

豆科基因组预训练大模型项目。

## 当前目标

构建一个面向豆科（Fabaceae/Leguminosae）的 DNA foundation model，用于学习豆科跨属、跨物种的基因组序列模式，并支持基因结构识别、功能区域表征、TE/重复序列识别、启动子/调控区域建模、基因功能预测和大豆育种相关下游任务。

训练资源策略：优先让正式方案能在 2 张 A100 40G GPU 范围内启动和推进；如果正式 token budget、上下文长度或吞吐要求超出该配置，则扩展到更多 GPU，而不是降低为非正式小模型。当前方案已经调整为结构注释驱动路线：正式训练只使用有结构注释的 251 个基因组，并对输入片段做过滤，不全量输入所有非编码区。

## 当前数据状态

更新时间：2026-06-08 09:24:33 CST

- 全局去冗余 assembly group：526 个。
- genome QC 通过且含 genome 的基因组：493 个。
- 正式训练使用的结构注释基因组：251 个。
- 正式训练覆盖属数：29 个。
- 正式训练基因组总长度：约 300.5 Gb。
- 功能注释覆盖：128 个基因组有功能注释。
- TE/repeat 注释覆盖：47 个基因组有 repeat 注释。
- 三类注释均齐全：20 个基因组。

关键索引：

- `data/metadata/legume_family_nonredundant_assemblies.tsv`
- `data/metadata/legume_family_nonredundant_files.tsv`
- `data/metadata/legume_family_nonredundant_duplicate_groups.tsv`

## 正式训练路线

本项目不做小测试模型作为最终路线。正式路线为：

1. 只使用 251 个有结构注释的去冗余豆科基因组作为正式训练集。
2. 训练时采用属级平衡和区域级加权采样，优先学习 CDS、剪接位点、启动子、UTR、intron、TE/repeat 等区域。
3. 使用双向、反向互补等变的长程 DNA 模型作为主干。
4. 使用结构注释和 TE/repeat 区域标签进行多任务预训练，不再设置功能注释和 gene family 继续预训练阶段。
5. 系统评估基因结构、剪接、CDS/frame、启动子、TE/repeat、大豆变异效应和 GWAS hit prioritization 等下游任务。

## 输入片段过滤

训练不会把所有序列都输入模型。功能区域高保留，普通远端非编码区只保留高质量代表子集：CDS、剪接位点、start/stop、UTR 和近端 promoter 保留 100%；普通 intron 内部保留约 20%；gene-proximal intergenic 保留约 20%；distal intergenic/far noncoding 只保留约 10% 且要求 N <= 2%、非低复杂度、非高度重复。

## 跨服务器训练数据搬运

如果在本服务器完成数据处理，再搬到其他服务器训练，过滤后推荐准备 **0.6-1.2 TB** 的训练数据包；最低 compact 可训练包约 **0.4-0.8 TB**，完整过滤后 shard 包约 **1.0-1.8 TB**。目标训练服务器建议至少预留 **2-4 TB** 可用空间，完整缓存和多 checkpoint 场景建议 **4-6 TB**。

详细方案见：

- `PLAN.md`
- `PROGRESS.md`

## GitHub 更新规则

每完成一个小阶段，更新 `PROGRESS.md`，每条进展必须带具体时间点。项目方案、模型结构、数据状态和训练结果同步提交到 GitHub。
