# 项目进展

## 2026-06-07 22:56:09 CST

- 确定项目名称为 `douke_genome`。
- 已完成豆科基因组数据全局去冗余索引整理。
- 当前可用于训练的去冗余豆科基因组为 493 个，覆盖 69 个属。
- genome QC 状态为 493/493 `ok`。
- 已确认注释覆盖：结构注释 251 个、功能注释 128 个、TE/repeat 注释 47 个、三类注释齐全 20 个。
- 确定训练策略：保留全部 493 个基因组，训练时使用属级平衡采样。
- 确定正式主模型路线：DoukeGenome-330M，双向、反向互补等变、长上下文 DNA 模型。
- 确定资源策略：优先适配 2 张 A100 40G GPU，使用 bf16、梯度检查点、DDP/ZeRO-2、梯度累积；如果正式训练吞吐不足，则扩展到更多 GPU，而不是改成非正式小模型。
- 创建 GitHub 核心文档：`README.md`、`PLAN.md`、`PROGRESS.md`。
- 按要求收敛 GitHub 内容：只放项目介绍、正式计划和单一进展文件；不上传原始数据、脚本、历史过程文档和大体积模型文件。

## 2026-06-07 23:42:26 CST

- 扩展 `PLAN.md` 为可执行级详细方案。
- 新增数据预处理流程：FASTA 标准化、注释标准化、assembly group 级切分、窗口化、shard 构建。
- 新增模型输入设计：单碱基 token、窗口样本字段、张量形态、mask 策略和 annotation-aware 标签。
- 明确模型架构依据：采用单碱基、长上下文、bidirectional MambaDNA、reverse-complement consistency，参考 Nucleotide Transformer、DNABERT-2、HyenaDNA、Caduceus、PlantCaduceus、Evo 2 和 DNA foundation model benchmark。
- 新增资源和耗时估算：数据工程约 2-4 天；2 张 A100 40G 下 Stage 1 约 23-46 天、Stage 2 约 5-14 天、初版总周期约 35-78 天；4 卡约 23-49 天；8 卡约 16-34 天。
- 新增存储估算：建议项目可用空间 4-8 TB。

## 2026-06-08 09:24:33 CST

- 按最新研究策略重写 `PLAN.md`：正式训练放弃没有结构注释的基因组，只使用 251 个结构注释可用基因组，覆盖 29 个属，总长度约 300.5 Gb。
- 将窗口 N 比例阈值从 `>20% 丢弃` 调整为正式训练默认 `N <= 5%`；训练中 5%-10% 仅允许小属或稀缺区域低权重救援，validation/test 中 `N > 5%` 全部丢弃。
- 强化数据泄漏控制：duplicate group、assembly、chromosome/scaffold、interval、gene、transcript 和 splice site 均不得跨 train/validation/test。
- 新增区域级采样比例：优先 CDS、splice site、promoter/TSS、UTR、intron、TE/repeat，再保留少量 intergenic 和 random genome coverage。
- 详细补充下游任务和基线比较：基因结构、剪接位点、CDS/frame、启动子、TE/repeat、基因家族/功能注释、大豆变异效应和 GWAS hit prioritization。
- 明确模型预计优势：预期在豆科结构注释相关任务、剪接、CDS/frame、跨属低标注迁移和大豆功能变异优先级排序上优于通用 DNA LM 和短上下文基线；不承诺在非豆科零样本、表达量精确预测和复杂农艺性状直接预测上全面领先。

## 2026-06-08 09:48:21 CST

- 按要求删除 `Stage 2: 功能和家族继续预训练`，不再使用功能注释、gene family 和 TE/repeat 子集做独立继续预训练。
- 从模型预训练目标、loss 权重、任务头和下游任务中移除 functional annotation multi-label learning 与 gene family contrastive learning。
- 保留 TE/repeat 作为结构区域监督和下游评估任务的一部分，不再作为独立 Stage 2 继续预训练。
- 训练阶段调整为：Stage 0 数据工程，Stage 1 结构注释驱动预训练，Stage 2 下游任务微调和系统评估。
- 总 token budget 调整为 Stage 1 的 130B tokens；预计总周期调整为 2 卡 25-54 天、4 卡 16-33 天、8 卡 10-23 天。

## 2026-06-08 09:50:28 CST

- 在 `PLAN.md` 新增“预训练输入、输出和 loss”专门小节。
- 明确预训练输入包括 `input_ids`、`attention_mask`、`mlm_mask`、`mlm_labels`、`region_labels`、`splice_labels`、`frame_labels`、`start_stop_labels`、`repeat_labels`、`rc_input_ids`、`next_window_input_ids` 和 `sample_weight`。
- 明确模型架构流：token embedding -> forward Mamba stream -> reverse-complement Mamba stream -> RC-equivariant bidirectional fusion -> task heads。
- 明确总 loss 公式：`L_mlm + L_region + L_cds_frame + L_splice + L_start_stop + L_promoter + L_repeat + L_rc + L_next_window`，并写明每项 loss 的适用 mask。

## 2026-06-08 09:54:55 CST

- 新增跨服务器搬运空间估算：最低可训练搬运包约 0.8-1.2 TB，推荐搬运包约 1.5-2.5 TB，完整处理缓存约 3-5 TB。
- 明确训练服务器磁盘建议：只训练至少 2 TB 可用空间，推荐稳定训练 4 TB，完整缓存和多 checkpoint 场景 6-8 TB。
- 明确不建议搬运动态 mask 后样本、RC 实体副本、next-window pair 实体副本和临时解压文件，避免不必要地放大搬运体积。

## 2026-06-08 10:07:34 CST

- 新增输入片段过滤策略：正式训练不再全量输入所有序列，而是先做硬质量过滤，再按基因组区域保留比例构建高价值窗口索引。
- 明确区域保留比例：CDS、splice、start/stop、UTR、近端 promoter 保留 100%；普通 intron 内部保留约 20%；gene-proximal intergenic 保留约 20%；distal intergenic/far noncoding 只保留约 10% 高质量代表窗口。
- 明确远端非编码区高质量标准：N <= 2%、无长 N、非低复杂度、非高度重复、GC 位于本 genome 的 5%-95% 分位范围。
- 新增相似窗口去冗余策略：普通 intergenic 和 repeat-rich 背景窗口相似度 >= 95% 时只保留代表窗口；CDS、splice、start/stop 不因相似性丢弃。
- 根据过滤策略更新存储估算：过滤后训练 shard 约 250-600 GB，compact sequence store + window index 约 150-350 GB；跨服务器 compact 推荐搬运包约 0.6-1.2 TB，full shard 推荐搬运包约 1.0-1.8 TB。

## 后续阶段

- 2026-06-08 之后：生成 structural-annotation-only 训练样本清单。
- 2026-06-08 之后：构建属级平衡 + 区域级加权采样器。
- 2026-06-08 之后：实现 FASTA 标准化、结构注释标准化、窗口化和多任务标签构建。
- 2026-06-08 之后：生成 leakage-safe split table，保证 duplicate group、assembly、interval、gene 不跨 split。
- 2026-06-08 之后：准备 DoukeGenome-330M 训练配置。
- 2026-06-08 之后：启动正式结构注释驱动预训练并记录 loss、mask accuracy、region F1、splice AUPRC、RC consistency 和下游验证指标。
