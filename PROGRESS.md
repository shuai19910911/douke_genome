# 项目进展

## 2026-06-07 22:56:09 CST

- 初始仓库名沿用早期占位名称，后续正式模型名调整为 `LegumeGenomeFM`。
- 已完成豆科基因组数据全局去冗余索引整理。
- 当前可用于训练的去冗余豆科基因组为 493 个，覆盖 69 个属。
- genome QC 状态为 493/493 `ok`。
- 已确认注释覆盖：结构注释 251 个、功能注释 128 个、TE/repeat 注释 47 个、三类注释齐全 20 个。
- 确定训练策略：保留全部 493 个基因组，训练时使用属级平衡采样。
- 确定正式主模型路线：LegumeGenomeFM-330M，双向、反向互补等变、长上下文 DNA 模型。
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

## 2026-06-08 10:22:28 CST

- 加大 TE/repeat 训练比例：有 repeat 注释的 TE/repeat interval 保留比例从 30% 提高到 50%，TE 边界和 gene/promoter 附近 TE 保留 100%。
- 将训练 batch 中 TE/repeat 目标比例从 10% 提高到 15%，同时设置 TE/repeat token 上限为 20%，避免 47 个 repeat 注释基因组过度支配训练。
- 为控制重复序列过拟合，保留 TE 相似窗口 >=95% 去冗余，并限制单个 repeat-annotated assembly 不超过 TE/repeat token 的 20%。
- 更新过滤后搬运空间估算：compact 最低包约 0.5-0.9 TB，推荐搬运包约 0.8-1.4 TB，完整过滤后 shard 包约 1.2-2.2 TB；训练服务器推荐预留 3-4 TB，完整缓存和多 checkpoint 场景 5-7 TB。

## 2026-06-08 10:39:24 CST

- 按用户确认的极简压缩策略更新训练计划：CDS、splice、start/stop、UTR 和 promoter 0-5 kb 保留 100%；promoter 5-20 kb 保留 15%；普通 intron 保留 10%；长 intron 保留 5%；gene-proximal intergenic 保留 10%；distal intergenic 保留 3%-5%；random genome coverage 保留 1%-2%。
- 保留 TE/repeat batch 目标比例 15%，但改为优先只存 TE interval index，训练时在线取片段，避免预先物化大体积 TE shard。
- 更新过滤后规模估算：sequence-equivalent 约 65-115 Gb；多尺度索引和标签开销后约 180-450 GB；compact sequence store + window index 约 120-280 GB。
- 更新跨服务器搬运估算：极简 compact 可训练包约 0.25-0.50 TB，推荐 compact 搬运包约 0.40-0.70 TB，带少量固定 shard 搬运包约 0.50-0.90 TB。
- 更新训练服务器磁盘建议：最低 1 TB，稳定训练 1.5-2 TB，多 checkpoint 场景 2-3 TB。

## 2026-06-08 11:18:59 CST

- 生成推荐 compact 搬运包的数据预处理脚本和 Slurm 脚本，本地脚本保存在 `scripts/`，GPU smoke test 脚本也已生成。
- 建立 compact 搬运包目录，包含 `manifests/`、`indexes/`、`files/genomes/`、`files/annotations/`、`files/repeats/`、`docs/` 和 `logs/`。
- 生成并保存技术路线图图片，并在 compact 搬运包的 `docs/` 中保留一份，方便后续随 compact 包搬运。
- 生成详细数据预处理技术路线文档：`docs/compact_preprocessing_route.md`，说明输入数据、过滤策略、目录结构、CPU Slurm 命令、GPU smoke test 命令、搬运命令和资源耗时估算。
- 已提交 manifest-only 预检查作业：`job_id=8462014`，分区 `q07`，节点 `cu16`，资源 `4 cores / 16G memory`；该作业用于先组织 compact 文件链接并生成清单，确认无误后再提交 30 核 150G 的完整索引构建作业。

## 2026-06-08 11:29:18 CST

- manifest-only 预检查作业 `job_id=8462014` 已完成。
- compact manifest 核对结果：251 个结构注释基因组、29 个属；`compact_manifest.tsv` 252 行，`selected_files.tsv` 550 行，`split.tsv` 252 行。
- 实际入选文件：251 个 genome、206 个 gff3、45 个 gff、47 个 repeat 注释文件；当前 compact 目录占用约 81G。
- leakage-safe split 当前分布：train 191、validation 33、test 27。
- 已提交完整 compact 索引构建作业：`job_id=8462433`，分区 `q07`，节点 `cu19`，资源 `30 cores / 150G memory`；该作业将生成 `sequence_index.tsv`、`filtered_windows.tsv` 和 `region_sampling_weights.tsv`。

## 2026-06-08 11:36:15 CST

- 将正式项目和模型名称从早期占位名称调整为更适合论文、GitHub 和模型发布的 `LegumeGenomeFM`。
- 正式主模型名更新为 `LegumeGenomeFM-330M`，中文描述为“豆科基因组基础模型”。
- 保留当前正在运行作业使用的本地 legacy 目录和脚本名，避免影响 `job_id=8462433` 的路径依赖；对外文档、论文和模型命名统一使用 `LegumeGenomeFM`。

## 2026-06-08 12:03:18 CST

- 按正式名称重新生成大尺寸技术路线图，标题更新为 `LegumeGenomeFM 豆科基因组基础模型`；本地项目和 compact 搬运包内均保留新版图片。
- 同步更新本地 compact 预处理说明和随包 `README_COMPACT_PACKAGE.md`，将对外标题和搬运目标目录统一为 `LegumeGenomeFM`。
- 完整 compact 索引构建作业 `job_id=8462433` 仍在 `cu19` 运行，已运行约 34 分钟。
- 当前 compact 包状态：`data/compact_douke_v1/` 占用约 81G；`filtered_windows.tsv` 已生成 1,725,169 行；`sequence_index.tsv` 仍在等待作业结束或缓冲刷新。

## 2026-06-08 12:19:40 CST

- 按要求重新生成更大的 `LegumeGenomeFM` 中文技术路线图，覆盖原本地图片。
- 将路线图加入 GitHub 跟踪范围：`docs/legumegenomefm_compact_technical_route.png`。
- 在 `README.md` 顶部新增“技术路线图”展示区，GitHub 首页可直接看到该图片。
- compact 搬运包内同步保留同一张路线图副本，路径为 `data/compact_douke_v1/docs/legumegenomefm_compact_technical_route.png`。

## 2026-06-08 13:16:37 CST

- 发现完整 compact 索引构建作业 `job_id=8462433` 实际主要是串行遍历 genome，已运行约 1 小时 48 分钟，`filtered_windows.tsv` 约 616 万行，预计整体耗时偏长。
- 按 q03-q08 优先策略改为 Slurm array 分片索引流程：251 个结构注释 genome 分别生成独立 part 文件，避免多个任务同时写同一个索引。
- 已生成 array 使用的 `compact_ids.txt`，共 251 个 compact genome ID。
- 已提交并行索引作业 `job_id=8462866`：分区 `q03,q04,q05,q07,q08`，每个 task `4 cores / 100G memory`，并发上限 `%6`。
- 已提交自动合并依赖作业 `job_id=8462867`：等待 `8462866` 全部成功后合并 `indexes_parallel/parts/` 到 `indexes_parallel/sequence_index.tsv`、`indexes_parallel/filtered_windows.tsv` 和 `indexes_parallel/region_sampling_weights.tsv`。
- 旧串行作业 `job_id=8462433` 暂未取消，避免直接丢弃已运行结果；正式 compact 索引以后优先使用并行流程产物 `data/compact_douke_v1/indexes_parallel/`。

## 2026-06-08 13:27:08 CST

- 根据每个计算节点约 `30 CPU / 150G memory` 的实际资源，取消低利用率并行作业 `8462866` 和依赖合并作业 `8462867`。
- 将分片索引 array 优化为每个 task `10 CPU / 50G memory`，并发上限从 `%6` 提高到 `%18`；理论上每个节点可同时运行 3 个 genome 分片，更接近满节点资源占用。
- 已提交优化后的并行索引作业 `job_id=8462868`，分区 `q03,q04,q05,q07,q08`；已有 task 开始运行。
- 已提交新的自动合并依赖作业 `job_id=8462872`，等待 `8462868` 全部成功后合并并行索引结果。
- 若个别超大 genome 分片因 50G 内存不足失败，后续只单独以 `100-150G` 内存重跑失败分片，不影响已完成 part。

## 2026-06-08 21:06:36 CST

- 将临时评估文档 `docs/TEMP_LEGUME_CONTEXT_LENGTH_STRATEGY_EVALUATION.md` 的推荐结论合并进正式 `PLAN.md`。
- 正式训练长度策略更新为同一个 `LegumeGenomeFM-330M` 的渐进式扩长：`Stage 1A 32 kb 主训练` -> `Stage 1B 64 kb 继续训练` -> `Stage 1C 128 kb long-context continue pretraining`。
- 明确不训练多个独立长度模型，也不把 `8 kb / 32 kb / 64 kb / 128 kb` 从第一步开始等比例完全混合。
- 新增每阶段 token 组成、短长度 replay、context_bucket/region_bucket 两级采样、同长度 micro-batch、gradient accumulation 层面混合、每长度 bucket loss 记录和进入下一阶段条件。
- 训练时间估算同步改为 Stage 1A/1B/1C 分段估算，Stage 1 总预算仍以约 `130B tokens` 为目标。

## 2026-06-08 23:18:50 CST

- 根据当前正式计划确认 CPU compact 预处理不需要重跑；已提交的并行索引作业 `job_id=8462868` 和依赖合并作业 `job_id=8462872` 继续作为正式流程保留。
- 取消旧串行 compact 索引作业 `job_id=8462433`，该作业已被并行分片流程替代，继续运行会浪费计算节点资源。
- 当前队列只保留正式并行索引和自动合并依赖作业。

## 2026-06-09 15:53:19 CST

- 并行 compact 索引流程已完成，最终合并作业 `job_id=8470464` 在 `cu59` 成功完成，退出码 `0:0`。
- 251 个 genome 分片全部合并成功：`sequence_parts=251`，`window_parts=251`。
- 最终 compact 训练索引位于 `data/compact_douke_v1/indexes_parallel/`；`filtered_windows.tsv` 约 `63G`，`sequence_index.tsv` 约 `57M`，`region_sampling_weights.tsv` 已生成。
- 最终统计：`sequence_rows=964,198`，`window_rows=802,272,042`。
- 按窗口长度统计：8 kb `221,605,555`，32 kb `203,818,407`，64 kb `193,139,777`，128 kb `183,708,303`。
- 按 split 统计：train `628,691,695`，validation `105,225,279`，test `68,355,068`。
- 原先 16 个失败分片已修复并重跑成功；失败原因是部分注释文件含非 UTF-8 字节以及部分 FASTA 存在空 header，已在本地预处理脚本中增加容错。

## 后续阶段

- 2026-06-09 之后：基于 `indexes_parallel/filtered_windows.tsv` 准备 LegumeGenomeFM-330M 的 Stage 1A/1B/1C 训练 sampler 配置。
- 2026-06-09 之后：生成训练前 QC 报告，重点核对窗口长度、region、split、genus 和 assembly 覆盖分布。
- 2026-06-09 之后：启动正式结构注释驱动预训练并记录 loss、mask accuracy、region F1、splice AUPRC、RC consistency 和下游验证指标。
