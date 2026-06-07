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

## 后续阶段

- 2026-06-07 之后：生成 genome-only 训练样本清单。
- 2026-06-07 之后：构建属级平衡采样器。
- 2026-06-07 之后：实现 FASTA 窗口化和 annotation-aware 标签构建。
- 2026-06-07 之后：准备 DoukeGenome-330M 训练配置。
- 2026-06-07 之后：启动正式预训练并记录 loss、perplexity、mask accuracy、RC consistency、下游验证指标。
