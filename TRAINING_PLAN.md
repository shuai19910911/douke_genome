# LegumeGenomeFM Training Plan

> 状态：证据与数据审计阶段（2026-07-18）。正式架构、tokenizer（序列离散化方式）、上下文、多尺度配额、参数量、训练目标和优化器均尚未冻结；本文不会把候选设想写成正式结论。

## 1. 目标与核心科学问题

LegumeGenomeFM旨在从多物种豆科参考基因组学习可跨物种、跨属和跨进化距离迁移的DNA表示。项目必须回答：豆科专属预训练相对通用及植物基因组模型是否具有公平可复现的优势；优势是否能通过物种/属/低同源/外部留出评测排除身份、重复序列、组装版本和同源泄漏；不同序列尺度是否对基因结构、调控、变异和育种任务产生等token、等算力条件下的真实增益。

## 2. 证据先行的执行顺序

1. 冻结检索截止日期、查询式、纳入/排除标准，并核验核心论文的DOI、Methods、补充方法和官方代码。
2. 通过SLURM建立原始文件清单，随后分阶段完成FASTA/GFF/VCF配套、组装质量、注释质量、重复与污染审计。
3. 基于真实可用碱基数、窗口长度分布、物种/属覆盖和算力推导token预算、容量及上下文。
4. 预注册数据划分、泄漏门禁、核心下游任务、baseline（对照模型）和统计方法。
5. 只冻结一个共享参数的正式模型、一套tokenizer、一套多尺度调度和一套训练目标。
6. 用同一正式代码完成RTX 2080 Ti短程验证、checkpoint（断点）恢复和可行时的DDP（分布式数据并行）启动验证。
7. 未经用户明确批准不使用A100；正式大规模预训练迁移至AutoDL。

## 3. Nature Portfolio系统检索协议

- 检索截止：执行日2026-07-18。
- 目标来源：Nature、Nature Genetics、Nature Biotechnology、Nature Methods、Nature Machine Intelligence、Nature Communications、Communications Biology、Scientific Data及检索命中的其他Nature Portfolio期刊。
- 补充来源：对架构、长序列算子、reverse complement（反向互补）、tokenizer、scaling（规模规律）、植物模型和泄漏控制不可替代的原始论文与官方代码。
- 查询族：DNA/genome language model、genomic foundation model、sequence-to-function、regulatory genomics、variant effect、cross-species pretraining、plant/crop genome foundation model、reverse-complement equivariance、nucleotide tokenizer、long-context architecture、genome scaling law、homology leakage。
- 纳入：与序列预训练、序列到功能、跨物种迁移、长上下文方法或严格评测直接相关，且元数据可核验。
- 排除：只有新闻稿/博客、无法确认论文身份、仅临床文本而无基因组序列方法、或不能支持本项目技术结论的记录。
- 当前状态：检索尚未完成；核心文献证据矩阵将在本文件中更新，不另建Markdown综述。

### 核心文献证据矩阵

| 文献 | 期刊/年份/DOI | 数据与物种 | tokenizer/架构/上下文 | 目标与算力 | 划分与泄漏控制 | 对本项目可复用内容 | 状态 |
|---|---|---|---|---|---|---|---|
| 尚待系统检索与逐篇核验 | 未核实 | 未核实 | 未核实 | 未核实 | 未核实 | 未确定 | 未开始 |

## 4. 当前数据事实与审计计划

已在原始数据根发现四个来源目录：`legume_family`、`legumeinfo`、`soyod`和`soyomics`。Phase 1通过SLURM核实3,841个目录项、3,427个普通文件、414个symlink和209,947,381,782 bytes（195.529 GiB）普通文件；文件名规则识别到1,289个FASTA与537个注释候选。物种数、属数、实际组装数、总碱基数、重复率、注释基因数、重复组装和可用token数仍待内容级审计。

阶段化审计：

1. Phase 1（已完成）：只读递归inventory（清单），记录相对路径、文件类型、大小、mtime、权限和不跟随的symlink（符号链接）；未在登录节点扫描序列内容。
2. Phase 2：按assembly并行解析FASTA，统计长度、contig、N、GC、IUPAC字符、软掩码、端粒/细胞器/未定位序列候选和SHA-256。
3. Phase 3：解析GFF/GTF/VCF，建立配套关系、基因/CDS/UTR计数、坐标合法性、序列ID覆盖、版本和许可证来源字段。
4. Phase 4：精确序列与反向互补去重；MinHash/比对近重复审计；材料、组装版本、物种、属和低同源隔离。
5. Phase 5：发布不可变数据manifest、split（划分）合同、checksum和可重建shard规范。

正式纳入/排除阈值必须等待Phase 2–4真实分布后冻结。当前不预设contig长度、N比例、上下文或采样权重。

## 5. 泄漏控制的冻结门槛

正式训练前必须验证：完全相同与反向互补重复、重叠窗口、近重复/高同源片段、同一材料多版本组装、下游标签区域、物种/属/进化分支留出及独立外部测试均有机器可读合同；任何门禁失败都阻止正式训练发布。

## 6. 正式模型与训练冻结门槛

冻结前必须同时具备：经核验文献矩阵、有效训练碱基/token规模、长度分布、泄漏后物种/属贡献、RTX 2080实测兼容路径、A100 40GB理论预算及至少meta-device（仅构建不分配权重）精确参数统计。最终只允许一个模型和以下唯一值：tokenizer、最大上下文、有限多尺度集合、各尺度token配额/阶段、训练目标、精确参数量、global batch token、优化器、学习率调度、总token与终止规则。

当前正式配置：**未冻结**。详见`MODEL_ARCHITECTURE.md`。

## 7. 预注册评测框架

核心benchmark（基准评测）将在预训练前冻结，至少完成12个高质量任务并覆盖六层：碱基/基因结构、顺式调控、表达/功能、变异效应、比较基因组迁移和育种应用。候选注册池不少于18个独立问题。每项必须定义困难负样本、输入尺度、标签来源、同物种/留一物种/留一属/低同源/外部划分、随机初始化同架构、通用模型、植物模型、参数或算力匹配对照、重复种子、置信区间、显著性检验和泄漏审计。正式测试集不用于逐checkpoint选择。

## 8. 计算、迁移与图件

- 登录节点仅做轻量检查、代码编辑、网络检索和SLURM控制。
- CPU全量处理优先使用非`cu`节点；当前唯一非`cu`分区为`fat`，资源不足时按记录后的规则回退到`q03/q04/q02/q05`。
- 当前GPU开发仅用RTX 2080 Ti；启动前核验显存、utilization（利用率）、compute进程和设备持有者。
- A100未获本阶段使用授权。
- `figures/figure_manifest.tsv`记录图号、脚本、源数据和输出；无真实结果时不生成伪数据图。
- AutoDL包必须提供锁定环境、manifest/checksum、自检、bootstrap、单/多卡启动和resume（恢复）命令。

## 9. 成功、失败与停止标准

最低成功要求是：严格外部/跨属/低同源测试上，相比强通用和植物模型在预注册任务族中呈现统计稳定且非单任务驱动的优势，同时训练稳定、无已知泄漏且可迁移复现。若豆科专属预训练在公平等算力比较中无一致收益、关键数据许可证不可用、泄漏无法消除、或算力预算与收益不匹配，则停止扩大训练并重新评估科学主张。数值阈值待数据规模和baseline实测后冻结。

## 10. 当前可执行命令

```sh
PYTHON_BIN=/path/to/python scripts/submit_raw_inventory.sh
```

资源：`fat`分区，2 CPU，8 GiB RAM，最长2小时，无GPU、无网络；只读扫描`DATA_ROOT`，输出相对路径manifest与SLURM日志。正式提交记录见`TRAINING_PROGRESS.md`。
