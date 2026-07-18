# LegumeGenomeFM Training Plan

> 状态：系统文献核验已完成、全量FASTA内容审计运行中（2026-07-18）。正式架构、tokenizer（序列离散化方式）、上下文、多尺度配额、参数量、训练目标和优化器仍未冻结；本文不会把候选设想写成正式结论。

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
- 机器检索：36个冻结查询族全部返回候选，合并得到2,958条唯一记录，Nature Portfolio题录粗筛154条。随后通过题名筛选、引文追踪和逐DOI/出版社页面核验，冻结30条核心白名单：19条核心方法、8条上下文证据、3条明确排除的Research Briefing/Highlight。
- 核验状态：30/30题名和标识通过；Nature记录进一步核验`citation_article_type`。Crossref未返回任何`update-to`关系；这表示没有Crossref登记的更新信号，不等价于覆盖所有撤稿数据库。
- 可重现产物：查询式、2,958条候选、30条核验记录和14篇方法矩阵分别见`metadata/literature_search_queries.json`、`data_manifests/literature_candidates.tsv`、`data_manifests/core_literature_verified.tsv`和`data_manifests/literature_evidence_matrix.tsv`。

### 核心文献证据矩阵

| 文献 | 期刊/年份/DOI | 数据与物种 | tokenizer/架构/上下文 | 目标与算力 | 划分与泄漏控制 | 对本项目可复用内容 | 状态 |
|---|---|---|---|---|---|---|---|
| Enformer | Nature Methods 2021; `10.1038/s41592-021-01252-x` | 人/鼠，多模态功能轨道 | 单碱基one-hot；7层卷积+11层Transformer；196,608 bp | 监督Poisson损失；64 TPU v3约3天 | 1-Mb人鼠同源连通分量不跨集合 | 同源分组拆分、长程调控基线 | 全文核验 |
| Nucleotide Transformer | Nature Methods 2024/2025; `10.1038/s41592-024-02523-z` | 3,202人基因组或850物种；多物种集不含植物 | 6-mer；6/12 kb；50M–2.5B Transformer | MLM；2.5B使用128张A100约28天 | 下游10-fold；论文承认部分比较有潜在预训练重叠 | 多物种多样性往往比单纯增参更重要 | 全文核验 |
| AgroNT | Communications Biology 2024; `10.1038/s42003-024-06465-2` | 48种食用植物；1,050万条6-kb序列 | 6-mer；RoBERTa；1B；约6 kb | MLM；8张A100约8天 | 未提供满足本项目要求的留属预训练证据 | 直接植物基线及数据混合参考 | 全文核验 |
| GPN | PNAS 2023; `10.1073/pnas.2311219120` | 7种Brassicales训练，完整留出拟南芥 | 单碱基；25M扩张卷积；512 bp | MLM；单A100约4天/seed | 真正leave-one-species-out（留一物种） | 小模型、物种均衡和变异效应强基线 | 全文核验 |
| Species-aware LM | Genome Biology 2024; `10.1186/s13059-024-03221-x` | 806种真菌，完整留出Saccharomyces属 | 重叠6-mer+species token；90M；300/1,000 nt | MLM；200k steps | 留一属；未知物种需近缘proxy token | 极低成本taxon conditioning（分类群条件）证据 | 全文核验 |
| PlantCaduceus | PNAS 2025; `10.1073/pnas.2421738122` | 16种被子植物；60.8 billion非N碱基 | 单碱基；RC等变MambaDNA；225M；512 bp | 双向语言建模；8张H100、750B tokens、约25天 | 跨物种评测并分析预训练多样性 | 单碱基+严格RC对称的植物强基线 | 全文核验 |
| Evo 2 | Nature 2026; `10.1038/s41586-026-10176-5` | 全生命域；8.8T+碱基 | 单碱基；StripedHyena 2；8,192→1M；7B/40B | 自回归；2.4T/9.3T tokens；千卡级 | 排除真核病毒；广泛外部评测 | 分阶段扩上下文和多尺度算子证据 | 全文核验；规模不可照搬 |
| Borzoi | Nature Genetics 2024/2025; `10.1038/s41588-024-02053-6` | 人/鼠RNA-seq及调控轨道 | one-hot；卷积+8层注意力+U-Net；524 kb→32 bp | 监督训练；2张A100约25天 | 人鼠同源区域同组划分 | RNA-seq/调控下游上限与40-GB显存边界 | 全文核验 |
| AlphaGenome | Nature 2025/2026; `10.1038/s41586-025-10014-0` | 人/鼠，11类模态 | one-hot；U-Net+Transformer；1 Mb→最高1 bp | 监督+蒸馏；单序列跨8个TPU v3 | 四折区间测试；all-fold teacher仅用于蒸馏 | 多分辨率输出和长上下文科学上限 | 全文核验；算力不可照搬 |
| DNA FM benchmark | Nature Communications 2025; `10.1038/s41467-025-65823-8` | 57个任务 | 比较DNABERT-2/GROVER/NT/Hyena/Caduceus | 冻结embedding+RF；单A100测速 | QTL用染色体nested CV；多数分类仍是随机70:30 | 多物种数据可增益；简单CNN常胜；无万能模型 | 全文核验；随机拆分结论降权 |
| GPN-MSA | Nature Biotechnology 2024/2025; `10.1038/s41587-024-02511-w` | 多物种比对 | 开放正文未提取完整架构 | 比对感知变异评分 | 使用正式变异benchmark | 正式variant baseline | DOI/Brief Communication核验 |
| HyenaDNA | NeurIPS 2023; `10.52202/075280-1872` | 人参考基因组 | 单碱基；隐式长卷积；最长1M | 自回归 | 原拆分须重新审计 | 低复杂度长上下文算子 | 官方论文核验 |
| Caduceus | ICML 2024; PMLR v235/schiff24a | 人参考基因组 | 单碱基；双向Mamba；RC参数共享/等变；最长131 kb | 双向语言建模 | 原拆分须重新审计 | RC对称状态空间骨干 | PMLR官方页面核验 |
| PlantGFM | Advanced Science 2026; `10.1002/advs.75772` | 植物基因组 | 报告多尺度整合；细节尚未从可访问全文提取 | 基因发现/生成 | 待代码与split审计 | 当前植物外部对手 | DOI核验；不得据摘要推断细节 |

### 由证据约束的设计结论（尚非正式架构）

1. **数据多样性优先于盲目增参。** NT、GPN、species-aware和2025 benchmark共同支持物种均衡及跨物种数据的价值；参数越大并不自动胜过小型任务模型。
2. **单碱基与RC一致性是植物模型的高价值归纳偏置。** GPN和PlantCaduceus直接支持单碱基表示；Caduceus系模型避免把反向互补当作两个无关模式。
3. **上下文应分阶段扩展。** Evo 2证明先短上下文预训练、再长上下文midtraining可行；Borzoi/AlphaGenome证明长程调控确有价值，但其算力不能在单A100上照搬。
4. **分类群条件必须可退化。** species token成本极低，但未知物种依赖proxy；本项目若采用taxon conditioning，必须含`unknown`和训练期dropout，不能让身份token替代序列学习。
5. **严格拆分高于榜单数字。** Enformer的同源连通分量、GPN整物种留出优于随机窗口；2025 benchmark中大量随机70:30结果只能作为任务可行性证据，不能证明跨物种泛化。
6. **单A100预算迫使模型远小于AgroNT/PlantCaduceus。** 1B AgroNT需8×A100约8天，225M PlantCaduceus需8×H100约25天；正式容量必须由清洗后token量和单A100实测吞吐共同反推。

## 4. 当前数据事实与审计计划

已在原始数据根发现四个来源目录：`legume_family`、`legumeinfo`、`soyod`和`soyomics`。Phase 1通过SLURM核实3,841个目录项、3,427个普通文件、414个symlink和209,947,381,782 bytes（195.529 GiB）普通文件；文件名规则识别到1,289个FASTA与537个注释候选。严格限定为普通文件、`file_type=fasta`且处于`genome/`目录后，Phase 2注册552个组装候选、172,309,972,336 bytes（160.476 GiB压缩体量），确定性分成6个约26.746 GiB的shard。物种数、属数、总碱基数、重复率、注释基因数和可用token数仍待内容级审计完成后确认。

阶段化审计：

1. Phase 1（已完成）：只读递归inventory（清单），记录相对路径、文件类型、大小、mtime、权限和不跟随的symlink（符号链接）；未在登录节点扫描序列内容。
2. Phase 2（运行中，SLURM array `8600499_[0-5]`）：按assembly并行解析552个FASTA，统计长度、contig、N、GC、IUPAC字符、软掩码、N50、压缩文件SHA-256及忽略换行/大小写的序列内容SHA-256。真实smoke已正确检出1个截断gzip，并在另一组装上完成PASS与3秒resume复用验证。
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
