# LegumeGenomeFM：豆科超长上下文基因组基础模型研究设计、数据合同与基准评测方案

**文档版本：** 1.0
**证据截止时间：** 2026-07-22
**项目状态：** `CANDIDATE_SET_READY_DATA_NOT_YET_PACKAGED`（候选数据集已完成质量筛选，但正式训练release尚未生成）
**候选模型：** LegumeGenomeFM-HierMamba
**适用范围：** 数据冻结、预训练、验证、测试、下游benchmark（基准评测）、关键外部模型比较、论文证据边界

> 重要证据边界：截至本文档日期，LegumeGenomeFM尚未开始正式预训练，也没有任何正式下游任务成绩；正式预训练和全部下游任务均未运行。因此，本文中的LegumeGenomeFM性能均标记为`NOT_RUN`或`N/A`，绝不把设计目标、CPU单元测试、数据QC（质量控制）通过率或其他论文的分数冒充本模型性能。外部模型数值均为其论文原协议下的文献报告值，不能与未来本项目同一split（数据划分）成绩直接混排。

## 1. 一页结论

本项目要回答的不是“能否再训练一个植物DNA模型”，而是三个更严格的问题：

1. **豆科领域专门化是否有效。** 在相同输入长度、相同标签、相同训练/验证/测试划分和相同下游head（任务头）预算下，约3.15亿参数的豆科模型能否在常规植物任务上不明显落后于AgroNT-1B、PlantCAD2、Nucleotide Transformer v2 500M，并在豆科根瘤共生、Rhg1/Rhg4胞囊线虫抗性、豆科巨型基因结构、豆科泛基因组SV/PAV（结构变异/存在缺失变异）、大豆成熟期与区域适应等任务上显著超过它们。
2. **256K上下文是否真的被利用。** 不能只证明模型“能放进256K”。必须在完全相同中心样本的1K/8K/32K/64K/128K/256K嵌套输入上，证明完整长上下文比截断、分块平均、随机flank（侧翼）或跨样本flank更好；否则超长架构没有生物学证据。
3. **跨属泛化是否真实。** 预训练阶段建议完全排除Phaseolus（菜豆属）、Arachis（花生属）和Vicia（野豌豆属）。Phaseolus作为cold-development（预训练未见属的开发验证集），Arachis与Vicia作为sealed cold test（封存跨属测试集）。这使最终结论不能依赖把测试属提前放进预训练。

当前已经真实完成：

- 92个候选基因组的BUSCO（保守单拷贝基因完整度）、污染和record-level QC全部闭合；
- 83个候选通过全部硬门禁，经过材料去重和方向/序列等价去重后选择74个source（来源基因组）；
- 74个source覆盖19个物种、13个属，具有91,922,061,939 bp可训练序列、124,098个可训练区间；
- 1K至128K均有74个source可用，256K有73个source可用；
- HierMamba模型、混合长度采样器、内容哈希门禁和候选preflight（训练前检查）已有代码与单元测试；
- 当前完整测试为152 passed。

当前尚未完成：

- cold-genus方案尚未写入schema-2正式release；
- HierMamba生产trainer（训练循环）尚未完成。现有`scripts/train_pretrain.py`仍连接固定长度的旧`LegumeGenomeModel`，不能直接执行候选HierMamba合同；
- H20上的Mamba-2/causal-conv依赖、显存、吞吐、DDP（分布式数据并行）和精确`named_parameters()`审计尚未完成；
- 所有正式预训练与下游benchmark均未运行。

因此，当前科学结论是“数据与候选架构已具备进入正式冻结前验证的条件”，不是“模型已经有效”。

## 2. 科学问题、假设与预注册成功标准

### 2.1 核心科学问题

- **Q1：** 豆科多物种、近重复受控的预训练，能否学习豆科特有的基因调控、共生、抗病与驯化序列语法？
- **Q2：** 在植物DNA模型通常只覆盖512 bp、6 kb、8,192 bp或约12.3 kb的条件下，32K–256K是否能恢复完整长基因、重复拷贝单倍型、远端调控和SV上下文？
- **Q3：** 模型在预训练未见属上是否仍能迁移，还是只记住了50个Glycine（大豆属）组装版本？
- **Q4：** 基因组基础模型embedding（向量表征）加入GBLUP/RR-BLUP等传统育种模型后，能否提高跨家系、跨区域和跨环境预测，而不是仅在随机拆分中利用亲缘关系泄漏？

### 2.2 预注册假设

- **H1，通用任务非劣：** 在共同可输入的1K或8K协议中，LegumeGenomeFM对主要AUPRC（精确率-召回率曲线下面积）、MCC（类别不平衡相关系数）、Spearman相关和R²指标，相对最佳植物专属baseline满足预注册非劣界。
- **H2，豆科专属优势：** 在根瘤细胞程序、Rhg1长重复单倍型、豆科巨型基因结构、Glycine WGD（全基因组复制）homeolog（同源复制基因）判别和区域适应多位点任务中，LegumeGenomeFM超过最佳“可合法执行且输入匹配”的外部模型。
- **H3，长上下文因果收益：** 完整32K–256K输入优于相同中心的8K截断，并且优势在flank shuffle（侧翼打乱）和cross-source flank（跨材料侧翼替换）中消失或显著下降。
- **H4，跨属迁移：** sealed Arachis/Vicia测试中的提升不能由测试属预训练暴露解释；所有模型均报告其预训练物种暴露状态。
- **H5，育种增量价值：** FM embedding加入标准亲缘关系/主成分/环境协变量后，仍对跨家系或跨区域测试提供可重复的增量，而不是取代一个被弱化的传统baseline。

### 2.3 “媲美”和“超过”的正式定义

| 结论用语 | 预注册判据 |
|---|---|
| 媲美/非劣 | 主要指标差值的95%配对bootstrap（自助法）置信区间下界高于预设非劣界；AUPRC/Macro-F1界为-0.02，AUROC/MCC界为-0.01，回归Spearman/R²界为-0.02 |
| 超过 | 相对最佳合格baseline的主要指标差值95%置信区间下界>0；同时AUPRC/Macro-F1绝对提升原则上≥0.03，或Spearman/R²绝对提升≥0.03；至少3个独立seed（随机种子）方向一致 |
| 原生长度覆盖优势 | baseline不能原生读取该长度时只记为`INELIGIBLE_NATIVE_CONTEXT`，不能把它计为0分，也不能据此声称性能胜利 |
| 真正长上下文收益 | full-context优于本模型8K截断、短模型tile-pooling（分块聚合）、flank shuffle和cross-source flank四类对照 |
| 不支持结论 | 只在单seed成立、测试集用于调参、近同源跨split、模型预训练暴露未知且未披露、或置信区间跨0 |

## 3. 数据来源、许可与质量控制

### 3.1 原始数据资产

项目保留约195.529 GiB原始数据，原始目录只读。原始清单含3,841项文件；584个genome来源通过基础门禁；574个来源具有annotation（注释）闭合信息；466个sequence store（序列存储）均为READY。正式数据筛选只使用来源元数据、公开/训练许可、基因组、GFF3注释及可再计算QC；没有下载FASTQ/BAM重做变异检测，也没有修改原始文件。

主要来源包括Legume Information System及其公开配套基因组/注释。每个候选必须通过：

1. provenance（来源）与许可门禁；
2. genome/annotation配对闭合；
3. genome BUSCO与protein BUSCO；
4. Tiara/UniVec/BLAST污染检查；
5. primary nuclear fraction（主核基因组比例）检查；
6. 材料、方向和近重复控制；
7. SHA-256内容寻址验证。

### 3.2 92个深度QC候选的结果

- BUSCO、污染和record级QC：92/92全部计算闭合；
- BUSCO硬门禁：84 PASS、8 FAIL；8个失败均为annotation BUSCO完整度<90%，虽然genome BUSCO仍为97.6%–99.0%；
- record级门禁：91 PASS、1 FAIL；唯一失败为主核基因组比例不足；
- 通过全部门禁：83个；
- 同材料替代版本排除：7个；
- 方向/序列等价替代版本排除：2个；
- 最终选择：74个source。

这74个source不是74个完全独立生物学样本。最终近重复权重对应约70个有效source；50/74属于Glycine。正式采样必须依赖物种、source容量和近重复组权重，不能按文件均匀抽样。

### 3.3 74-source组成

| 属 | 物种 | source数 | 可训练bp | 256K不重叠窗口 |
|---|---|---:|---:|---:|
| Arachis | A. hypogaea | 2 | 5,066,089,559 | 17,502 |
| Arachis | A. stenosperma | 1 | 1,231,966,351 | 3,793 |
| Cajanus | C. cajan | 1 | 510,632,555 | 0 |
| Cercis | C. chinensis | 1 | 331,320,899 | 1,254 |
| Cicer | C. arietinum | 1 | 457,434,276 | 3 |
| Glycine | G. max | 45 | 44,551,510,384 | 161,840 |
| Glycine | G. soja | 5 | 4,895,287,096 | 17,129 |
| Lablab | L. purpureus | 1 | 415,816,948 | 1,449 |
| Lens | L. culinaris | 1 | 3,488,978,896 | 11,030 |
| Lens | L. ervoides | 1 | 2,751,410,373 | 9,870 |
| Lupinus | L. albus | 1 | 431,633,093 | 1,610 |
| Medicago | M. truncatula | 4 | 1,660,386,676 | 5,405 |
| Phaseolus | P. acutifolius | 1 | 502,359,084 | 1,727 |
| Phaseolus | P. lunatus | 1 | 511,920,544 | 1,846 |
| Phaseolus | P. vulgaris | 3 | 1,625,014,406 | 5,869 |
| Trifolium | T. pratense | 1 | 406,190,051 | 1,431 |
| Vicia | V. faba | 2 | 22,121,774,076 | 75,999 |
| Vigna | V. angularis | 1 | 462,192,235 | 1,514 |
| Vigna | V. radiata | 1 | 500,144,437 | 1,900 |
| **合计** | **19物种/13属** | **74** | **91,922,061,939** | **321,171** |

全部组装的染色体尺度证据目前为`structural_proxy`（结构代理），即由伪染色体结构、N50、大序列比例和primary fraction支持；不能在论文中写成官方`Chromosome`认证。

## 4. 训练、验证与测试数据划分

### 4.1 推荐的严格跨属合同

本报告建议但尚未写入正式schema-2 release的拆分如下：

| 集合 | 属 | source | 物种 | 可训练bp | 用途 |
|---|---|---:|---:|---:|---|
| 预训练池 | 除Arachis、Phaseolus、Vicia外的10属 | 64 | 13 | 60,862,937,919 | span-MLM预训练、ID验证与ID测试 |
| cold-development | Phaseolus | 5 | 3 | 2,639,294,034 | 模型选择、跨属开发，不进入预训练 |
| sealed cold test | Arachis + Vicia | 5 | 3 | 28,419,829,986 | 最终一次跨属/巨型基因组测试，不参与模型选择 |

选择逻辑：

- Phaseolus含3个物种和5个组装，适合稳定的unseen-genus开发；
- Arachis同时含栽培种和野生种，适合跨属、驯化与泛基因组任务；
- Vicia贡献超大基因组和大量长区间，是对256K是否能迁移到预训练未见巨型基因组的强测试；
- 这项设计故意让本模型处于更严格条件。PlantCAD2语料已明确含A. hypogaea和V. faba，因此未来比较必须把它标为“baseline对测试属可能已预训练暴露”，而不是假装所有模型都完全cold。

### 4.2 每个集合的正式窗口容量

以下是finalizer产生的`eligible_nonoverlap_windows`（可用不重叠窗口数），不是随机切窗可能产生的所有重叠窗口数：

| 集合 | 1K | 8K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|
| 74-source总容量 | 89,706,255 | 11,161,183 | 2,753,850 | 1,358,122 | 665,151 | 321,171 |
| 64-source预训练池 | 59,387,583 | 7,382,203 | 1,818,464 | 896,712 | 440,553 | 214,435 |
| Phaseolus cold-development | 2,576,726 | 321,442 | 79,812 | 39,522 | 19,425 | 9,442 |
| Arachis+Vicia sealed test | 27,741,946 | 3,457,538 | 855,574 | 421,888 | 205,173 | 97,294 |

256K在预训练池中有63/64个source可用；C. cajan不满足256K，但仍参加1K–128K。任何长桶不得因为某个source容量不足而回退到非法短窗口后仍记作长窗口。

### 4.3 预训练池内部坐标拆分

正式release builder仍需实现以下合同：

1. 对64个预训练source按染色体同源组件而不是随机窗口拆分，目标base mass（碱基量）为90% train、5% ID-validation、5% ID-test；
2. 同一同源染色体组件、近重复材料和重叠256K邻域不得跨split；
3. 在split边界设置至少256K quarantine buffer（隔离缓冲）；
4. 所有短窗口继承其最长父区间的split，不能分别随机分配；
5. exact count（精确窗口数）只能由schema-2 receipt生成。本文不预填尚未构建的90/5/5计数。

Phaseolus可用于选择checkpoint和任务超参数；Arachis/Vicia只能在全部选择冻结后解封一次。正式test不得在每个checkpoint反复运行。

## 5. 上下文长度混合与训练数据量

### 5.1 单一连续run的长度配方

候选合同采用一个连续optimizer/scheduler/checkpoint lineage（优化器/学习率/检查点谱系），不做“先短后长再重置”的多阶段模型。按处理token（这里一个DNA碱基约等于一个token）分配：

| 长度 | token占比 | 生物学侧重 |
|---:|---:|---|
| 1,024 | 10% | motif（基序）、剪接点、起止位点、局部调控 |
| 8,192 | 15% | 完整小基因、promoter（启动子）、近端调控；与PlantCAD2共同长度 |
| 32,768 | 20% | 中长基因、局部重复、Rhg1重复单元 |
| 65,536 | 20% | 长基因、基因簇、SV侧翼 |
| 131,072 | 20% | 巨型基因、长重复结构、较远调控 |
| 262,144 | 15% | 完整复杂区域、长PAV/SV、跨多个局部模块 |

选择长度后先按物种容量的0.3次方采样物种，再按source碱基容量0.5次方并除以近重复组大小0.5次方采样source。这样既不让Glycine的50个版本按数量支配训练，也不让单个Vicia超大基因组按bp完全支配训练。

### 5.2 三档token预算及窗口draw数

H20吞吐未测，不能现在把预算伪装成已冻结值。本报告预注册三档，由H20 Gate选择：

- minimum viable：30B输入token；
- target：100B输入token；
- stretch：200B输入token。

100B target对应计划抽样窗口数：

| 长度 | 分配token | 计划窗口draw数 |
|---:|---:|---:|
| 1K | 10B | 9,765,625 |
| 8K | 15B | 1,831,055 |
| 32K | 20B | 610,352 |
| 64K | 20B | 305,176 |
| 128K | 20B | 152,588 |
| 256K | 15B | 57,220 |
| **合计** | **100B** | **12,722,016** |

30B和200B分别约需3,816,605和25,444,031次窗口draw。draw数是训练采样次数，不是独立生物学样本数。`tokens_seen`只计原始输入一次；模型为RC conjoin（正反向互补合并）进行两次整模型前向，实际计算量约再乘2。

## 6. 模型架构：LegumeGenomeFM-HierMamba

### 6.1 输入与tokenizer

词表为7个单碱基token：`PAD、MASK、A、C、G、T、N`。优势是坐标精确、单碱基预测直接、不同长度统一；代价是序列比6-mer模型长约6倍。输入必须在1,024–262,144 bp之间且能被128整除。

### 6.2 层次化local encoder

输入`[B,L]`经过128维embedding后转为`[B,128,L]`。8个尺度如下：

| 尺度 | 通道 | 序列长度 | residual block数 |
|---:|---:|---:|---:|
| 0 | 128 | L | 2 |
| 1 | 192 | L/2 | 2 |
| 2 | 256 | L/4 | 2 |
| 3 | 384 | L/8 | 2 |
| 4 | 512 | L/16 | 2 |
| 5 | 768 | L/32 | 2 |
| 6 | 1024 | L/64 | 2 |
| 7 | 1024 | L/128 | 2 |

每次下采样使用`Conv1d(kernel=4,stride=2,padding=1)`；每个local residual block（局部残差块）使用depthwise Conv1d、RMSNorm、4倍通道扩展、GELU和投影。16个encoder block完成128倍压缩。

### 6.3 24层双向Mamba-2 global core

压缩后global token长度为：1K→8、8K→64、32K→256、64K→512、128K→1,024、256K→2,048。这样全局层始终在最多2,048个latent token（潜在标记）上计算，而不是直接在262,144个碱基上运行24层。

每个global block参数：

- `d_model=1024`
- `d_state=128`
- `d_conv=4`
- `expand=2`
- `headdim=64`
- `ngroups=8`
- `chunk_size=256`
- 层数24

每层含forward scan和reverse scan；反向序列运行后再翻转对齐。两个方向共享`in_proj`和`out_proj`，但保留各自的卷积与状态参数，输出相加并走残差连接。

### 6.4 U-Net式decoder与单碱基输出

Global输出与最深encoder skip相加，再经过7次`ConvTranspose1d`上采样。每层与对应encoder skip拼接，经1×1卷积融合和local residual blocks恢复到`[B,L,128]`。MLM head输出7类且与输入embedding权重绑定。

### 6.5 严格RC一致性

模型不仅在global层内双向扫描，还对整个输入及其reverse complement（反向互补）各运行一次完整backbone，将RC输出翻转并按A↔T、C↔G对齐后平均。由此最终logits理论上满足方向一致性。代价是整模型约2倍前向计算；必须通过H20吞吐验证确认可承受。

### 6.6 参数量

| 组件 | 参数量 | 状态 |
|---|---:|---|
| embedding、local encoder、down/up-sampling、skip fusion、decoder、norm | 111,470,528 | 代码结构精确计数；不含Mamba-2参数 |
| 24层双向Mamba-2 mixer | 203,198,976 | 按固定commit参数公式、`d_state=128`及共享in/out projection推导 |
| **候选总量** | **314,669,504（约314.67M）** | 设计推导值 |

最终论文参数量只能使用H20目标环境成功实例化后的`unique_trainable_parameter_count()`与逐参数清单。314.67M不能替代生产依赖实测；如果固定commit的实现、bias或fused module改变，必须重新审计。

## 7. 预训练任务

### 7.1 当前代码已实现任务

当前采样器实现15% span-MLM（跨度掩码语言模型）：

- 选中位置的80%替换为MASK；
- 10%替换为随机碱基；
- 10%保持原碱基；
- 当前span平均长度为3 bp；
- loss按全局有效masked token总数求和后归一化；
- `tokens_seen`、每长度token计数、采样器状态、优化器、scheduler和RNG（随机数）全部应进入checkpoint。

### 7.2 正式训练前必须增加的长程masking spike

仅使用平均3 bp短span可能让local encoder解决大部分问题，无法证明global 256K被训练。正式合同建议在不增加新head、不改变单一run的前提下，把同一MLM loss改为multi-scale span mixture（多尺度跨度混合）：

| mask模式 | batch占比建议 | 平均span | 上限 | 目的 |
|---|---:|---:|---:|---|
| short | 70% | 3 bp | 16 bp | motif和单碱基语法 |
| medium | 20% | 32 bp | 256 bp | exon片段、短调控模块 |
| long | 10% | 256 bp | min(4,096 bp, L/16) | 强迫模型利用较远上下文恢复连续缺失块 |

总mask比例仍为15%。这只是待spike验证的推荐，不是已实现事实。通过标准是：在32K–256K上，multi-scale masking相对mean-3-only提高held-out long-span恢复且不显著损害短任务；否则回退到更简单目标。

### 7.3 不采用的伪任务

不把“判断两个随机片段是否来自同一染色体”“预测人工打乱顺序”等容易学习GC、重复比例或数据来源捷径的任务作为主预训练目标。长程能力应由真实下游任务和flank干预证明，而不是靠合成分类准确率自证。

## 8. 优化、分布式训练与H20 Gate

候选优化器为AdamW；当前建议`lr=2e-4`、`betas=(0.9,0.95)`、`weight_decay=0.1`、`grad_clip=1.0`、BF16、activation checkpointing（激活重计算）。学习率按token而非step调度；warmup暂为总token的1%。这些都必须在H20实测后冻结。

正式启动的硬门禁：

1. 2–3张H20 96GB均通过ECC/Xid/显存和进程检查；
2. 固定commit的`mamba_ssm`和`causal_conv1d`可导入；
3. 1K–256K逐长度forward/backward成功；
4. 自动选择每长度micro-batch和gradient accumulation，使global token batch接近目标；
5. 2卡和3卡DDP的loss归一化与单卡对齐；
6. 生成参数量、峰值显存、tokens/s、samples/s、masked tokens/s和预计总时长receipt；
7. 训练数据状态必须是`TRAINING_DATASET_READY`，而不是当前候选状态；
8. 生产HierMamba trainer必须替换当前固定长度legacy trainer并通过resume等价测试。

## 9. 关键外部baseline

### 9.1 必须纳入的四个核心baseline

| 模型 | 参数 | 原生最大输入 | 预训练侧重 | 主要比较角色 |
|---|---:|---:|---|---|
| AgroNT-1B | 1B | 6,000 bp | 48种食用植物、6-mer、15% MLM | 最重要农业/植物专属baseline；通用任务与豆科表达任务 |
| PlantCAD2-Large | 694M（预印本正文；摘要写676M） | 8,192 bp | 65种被子植物、基因中心窗口、单碱基RC模型 | 最强直接植物专属baseline；Small 88M与Medium 311M用于规模匹配 |
| NT-v2-500M-multi-species | 500M | 12,282 bp | 850个多物种基因组、6-mer MLM | 强通用多物种Transformer baseline |
| Evo-2-7B | 7B | 1,000,000 bp | 128,000个基因组、8.8T token、自回归 | 能原生覆盖256K的强通用超长baseline |

Evo-2-40B只有在2–3×H20真实推理gate通过时加入；否则记录`INFEASIBLE_ON_DECLARED_HARDWARE`，不以7B替代后仍称作40B结果。

### 9.2 补充baseline与非基础模型控制

- PlantCaduceus-l32（225M、512 bp）：植物RC-equivariant（反向互补等变）短上下文专家；
- GPN-Brassicales（官方权重65,880,071参数、512 bp）：植物变异效应短上下文专家；
- Caduceus（最长公开论文checkpoint 131K）：长程RC架构控制；
- HyenaDNA（最高1M）：长程非植物预训练控制；
- DNABERT-2：短上下文BPE DNA模型；
- task-specific CNN、ResNet、BiLSTM和Transformer；
- GBLUP、RR-BLUP、BayesB、XGBoost/LightGBM：育种任务必要传统baseline；
- one-hot k-mer、GC/repeat/length、注释密度等简单baseline，用于发现数据捷径。

### 9.3 长度公平协议

每个长任务必须同时报告：

1. **Matched-1K/8K：** 所有模型读取完全相同中心区域；
2. **Native-full：** 能原生读取全长的模型读取32K–256K；
3. **Tile-pooling：** 短模型把长区域切成其原生长度，使用同一参数预算的set-pooling head聚合；
4. **Our-truncated：** LegumeGenomeFM也截为1K/8K，分离领域预训练优势与上下文优势；
5. **Intervention controls：** flank shuffle、cross-source flank和距离保持但内容替换。

不能把AgroNT或PlantCAD2“无法原生读取256K”写成性能低；只能说它们需要分块、不能在encoder内发生跨块交互。真正性能胜负由tile-pooling对照决定。

## 10. 文献报告性能：只能作为参考锚点

### 10.1 AgroNT原论文

AgroNT论文报告：alternative polyadenylation（可变多聚腺苷化）AUROC约0.89–0.96、AUPRC约0.82–0.93；Arabidopsis donor/acceptor剪接AUROC约0.97/0.95、AUPRC约0.97/0.98；promoter strength R²约0.70/0.73；terminator strength R²约0.67/0.77；多物种gene expression R²约0.419–0.621；跨物种chromatin accessibility AUPRC约0.51–0.67。它们来自AgroNT自有数据和split，不是本项目成绩。

### 10.2 PlantCAD2预印本中的关键植物对照

| 任务 | PlantCAD2文献值 | 对照文献值 | 说明 |
|---|---:|---:|---|
| G. max ACR | AUPRC 0.5111（Large） | AgroNT 0.4159 | PlantCAD2 Supplementary Table 8 |
| P. vulgaris ACR | AUPRC 0.4854（Large） | AgroNT 0.4367 | 豆科但仍为原论文协议 |
| 大豆NAM表达二分类 | AUROC 0.8538（Large） | AgroNT 0.8190 | 表达head协议不同于本报告未来任务 |
| 大豆NAM表达回归 | absolute Spearman 0.6341（Medium） | AgroNT 0.6164 | 最佳PlantCAD2规模不是Large |
| 玉米translation回归 | Spearman 0.3212（Medium） | AgroNT 0.1808 | 原论文finetuning |
| Andropogoneae conservation | AUROC 0.7245（Large） | Evo-2-7B 0.6912 | zero-shot，8,192 bp |
| Poaceae TIS | accuracy 0.6703（Large） | Evo-2-7B 0.5335 | Evo不是所有植物任务都占优 |
| Poaceae non-TIS | accuracy 0.7125（Large） | Evo-2-7B 0.8216 | 反例：Evo明显更好 |
| SV effect | AUROC/AUPRC 0.8452/0.8410（Large） | Evo-2-7B 0.8395/0.7706 | 预印本任务协议 |

PlantCAD2为2025年预印本；其摘要称Large为676M，正文、图和公开checkpoint命名为694M。本文保留这个内部不一致，不擅自选择一个“更漂亮”的数字。

### 10.3 NT-v2与PlantCaduceus

NT-v2论文在其人类剪接benchmark中报告500M多物种模型top-k retrieval约96%、PR-AUC约0.98。PlantCaduceus论文报告Arabidopsis constraint validation AUROC/AUPRC 0.896/0.876、向maize转移0.829/0.797。二者都不构成豆科长上下文直接证据。

### 10.4 本项目目前的全部性能

| 类别 | 当前结果 |
|---|---|
| 数据QC | 已完成：92/92计算闭合；84 BUSCO gate PASS；91 record-QC PASS；最终74-source |
| 模型CPU/单元测试 | 已完成：项目完整测试152 passed |
| H20峰值显存/吞吐 | `NOT_RUN` |
| MLM validation loss/accuracy | `NOT_RUN` |
| 1K–256K上下文收益 | `NOT_RUN` |
| 所有下游任务 | `NOT_RUN` |
| 与AgroNT/PlantCAD2/NT-v2/Evo2同协议比较 | `NOT_RUN` |

“152 passed”只说明代码测试通过，不是模型性能。

## 11. 通用任务组：与关键baseline正面对齐

### C01：Masked-base recovery（掩码碱基恢复）

- **问题：** 模型是否学会不同长度、不同物种的DNA条件分布。
- **数据：** ID-validation、Phaseolus cold-development、sealed Arachis/Vicia；直接来自训练release未使用区间。
- **长度：** 1K/8K/32K/64K/128K/256K。
- **标签：** 被mask的真实A/C/G/T；N不计主要准确率。
- **指标：** masked NLL、accuracy、按GC/repeat/基因区分层的accuracy、RC consistency error。
- **baseline：** MLM模型用相同mask；Evo2用条件/伪似然单独报告，不能把AR NLL与MLM NLL直接混为一个排名。
- **当前样本数：** capacity已给出；正式mask实例数由release和token预算决定。
- **状态：** `NOT_RUN`。

### C02：Splice donor/acceptor（剪接供体/受体）

- **单位：** 以注释转录本的真剪接位点为正例；负例按染色体、GC、基因内位置和二核苷酸匹配。
- **输入：** 512 bp、1K、8K；额外对长内含子使用32K。
- **拆分：** 先按orthogroup（直系同源组）聚类，再按染色体/属拆分；同源基因不得跨split。
- **指标：** AUPRC主指标；MCC、AUROC和calibration（校准）次指标。
- **baseline：** AgroNT、PlantCAD2、NT-v2、PlantCaduceus、GPN、CNN。
- **原始标签上限：** 74套GFF含4,823,254条mRNA、25,931,250条CDS记录；注释格式异质，正式计数必须经过主转录本与exon重建。
- **状态：** dataset builder未实现，性能`N/A`。

### C03：TIS/TTS与gene-boundary（翻译起点/终点与基因边界）

- **输入：** 1K与8K中心窗口；长基因另做32K完整locus（位点）输入。
- **标签：** 主转录本CDS起止、gene start/end；negative按同一基因附近匹配。
- **指标：** AUPRC、MCC、±1/±3/±10 bp boundary F1。
- **防泄漏：** 不允许同一个基因的不同isoform跨split。
- **baseline：** PlantCAD2、AgroNT、NT-v2、Evo2、CNN。
- **状态：** `NOT_RUN`。

### C04：ACR/CRE（开放染色质/顺式调控元件）

- **来源：** GEO GSE128434/PRJNA527732，覆盖13种植物的ATAC/DNase、histone mark与expression；重点使用G. max和P. vulgaris；可复用AgroNT Plant Genomic Benchmark公开FASTA。
- **输入：** 中心600 bp、1K、8K及可用时32K嵌套上下文。
- **标签：** peak中心为正例；负例按GC、可比对性、重复和距TSS距离匹配。
- **拆分：** 染色体留出、物种留出；不沿用不透明随机split。
- **指标：** AUPRC主指标、AUROC、MCC、Brier score（概率校准误差）。
- **科学重点：** 8K匹配PlantCAD2，32K检验远端侧翼是否增益。
- **状态：** 公共来源已核验，数据未导入，性能`N/A`。

### C05：Gene expression（基因表达）

- **来源：** AgroNT Plant Genomic Benchmark、SoyBase expression atlas及公开豆科组织表达矩阵。
- **输入：** TSS上下游1K/8K；完整基因+侧翼32K–256K。
- **标签：** log-normalized expression；同时做expressed/not-expressed二分类。
- **拆分：** gene-family split、染色体split和species transfer；组织条件通过显式tissue embedding输入，不能让同一序列在没有条件变量时对应多个互相矛盾标签。
- **指标：** Spearman、Pearson、R²、binary AUROC/AUPRC。
- **baseline：** AgroNT、PlantCAD2、NT-v2、Evo2、Enformer式CNN、promoter k-mer。
- **状态：** `NOT_RUN`。

### C06：Constraint/variant effect（保守约束/变异效应）

- **来源：** Soybean pangenome、SoySNP50K/GmHapMap和公开处理后VCF；不下载FASTQ/BAM重呼叫。
- **任务：** 常见/稀有变异区分、保守位点打分、已知功能等位基因排序。
- **输入：** 512 bp–8K用于单变异；32K–256K用于SV或单倍型。
- **拆分：** 变异附近同源块、材料和染色体隔离；MAF和population structure分层。
- **指标：** AUROC/AUPRC、Spearman、MRR/Hits@K。
- **baseline：** PlantCaduceus、GPN、PlantCAD2、NT-v2、Evo2、SnpEff/保守分数。
- **状态：** `NOT_RUN`。

### C07：Whole-gene segmentation（完整基因分割）

- **标签：** intergenic、UTR、CDS、intron及方向；先把不同GFF schema规范化。
- **原始上限：** 3,690,769个gene row；其中gene span≥8K为346,276、≥32K为14,412、≥64K为2,570、≥128K为533、≥256K为67。它们包含多组装与近重复，不能直接当独立样本数。
- **输入：** 覆盖完整gene+两侧flank的最小嵌套桶。
- **指标：** base-level macro-F1、boundary F1、完整转录本exact match、长内含子召回。
- **baseline：** AUGUSTUS/Helixer等专用注释器、PlantCAD2 tile、Evo2、Caduceus/HyenaDNA。
- **状态：** source标签已存在，标准化builder未实现，性能`N/A`。

### C08：Ortholog/core-accessory retrieval（直系同源与核心/可变基因检索）

- **来源：** soybean/peanut pangenome及LIS orthology；标签只使用发布的orthogroup/PAV表或冻结工具版本重新计算。
- **输入：** full gene+flank，8K–256K。
- **任务：** ortholog retrieval、core/soft-core/distributed/private分类、跨属zero-shot。
- **指标：** Recall@K、MRR、AUROC/AUPRC。
- **baseline：** DIAMOND/MMseqs2蛋白同源、k-mer、AgroNT/PlantCAD2/NT-v2/Evo2 embeddings。
- **状态：** `NOT_RUN`。

## 12. 超长与复杂区域任务组

### L01：Cold-genus giant-gene reconstruction（未见属巨型基因重建）

这是最直接检验256K价值的任务。

- **开发：** Phaseolus有356个gene span≥32K、44个≥64K、6个≥128K、1个≥256K的原始gene row；
- **封存测试：** Arachis+Vicia有425个≥32K、85个≥64K、21个≥128K、1个≥256K；
- **训练：** 预训练池有13,631个≥32K、2,441个≥64K、506个≥128K、65个≥256K的原始gene row；
- **关键限制：** 正式样本数必须在主转录本、orthology、近重复和注释一致性过滤后重新出receipt；
- **输出：** CDS/intron/UTR base segmentation与边界；
- **比较：** PlantCAD2/AgroNT/NT-v2原生截断、tile-pooling；Evo2/HyenaDNA原生长输入；本模型1K/8K/全长三档；
- **主要结论：** 只有全长相对tile和本模型8K均显著提升，才证明层次化global core有用。

### L02：Nested distal-context ACR/expression（嵌套远端调控上下文）

对同一个ACR或gene中心固定标签，生成1K→256K同源嵌套窗口。统计性能随长度的饱和曲线；以flank shuffle、distance-preserved random flank和另一accession同源区域替换判断模型是否真正使用侧翼内容。不能用不同样本组成的“长桶比短桶好”替代嵌套证据。

### L03：Pangenome SV/PAV breakpoint and effect（泛基因组结构变异断点与效应）

- **来源：** 2020 soybean pangenome公开PAV/SV表；915-accession soybean pangenome；2025 peanut pangenome的8个高质量基因组、269个accession及公开SV-GWAS结果。
- **输入：** 两侧各16K/32K/64K形成32K–128K窗口；复杂SV可用256K。
- **任务：** breakpoint真/假、SV是否落在基因/调控域、core/accessory变化、表型关联候选排序。
- **防捷径：** 负例匹配重复类别、长度、染色体和组装可比对性；同一SV family不得跨split。
- **baseline：** sequence-only FM、SV长度/类别/TE注释模型、PlantCAD2 tile、Evo2 full。

### L04：Glycine WGD homeolog disambiguation（大豆复制同源基因判别）

- **问题：** 大豆近期WGD产生大量相似homeolog。短编码序列可能无法区分其不同调控和synteny（共线性）背景。
- **标签：** SoyBase/LIS发布的syntenic homeolog与ortholog关系。
- **输入：** gene-only、8K、32K、128K嵌套；成对或检索式任务。
- **拆分：** 整个homeolog family作为一个group；不同组装的同一材料归同group。
- **指标：** pair AUROC、Recall@1/5、MRR。
- **专属性：** 这是Glycine复制历史与多组装语料直接相关的任务；PlantCAD2/AgroNT可做短窗口，但不能原生利用>12K syntenic flank。

### L05：TE–gene boundary in giant legumes（巨型豆科TE-基因边界）

Vicia/Lens等大基因组中LTR retrotransposon（长末端重复逆转座子）造成超长intergenic和intron结构。任务以冻结EDTA/RepeatMasker结果与人工高置信交集为标签，做TE边界、嵌套TE和gene/TE冲突分类。该任务当前为`BLOCKED_DATASET_NOT_BUILT`，只能作为次要探索任务，不能用单一工具伪标签形成论文主结论。

## 13. 豆科专属任务组

### S01：Soybean nodule cell-program prediction（大豆根瘤细胞程序预测）

- **来源：** GSE226149/PRJNA938968单核图谱；14,369个root nuclei（根细胞核）、16个root cluster；7,830个nodule nuclei、11个nodule cluster；含infected、non-fixing、fixing和senescing等根瘤状态。
- **标签构建：** 以公开count matrix重新做预注册差异分析，得到每个gene的root-specific、infection、non-fixing、fixing、senescing或shared标签；不用论文图中手抄少量marker充当大数据集。
- **输入：** promoter 1K/8K、完整gene 32K、最大256K；相同gene嵌套比较。
- **拆分：** orthogroup+染色体；测试gene family在训练中完全缺席；单细胞样本的相同gene不能跨split。
- **指标：** macro-AUPRC、macro-F1、层次分类F1；类别不平衡下不以accuracy为主。
- **baseline：** PlantCAD2、AgroNT、NT-v2、Evo2、表达共变网络、promoter k-mer/CNN。
- **预期优势来源：** 豆科/大豆序列语法和长gene/flank，而不是声称DNA序列可以在没有condition变量时预测任意细胞状态。

### S02：Nodulation/symbiosis gene-function retrieval（根瘤共生基因功能检索）

- **来源：** Roy等2020综述及LIS/SoyBase/Medicago/lotus正式注释；构建人工核验的功能层次：Nod factor perception、common symbiosis signalling、infection thread、nodule organogenesis、autoregulation、nitrogen fixation/metabolism。
- **任务：** 低样本hierarchical classification（层次分类）与query-to-gene retrieval；NFR1/NFR5、SYMRK、CASTOR/POLLUX、CCaMK、CYCLOPS、RPG、ERN、NIN、NF-Y、SUNN/HAR1等只是已知锚点，不直接扩充成重复正例。
- **输入：** full gene+up/downstream flank 8K–128K。
- **拆分：** leave-one-genus-out；同一orthogroup全部同split。
- **指标：** macro-AUPRC、hierarchical F1、MRR/Recall@K。
- **风险：** 已知功能基因数量有限。因此主结论应是few-shot retrieval（少样本检索），不能训练大分类器后宣称高准确率。

### S03：Rhg1/Rhg4–SCN resistance（大豆胞囊线虫抗性）

- **生物学：** Rhg1位于chr18，约31 kb重复单元的copy number和序列共同影响抗性；Peking型低拷贝rhg1-a通常依赖chr8的Rhg4，高拷贝rhg1-b具有不同遗传模式。
- **来源A：** Cook等2012已验证Rhg1多基因CNV机制；
- **来源B：** 2024公开研究含100个育种系、HG type 2.5.7与7、两次独立实验、Female Index及四级R/MR/MS/S标签，并用12个已知copy-number accession验证qPCR，相关r=0.994；
- **任务1：** 32K–128K Rhg1 locus的copy/haplotype class；
- **任务2：** Rhg1+Rhg4双locus embedding后预测不同HG type的Female Index或ordinal resistance；
- **baseline：** CN-only、SNP-only、Rhg1/Rhg4规则模型、GBLUP、AgroNT/PlantCAD2 tile、Evo2 full、LegumeGenomeFM；
- **关键限制：** 100个育种系只有在公开处理后基因型能与表型可靠连接时才进入sequence benchmark；否则只用于表型关联参考，不能从没有序列的材料“生成”locus。
- **指标：** Spearman、ordinal macro-F1、每HG type分层AUPRC；按实验run bootstrap。

### S04：Soybean flowering and geographic adaptation（大豆开花与区域适应）

两个互补数据集：

1. **915-accession flowering panel：** 673 landrace、64 modern、37 old cultivar、129 wild，来自41个国家；公开genotype、flowering phenotype和局部单倍型；
2. **USDA landscape genomics：** 17,019个中国来源genebank accession，SoySNP50K原始42,080 marker；合并欧洲材料后的数据为20,269 accession、41,084 marker；含六个生长区域、环境PC、passport和maturity group。公开入口为SoyBase、Figshare和Zenodo 10.5281/zenodo.6126368。

任务：

- E1–E4、J、Tof等已知位点周围单倍型的flowering-time回归；
- 多位点set encoder预测maturity group和环境PC；
- 中国不同区域→北美/欧洲的严格geographic transfer；
- 已知适应候选位点ranking，而非简单预测“国家”从而学习population structure。

baseline必须包括population structure PCs、kinship、SoySNP50K GBLUP和环境协变量。FM只有在这些协变量之上仍有增量才算有效。

### S05：Legume domestication causal-variant ranking（豆科驯化因果变异排序）

围绕大豆pod shattering、determinate growth和maturity等有实验支持的locus构建小规模ranking任务。每个已知候选配同一LD block、相同MAF和功能类别的hard negatives；按locus而不是variant随机拆分。指标为MRR、Recall@K和候选排序percentile。该任务用于验证模型能否优先已知因果位点，不适合用少量正例做高容量分类器。

### S06：Peanut pangenome SV-to-trait（花生泛基因组SV到性状）

- **来源：** 8个高质量peanut genome、269个resequenced accession；公开NGDC项目PRJCA029798、PRJCA029800、PRJCA029802、PRJCA030060及PeanutPan结果站；
- **标签：** core/distributed/private gene family、已发布SV、seed length/weight和已验证AhCKX6/AhARF2-2候选；
- **任务：** cold Arachis上的PAV/SV effect、候选排序和seed trait增量预测；
- **公平性：** LegumeGenomeFM预训练完全排除Arachis；PlantCAD2公开语料包含A. hypogaea。暴露差异必须在结果表中单列。

### S07：Legume NLR-cluster integrity（豆科免疫受体簇完整性）

利用发布注释与冻结NLR-Annotator交集构建NLR cluster boundary、intact/pseudogene和core/accessory任务，输入64K–256K。由于工具标签可能把模型训练成“NLR-Annotator模仿器”，该任务只能是探索/方法学补充；若没有独立人工或功能标签，不进入主结论。

## 14. 育种与基因型到表型任务

### G01：SoyNAM跨家系基因组预测

- **来源：** SoyNAM由40个diverse parent与共同亲本杂交，含5,600个RIL（重组自交系）；41个亲本深度测序；公开SoyNAM6K和525,772个高置信imputed SNP；
- **性状：** 只纳入公开、样本ID闭合、环境元数据完整的产量、蛋白、油分、成熟期等性状；每个性状单独出数据receipt；
- **输入：** 每个SNP周围1K/8K embedding，按LD block聚合；长候选区用32K–128K；
- **拆分：** leave-family-out为主，held-environment为补充；禁止随机个体拆分；
- **baseline：** family mean、GBLUP、RR-BLUP、BayesB、XGBoost/LightGBM、MLP、AgroNT/PlantCAD2/NT-v2/Evo2 embedding；
- **模型比较：** `GBLUP + covariates`对`GBLUP + covariates + FM features`，而不是只比较FM与弱MLP；
- **指标：** Pearson、Spearman、R²、RMSE；按family和environment分层bootstrap。

### G02：Region-aware adaptation prediction（区域感知适应预测）

使用17,019 USDA中国来源accession与六区域环境信息，构建maturity/environment回归和跨区域测试。训练中显式加入PC/kinship，测试模型是否通过长单倍型与多位点embedding提供增量。必须报告每个区域性能、校准和最差区域，而不能只给全体平均值。人口结构高度可预测地理来源，因此“区域分类准确率”不能作为主要生物学结论。

## 15. 任务状态与未来结果表

| ID | 任务 | 主指标 | 当前LegumeGenomeFM | 关键baseline | 当前证据状态 |
|---|---|---|---|---|---|
| C01 | masked-base recovery | masked NLL/accuracy | N/A | PlantCAD2/NT/AgroNT/Evo2 | release未建 |
| C02 | splice | AUPRC/MCC | N/A | AgroNT/PlantCAD2/NT | builder未建 |
| C03 | TIS/TTS/boundary | AUPRC/boundary F1 | N/A | PlantCAD2/Evo2 | builder未建 |
| C04 | ACR/CRE | AUPRC | N/A | PlantCAD2/AgroNT | 公共来源已核验 |
| C05 | expression | Spearman/R² | N/A | PlantCAD2/AgroNT | 数据未导入 |
| C06 | constraint/variant | AUROC/AUPRC/MRR | N/A | PlantCaduceus/GPN/Evo2 | 数据未导入 |
| C07 | whole-gene segmentation | macro-F1/boundary F1 | N/A | Helixer/Evo2/tile | 原始GFF可用 |
| C08 | ortholog/core-accessory | Recall@K/AUPRC | N/A | DIAMOND/PlantCAD2 | 数据未导入 |
| L01 | cold giant gene | macro-F1 | N/A | Evo2/Hyena/tile | 原始上限已统计 |
| L02 | distal context | ΔAUPRC/ΔR² | N/A | nested/tile controls | 未运行 |
| L03 | SV/PAV | AUPRC/MRR | N/A | Evo2/PlantCAD2 | 来源已核验 |
| L04 | WGD homeolog | MRR/Recall@K | N/A | MMseqs2/FM | 标签未冻结 |
| L05 | TE-gene boundary | boundary F1 | N/A | EDTA/long models | blocked |
| S01 | nodule cell program | macro-AUPRC | N/A | PlantCAD2/AgroNT | GSE226149已核验 |
| S02 | symbiosis retrieval | MRR/hierarchical F1 | N/A | protein homology/FM | curated set未建 |
| S03 | Rhg1/Rhg4-SCN | Spearman/ordinal F1 | N/A | CN/SNP rules/FM | 表型来源已核验 |
| S04 | flowering/adaptation | Spearman/R² | N/A | GBLUP/PlantCAD2 | 数据入口已核验 |
| S05 | domestication ranking | MRR/Recall@K | N/A | conservation/FM | curated set未建 |
| S06 | peanut SV-to-trait | AUPRC/R² | N/A | GBLUP/Evo2 | 来源已核验 |
| S07 | NLR cluster | boundary F1/AUPRC | N/A | NLR-Annotator/FM | exploratory |
| G01 | SoyNAM | R²/Spearman | N/A | GBLUP/RR-BLUP | 5,600 RIL来源已核验 |
| G02 | regional adaptation | R²/worst-region | N/A | kinship+PC/GBLUP | 17,019 panel已核验 |

本表就是“目前所有任务性能”的完整状态：没有运行的任务必须继续是N/A，后续每次只允许由带hash的result receipt更新。

## 16. 统计设计、泄漏控制与报告规范

### 16.1 随机种子与重复

- medium/large正式任务：至少3个seed；
- 小样本豆科任务：5个seed或nested cross-validation；
- 最高单seed与多seed均值±标准差分别报告；
- 缺失seed不得用已有seed均值补齐。

### 16.2 split单位

按任务选择gene family、orthogroup、chromosome homology component、accession family、region或environment作为group。任何比窗口更高层的生物学实体必须先分组后拆分。随机窗口split只允许用于非正式smoke test。

### 16.3 置信区间与多重比较

- 分类：group-stratified paired bootstrap 10,000次；
- 回归：按family/accession/environment分层bootstrap；
- AUROC可辅以DeLong，但不替代group bootstrap；
- 多任务主终点采用Holm校正；探索任务用Benjamini-Hochberg FDR并明确标为探索；
- 同时报告effect size和95% CI，不以单个P值判胜负。

### 16.4 checkpoint纪律

- 每个checkpoint可运行轻量MLM/健康诊断；
- 中等validation benchmark只运行validation-best、step5000/7000或预注册候选；
- formal test不在每个checkpoint运行；
- sealed Arachis/Vicia只在模型、head、阈值和seed列表冻结后解封。

### 16.5 baseline暴露审计

每个baseline记录：论文、checkpoint ID、commit、权重SHA、license、参数量、原生最大长度、tokenizer、预训练物种、测试物种是否暴露、是否full fine-tune/LoRA/frozen probe、峰值显存和运行失败原因。无法确认暴露时标`UNKNOWN`，不能写成unseen。

## 17. 当前实现缺口与执行路线

### Gate 1：正式数据release

- 审核并冻结Phaseolus cold-development、Arachis/Vicia sealed-test；
- 构建schema-2 manifest、坐标split、SHA与窗口receipt；
- 将状态从当前候选态升级为`TRAINING_DATASET_READY`；
- 在这之前不启动正式训练。

### Gate 2：HierMamba trainer

- 将mixed-length sampler、multi-scale span masking、HierMamba forward、DDP全局masked-token归一化、token scheduler和resume状态接入一个生产trainer；
- 替换当前连接legacy `LegumeGenomeModel`的脚本；
- 做单卡/双卡loss等价、断点续训等价和RC一致性测试。

### Gate 3：H20实测

- 安装锁定Mamba-2和causal-conv；
- 逐长度测forward/backward、micro-batch、峰值显存与吞吐；
- 确认314.67M推导参数与实例化唯一参数量一致；
- 在30B/100B/200B三档中冻结可完成的正式token预算。

### Gate 4：预训练与候选选择

- 单一连续run；
- 依据ID-validation和Phaseolus cold-development选择checkpoint；
- 不访问Arachis/Vicia sealed test；
- 生成模型卡、训练receipt和失败恢复记录。

### Gate 5：分层benchmark

1. 先运行C01–C04和L01，证明模型可用且长上下文有因果收益；
2. 再运行S01、S03、S04和G01四个核心论文任务；
3. S02/S05/S07只有在标签质量门禁通过后升级；
4. 最后一次解封cold test并生成正式结果表。

## 18. 风险与停止条件

- **Glycine过多：** 50/74 source来自Glycine；必须报告采样后实际token份额，并做去掉多组装或每物种单代表消融。
- **Vicia数据量巨大：** cold test占约28.42 Gbp；它是严格测试资产，不得因“训练缺数据”回流预训练。
- **短MLM无法训练长程：** 若multi-scale masking和context intervention均无收益，应缩短模型主张，而不是继续扩大长度。
- **基线暴露不公平：** 不隐藏PlantCAD2见过Arachis/Vicia等事实；报告ours-cold/baseline-seen。
- **标签伪造风险：** NLR/TE工具伪标签不能形成主要生物学结论。
- **育种人口结构泄漏：** 若FM增益在加入kinship/PC后消失，结论应是“未提供独立增量”。
- **硬件不支持：** 若256K backward在H20不稳定或吞吐不可接受，应保留数据与模型证据，降低正式长度或停止，而不是声称已完成256K训练。
- **性能未达标：** 通用任务非劣和至少两个高优先级豆科专属任务显著超过baseline，才支持“豆科基础模型”主张；否则应作为长上下文架构/数据资源论文重新定位。

## 19. 可复核机器证据

- `data_manifests/data_refinement_final.summary.json`：74-source最终QC与容量；
- `data_manifests/data_refinement_final.selected.tsv`：source、材料、BUSCO、污染与权重；
- `data_manifests/data_refinement_final.contexts.tsv`：六长度正式容量；
- `research/proposed_pretraining_split_evidence.json`：本文cold split的source/bp/window和GFF上限统计；状态明确为`PROPOSED_NOT_FROZEN`；
- `research/baseline_model_evidence.tsv`：关键baseline参数、长度、许可、论文与文献值；
- `configs/pretrain_h20_candidate.yaml`：候选架构/优化器；
- `configs/evaluation_matrix.yaml`：现有机器评测合同；
- `src/legumegenomefm/hiermamba.py`：候选模型实现；
- `src/legumegenomefm/training_data.py`：混合长度采样与release验证；
- `src/legumegenomefm/training.py`：当前legacy trainer，正式训练前必须替换。

## 20. 参考文献与数据入口

1. Mendoza-Revilla, J. et al. AgroNT: a foundational large language model for agriculture. *Communications Biology* 7, 835 (2024). https://doi.org/10.1038/s42003-024-06465-2
2. Zhai, J. et al. A DNA language model based on multispecies alignment predicts the effects of noncoding variants in plants. *PNAS* 122, e2421738122 (2025). https://doi.org/10.1073/pnas.2421738122
3. Zhai, J. et al. PlantCAD2: A Long-Context DNA Language Model for Cross-Species Functional Annotation in Angiosperms. bioRxiv (2025). https://doi.org/10.1101/2025.08.27.672609
4. Dalla-Torre, H. et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nature Methods* 22, 287–297 (2025). https://doi.org/10.1038/s41592-024-02523-z
5. Brixi, G. et al. Genome modelling and design across all domains of life with Evo 2. *Nature* (2026). https://doi.org/10.1038/s41586-026-10176-5
6. Schiff, Y. et al. Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling. *ICML* (2024). https://proceedings.mlr.press/v235/schiff24a.html
7. Nguyen, E. et al. HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. *NeurIPS* (2023). https://doi.org/10.48550/arXiv.2306.15794
8. Benegas, G., Batra, S. S. & Song, Y. S. DNA language models are powerful predictors of genome-wide variant effects. *PNAS* 120, e2311219120 (2023). https://doi.org/10.1073/pnas.2311219120
9. Dao, T. & Gu, A. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *ICML* (2024). https://doi.org/10.48550/arXiv.2405.21060
10. Liu, Y. et al. Pan-Genome of Wild and Cultivated Soybeans. *Cell* 182, 162–176.e13 (2020). https://doi.org/10.1016/j.cell.2020.05.023
11. Cervantes-Pérez, S. A. et al. Single-cell transcriptome atlases of soybean root and mature nodule reveal new regulatory programs that control the nodulation process. *Plant Communications* 5, 100984 (2024). https://doi.org/10.1016/j.xplc.2024.100984 ；GEO GSE226149；BioProject PRJNA938968
12. Roy, S. et al. Celebrating 20 Years of Genetic Discoveries in Legume Nodulation and Symbiotic Nitrogen Fixation. *The Plant Cell* 32, 15–41 (2020). https://doi.org/10.1105/tpc.19.00279
13. Cook, D. E. et al. Copy number variation of multiple genes at Rhg1 mediates nematode resistance in soybean. *Science* 338, 1206–1209 (2012). https://doi.org/10.1126/science.1228746
14. Poudel, D. et al. Copy number variations at the Rhg1 locus and their relationship with resistance to soybean cyst nematode. *Frontiers in Plant Science* 15, 1504932 (2024). https://doi.org/10.3389/fpls.2024.1504932
15. Mohamedikbal, S. et al. Local haplotyping reveals insights into the genetic control of flowering time variation in wild and domesticated soybean. *The Plant Genome* 17, e20528 (2024). https://doi.org/10.1002/tpg2.20528
16. Haupt, M. & Schmid, K. J. Using landscape genomics to infer genomic regions involved in environmental adaptation of soybean genebank accessions. *BMC Plant Biology* 25, 1175 (2025). https://doi.org/10.1186/s12870-025-07202-5
17. Song, Q. et al. Genetic Characterization of the Soybean Nested Association Mapping Population. *The Plant Genome* 10 (2017). https://doi.org/10.3835/plantgenome2016.10.0109
18. Zhao, K. et al. Pangenome analysis reveals structural variation associated with seed size and weight traits in peanut. *Nature Genetics* 57, 1250–1261 (2025). https://doi.org/10.1038/s41588-025-02170-w
19. Brown, A. V. et al. A new decade and new data at SoyBase, the USDA-ARS soybean genetics and genomics database. *Nucleic Acids Research* 49, D1496–D1501 (2021). https://doi.org/10.1093/nar/gkaa1107
20. Lu, Z. et al. The prevalence, evolution and chromatin signatures of plant regulatory elements. *Nature Plants* 5, 1250–1259 (2019). GEO GSE128434; BioProject PRJNA527732.

## 21. 最终声明

本研究设计已经给出可执行的数据、架构、任务、baseline、拆分、统计和失败合同；但它仍处于“设计与证据冻结前”而非“结果完成”阶段。正式论文中只有运行后由机器receipt支持的值可以替换`N/A`。项目的目标是：在共同短上下文任务上对关键植物/通用模型保持非劣，在预注册的豆科共生、抗病、巨型基因、复杂结构和区域适应任务上取得可重复优势；如果真实实验不支持这一目标，应明确报告失败并停止扩大主张。
