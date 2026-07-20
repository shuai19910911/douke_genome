# LegumeGenomeFM：研究与预训练总计划

> 文档状态：**候选合同（尚不可启动正式训练）**，最后更新：2026-07-20。固定上下文集合为1K、8K、32K、64K、128K和256K；数据最终source数、模型精确参数量、H20实测配置和总token预算仍在冻结门禁内。任何`superseded`（已废弃）短上下文配置均不得启动。

## 1. 项目定位与科学假设

LegumeGenomeFM是面向豆科植物、以单碱基DNA为输入的多物种基因组基础模型。项目不以“模型更大”作为创新，而以高质量豆科语料、超长上下文、反向互补一致性和严格跨物种评测共同回答以下问题：

1. 豆科专属多物种预训练能否优于通用DNA模型、单物种模型和同预算随机初始化模型？
2. 提升能否迁移到未参与预训练的物种、属、低同源基因家族和材料，而非来自组装版本或近重复泄漏？
3. 1K至256K的连续混合上下文能否同时改善碱基、区域、基因、变异、调控和育种层任务？
4. 模型是否学习到可解释的剪接、编码、启动子、重复序列、保守性和长距离调控规律？
5. 性能增益是否在多个seed（随机种子）、严格分组划分和外部基线下稳定，并足以抵消额外计算成本？

预注册主假设是：在相同标注数据和调参预算下，豆科多物种预训练模型在cold-genus（整属留出）及low-homology（低同源）测试上的主指标优于最强可复现外部基线。若仅随机划分有效、严格划分无效，则核心科学假设不成立。

## 2. 文献检索与证据边界

### 2.1 系统检索方法

检索截止日为2026-07-18。机器记录见：

- 检索式：`metadata/literature_search_queries.json`；
- 候选与检索摘要：`data_manifests/literature_candidates.tsv`、`data_manifests/literature_search.summary.json`；
- 核心文献元数据核验：`data_manifests/core_literature_verified.tsv`；
- Methods级证据矩阵：`data_manifests/literature_evidence_matrix.tsv`。

共执行36组Crossref检索，得到2,958个去重候选，其中154个Nature Portfolio候选；30个核心/背景记录完成DOI、期刊、题目和链接核验，失败数为0。纳入标准为原始研究、方法或正式会议论文，且直接涉及DNA基础模型、长上下文、跨物种、变异效应、反向互补或泄漏控制；新闻、Research Highlight和Research Briefing不作为模型设计主证据。

### 2.2 影响本项目的核心证据

| 工作 | 关键设计证据 | 本项目采用内容 | 不直接照搬的原因 |
|---|---|---|---|
| Enformer，Nature Methods，DOI `10.1038/s41592-021-01252-x` | 196,608 bp输入、分层降采样、同源连通分量划分 | 分层局部—全局表示、同源组不跨split | 人/鼠监督轨迹，不是豆科自监督模型 |
| Nucleotide Transformer，Nature Methods，DOI `10.1038/s41592-024-02523-z` | 多物种多样性、严格模型规模比较 | 多物种平衡和外部基线 | 6-mer会降低单碱基对齐；语料不含植物 |
| AgroNT，Communications Biology，DOI `10.1038/s42003-024-06465-2` | 48种可食植物、植物专属预训练 | 主要植物外部基线 | 约6 kb上下文且无精确RC等变 |
| GPN，PNAS，DOI `10.1073/pnas.2311219120` | 整物种留出、单碱基MLM、植物变异预测 | zero-shot变异基线和整物种留出 | 512 bp上下文、Brassicales范围较窄 |
| PlantCaduceus，PNAS，DOI `10.1073/pnas.2421738122` | 植物单碱基、双向状态空间、RC处理 | RC机制和跨物种基线 | 512 bp预训练上下文，不能证明256K可行 |
| Borzoi，Nature Genetics，DOI `10.1038/s41588-024-02053-6` | 524,288 bp、多尺度编码/解码 | 超长序列分层设计 | 人/鼠监督模型，训练成本和输出不同 |
| AlphaGenome，Nature，DOI `10.1038/s41586-025-10014-0` | 1 Mb、U-Net多尺度、细粒度输出 | 单碱基局部skip与压缩全局路径 | 闭源级算力、监督多组学，不适合作为预训练实现模板 |
| Evo 2，Nature，DOI `10.1038/s41586-026-10176-5` | 单碱基、最长1 Mb、混合长卷积/注意力 | 长上下文能力上界与效率对照 | 万亿token和千卡级规模不可复制 |
| DNA FM benchmark，Nature Communications，DOI `10.1038/s41467-025-65823-8` | 简单CNN可胜过冻结FM，随机split偏乐观 | 强任务基线、严格分组split | 其部分数据缺乏染色体信息 |
| Caduceus，ICML 2024 | 双向Mamba与精确RC参数共享 | RC数学合同和外部基线 | 官方实现基于原始Mamba；Mamba-2替换必须重新验证 |
| HyenaDNA，NeurIPS 2023 | 单碱基、最长1M的长卷积 | 同参数/同token架构比较器 | 人类单参考语料且无精确RC |

官方代码审计和commit固定于`research/ultralong_architecture_evidence.json`。当前唯一架构方向是“单碱基局部编码/解码＋128 bp潜变量＋双向Mamba-2全局核心＋整模型RC对称”。纯全注意力、已撤销的16K卷积模型，以及未经H20编译/实测的Mamba-2声明均不具备冻结资格。

## 3. 数据现状：已核实与未核实严格分开

### 3.1 已核实原始数据

- `data/raw/`只读盘点：3,841项，约195.529 GiB；原始文件不得删除、原地修复或改名。
- 普通FASTA：552个候选，550个PASS，2个截断gzip永久fail-closed。
- ZIP容器：201个，197个PASS，4个非法容器永久排除。
- ZIP genome：34个PASS，其中30个与普通来源exact duplicate（精确重复），4个为新增T2T候选。
- 统一genome catalog：584个PASS来源、483个exact-unique序列。
- 统一annotation catalog：574个来源，其中301个主gene model，共15,973,108个gene feature。
- 已建立466个2-bit sequence store、canonical MinHash签名和orientation-invariant identity（不受序列名称、contig顺序和每条contig方向影响）。

### 3.2 当前精简审计状态

机器输入为`data_manifests/data_refinement_candidates.tsv`及其summary。初始结构、注释和序列门禁得到167个需深审候选：

- FASTA–GFF坐标闭合：165 PASS、2 FAIL；失败项存在未知seqid和越界feature，必须排除。
- 来源与许可证证据：94 PASS；67个候选进入`LICENSE_REVIEW_REQUIRED`（含41个仅有NCBI assembly report、但submitter权利未解析的候选），3个数据库README元数据不完整，3个来源缺少可审计resolver。未取得明确授权的候选不能进入正式核心集。
- BUSCO和污染审计任务集：92个许可证明确且坐标闭合的候选；genome BUSCO、annotation BUSCO、Tiara与UniVec正在通过SLURM执行。
- 中黄13的显式material alias已统一：`zh13`、`whfsgmzh1310`、`zh13iga1005`、`gmaxzh13`和`gmaxzh13v20`视为同一材料，最终只允许一个代表。
- 最终source数、物种/属数、可训练碱基数和各上下文容量尚未产生，不得引用第一版约140个候选作为最终发布数字。

QV（共识质量值）仅在来源提供可审计值或存在原始读段时可报告；当前缺失项标为`UNAVAILABLE_NO_RAW_READS_OR_SOURCE_QV`，绝不默认通过或伪造数值。`lowercase_count`只能表示FASTA软掩码字符，不能替代RepeatMasker意义上的完整重复序列比例。

## 4. 正式纳入、排除与代表选择

### 4.1 Assembly门禁

纳入至少满足以下一类证据：

1. NCBI `Complete Genome`；
2. NCBI `Chromosome`；
3. 来源报告明确T2T且来源可审计；
4. 缺少官方level时，数据库README/论文明确pseudomolecule或染色体挂载，同时N50≥10 Mb、至少80%碱基位于≥10 Mb序列。

第4类只记为`structural_proxy`（结构代理），不得写成官方Chromosome。FASTA名称中含“T2T”但无出处，仅是待核验标签，不自动通过。

来源还必须具有可审计的公开使用许可。当前自动allowlist只接受规范化后的`public + open`；包含`usage agreement`、`restricted`或缺失许可证的记录先排除并进入人工审核，不能因文件可下载就推定可用于模型预训练。NCBI政策明确说明submitter并未把权利转移给NCBI，因此NCBI assembly可用于证明组装级别，但单凭NCBI可下载不能自动通过训练许可；政策证据锁定在`metadata/ncbi_molecular_database_policy.json`。

### 4.2 Annotation门禁

- genome与annotation必须来自同一assembly/version或可信数据库bundle；
- 主gene model且gene数>0；
- malformed line、非法坐标、非法strand/phase、重复gene/transcript ID均为0；
- GFF所有seqid必须映射到FASTA，所有feature坐标闭合；
- annotation BUSCO complete≥90%。

### 4.3 序列与污染门禁

- genome BUSCO complete≥90%；
- N比例≤10%；
- primary nuclear bases占assembly≥90%；
- 只保留assembled nuclear chromosome或有充分证据的长pseudomolecule；
- organelle、plasmid、alternate locus、decoy和小型unplaced scaffold排除；
- Tiara在primary nuclear区域的有效覆盖≥95%，prokaryotic分类比例≤0.5%；
- UniVec高置信命中比例≤0.01%；
- 可定位污染区间进入mask，不跨污染区采样。

污染审计的正式软件组合固定为`quay.io/biocontainers/tiara:1.0.3`构建的Singularity SIF、`soygenome_qc`中的BLAST 2.17.0+和本地格式化的NCBI UniVec Core；不依赖单独的`soygenome_contam` conda环境。Tiara SIF、UniVec全部数据库文件和`blastn`二进制由`contamination_references.receipt.json`及READY绑定，finalizer聚合前执行完整SHA-256复核。新shard直接记录reference receipt SHA；在receipt引入前产生的55个PASS shard不改写历史JSON，而由独立legacy-binding receipt按shard SHA、精确命令路径和“参考文件先于shard生成”的mtime证据绑定。

### 4.4 去重与同材料唯一版本

按以下顺序选择：

1. exact/orientation等价组只留一个；
2. 同物种同材料只留一个；
3. 质量排序为已核验T2T/Complete Genome > official Chromosome > 有出处的structural proxy；
4. 同级比较BUSCO、QV（若可用）、primary nuclear比例、N/gap、长窗口容量、N50、contig数和superseded状态；
5. 不同材料即使MinHash相似也不自动删除，只按near-duplicate group降低总采样权重；
6. 不同taxon却高度相似时先作为错标/镜像/污染红灯调查。

多倍体的不同染色体和亚基因组属于同一assembly，不按材料重复删除。

## 5. 可重现的数据处理链

正式顺序为：raw只读inventory → 容器/FASTA完整性 → annotation结构审计 → unified catalog → exact/orientation identity → 全局MinHash → assembly evidence → FASTA–GFF闭合 → BUSCO → Tiara/UniVec → primary nuclear record policy → material representative → 每长度容量 → split → release。

每个阶段必须满足：

- 输入相对路径、SHA-256、实现closure hash和配置hash可追溯；
- worker只写独立JSON shard，聚合器严格检查missing/extra/duplicate；
- 失败不修改raw；重跑只补缺失或无效shard；
- 正式TSV/JSON使用临时文件、`fsync`和原子rename；
- READY最后写入，且只在所有hash和计数闭合后产生；
- 旧READY和旧440-source manifest已删除，不得作为训练入口。

## 6. 正式语料与窗口合同

### 6.1 Tokenizer

候选正式tokenizer为单碱基7-token词表：`PAD`、`MASK`、`A`、`C`、`G`、`T`、`N`。训练主语料只从ACGT连续区间采样；外部序列中的其他IUPAC歧义码统一映射到N并从MLM标签中忽略。A↔T、C↔G，PAD/MASK/N自互补。词表、映射和RC置换在配置与checkpoint receipt中绑定。

### 6.2 上下文集合

正式长度集合固定为：

`[1,024, 8,192, 32,768, 65,536, 131,072, 262,144] bp`

每个长度独立建立eligible-source和non-overlap capacity；不得要求所有source都有256K窗口。窗口不跨染色体、不跨contig、不跨N/歧义区和污染mask。最终训练可重复采样，但统计有效容量时使用非重叠窗口，避免重叠滑窗虚增数据量。

最终schema-2 manifest必须同时记录FASTA record-local坐标`record_start_0based`和packed store全局坐标`store_start`。污染mask先在record-local坐标系扣除，再转换到store坐标；sampler只能从最终`trainable_intervals`解码，不得退回store原始callable interval。builder与preflight均复核contig边界、callable包含关系、区间不重叠及六长度capacity。

### 6.3 混合长度调度

只训练一个共享参数模型、一个optimizer、一个scheduler和一条checkpoint lineage。候选token占比为10%/15%/20%/20%/20%/15%，对应1K/8K/32K/64K/128K/256K；这是H20 profiling前的预注册候选，只有在各长度容量和吞吐门禁通过后才改为`frozen`。

长度按token而非样本分配。每个optimizer step由rank 0使用可恢复counter-based RNG选择长度并广播，所有DDP rank及该step全部microstep使用同一长度。候选每GPU microbatch为256/32/8/4/2/1个序列，使每个长度均为262,144输入token/GPU/microstep。若H20实测不通过，只能调整microbatch与梯度累积，不能删除256K、改变模型结构或悄悄改变token占比。

source采样先按species平衡，再按该长度的clean capacity分配，并以`1 / selected-near-group-size`降低近重复组权重。任何cold-genus/test source的训练权重严格为0。

## 7. 唯一模型方向与冻结门禁

详细结构见`MODEL_ARCHITECTURE.md`。候选名称为**LegumeGenomeFM-HierMamba**：

- 单碱基局部encoder；
- 总stride 128的分层压缩；
- 最长2,048个全局latent token；
- 双向Mamba-2全局核心；
- U-Net式decoder和base-resolution skip；
- 同一完整backbone对正向与RC输入运行，RC对齐logits取均值，保证输出对称；
- base-resolution span MLM。

在以下证据齐全前，名称后必须标记`candidate`，`contract_status`不得为`frozen`：

1. 实现完成并输出逐模块、去重后的精确参数量；
2. 与同参数Hyena global core完成同硬件spike，唯一kernel按验证集前置规则选择；
3. H20真实型号、卡数、显存、compute capability、CUDA、驱动、互联和空闲显存记录完成；
4. 1K至256K全部完成BF16 forward/backward/optimizer实测；
5. 2卡与3卡DDP长度同步、global masked-token loss和checkpoint恢复闭合；
6. 256K无OOM/NaN/Inf且峰值显存保留≥10%安全余量；
7. 训练配置、实现closure、数据release和环境锁定hash共同写入receipt。

## 8. 预训练目标与优化语义

唯一主目标为span MLM：候选mask比例15%，span长度服从均值3 bp的截断几何分布；80%替换MASK、10%随机ACGT、10%保留原token。只在选中且标签为ACGT的位置计算交叉熵。

全局loss不是“各rank loss平均”，而是：

`sum(all masked-token negative log-likelihood) / sum(all valid masked-token count)`。

分子和分母跨rank all-reduce，防止不同物种、不同mask数和最后batch产生权重偏差。只有有限梯度且optimizer真正更新后，`tokens_seen`、每长度token计数和step才增加。overflow/OOM重试不得计入预算。

优化器候选为AdamW，`betas=(0.9, 0.95)`、weight decay 0.1、global gradient norm 1.0、BF16。peak LR、warmup token数、global batch tokens、总token预算和最小LR只能在精确参数量、最终语料容量与H20 tokens/s实测后冻结；当前不沿用旧100B或旧三阶段学习率。

## 9. 训练预算如何冻结

总预算必须同时满足：

1. 数据：报告总clean capacity及按物种/长度的有效重复次数；
2. 优化：每个长度至少获得足以稳定验证loss的更新数；
3. 算力：按H20实测端到端tokens/s（含dataloader、RC双路、反向和checkpoint）计算walltime；
4. 成本：保留10%失败/恢复余量；
5. 统计：预留至少三个下游seed和消融预算，不能把全部GPU预算耗在单次预训练。

冻结receipt应包含`total_tokens`、六长度token整数配额、`global_batch_tokens`、每种world size的梯度累积、预计optimizer steps、实测tokens/s、预计小时和费用区间。预算必须整除global batch或明确原子尾batch规则。

## 10. Split与泄漏控制

最终过滤后重新构建split，旧六个cold genus不自动继承。至少维护四层报告：

- in-distribution：材料、染色体和同源组件分组；
- leave-one-species-out；
- cold-genus：完整属不参与预训练与下游训练；
- low-homology：train/test蛋白身份<30%或使用预注册的序列同源组件。

以下组件不得跨正式train/validation/test：orientation identity、exact duplicate、near-duplicate、material alias、染色体同源块和gene/protein homology。正式test在模型、超参数、checkpoint选择规则和分析代码冻结后一次性运行；训练期只用diagnostic和validation，禁止逐checkpoint看formal test。

## 11. 下游任务预注册

机器合同为`configs/evaluation_matrix.yaml`，当前状态是`pending_refined_data_and_h20_freeze`。核心任务覆盖至少六个层级：

1. 碱基层：splice donor/acceptor、translation start/stop；
2. 区域层：promoter proxy、exon/intron与gene-body segmentation；
3. 基因层：coding frame、ortholog retrieval、低同源GO功能；
4. 变异层：zero-shot constraint、实验支持的deleterious variant；
5. 调控层：ATAC/DNase、TF binding、跨组织gene expression；
6. 育种层：多环境genomic prediction与allele-specific expression；
7. 长上下文/比较基因组：最长256K promoter–region link和synteny break。

`pending_official_*`任务在官方来源、许可证、样本量和split无法冻结时不得进入主结果。可从当前严格GFF直接构建的任务先完成；缺可信数据源时停止盲目搜索，列为扩展或探索性任务。

## 12. Baseline、消融与公平比较

### 12.1 外部基线

优先比较GPN、AgroNT、Nucleotide Transformer v2 50M、PlantCaduceus、Caduceus和HyenaDNA；PlantGFM仅在公开权重、许可证、输入语义和推理代码核验后进入主表。每个基线使用其正式tokenizer和推荐输入，不用不公平截断或未训练随机头代替。

### 12.2 任务专用基线

每项任务至少包含one-hot CNN、k-mer gradient boosting和成熟生物信息学工具。冻结embedding与full fine-tune分别报告；训练预算、标注数据、早停次数和超参数搜索次数相同。

### 12.3 预注册消融

- random initialization，同架构；
- 单物种预训练，同token预算；
- 无RC对称；
- 仅8K上下文，同token预算；
- 不做species/near-group平衡；
- 去除Mamba-2 global core；
- 去除base-resolution skip；
- latent stride 64/128/256（只在架构冻结前的小规模validation spike，不进入多个正式模型）；
- Mamba-2与Hyena global core同参数/同token比较。

正式主模型只有一个；消融是解释实验，不是并列正式方案。

## 13. 统计分析

- 主下游任务默认至少3个独立seed；只报告单seed最高值属于探索性结果，不能替代均值±标准差/置信区间。
- 分类：auPRC为类别不平衡主指标，同时报告auROC、MCC和校准；回归：Pearson、Spearman、R²及normalized RMSE；分割：macro-F1、boundary-F1和IoU。
- 模型差异使用同一fold/seed的配对bootstrap或置换检验；报告效应量和95% CI，不只报P值。
- 多任务主检验控制FDR；预注册主终点与探索性分析分开。
- 任何样本、物种或任务删除均记录原因；不得根据test表现后验改变主指标。
- 报告总体、逐物种、逐属、同源区间、重复区、基因长度和等位频率分层结果。

## 14. 可解释性与生物学验证

候选方法包括in-silico mutagenesis、integrated gradients、motif enrichment、embedding邻域、长距离遮挡和RC一致性。解释结果必须：

- 使用独立test或冻结案例；
- 与已知剪接/启动子/TF motif及保守性比较；
- 对重复序列和GC混杂做匹配对照；
- 不把attention/SSM响应直接称为因果机制；
- 重要候选若无外部实验，只表述为计算假设。

## 15. 图件与source data预注册

主图候选：

1. 数据来源、审计、去重和最终物种树；
2. HierMamba架构与六长度连续训练；
3. 预训练稳定性、H20吞吐/显存与scaling；
4. 碱基/区域/基因核心任务；
5. cold-species/cold-genus/low-homology泛化；
6. 变异、调控和育种任务；
7. 机制解释和跨物种案例。

补充图不少于12张独立信息图，覆盖assembly/annotation质量、BUSCO、污染mask、material alias、near duplicate、各长度容量、采样分布、seed、校准、失败类别、逐物种结果、显存/吞吐、消融和数据泄漏审计。所有图由`figures/source_data/`中的冻结TSV/CSV和`scripts/figures/`重建，输出PNG+PDF；正式数据出现前不制作虚假训练曲线。

## 16. H20、AutoDL与环境迁移

目标硬件由用户提供为2–3张NVIDIA H20、每张96 GB，但尚未实机核验。首次登录必须保存匿名化probe receipt：GPU型号/数量/显存、compute capability、驱动、CUDA、PyTorch、NCCL、GPU互联和每卡空闲显存。不得公开主机、GPU UUID或凭据。

Mamba-2生产路径依赖固定commit/版本的`mamba-ssm`、`causal-conv1d`和Triton；先在隔离环境编译，不能破坏共享环境。迁移包必须包含：

- Git跟踪代码；
- 最终data release和被选store；
- 环境lock及离线wheel/archive；
- 模型/训练/config hash；
- bootstrap、deep verifier和launcher；
- 不含raw、旧store、日志、checkpoint或安全文件。

release采用manifest＋SHA-256＋最后READY。迁移后先deep verify，再离线安装/import，再执行1K→256K profiling；任何一步失败均不得启动正式run。

## 17. 成功、失败与停止标准

### 成功

- 数据release所有门禁PASS且每个来源可解释；
- 256K真实H20训练步、2卡/3卡DDP、checkpoint/resume通过；
- 主任务严格split下稳定优于最强公平基线；
- 至少一个长上下文任务显示相对短上下文的预注册增益；
- 效应跨seed且具有可信CI，不依赖单物种或单任务。

### 失败/停止

- 最终高质量语料缺乏足够物种/属覆盖，无法构建cold-genus；
- 256K在H20上无法保留安全显存或吞吐使预算不可执行；
- Mamba-2/Hyena生产kernel均无法稳定编译或出现不可接受数值偏差；
- 严格split下主模型不优于简单CNN/外部基线；
- 关键变异、调控或育种任务缺可信官方数据；
- test泄漏、split污染或数据来源许可无法修复。

遇到真实blocker时冻结阶段结果、输出缺失清单并停止盲目计算；不得通过放宽test、删除失败seed或降低质量门槛伪造成功。

## 18. 执行顺序与里程碑

1. 完成许可证明确开放且坐标闭合的92个候选的BUSCO和污染审计；
2. 应用material alias、质量排序和primary nuclear policy，原子生成最终候选集；
3. 重建六长度capacity、cold-genus、low-homology split和data release；
4. 只在schema-2 release READY、所有QC作业结束并完成dry-run后，用受限GC删除未引用2-bit store；GC只能作用于`data/processed/sequence_store/<16位candidate_id>`，先写plan、后写receipt，raw永远不在作用域；
5. 实现HierMamba同一正式结构和精确参数统计；
6. 构建Mamba-2/Hyena等预算H20 spike，冻结唯一global kernel；
7. 冻结六长度token配额、global batch、LR和总预算；
8. 完成单卡、2卡、3卡、checkpoint/resume和数值一致性；
9. 构建核心下游数据和sealed test；
10. 生成AutoDL自包含release并deep verify；
11. 启动唯一连续预训练run；
12. 按预注册checkpoint规则评测并完成论文级统计、图件和发布。

## 19. 当前可运行的关键命令

以下命令均从项目根目录执行；正式训练命令在合同冻结前故意不存在。

```sh
# 查看本人的SLURM任务；当前CPU任务限定q02–q05
squeue -u "$USER"

# 定点测试
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_data_refinement.py tests/test_slurm_scripts.py

# 所有BUSCO/污染shard完成后聚合（输入必须完整，否则fail-closed）
PYTHONPATH="$PWD/src" python scripts/merge_busco_mode_shards.py \
  --tasks data_manifests/data_refinement_busco_tasks.tsv \
  --protein-dir workspace/data_refinement_busco_protein_shards \
  --genome-dir workspace/data_refinement_busco_genome_shards \
  --combined-dir workspace/data_refinement_busco_shards \
  --lineage-ready data_manifests/busco_lineage_eudicots_odb10.READY

# 全测试；正式发布前必须通过
PYTHONPATH="$PWD/src" python -m pytest
```

任何配置只有`contract_status: frozen`且对应receipt闭合时才可进入launcher。