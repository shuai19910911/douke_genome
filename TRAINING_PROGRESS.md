# LegumeGenomeFM：实时执行进度

> 最后更新：2026-07-21 23:04 CST（UTC+08:00）。当前阶段：**精简数据深审收尾＋超长架构候选实现**。正式训练未开始；旧440-source release和旧89M/16K训练合同均已撤销。

## 1. 状态总览

| 工作流 | 状态 | 当前证据/说明 |
|---|---|---|
| raw只读inventory | 已完成 | 3,841项，约195.529 GiB；`data_manifests/raw_inventory.tsv` |
| FASTA/ZIP完整性 | 已完成 | 普通FASTA 550 PASS/2 FAIL；ZIP 197 PASS/4 invalid |
| unified genome/annotation | 已完成 | genome 584 PASS；annotation 574来源、301主模型 |
| exact/orientation/MinHash | 已完成 | 466个store和全局相似度证据 |
| assembly/annotation初筛 | 已完成 | 167个深审候选；`data_refinement_candidates*` |
| FASTA–GFF闭合 | 已完成 | 165 PASS、2 FAIL |
| source provenance/license | 已完成但有排除 | 94 PASS；67需许可证审核、3 incomplete、3 no resolver |
| BUSCO | 接近完成 | protein 92/92 PASS；genome 86/92 PASS；当前85/92已合并双模式shard |
| Tiara/UniVec污染审计 | 已完成 | 92/92 PASS；55个legacy-hash-bound、37个direct-receipt-bound |
| 最终代表与六长度capacity | 等待自动聚合 | 只剩6个genome BUSCO缺口；中黄13 alias已统一 |
| 精简data release | 未开始 | 旧440-source READY/manifest已删除 |
| Nature/架构证据 | 已完成 | 36检索、2,958候选、30个核心记录验证；官方代码commit已固定 |
| HierMamba候选配置/实现 | 已完成候选代码 | 256K、stride 128、Mamba-2双向core、RC conjoin |
| 生产Mamba-2/H20 profile | 阻塞 | 当前环境无`mamba_ssm`/`causal_conv1d`；目标H20尚未probe |
| 正式参数量与训练预算 | 未冻结 | preflight要求整数参数量和H20 PASS receipt |
| 正式预训练 | 未开始 | 无正式PID、日志或checkpoint |

## 2. 数据审计实况

### 已核实

- 普通FASTA：552个候选，550 PASS，2个截断gzip永久排除。
- ZIP：201个，197 PASS，4个非法容器永久排除。
- ZIP genome：34 PASS，其中30个为普通来源exact duplicate，4个为新增T2T候选。
- unified genome：584个PASS来源、483个exact-unique序列。
- unified annotation：574个来源、301个主gene model、15,973,108个gene feature。
- sequence store：466/466 READY；原始和processed store当前保留，因为深审任务仍需使用。
- annotation closure：167个候选中165 PASS、2 FAIL；失败候选存在未知seqid和越界feature。
- source provenance/license：167个候选中94 PASS；67个候选在取得明确授权前进入`LICENSE_REVIEW_REQUIRED`，其中41个只有NCBI assembly report。NCBI政策说明submitter权利未转移给NCBI，因此assembly report只证明组装级别，不自动授予训练许可；另有3 incomplete、3 no resolver。

### 当前深审任务集

`data_manifests/data_refinement_busco_tasks.tsv`含92个候选，即同时通过annotation closure、source provenance和明确`public + open`许可证门禁的候选，覆盖27个物种、13个属。最终source数尚未产生。

中黄13材料已通过显式alias统一：`zh13`、`whfsgmzh1310`、`zh13iga1005`、`gmaxzh13`、`gmaxzh13v20`最终只能保留一个代表；不采用模糊substring规则，因此不会误并`Zhongmu_No_1`等其他材料。

## 3. SLURM状态

用户当前资源硬规则：只允许向`q02,q03,q04,q05`提交；`cu`和`fat`均禁止且不得作为回退。所有项目submitter、sbatch默认值和Python repair/controller均已锁死到该允许列表，不能通过`PARTITION(S)`环境变量绕过。

当前持久链：

| Job | 作用 | CPU/内存 | 分区 | 状态（2026-07-21 23:04 CST） |
|---:|---|---|---|---|
| 8605100 | 首轮Tiara/UniVec补算 | 4 CPU / 32G | q02 | COMPLETED |
| 8605101 | 首轮protein BUSCO补算 | 4 CPU / 8G | q02 | COMPLETED |
| 8605102 | 首轮genome BUSCO tiny补算 | 4 CPU / 48G | q02 | COMPLETED |
| 8605103 | 持久split-QC控制器 | 1 CPU / 2G | q02 | RUNNING；自动补最后6项 |
| 8606891 | 当前medium genome BUSCO批次 | 4 CPU / 128G | q02–q05 | 2 RUNNING、1 PENDING、1已完成 |
| 8605119 | BUSCO/污染/最终代表/capacity聚合 | 4 CPU / 32G | q02–q05 | PENDING (afterok 8605103) |

controller日志已推进至iteration 7且stderr为空；污染和protein BUSCO已闭合，当前medium批次结束后将继续补剩余large候选。`q02–q05`均有idle/mix节点，但绝不回退`cu/fat`。Hermes cron `c9692ecd758e`每10分钟检查资源、项目链和分区门禁。

## 4. 本轮代码与合同

### 数据精简

- 新增assembly/annotation/来源/BUSCO/污染/最终选择模块和回归测试；
- worker独立写JSON shard，聚合器严格检查missing/extra/duplicate；
- BUSCO在节点scratch目录运行，不再向项目根散落`busco_*.log`；
- `eudicots_odb10`（2024-01-08，2,326 markers）已冻结为4,662文件、740,258,216字节的receipt/READY；path-size inventory与完整tree SHA-256均PASS。每个worker shard必须绑定当前receipt，聚合器逐mode核对BUSCO 5.8.3和lineage元数据；旧2个无receipt shard已删除重算。
- 污染生产组合已冻结：Tiara 1.0.3 SIF（587,227,136字节）、UniVec Core（3,155条/688,131 bp，BLASTDB v5）和BLAST 2.17.0+均由receipt/READY绑定并完成full hash验证。55个历史PASS shard逐个核对命令路径、生成时间和shard SHA后写入独立legacy-binding receipt，未篡改原始结果；新worker直接写reference receipt SHA。
- 来源、annotation、BUSCO、污染和最终selection均fail-closed；
- material alias用于最终选择，并保留原始material key用于审计。

### 超长架构

候选机器合同：`configs/pretrain_h20_candidate.yaml`。

- 上下文：1K/8K/32K/64K/128K/256K；
- 每GPU每microstep固定262,144输入token；
- 单一optimizer/scheduler/checkpoint lineage；
- 128-bp层次化latent，256K对应2,048个global token；
- 24层双向Mamba-2 global core；
- U-Net base-resolution decoder；
- 完整模型正向/RC双路logit对齐均值；
- loss接口返回`loss_sum`和`masked_token_count`，供DDP全局归一化。

生产Mamba-2缺失时模型在大参数分配前立即拒绝，不以Identity或旧卷积替代。preflight只有在`contract_status: frozen`、参数量整数、H20六长度profile、2/3卡DDP、显存余量、kernel对比和新data release均PASS时才放行。

## 5. 文档与清理

- `TRAINING_PLAN.md`已从头重写，覆盖科学问题、Nature证据、数据、训练、评测、统计、图件、风险、迁移与停止标准。
- `MODEL_ARCHITECTURE.md`已重写为唯一HierMamba候选，旧89M数字不再作为正式合同。
- 旧`README.md`已删除，项目最终只保留任务要求的三个Markdown文档。
- 旧440-source `TRAINING_DATASET_READY`、dataset、release和summary已删除。
- 旧121-GB AutoDL release和旧环境验收payload已不存在。
- 已删除21个根目录BUSCO日志、旧pilot目录、失败/取消数组日志和Python cache，共81个无用路径；两轮许可证门禁收紧后又删除46个已排除候选shard（约5.78 MB）。当前仅保留92项任务集实际引用的shard。
- 466个sequence store暂不垃圾回收，因为92项深审仍依赖；最终引用闭合后才删除未引用store。

## 6. 环境、Git与测试

- Python：3.10.20；
- PyTorch：已安装（生产训练版本尚未因H20重新锁定）；
- Triton：已安装；
- `mamba_ssm`：未安装；
- `causal_conv1d`：未安装；
- A100：本轮未使用；
- H20：尚未使用/尚未probe。
- 独立QC环境`soygenome_qc`：Python 3.13.14、BUSCO 5.8.3、gffread 0.12.7、seqkit 2.13.0；BUSCO import、`run_BUSCO.main`、`--version`和`--help`均按正式worker的`QC_ENV/bin`优先PATH语义通过。直接执行绝对入口但不设置环境PATH会错误解析到外部Python，因此不作为有效调用方式。
- 污染正式路径为Tiara Singularity容器＋`soygenome_qc`中的BLAST，不需要独立`soygenome_contam`。后者在CentOS 7/glibc 2.17上因Tiara旧Numba与BLAST新zlib约束不可同时求解，失败后未生成目录、无残留，也未影响生产任务。

Git：`main`；每个里程碑均核对本地HEAD与`origin/main`一致，除正在编辑的进度同步外工作区保持干净。

本轮真实定点测试：

- 数据精简单文件：20 passed；
- HierMamba＋training preflight：13 passed；
- 最近一次preflight单文件：10 passed；
- 完整`pytest`：148 passed（37.73 s）；同时BUSCO 708 MiB full-tree verifier、Tiara/UniVec/BLAST full verifier、Python编译、shell语法、全项目SLURM分区门禁和`git diff --check`均PASS。上一轮YAML/JSON全解析亦PASS。

本轮在release闭包审计中发现并修复一个关键坐标问题：sequence store的callable interval使用全局packed offset，而Tiara/UniVec mask使用单条FASTA record本地坐标。finalizer现在先将callable interval转换为record-local坐标再扣mask，并同时输出`record_start_0based`与`store_start`。新增schema-2 manifest只嵌入这些最终区间，sampler和preflight均验证坐标、callable包含关系与六长度capacity。

## 7. 当前阻塞

1. 还差6个genome BUSCO shard；当前controller正在自动补齐，不需要人工重提。
2. 最终精简source数、六长度capacity、cold-genus和release hash尚未产生；必须等待8605119完成并验证READY。
3. 目标H20环境尚不可访问，无法冻结Mamba-2生产依赖、精确参数量、BF16显存/吞吐、global batch和总token预算。
4. 正式架构图应在参数与shape实测冻结后生成，避免图文数字先于代码。

## 8. 下一步唯一优先任务

让8605103补完最后6个genome BUSCO并闭合92个双模式shard；随后8605119将自动复核BUSCO lineage及Tiara/UniVec/BLAST参考，生成BUSCO/污染/最终代表/六长度capacity机器产物。验证READY后再冻结cold genera并构建schema-2精简data release；全程只用q02–q05。