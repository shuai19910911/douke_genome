# LegumeGenomeFM Training Progress

## 当前状态

- 最后更新：2026-07-18 17:14:18 CST（UTC+08:00）。
- 当前阶段：系统文献检索与核心全文核验已完成；Phase 2全量FASTA内容审计运行中。
- Git：`main`；已发布首个里程碑`2e5a1ff`，本轮文献/Phase 2代码修改待独立里程碑提交。
- 正式架构：未冻结。
- 正式训练：未开始。
- 下一步唯一优先任务：持续监控并完成SLURM array `8600499_[0-5]`，聚合552个FASTA结果并冻结可用组装/重复/损坏清单。

## 已核实的计算环境

### 登录与SLURM

- 登录角色：36 CPU cores，251 GiB RAM，无swap；共享BeeGFS约6.1 PiB，审计时使用率89%。
- 可见分区：`cu`、`fat`、`q02`、`q03`、`q04`、`q05`；均无固定time limit。所有普通计算节点为36 cores/190000 MB，`fat`为128 cores/309000 MB。
- 审计时`fat`为mixed；`q02/q03/q04`存在idle节点；`q05`为mixed。
- 初始审计时用户SLURM队列为空；当前6个Phase 2 array task正在`q03`运行。
- 历史项目SLURM脚本：清空后的仓库中不存在。

### RTX 2080 Ti开发节点

- 实测7张NVIDIA GeForce RTX 2080 Ti，每张11264 MiB，compute capability 7.5，driver 535.104.05。
- 审计时物理GPU 0–4为空闲；5和6已有任务，未干扰、未终止。
- 项目尚未启动GPU进程；单卡和多卡验证均未开始。
- 节点与登录节点访问同一项目文件；共享环境目录存在。

### A100节点

- 仅执行只读资源查询；**未使用A100**。
- 允许的物理GPU 2为NVIDIA A100-SXM4-40GB，审计时仅余约2.2 GiB并已有任务，当前不可用于项目。
- 任何A100计算前必须单独通知用户并等待明确批准。

### Python与依赖

- `douke_genomemodel`环境存在，Python 3.10.20。
- 该环境当前未安装PyTorch及正式训练依赖；环境管理CLI未在当前shell PATH中发现。
- 未修改环境。正式依赖安装前仍需保存完整包清单并建立可恢复锁文件。
- 当前inventory测试借用已有只读开发Python环境执行；这不是正式训练环境验收。

## 已发现的数据

原始数据根存在，轻量顶层检查发现：

| 顶层目录 | 轻量观察 | 完整统计状态 |
|---|---|---|
| `legume_family` | 2,808条记录；普通文件157.882 GiB | Phase 1已核实 |
| `legumeinfo` | 621条记录；普通文件16.586 GiB | Phase 1已核实 |
| `soyod` | 208条记录；普通文件11.027 GiB | Phase 1已核实 |
| `soyomics` | 204条记录；普通文件10.033 GiB | Phase 1已核实 |

Phase 1共记录3,841个目录项，其中3,427个普通文件、414个symlink（符号链接）、0个特殊文件；普通文件精确总量209,947,381,782 bytes（195.529 GiB）。按文件名识别到1,289个FASTA候选、537个注释候选、155个BED类区间文件、49个checksum文件、201个archive及1,196个其他文件；这些是候选数量，尚未去除symlink、同一文件多后缀或重复版本。物种/属数、总碱基、FASTA/GFF/VCF配套、质量、重复、许可证和有效token仍待Phase 2–4核实。登录节点未进行全量序列扫描。

Phase 2从Phase 1中严格选择`kind=file`、`file_type=fasta`且路径位于`genome/`目录的记录，得到552个真实组装候选、172,309,972,336 bytes（160.476 GiB压缩数据）。先前宽松文件名筛选多出的144条均为`cds_from_genomic`或`rna_from_genomic`，已正确排除。候选按压缩字节LPT（最长处理时间优先）确定性分为6个约26.746 GiB的shard。

已验收产物：

- `data_manifests/raw_inventory.tsv`：642,393 bytes，3,842行（含表头），只含相对路径；
- `data_manifests/raw_inventory.summary.json`；
- inventory SHA-256：`06f2e74f46a76f302aef45b6243c3498823bf9bff253509d900e77358ca92617`，独立重算一致；TSV无尾随空白。

## 已完成代码与验证

已完成：

- `src/legumegenomefm/data_inventory.py`：确定性只读目录扫描；不跟随符号链接；输出相对路径；绝对symlink目标只保留hash；原子写入TSV/JSON。
- `scripts/inventory_raw_data.py`：Phase 1 CLI入口。
- `scripts/slurm/inventory_raw_data.sbatch`：POSIX `sh`、默认`fat`分区的SLURM入口。
- `scripts/submit_raw_inventory.sh`：运行时注入项目/数据/输出/Python路径，并设置绝对SLURM日志路径。
- `configs/data_audit.yaml`：阶段与资源合同。
- 三个唯一Markdown和figure manifest初始骨架。
- `src/legumegenomefm/assembly_audit.py`：候选注册、6-way LPT分片、gzip/plain单遍流式FASTA解析、N50/组成统计、文件与规范化序列双SHA-256、路径边界及实现hash绑定的resume。
- `scripts/build_assembly_registry.py`、`scripts/audit_fasta_shard.py`、`scripts/slurm/audit_fasta_shards.sbatch`和`scripts/submit_fasta_qc.sh`：Phase 2正式链路。
- `src/legumegenomefm/literature.py`、`scripts/retrieve_literature.py`和`scripts/verify_core_literature.py`：查询、去重、逐DOI/出版社页面题名与article type核验。

实际测试：

```text
pytest -q -p no:cacheprovider tests
31 passed in 1.85s
/bin/sh -n scripts/submit_raw_inventory.sh scripts/submit_fasta_qc.sh及两个sbatch入口
Python py_compile: PASS
```

测试按TDD执行：inventory、SLURM、文献、assembly注册/解析/分片/resume均先观察预期RED，再实现GREEN。

## SLURM任务

| Job ID | 分区/节点 | 资源 | 状态 | 时间/退出码 | 说明 |
|---|---|---|---|---|---|
| `8600458` | `fat`/未分配 | 2 CPU，8 GiB，2 h | CANCELLED | 0 s；用户取消 | `fat`已有更早任务排队且本任务持续因Priority等待，按预设回退规则取消；未产生数据 |
| `8600479` | `q03`/`cu25` | 2 CPU，8 GiB，2 h | COMPLETED | 21 s；`0:0`；MaxRSS 1,148 KiB | Phase 1内容正确；提交检查发现空末列形成尾随tab，未作为最终manifest发布 |
| `8600480` | `q03`/`cu25` | 2 CPU，8 GiB，2 h | COMPLETED | 7 s；`0:0`；MaxRSS 1,148 KiB | 修复TSV列顺序后原子重建；计数/字节不变，stderr为空，最终manifest通过 |
| `8600492_0` | `q03`/`cu25` | 1 CPU，8 GiB | FAILED（预期数据发现） | 23 s；`1:0`；MaxRSS 968 KiB | 最小真实候选触发`EOFError`，确认该29,490,750-byte gzip被截断；解析器正确fail-closed |
| `8600494_0` | `q03`/`cu25` | 1 CPU，8 GiB | COMPLETED | 63 s；`0:0`；MaxRSS 13,516 KiB | 83,059,820-byte NCBI组装真实PASS，得到8条序列、391,466,139 bases及双hash |
| `8600498_0` | `q03`/`cu25` | 1 CPU，8 GiB | COMPLETED | 3 s；`0:0`；MaxRSS 964 KiB | 同一smoke重跑，`reused_count=1`，未重复读取FASTA |
| `8600499_[0-5]` | `q03`/`cu25` | 每task 1 CPU、8 GiB、2 d | RUNNING | 提交16:36 CST | 552个候选、6 shard、160.476 GiB压缩FASTA全量只读审计 |

CPU任务先尝试了唯一非`cu`节点分区`fat`；确认其已有4个更早任务排队且分配内存约2.7 TB后才回退至有idle节点的`q03`。

## 文献检索状态

- 状态：**COMPLETED**。检索截止日期2026-07-18；36个Crossref查询族全部返回候选。
- `data_manifests/literature_candidates.tsv`收录2,958条去重记录；Nature Portfolio期刊名宽松候选154条；候选manifest SHA-256为`980447eb49e6b72f81510bcc2ca6e35a2af179498e36fc469f683b3eb23660eb`。
- 引文追踪补回初始宽查询排名漏掉的Evo 2、AgroNT、PlantCaduceus、GPN、PlantGFM和species-aware论文，并将对应精确查询纳入统一重跑，未手工拼接结果。
- 30条白名单逐源核验全部PASS：19条核心、8条上下文、3条明确排除的Research Briefing/Highlight；Nature记录额外核验article type。
- 首轮核验真实捕获11个不精确题名/article type和1个无效DOI；按出版社元数据修正后30/30通过。Crossref更新关系计数均为0。
- 14篇架构关键论文已完成训练数据、tokenizer、上下文、参数/算力、拆分、证据与局限矩阵；开放全文无法确认的字段明确标为`not extracted`。

## GPU、checkpoint与迁移状态

| 项目 | 状态 |
|---|---|
| RTX 2080 Ti单卡 | 未开始 |
| RTX 2080 Ti多卡 | 未开始 |
| A100 | 未使用；未获本阶段批准 |
| checkpoint保存/恢复 | 未开始 |
| AutoDL环境锁 | 未开始 |
| AutoDL bootstrap | 未开始 |
| 正式数据shard | 未开始 |

## 图件状态

- 主图：0张。
- 补充图：0张。
- 训练诊断图：0张。
- Nature级验收：未开始。
- 未使用占位或伪造数据生成图件。

## 当前阻塞

1. 正式模型选择仍被真实数据规模/长度/重复/泄漏审计有意阻断；文献门槛已解除，但不得提前填写参数。
2. `douke_genomemodel`缺PyTorch与训练依赖，需在保存环境基线后修复。
3. A100物理2号卡当前繁忙且本阶段未经使用授权；不影响Phase 1 CPU审计。

## 操作日志

- 2026-07-18 15:26 CST：检查Git、SLURM节点/分区、用户队列、登录节点CPU/RAM/磁盘。
- 2026-07-18 15:26 CST：只读检查RTX 2080 Ti和A100资源；未启动或终止任何GPU任务。
- 2026-07-18 15:30 CST：确认目标环境仅含Python基础包且缺PyTorch；确认仓库中无旧SLURM脚本。
- 2026-07-18 15:39 CST：Phase 1先提交`fat`；因Priority等待且存在更早队列，精确取消Job `8600458`。
- 2026-07-18 15:49 CST：回退到`q03`提交Job `8600479`，21秒完成并通过manifest验收。
- 2026-07-18 15:50 CST：预提交`git diff --check`捕获尾随tab；TDD加入回归测试，Job `8600480`重建最终manifest并通过。
- 2026-07-18 16:18–16:33 CST：真实FASTA smoke Job `8600492`检出截断gzip；切换到NCBI组装后Job `8600494`通过，Job `8600498`验证resume。
- 2026-07-18 16:36 CST：确认`fat`仍有更早队列后，将6-way全量Phase 2 array `8600499`提交到`q03`。
- 2026-07-18 17:13 CST：完成36查询、2,958候选、30/30逐源核验和14篇全文方法矩阵；文献任务关闭。
- 2026-07-18 17:14 CST：全套31项测试和Python/POSIX语法验证通过。
