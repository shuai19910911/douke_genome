# LegumeGenomeFM Training Progress

## 当前状态

- 最后更新：2026-07-18 23:35:28 CST（UTC+08:00）。
- 当前阶段：普通FASTA、SoyOD ZIP及ZIP内部genome内容审计均已完成；正在统一来源并准备近重复/泄漏审计。
- Git：`main`；数据审计里程碑`d9122bc`已发布并与`origin/main`一致。
- 正式架构：未冻结。
- 正式训练：未开始。
- 下一步唯一优先任务：发布584个PASS genome source的统一机器清单，补审37个有效ZIP GFF，并按材料/精确内容/反向互补/近重复冻结训练代表组装与泄漏合同。

## 已核实的计算环境

### 登录与SLURM

- 登录角色：36 CPU cores，251 GiB RAM，无swap；共享BeeGFS约6.1 PiB，审计时使用率89%。
- 可见分区：`cu`、`fat`、`q02`、`q03`、`q04`、`q05`；均无固定time limit。所有普通计算节点为36 cores/190000 MB，`fat`为128 cores/3,096,000 MB。
- 审计时`fat`为mixed；`q02/q03/q04`存在idle节点；`q05`为mixed。
- 初始审计时用户SLURM队列为空；本轮数据审计CPU任务目前已全部终止并验收。
- 历史项目SLURM脚本：清空后的仓库中不存在。

### RTX 2080 Ti开发节点

- 实测7张NVIDIA GeForce RTX 2080 Ti，每张11264 MiB，compute capability 7.5，driver 535.104.05。
- 初始审计时物理GPU 0–4为空闲；5和6已有任务。依赖安装后重新门禁时，仅一张卡满足无compute进程、1 MiB、0%利用率；其余卡全部避开，未干扰、未终止任何进程。
- 已在动态选择的空闲RTX 2080 Ti上完成20.12 MiB峰值的PyTorch FP16矩阵smoke：结果finite、compute capability 7.5；正式模型单卡/多卡验证尚未开始。
- 节点与登录节点访问同一项目文件；共享环境目录存在。

### A100节点

- 仅执行只读资源查询；**未使用A100**。
- 允许的物理GPU 2为NVIDIA A100-SXM4-40GB，审计时仅余约2.2 GiB并已有任务，当前不可用于项目。
- 任何A100计算前必须单独通知用户并等待明确批准。

### Python与依赖

- `douke_genomemodel`环境存在，Python 3.10.20；安装前基线已保存。
- 已用`uv pip`安装并实测PyTorch `2.5.1+cu124`、NumPy `1.26.4`、PyYAML `6.0.3`及测试栈；CUDA 12.4在RTX 2080 Ti driver 535.104.05上真实运行通过。
- `environment/requirements.lock.txt`已将本地conda direct-reference规范化为`name==version`，不含服务器绝对路径；`runtime_manifest.json`记录CPU/CUDA验收。
- 正式环境最新全套测试：60 passed in 1.13 s。

## 已发现的数据

原始数据根存在，轻量顶层检查发现：

| 顶层目录 | 轻量观察 | 完整统计状态 |
|---|---|---|
| `legume_family` | 2,808条记录；普通文件157.882 GiB | Phase 1已核实 |
| `legumeinfo` | 621条记录；普通文件16.586 GiB | Phase 1已核实 |
| `soyod` | 208条记录；普通文件11.027 GiB | Phase 1已核实 |
| `soyomics` | 204条记录；普通文件10.033 GiB | Phase 1已核实 |

Phase 1共记录3,841个目录项，其中3,427个普通文件、414个symlink（符号链接）、0个特殊文件；普通文件精确总量209,947,381,782 bytes（195.529 GiB）。按文件名识别到1,289个FASTA候选、537个注释候选、155个BED类区间文件、49个checksum文件、201个archive及1,196个其他文件；这些是候选数量，尚未去除symlink、同一文件多后缀或重复版本。物种/属数、总碱基、FASTA/GFF/VCF配套、质量、重复、许可证和有效token仍待Phase 2–4核实。登录节点未进行全量序列扫描。

Phase 2从Phase 1中严格选择`kind=file`、`file_type=fasta`且路径位于`genome/`目录的记录，得到552个普通组装候选、172,309,972,336 bytes（160.476 GiB压缩数据）。550个完整通过、2个gzip压缩流截断；无missing、无非法DNA字符、无重复header和空序列。550个PASS source共609,760,149,233 symbols；精确内容去重后479个唯一序列、527,173,283,543 symbols。存在66个精确重复组、137个成员，其中60组跨来源。

Annotation审计已完成：537/537个GFF/GTF候选PASS且无missing；264个主基因模型共12,856,331个gene feature。4个“主模型”实际无gene feature；31个文件含malformed line，6个含非法坐标，23个含重复gene ID，均已逐文件定位。修复跨来源assembly-stem规则后537/537均有候选组装配套；内容级seqid覆盖仍需FASTA header/length二次门禁。

Taxon metadata已完成：552/552个普通组装候选均解析物种，覆盖141 species、69 genera。来源为路径显式472条、SoyOmics官方API 27条、LegumeInfo注释5条、由显式taxon动态学习的序列前缀45条、同材料跨来源3条。SoyOmics官方API核实27个组装（24 `Glycine max`、3 `Glycine soja`）。

Phase 1另有201个SoyOD ZIP（10.469 GiB）。197个archive通过SHA-256与全部member CRC，4个具有ZIP local header但压缩成员流和中央目录均真实截断，无法安全恢复；其中损失2个genome、1个protein和1个GFF archive。34个有效genome member随后以34-way array完整扫描：34/34 PASS，共34,376,817,912 symbols。与普通FASTA比较，30个是精确内容镜像，4个T2T组装为新增唯一序列。统一计算得到584个PASS source、483个精确唯一序列和531,235,812,762个唯一symbols；另有2个普通FASTA FAIL。

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
- `annotation_audit.py`/`annotation_summary.py`及4-way SLURM链：537个GFF/GTF流式解析、配套、实现hash resume与严格聚合。
- `metadata_integration.py`、SoyOmics官方API抓取器及assembly metadata CLI：552条taxon provenance（来源）完整清单。
- `archive_audit.py`与ZIP SLURM链：archive SHA-256、全member CRC、路径穿越/加密/压缩比门禁和resume。
- `audit_summary.py`及FASTA聚合CLI：PASS/FAIL/MISSING账目、实现hash一致性和精确序列重复组。
- `archive_sequence_audit.py`/`archive_sequence_summary.py`及34-way SLURM链：ZIP genome成员身份、CRC、临时提取、FASTA QC、resume与严格聚合；临时文件自动清理。

实际测试：

```text
douke_genomemodel/bin/python -m pytest -q -p no:cacheprovider tests
60 passed in 1.13s
/bin/sh -n 所有submitter与sbatch入口
Python py_compile: PASS
```

测试按TDD执行：inventory、文献、FASTA、Annotation、metadata、ZIP、聚合与SLURM链均先观察预期RED，再实现GREEN。

## SLURM任务

| Job ID | 分区/节点 | 资源 | 状态 | 时间/退出码 | 说明 |
|---|---|---|---|---|---|
| `8600458` | `fat`/未分配 | 2 CPU，8 GiB，2 h | CANCELLED | 0 s；用户取消 | `fat`已有更早任务排队且本任务持续因Priority等待，按预设回退规则取消；未产生数据 |
| `8600479` | `q03`/`cu25` | 2 CPU，8 GiB，2 h | COMPLETED | 21 s；`0:0`；MaxRSS 1,148 KiB | Phase 1内容正确；提交检查发现空末列形成尾随tab，未作为最终manifest发布 |
| `8600480` | `q03`/`cu25` | 2 CPU，8 GiB，2 h | COMPLETED | 7 s；`0:0`；MaxRSS 1,148 KiB | 修复TSV列顺序后原子重建；计数/字节不变，stderr为空，最终manifest通过 |
| `8600492_0` | `q03`/`cu25` | 1 CPU，8 GiB | FAILED（预期数据发现） | 23 s；`1:0`；MaxRSS 968 KiB | 最小真实候选触发`EOFError`，确认该29,490,750-byte gzip被截断；解析器正确fail-closed |
| `8600494_0` | `q03`/`cu25` | 1 CPU，8 GiB | COMPLETED | 63 s；`0:0`；MaxRSS 13,516 KiB | 83,059,820-byte NCBI组装真实PASS，得到8条序列、391,466,139 bases及双hash |
| `8600498_0` | `q03`/`cu25` | 1 CPU，8 GiB | COMPLETED | 3 s；`0:0`；MaxRSS 964 KiB | 同一smoke重跑，`reused_count=1`，未重复读取FASTA |
| `8600499_[0-5]` | `q03`/`cu25` | 每task 1 CPU、8 GiB、2 d | 5 COMPLETED；1数据FAIL | 5:08–5:19 h | 552个候选均有终态；550 PASS、2个截断gzip FAIL；严格聚合无missing |
| `8600519_0`/`8600520_0` | `q04`/`cu42` | 1 CPU、8 GiB | COMPLETED | 各1 s；`0:0` | 真实GFF smoke PASS并验证`reused_count=1` |
| `8600521_[0-3]` | `q04`/`cu42` | 每task 1 CPU、8 GiB | COMPLETED | 12–15 min；`0:0` | 首次537文件全量审计；真实数据揭示跨来源配套规则缺口，结果未作为最终hash发布 |
| `8600525_[0-3]` | `q04`/`cu42` | 每task 1 CPU、8 GiB | COMPLETED | 12–15 min；`0:0` | 修复`gnmHiC_1`及跨来源stem规则后重跑；537/537 PASS、537/537 paired |
| `8600531`/`8600533` | `q02`/`cu14` | 1 CPU、8 GiB | COMPLETED | 各2 s；`0:0` | 单ZIP CRC/safety smoke PASS并验证resume |
| `8600538` | `q02`/`cu14` | 1 CPU、8 GiB、1 d | 数据FAIL | 4:53 min；`1:0` | 197 archive PASS；4个截断ZIP FAIL；有效member无CRC failure、加密或不安全路径 |
| `8602923_0` | `q02`/`cu10` | 1 CPU、4 GiB | COMPLETED | 2:34 min；`0:0`；MaxRSS 166,844 KiB | ZIP genome真实smoke PASS |
| `8602925_[0-33]` | `q02`/`cu06–cu10` | 每task 1 CPU、4 GiB | 34/34 COMPLETED | 最大3:18 min；全部`0:0` | 34个有效ZIP genome并行内容审计；未使用`fat` |

CPU并行策略已按用户修正：普通CPU/IO任务可在`q02–q05`按实时空闲资源超过6路并行；`fat`是稀缺大内存节点，默认不占用，仅在实测内存超出普通节点容量且确认空闲时考虑。

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
| RTX 2080 Ti环境/CUDA smoke | PASS；PyTorch 2.5.1+cu124、FP16 finite |
| RTX 2080 Ti正式模型单卡 | 未开始 |
| RTX 2080 Ti多卡 | 未开始 |
| A100 | 未使用；未获本阶段批准 |
| checkpoint保存/恢复 | 未开始 |
| Python/CUDA依赖锁 | 已生成并去除本地绝对路径；AutoDL尚未验收 |
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
2. 34个ZIP genome已审计并完成精确内容比较，但统一机器清单、37个有效ZIP GFF内容审计、反向互补和近重复聚类尚未冻结。
3. A100物理2号卡初始审计时繁忙且本阶段未经使用授权；不影响CPU审计和2080开发验证。

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
- 2026-07-18 17:40–18:36 CST：完成537个Annotation两轮全量审计；第二轮修复跨来源配套后537/537 PASS和paired，严格聚合发布。
- 2026-07-18 18:10 CST：SoyOmics官方API核实27个assembly；整合后552/552组装taxon已解析，覆盖141 species、69 genera。
- 2026-07-18 18:25 CST：正式环境依赖安装完成；动态GPU门禁后仅用一张空闲RTX 2080 Ti完成CUDA smoke，其他GPU全部避开。
- 2026-07-18 18:48 CST：ZIP audit smoke和resume通过；18:53提交201个archive全量Job `8600538`。
- 2026-07-18 18:53 CST：正式环境全套50项测试通过。
- 2026-07-18 21:44–21:55 CST：552个普通FASTA全部获得终态；550 PASS、2个截断gzip FAIL；修复聚合器与producer schema不一致后严格聚合通过。
- 2026-07-18 23:04 CST：确认ZIP全量Job已在18:58完成；197 PASS、4个截断源文件FAIL，所有有效member CRC通过。
- 2026-07-18 23:26 CST：按新并行策略检查`q02–q05`，不使用`fat`；ZIP genome smoke `8602923_0`通过。
- 2026-07-18 23:30–23:34 CST：34-way Job `8602925`在`q02`五个普通节点完成，34/34 PASS；30个与普通来源精确重复，4个T2T为新增唯一序列。
- 2026-07-18 23:35 CST：全套60项测试通过；全部当前CPU任务结束。
