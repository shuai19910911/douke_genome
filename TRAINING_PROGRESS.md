# LegumeGenomeFM Training Progress

## 当前状态

- 最后更新：2026-07-18 15:50:44 CST（UTC+08:00）。
- 当前阶段：Phase 1原始数据inventory已完成；系统文献候选检索与Phase 2序列质量审计准备中。
- Git：`main`；开始工作时本地与`origin/main`一致，工作树干净，远端为无父空根。
- 正式架构：未冻结。
- 正式训练：未开始。
- 下一步唯一优先任务：从已冻结inventory构建去重后的assembly候选注册表，并提交Phase 2 FASTA质量审计任务。

## 已核实的计算环境

### 登录与SLURM

- 登录角色：36 CPU cores，251 GiB RAM，无swap；共享BeeGFS约6.1 PiB，审计时使用率89%。
- 可见分区：`cu`、`fat`、`q02`、`q03`、`q04`、`q05`；均无固定time limit。所有普通计算节点为36 cores/190000 MB，`fat`为128 cores/309000 MB。
- 审计时`fat`为mixed；`q02/q03/q04`存在idle节点；`q05`为mixed。
- 用户SLURM队列：空。
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

实际测试：

```text
pytest -q -p no:cacheprovider tests
19 passed in 0.66s
/bin/sh -n scripts/submit_raw_inventory.sh
/bin/sh -n scripts/slurm/inventory_raw_data.sbatch
Python py_compile: PASS
```

测试按TDD执行：inventory import缺失和SLURM脚本缺失均先观察到预期RED，再实现GREEN。

## SLURM任务

| Job ID | 分区/节点 | 资源 | 状态 | 时间/退出码 | 说明 |
|---|---|---|---|---|---|
| `8600458` | `fat`/未分配 | 2 CPU，8 GiB，2 h | CANCELLED | 0 s；用户取消 | `fat`已有更早任务排队且本任务持续因Priority等待，按预设回退规则取消；未产生数据 |
| `8600479` | `q03`/`cu25` | 2 CPU，8 GiB，2 h | COMPLETED | 21 s；`0:0`；MaxRSS 1,148 KiB | Phase 1内容正确；提交检查发现空末列形成尾随tab，未作为最终manifest发布 |
| `8600480` | `q03`/`cu25` | 2 CPU，8 GiB，2 h | COMPLETED | 7 s；`0:0`；MaxRSS 1,148 KiB | 修复TSV列顺序后原子重建；计数/字节不变，stderr为空，最终manifest通过 |

CPU任务先尝试了唯一非`cu`节点分区`fat`；确认其已有4个更早任务排队且分配内存约2.7 TB后才回退至有idle节点的`q03`。

## 文献检索状态

- 已冻结检索截止日期2026-07-18和26个Crossref查询族。
- `data_manifests/literature_candidates.tsv`收录2,264个去重候选；其中122条由期刊名规则标为Nature Portfolio宽松候选。
- 122不是纳入论文数；标题初筛、摘要/全文筛选、逐DOI元数据核验、Methods/补充方法和官方代码提取仍在进行。
- 候选manifest SHA-256：`35b9648ca4109eda89479bffa87216ded7cfcd38ba4407a946923f8ded5e1089`。

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

1. 正式模型选择被真实数据规模/长度/泄漏审计和系统文献检索有意阻断；不得提前填写参数。
2. `douke_genomemodel`缺PyTorch与训练依赖，需在保存环境基线后修复。
3. A100物理2号卡当前繁忙且本阶段未经使用授权；不影响Phase 1 CPU审计。

## 操作日志

- 2026-07-18 15:26 CST：检查Git、SLURM节点/分区、用户队列、登录节点CPU/RAM/磁盘。
- 2026-07-18 15:26 CST：只读检查RTX 2080 Ti和A100资源；未启动或终止任何GPU任务。
- 2026-07-18 15:30 CST：确认目标环境仅含Python基础包且缺PyTorch；确认仓库中无旧SLURM脚本。
- 2026-07-18 15:39 CST：Phase 1先提交`fat`；因Priority等待且存在更早队列，精确取消Job `8600458`。
- 2026-07-18 15:49 CST：回退到`q03`提交Job `8600479`，21秒完成并通过manifest验收。
- 2026-07-18 15:50 CST：预提交`git diff --check`捕获尾随tab；TDD加入回归测试，Job `8600480`重建最终manifest并通过。
- 2026-07-18：以TDD实现Phase 1 inventory、文献元数据处理和portable SLURM路径；19项测试通过。
- 2026-07-18：完成26个Crossref查询，生成2,264条可审计候选；筛选与全文核验尚未完成。
