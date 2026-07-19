# LegumeGenomeFM-89M：正式预训练计划

> 状态：**FROZEN（已冻结）**，2026-07-19。数据、模型、目标、三个stage、global token batch、总预算、checkpoint语义和评测泄漏合同均已固定。实际开始仍必须通过AutoDL release deep verification（深度校验）、离线环境重定位验证和目标GPU preflight（启动前门禁）。

## 1. 正式输入数据

机器入口：`data_release/training_dataset.json`。

| 字段 | 冻结值 |
|---|---:|
| RC归一化后的source数 | 440 |
| pretrain source | 337 |
| cold-genus holdout source | 103 |
| genus数 | 69 |
| species标签数 | 140 |
| pretrain callable bases | 372,953,588,394 |
| cold-genus callable bases | 99,581,789,352 |
| 总callable bases | 472,535,377,746 |
| 被选2-bit数据大小 | 124,660,775,084 bytes |
| manifest SHA-256 | `d154f7a4d0dd3bad2b556ec15188aa24c7d6d490cb5900a4fea3723751571bb3` |
| release receipt SHA-256 | `d6712a0207f10e89cb91f32c06cc9ff8b6f627557b963bb3cad41ea1ce7f3fdf` |

发布链由`training_dataset.release.json`绑定`data_freeze.yaml`、466-source sketch registry、全局108,345对MinHash比较、RC identity结果、dataset和summary；`TRAINING_DATASET_READY`最后写入。训练preflight逐一验证440个store的`READY`与manifest hash。

### 1.1 冻结审计结果

- 466个候选sequence store全部READY；
- RC/contig-order/name不敏感的全序列identity得到440组，其中26个重复组、52个成员，只保留每组一个代表；
- 31-mer MinHash对466个候选执行全局108,345对比较，而不是只在同species内比较；
- 发现2对跨taxon标签的近重复（Pisum sativum与Lathyrus oleraceus），已进入同一near-duplicate group；
- 66个near-duplicate group包含187个成员；
- 无RC group或near-duplicate group跨越pretrain/cold-genus边界；
- 574个annotation来源的统一catalog含537个普通文件和37个ZIP成员；301个primary-gene-model来源，共15,973,108个gene feature。

### 1.2 Cold-genus合同

完整留出的六个属为：`Arachis`、`Cercis`、`Chamaecrista`、`Cicer`、`Lupinus`、`Vigna`。这些source的`sampling_weight`严格为0，不进入预训练更新，只作为后续跨属泛化合同的一部分。训练不得根据cold-genus或正式test标签调模型超参数。

### 1.3 采样权重

正式pretrain source先按species做近似均衡，再按near-duplicate group size和material-version group size惩罚，最后归一化到总和1。这样可防止拥有大量版本或重复下载的物种支配token流。每次抽样先选source，再在其长度加权的callable interval中选不跨contig的窗口。随机流由checkpoint中的sampler/RNG状态控制。

## 2. 目标与模型

- 模型：`LegumeGenomeFM-89M`，88,946,028参数；完整定义见`MODEL_ARCHITECTURE.md`。
- Tokenizer：17-token单碱基IUPAC词表；正式训练窗口只含ACGT callable bases。
- 目标：span MLM（连续片段掩码预测），mask ratio 0.15，mean span length 3。
- RC语义：正向与反向互补共享主干并对齐平均，训练和推理一致。
- 正式精度：BF16；目标GPU必须为compute capability 8.0或更高。
- 并行：单进程单GPU的DDP rank，使用`torchrun`；不采用参数分片或pipeline parallel。

## 3. 三阶段100B-token课程

| Stage | Config | Context | Micro-batch/GPU | Global batch tokens | Optimizer steps | Token budget | Peak LR |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `configs/pretrain_stage1.yaml` | 1,024 | 8 | 262,144 | 133,514 | 34,999,894,016 | 3e-4 |
| 2 | `configs/pretrain_stage2.yaml` | 4,096 | 2 | 524,288 | 66,757 | 34,999,894,016 | 2e-4 |
| 3 | `configs/pretrain_stage3.yaml` | 16,384 | 1 | 524,288 | 57,220 | 29,999,759,360 | 1e-4 |

总计：257,491 optimizer steps、99,999,547,392 tokens。每个stage的token预算都恰好被global batch tokens整除。

梯度累积步数由以下公式唯一决定：

`gradient_accumulation = global_batch_tokens / (context_length × micro_batch_per_gpu × world_size)`

结果必须是正整数，否则preflight拒绝启动。典型8-GPU配置中，Stage 1/2/3的累积步数分别为4、8、4。

## 4. 优化和调度

- AdamW：`betas=(0.9, 0.95)`、`weight_decay=0.1`；
- 2% linear warmup后cosine decay，minimum LR ratio 0.1；
- global gradient norm clip 1.0；
- activation checkpointing开启；
- 正式dropout为0；
- 日志每10步，checkpoint每500步；最终不足500步的尾部也必须写最终checkpoint。

Stage之间是**initialize（仅加载模型参数）**：Stage 2从Stage 1最终checkpoint初始化，Stage 3从Stage 2初始化，并为新stage重新创建optimizer/scheduler/scaler。Stage内部中断是**resume（完整恢复）**：恢复模型、optimizer、scheduler、scaler、step、tokens_seen、global_microstep、RNG和sampler状态。两者不可混用。

## 5. Checkpoint与失败语义

每个checkpoint使用临时目录写入，计算`state.pt` SHA-256并生成`receipt.json`，最后写`READY`后原子rename。恢复前必须验证READY、receipt、state hash、config hash和实现closure hash。

- FP16/BF16出现非有限loss或梯度时，本次更新不得增加optimizer step或tokens_seen；
- world size可以在恢复时变化，但global batch tokens与配置合同不变；
- `MODE=fresh`拒绝已有output目录，防止覆盖旧run；
- `MODE=resume`要求output内已有READY checkpoint；
- `MODE=initialize`要求显式`INITIALIZE_FROM`且新output不存在；
- 校验失败只停止本任务，不删除已有checkpoint或数据。

## 6. AutoDL启动流程

release布局为`AUTODL_RELEASE_MANIFEST.json + project/`。正式release携带：Git跟踪代码、冻结manifest、440个被选2-bit store和可重定位的离线CUDA环境归档。

```sh
# 1. 迁移后先做完整校验（首次/传输后必须deep）
python3 project/scripts/verify_autodl_release.py \
  --release-root "$PWD" --deep

# 2. 若尚未创建环境，离线解包并重定位
sh project/environment/bootstrap_autodl.sh

# 3. Stage 1 fresh start；NPROC_PER_NODE改为实际可用GPU数
MODE=fresh NPROC_PER_NODE=8 CONFIG=project/configs/pretrain_stage1.yaml \
  project/scripts/autodl_launch.sh

# Stage 1中断后的完整恢复
MODE=resume NPROC_PER_NODE=8 CONFIG=project/configs/pretrain_stage1.yaml \
  project/scripts/autodl_launch.sh

# Stage 2模型初始化示例
MODE=initialize INITIALIZE_FROM=/absolute/path/to/stage1/final_checkpoint \
  NPROC_PER_NODE=8 CONFIG=project/configs/pretrain_stage2.yaml \
  project/scripts/autodl_launch.sh
```

`autodl_launch.sh`会先运行release quick verification，再检查dataset release、440个store、模式互斥、world-size整除、CUDA数量、BF16能力以及每卡至少6,000 MiB空闲显存，并写原子preflight receipt；只有全部PASS才执行`torchrun`。

## 7. 评测与模型选择

正式矩阵位于`configs/evaluation_matrix.yaml`，包含20项任务和内部/外部/任务专用基线。共同泄漏分组为orientation group、near-duplicate group、material key和chromosome-homology component。

- 训练期间可在checkpoint上运行轻量diagnostic，但不得用formal test选择模型；
- medium validation benchmark只运行预选候选checkpoint；
- formal test在模型、超参数和选择规则全部冻结后仅运行一次正式流程；
- 必报in-distribution、leave-one-species-out、cold-genus、low-homology，以及frozen encoder/full finetune；
- 外部模型只在公开权重、许可证、输入语义和可复现实装核验后进入主表；
- 任何smoke loss或单步结果都不是论文性能。

## 8. 开始训练的最终判定

只有以下条件同时满足才可称为“可以开始正式训练”：

1. Git main与release manifest绑定同一commit；
2. dataset release READY、440个store和125-GB级payload全部deep PASS；
3. 离线环境archive解包、`conda-unpack`和核心import PASS；
4. AutoDL目标GPU通过BF16/显存/world-size preflight；
5. fresh output目录不存在，或resume/initialize路径明确且READY；
6. 启动命令和preflight receipt已归档。

AutoDL具体GPU型号和数量在迁移前未知，因此本文不虚构吞吐、walltime或费用；首次目标机preflight和短profiling之后再补实测ETA。