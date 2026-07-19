# LegumeGenomeFM-89M：训练准备进度

> 记录日期：2026-07-19。当前状态为**数据/模型/代码已冻结，等待最终commit后的AutoDL release acceptance与目标GPU preflight**。尚未创建正式100B-token训练进程；RTX 2080 Ti结果均为真实工程验证，不冒充正式预训练。

## 1. 状态总览

| 工作流 | 状态 | 可核验证据 |
|---|---|---|
| 原始FASTA/ZIP genome审计 | PASS | `data_manifests/*fasta*`、`archive_genome*` |
| 普通/ZIP annotation统一审计 | PASS | `unified_annotation_catalog.tsv`及summary |
| 466个2-bit sequence store | PASS | 每store的`manifest.json`和`READY` |
| 全局MinHash近重复 | PASS | `genome_similarity*`；466个候选、108,345对 |
| RC/contig顺序/名称不敏感identity | PASS | `orientation_identity*`；466→440代表 |
| 冻结训练dataset | READY | `data_release/training_dataset*`、`TRAINING_DATASET_READY` |
| 唯一正式架构/tokenizer/100B预算 | FROZEN | `MODEL_ARCHITECTURE.md`、三个pretrain YAML |
| 单GPU 1/4/16 kb训练步骤 | PASS | `gpu_validation.summary.json` |
| Checkpoint完整恢复 | PASS | 同上；step 1→step 2状态闭合 |
| 双GPU DDP | PASS | 同上；world size 2、独立物理GPU |
| AutoDL builder/verifier/launcher | IMPLEMENTED | `scripts/build_autodl_release.py`等 |
| 离线CUDA环境归档 | BUILT | Git外`workspace/autodl_assets/`，约3.192 GB |
| 最终release deep verification | POST-COMMIT GATE | release manifest与SLURM验收日志，Git外保存 |
| 目标AutoDL BF16 GPU preflight | WAITING FOR TARGET | 目标机上由`autodl_launch.sh`执行 |
| 正式Stage 1训练 | NOT STARTED | 无正式PID、日志或checkpoint |

## 2. 数据冻结结果

原始来源经过普通文件、ZIP成员、内容SHA、gzip/CRC、FASTA结构、annotation结构、exact content、MinHash、RC identity、材料版本和cold-genus边界审计。最终正式dataset：

- 440个RC归一化source；
- 337个pretrain source，372,953,588,394个callable bases；
- 103个cold-genus source，99,581,789,352个callable bases；
- 总计472,535,377,746个callable bases；
- 六个完整留出属：Arachis、Cercis、Chamaecrista、Cicer、Lupinus、Vigna；
- 训练dataset SHA-256：`d154f7a4d0dd3bad2b556ec15188aa24c7d6d490cb5900a4fea3723751571bb3`；
- READY-last release receipt SHA-256：`d6712a0207f10e89cb91f32c06cc9ff8b6f627557b963bb3cad41ea1ce7f3fdf`。

关键修正：近重复初版只在相同binomial species内比较，无法证明跨taxon/cold-genus无污染。回归测试先改为跨taxon相同assembly必须同组并观察RED，随后将算法改为466个候选全局两两比较。最终发现2对跨taxon标签近重复；无near group或RC group跨pretrain/cold边界。

统一annotation catalog覆盖574个来源（537个普通文件、37个ZIP成员），其中301个primary gene model，共15,973,108个gene feature；568个来源有明确genome配对，6个没有。

## 3. 正式模型与训练合同

唯一正式模型为LegumeGenomeFM-89M：17-token单碱基词表、640 hidden size、18个分层膨胀卷积block、精确RC输出对称、88,946,028参数、最大16,384 bp上下文。

三个共享参数stage：

1. 1,024 bp：34,999,894,016 tokens，133,514步；
2. 4,096 bp：34,999,894,016 tokens，66,757步；
3. 16,384 bp：29,999,759,360 tokens，57,220步。

总预算99,999,547,392 tokens、257,491个完整optimizer steps。Stage 3曾发现max token不被global batch整除、会隐式多跑尾部batch；现已修为精确整步值，并加入合成与三个正式YAML的回归门禁。

## 4. 真实GPU证据

`data_manifests/gpu_validation.summary.json`记录：

- RTX 2080 Ti、PyTorch 2.5.1+cu124；
- 1 kb、4 kb和16 kb forward/backward/optimizer均PASS；
- 三种长度`rc_max_abs_error=0.0`；
- 16 kb峰值allocated约3.49 GB、reserved约3.68 GB；
- 25%显存配额出现预期OOM，35%配额PASS；
- 单卡完整checkpoint resume从1,024 tokens恢复到2,048 tokens；
- 双卡DDP使用两个独立物理GPU，world size 2；
- FP16 overflow会重试，不错误增加optimizer step或tokens_seen；
- A100未使用，未伪造A100吞吐。

## 5. AutoDL迁移闭包

release builder只从`git archive HEAD`提取跟踪文件，防止把secret、未提交代码或本地checkpoint带入包中。数据store按hardlink→reflink→copy回退，跨文件系统copy分支已经KB级真实end-to-end deep smoke；同文件系统正式staging可使用hardlink而不复制约125 GB payload。

release还携带由当前PyTorch/CUDA环境产生的可重定位`conda-pack`归档。bootstrap优先离线解包，执行`conda-unpack`和本地editable install；无归档时才回退wheelhouse/uv。

最终release必须在包含本文件的commit之后构建，因此它的`AUTODL_RELEASE_MANIFEST.json`、deep-verification输出和环境重定位receipt有意保存在Git外。这样避免“先写成功文档、再改变commit、导致release绑定旧commit”的循环。

## 6. 测试状态

最新全套回归：**98 passed**。范围包括FASTA/ZIP、2-bit store、streaming producer、全局相似度、RC identity、sampler、tokenizer/MLM、RC模型、checkpoint、resume、overflow/DDP语义、AutoDL跨盘staging、release verifier、dataset preflight和整步预算。

正式dataset另完成逐store preflight：440/440 READY/hash一致，337个pretrain权重为正、103个cold-genus权重为0，sampling weight总和在浮点容差内为1。

## 7. 下一动作与停止边界

提交本轮冻结产物后，按顺序执行：

1. 从该commit构建正式AutoDL release；
2. 在计算节点完成quick + deep验证（包括全部store packed SHA）；
3. 完成离线环境解包/重定位/import验证；
4. 将release迁移到AutoDL；
5. 在目标GPU运行BF16、显存和world-size preflight；
6. 只有preflight PASS后才以`MODE=fresh`启动Stage 1。

如果目标AutoDL GPU尚未提供，工程应停在“可迁移、待目标机preflight”，不在2080 Ti上误启动正式BF16 run，也不把验证checkpoint称为训练开始。