# LegumeGenomeFM-89M

LegumeGenomeFM-89M是一个面向豆科多物种基因组的单碱基基础模型项目。正式模型包含88,946,028个参数，以1 kb、4 kb和16 kb三个上下文stage完成约100B token的span-MLM（连续片段掩码预测）预训练，并在输出层严格满足reverse-complement symmetry（反向互补对称）。

## 当前状态

- 数据合同：已冻结；440个RC归一化source，其中337个用于预训练、103个属于六个完整cold-genus（整属留出）集合。
- 模型合同：已冻结；唯一正式模型为LegumeGenomeFM-89M。
- 工程验证：RTX 2080 Ti上的1/4/16 kb训练步骤、checkpoint恢复和双GPU DDP均PASS；A100未用于验证。
- AutoDL：提供自包含release builder、quick/deep verifier、3.192-GB可重定位离线CUDA环境和fresh/resume/initialize启动门禁。
- 正式预训练：尚未把smoke结果冒充正式run；只有目标AutoDL GPU preflight通过后才允许启动。

完整说明见：

- `TRAINING_PLAN.md`：数据、100B-token课程、采样、checkpoint、启动和评测合同；
- `MODEL_ARCHITECTURE.md`：tokenizer、88.9M模型、RC语义、多尺度与GPU证据；
- `TRAINING_PROGRESS.md`：已完成里程碑、机器产物和剩余门禁；
- `configs/evaluation_matrix.yaml`：20项下游任务、泄漏分组和baseline合同。

## 冻结数据

正式入口为`data_release/training_dataset.json`，其SHA-256为：

`d154f7a4d0dd3bad2b556ec15188aa24c7d6d490cb5900a4fea3723751571bb3`

数据发布回执为`data_release/training_dataset.release.json`，`TRAINING_DATASET_READY`最后写入。Git仓库只保存代码、配置和轻量machine-readable manifests（机器可读清单）；约125 GB的2-bit sequence stores、原始数据、离线环境和checkpoint均不提交Git。

数据冻结链包括：

1. 466/466个sequence store READY；
2. RC/contig-order/name不敏感的全序列identity去重为440个代表；
3. 466个候选的108,345对全局MinHash比较；
4. material-version（同材料多版本）与species-balanced（物种平衡）采样；
5. `Arachis/Cercis/Chamaecrista/Cicer/Lupinus/Vigna`六属完整留出；
6. 440个store manifest hash和发布READY闭包。

## 代码结构

```text
configs/                     冻结数据、三个预训练stage和评测合同
src/legumegenomefm/          tokenizer、2-bit store、sampler、模型与训练器
data_manifests/              聚合审计证据和GPU验证summary
data_release/                正式训练dataset与READY-last回执
environment/                 依赖锁和AutoDL bootstrap
scripts/                     数据生产、训练、release构建/验证入口
scripts/slurm/               CPU生产和大文件验收的POSIX SLURM脚本
tests/                       单元、积分、恢复和分布式语义回归
```

## 本地测试

项目验证环境为Python 3.10、PyTorch 2.5.1+cu124。不要从Git下载原始数据；测试只使用临时小型fixtures（夹具）。

```sh
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
/path/to/python -m pytest -q -p no:cacheprovider tests
```

## AutoDL release

release根目录结构：

```text
AUTODL_RELEASE_MANIFEST.json
project/
  data/processed/sequence_store/   # 440个被选store
  data_release/
  environment/douke_genomemodel.tar.gz
  scripts/
  src/
```

迁移后必须先deep verification（深度校验）再解包环境：

```sh
python3 project/scripts/verify_autodl_release.py --release-root "$PWD" --deep
sh project/environment/bootstrap_autodl.sh
```

`bootstrap_autodl.sh`优先离线解包`douke_genomemodel.tar.gz`，执行`conda-unpack`并本地安装项目；只有release未携带归档时才回退到wheelhouse或联网uv安装。

正式Stage 1启动：

```sh
MODE=fresh NPROC_PER_NODE=8 CONFIG="$PWD/project/configs/pretrain_stage1.yaml" \
  "$PWD/project/scripts/autodl_launch.sh"
```

Stage内恢复：

```sh
MODE=resume NPROC_PER_NODE=8 CONFIG="$PWD/project/configs/pretrain_stage1.yaml" \
  "$PWD/project/scripts/autodl_launch.sh"
```

Stage切换：

```sh
MODE=initialize INITIALIZE_FROM=/absolute/path/to/stage1/final_checkpoint \
  NPROC_PER_NODE=8 CONFIG="$PWD/project/configs/pretrain_stage2.yaml" \
  "$PWD/project/scripts/autodl_launch.sh"
```

launcher会在执行`torchrun`前验证release、数据回执、440个store、启动模式、global batch整除、完整optimizer-step预算、CUDA数量、BF16能力和空闲显存。任何失败都会在创建正式训练进程前退出。

## 结果解释

当前GPU loss仅证明管线能够真实forward/backward、更新、保存和恢复，不代表生物学性能。正式结论必须来自冻结评测矩阵，特别是cold-genus、low-homology（低同源）、leave-one-species-out（留一物种）和sealed formal test（密封正式测试）；禁止逐checkpoint反复查看formal test。
