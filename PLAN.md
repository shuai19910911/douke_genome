# DoukeGenome 豆科基因组预训练大模型正式训练方案

更新时间：2026-06-07 23:42:26 CST

## 1. 项目目标

训练一个面向豆科（Fabaceae/Leguminosae）的 DNA foundation model。当前不使用 SNP/INDEL 变异矩阵，只使用已下载的豆科基因组序列及结构注释、功能注释、TE/repeat 注释等信息。

目标不是训练一个只适合大豆的模型，而是先训练豆科通用模型，再面向大豆和其他豆科作物任务继续适配。

最终模型名称：DoukeGenome-330M。

## 2. 设计依据

本方案参考当前 DNA foundation model 的前沿方向：

- Nucleotide Transformer：大规模 DNA 预训练可迁移到多种基因组预测任务，模型规模覆盖 50M 到 2.5B 参数，并使用 3,202 个人类基因组和 850 个跨物种基因组。
- DNABERT-2：多物种 DNA 预训练和高效 tokenization 能降低计算成本，但 BERT 类模型上下文长度仍不适合本项目的长 intron、TE 和基因邻域建模。
- HyenaDNA：证明单碱基分辨率下可以把 DNA 上下文扩展到 1M token 级别，解决 Transformer 在长序列上的二次复杂度限制。
- Caduceus：提出双向、反向互补等变的长程 DNA 模型。DNA 不是单向文本，上下游序列和 reverse-complement 对称性必须进入架构设计。
- PlantCaduceus：证明跨物种植物 DNA 模型可在有限标注条件下提升植物功能区域预测，最贴近本项目的豆科跨属场景。
- Evo 2：代表最新一代单碱基、长上下文、跨生命域 genome model，说明前沿方向是长上下文、多尺度、可预测也可生成的 genome foundation model。
- 2025 年 DNA foundation model benchmark 结果提示：通用 DNA 模型并非所有任务都自动最优，尤其 gene expression、QTL/variant effect 等任务仍需要领域数据、任务头和专门微调。因此 DoukeGenome 必须设计豆科专属数据采样和下游验证。

结论：DoukeGenome 不采用短上下文 BERT 作为主干，正式主干采用“单碱基 token + 长上下文 + bidirectional SSM/MambaDNA + reverse-complement consistency”的路线。

参考来源：

- https://www.nature.com/articles/s41592-024-02523-z
- https://arxiv.org/abs/2306.15006
- https://papers.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html
- https://arxiv.org/abs/2403.03234
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12184517/
- https://www.nature.com/articles/s41586-026-10176-5
- https://www.nature.com/articles/s41467-025-65823-8

## 3. 当前数据

当前全局去冗余结果：

```text
去冗余 assembly group: 526
可训练 genome: 493
覆盖属数: 69
genome QC: 493/493 ok
可训练 genome 总长度: 554.1 Gb
平均 genome 长度: 1.12 Gb
```

注释覆盖：

```text
结构注释: 251 个基因组
功能注释: 128 个基因组
TE/repeat 注释: 47 个基因组
三类注释齐全: 20 个基因组
三类注释均无: 242 个基因组
```

主要属分布：

```text
Glycine 149
Medicago 72
Arachis 38
Vigna 38
Phaseolus 23
Trifolium 21
Cicer 15
Lathyrus 9
Vicia 9
Ormosia 8
Cajanus 7
Lupinus 7
```

主要属碱基规模：

```text
Glycine: 150.8 Gb
Medicago: 51.5 Gb
Arachis: 77.3 Gb
Vigna: 19.7 Gb
Phaseolus: 12.3 Gb
Trifolium: 13.6 Gb
Cicer: 7.7 Gb
Lathyrus: 40.7 Gb
Vicia: 67.3 Gb
```

## 4. 数据预处理总流程

正式训练前先把数据整理为三层产物：

```text
raw genome/annotation files
  -> clean genome and annotation index
  -> window-level training shards
  -> streaming dataloader batches
```

### 4.1 输入清单

输入索引来自本地：

```text
data/metadata/legume_family_nonredundant_assemblies.tsv
data/metadata/legume_family_nonredundant_files.tsv
data/metadata/legume_family_nonredundant_duplicate_groups.tsv
```

只使用：

```text
has_genome = yes
genome_qc_status = ok
```

每条 genome 记录需要保留：

```text
assembly_id
species
genus
source
accession
local_fasta_path
total_bp
n50
n_percent
annotation_paths
duplicate_group_id
```

### 4.2 FASTA 标准化

对 493 个 genome 执行：

```text
1. 解压或流式读取 FASTA。
2. header 标准化为 assembly_id|seq_id。
3. 碱基统一大写。
4. 非 A/C/G/T/N 字符转为 N。
5. 长度 < 1 kb 的 contig 默认不进入预训练。
6. 连续 N 超过 5 kb 的区域作为窗口切分边界。
7. 输出 clean FASTA 或 indexed FASTA。
8. 生成 .fai、sequence length table、checksum table。
```

推荐产物：

```text
data/processed/genomes/clean_fasta/
data/processed/genomes/fai/
data/processed/index/genome_sequences.tsv
```

预计资源和时间：

```text
输入规模: 约 554 Gb DNA bases，压缩文件约 100-250 GB 量级，视来源而定
CPU: 16-32 cores
内存: 64-128 GB
磁盘临时空间: 1.0-1.5 TB
预计时间: 6-18 小时
瓶颈: 解压和并行 IO
```

### 4.3 注释标准化

对有注释的 genome 执行：

```text
1. GFF3/GTF/BED 坐标统一为 0-based half-open 内部格式。
2. seqid 映射到标准化 FASTA header。
3. gene/mRNA/exon/CDS/UTR/intron 层级展开。
4. functional annotation 映射到 gene_id。
5. gene family 映射到 gene_id 或 protein_id。
6. TE/repeat 注释转为 per-base 或 interval label。
7. 坐标越界、孤立 transcript、缺失 parent 的记录单独标记，不直接用于监督。
```

输出标签类型：

```text
gene
mRNA
exon
intron
CDS
UTR
splice_donor
splice_acceptor
start_codon_window
stop_codon_window
repeat_or_TE
functional_terms
gene_family
```

预计资源和时间：

```text
CPU: 16-32 cores
内存: 64-128 GB
磁盘临时空间: 300-800 GB
预计时间: 8-24 小时
瓶颈: GFF3 层级解析、seqid 对齐、区间排序
```

### 4.4 数据切分

切分单位必须是 assembly group，不是窗口。避免同一 assembly 的窗口同时进入 train 和 test。

推荐：

```text
train: 80% assembly groups
validation: 10% assembly groups
test: 10% assembly groups
```

规则：

```text
1. duplicate group 不跨 split。
2. genus-stratified split。
3. Glycine 内部单独留出一批 accession 做 soybean holdout。
4. 小属优先保证至少进入 validation 或 test。
5. annotation-rich genome 在 train/validation/test 中均保留。
```

预计时间：

```text
CPU: 4-8 cores
内存: < 32 GB
预计时间: 0.5-2 小时
```

## 5. 训练样本构建

### 5.1 窗口策略

正式训练采用多尺度窗口：

```text
8,192 bp
32,768 bp
65,536 bp
131,072 bp
```

窗口生成：

```text
1. 对每条 chromosome/scaffold 按窗口长度切片。
2. train split 使用随机 offset 和随机 strand。
3. validation/test 使用固定 offset，保证可复现。
4. 窗口中 N 比例 > 20% 默认丢弃。
5. 窗口中有效 A/C/G/T 少于 70% 默认丢弃。
6. 对 annotation-aware 阶段，优先采样包含 gene、splice site、repeat、functional term 的窗口。
```

### 5.2 样本格式

每个训练样本包含：

```yaml
sample_id: string
assembly_id: string
species: string
genus: string
seq_id: string
start: int
end: int
strand: + or -
input_ids: uint8 array
attention_mask: bool array
mlm_mask: bool array
labels_mlm: int array
labels_region: optional uint8 array
labels_splice: optional uint8 array
labels_repeat: optional uint8 array
gene_ids: optional list
functional_terms: optional sparse list
gene_family_ids: optional list
```

单碱基 token：

```text
A=0
C=1
G=2
T=3
N=4
MASK=5
PAD=6
BOS=7
EOS=8
```

输入模型前的张量形态：

```text
input_ids: [batch, seq_len]
mlm_labels: [batch, seq_len]
region_labels: [batch, seq_len, num_region_labels]
sample_weight: [batch]
genus_id: [batch]
```

### 5.3 Mask 策略

DNA-only 阶段：

```text
mask ratio: 15%
span length mixture: 3, 6, 12, 24, 48, 96 bp
80% -> MASK token
10% -> random A/C/G/T/N
10% -> unchanged
```

补充增强：

```text
random reverse-complement: 50%
random shift for train windows
random N dropout: low probability, only for robustness
```

### 5.4 Shard 格式

推荐使用 WebDataset tar shards 或 mmap 二进制 shards：

```text
shard size: 1-4 GB
records per shard: 按 seq_len 决定
compression: zstd 可选；若 IO 充足，训练 shard 不压缩更快
index: shard_id, sample_count, token_count, genus histogram
```

预计窗口和 token 规模：

```text
单轮全基因组有效碱基: 约 554B bases
按 32 kb 无重叠窗口: 约 16.9M windows
按 64 kb 无重叠窗口: 约 8.45M windows
考虑过滤和多尺度采样后，训练不是穷举固定窗口，而是流式随机窗口采样。
正式 token budget: 180B DNA-only + 20B-40B annotation-aware
```

预处理资源：

```text
CPU: 32 cores 推荐
内存: 128 GB 推荐
磁盘: 2-4 TB 推荐
预计时间: 1-3 天
```

## 6. 数据采样策略

正式策略：保留全部 493 个基因组，但训练时做属级平衡采样。

不建议直接按原始数量训练，因为 Glycine、Medicago 等会主导模型；也不建议每属硬性等量，因为小属会被重复过度采样。

训练 batch 来源：

```text
60% genus-balanced genome sampling
30% natural genome-count sampling
10% annotation-rich genome sampling
```

属级权重：

```text
weight(genus) = 1 / sqrt(number_of_genomes_in_genus)
```

窗口级权重：

```text
base_weight = genus_weight
+ gene_window_bonus
+ splice_site_bonus
+ repeat_window_bonus
+ annotation_rich_bonus
```

约束：

```text
任何单属在一个 epoch 的 token 占比不超过 20%-25%
Glycine 在 Stage 1 不超过 25%
annotation-rich 采样只用于增强，不替代 genome-only 随机覆盖
```

## 7. 模型架构

模型名称：DoukeGenome-330M。

主干：RC-equivariant bidirectional MambaDNA。

### 7.1 为什么选择这个架构

豆科真核基因组包含长 intron、大量 repeat/TE、基因邻域效应和远距离调控。纯 Transformer 在 64 kb-128 kb 长度下计算成本过高；短上下文 BERT 会错过大量长程信息。

因此采用：

```text
Mamba/SSM: 长序列线性或近线性复杂度
Bidirectional: 同时建模上下游序列
RC-equivariant: 显式约束 DNA reverse-complement 对称性
Single nucleotide token: 保留单碱基分辨率
```

### 7.2 主配置

```yaml
model_name: DoukeGenome-330M
architecture: bidirectional_rc_equivariant_mamba_dna
vocab_size: 9
d_model: 1024
n_layers: 36
ssm_state_size: 128
expand_factor: 2
conv_kernel_size: 7
dropout: 0.05
embedding_dropout: 0.05
norm: rmsnorm
activation: silu
precision: bf16
reverse_complement_equivariance: true
bidirectional_context: true
max_context_length: 131072
target_parameters: approximately 300M-360M
```

### 7.3 模型输入到输出

```text
input_ids [B, L]
  -> nucleotide embedding [B, L, D]
  -> forward MambaDNA stream
  -> reverse-complement MambaDNA stream
  -> bidirectional fusion
  -> contextual representation [B, L, D]
  -> task heads
```

任务头：

```text
mlm_head: [B, L, 9]
region_head: [B, L, region_labels]
splice_head: [B, L, donor/acceptor]
repeat_head: [B, L, repeat_labels]
gene_pooling_head: [B, D] or [num_genes, D]
function_head: multi-label
gene_family_head: contrastive embedding
```

### 7.4 对照模型

正式项目至少保留以下对照，不作为主模型：

```text
DNABERT-2 embedding baseline
Nucleotide Transformer embedding baseline
PlantCaduceus/Caduceus-style checkpoint baseline if available
CNN or small Transformer supervised baseline for annotation tasks
```

## 8. 训练阶段

### Stage 0: 数据工程和样本构建

目标：生成可复现训练 shard。

输出：

```text
clean genome index
annotation interval index
train/val/test split
genome-only shards
annotation-aware shards
sampling weight tables
```

资源和时间：

```text
CPU: 32 cores
内存: 128 GB
磁盘: 2-4 TB
时间: 2-4 天
```

### Stage 1: genome-only 预训练

输入：493 个 QC=ok 的 genome FASTA 生成的随机窗口。

目标函数：

```text
span masked nucleotide modeling
reverse-complement consistency
next-window contrastive learning
```

token budget：

```text
8 kb: 30B tokens
32 kb: 60B tokens
64 kb: 60B tokens
128 kb: 30B tokens
total: 180B tokens
```

训练参数：

```yaml
optimizer: AdamW
peak_lr: 2.0e-4
min_lr: 2.0e-5
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_ratio: 0.02
schedule: cosine
gradient_clip: 1.0
precision: bf16
activation_checkpointing: true
```

批量建议：

```yaml
8192 bp:
  micro_batch_per_gpu: 16
  gradient_accumulation_steps: 16
32768 bp:
  micro_batch_per_gpu: 4
  gradient_accumulation_steps: 32
65536 bp:
  micro_batch_per_gpu: 2
  gradient_accumulation_steps: 48
131072 bp:
  micro_batch_per_gpu: 1
  gradient_accumulation_steps: 64
```

### Stage 2: annotation-aware 继续预训练

输入：有注释的 genome windows。

任务：

```text
gene/intergenic classification
exon/intron/CDS/UTR/repeat multi-label prediction
splice donor/acceptor prediction
start/stop codon neighborhood prediction
gene family contrastive learning
functional annotation multi-label learning
TE/repeat region prediction
```

token budget：

```text
20B-40B tokens
```

策略：

```text
1. 冻结底部 12 层，训练 annotation heads 1-2 epoch。
2. 解冻全模型，以较低学习率继续训练。
3. 每个 batch 中 genome-only loss 和 annotation loss 混合，避免注释偏向少数属。
```

### Stage 3: 豆科作物和大豆适配

重点任务：

```text
soybean promoter model
nodulation-related gene prioritization
domestication/selection region embedding
TE insertion effect representation
variant effect scoring with ref/alt windows
GWAS hit prioritization
```

## 9. GPU 资源和训练时间估算

下面是工程估算，不是承诺值。真实速度取决于实现、IO、Mamba kernel、checkpointing、shard 格式和集群通信。

### 9.1 推荐启动配置

```yaml
preferred_start: 2 x A100 40G
scale_out_if_needed: 4-8 x A100 40G or equivalent
precision: bf16
parallelism: DDP or DeepSpeed ZeRO-2
activation_checkpointing: true
gradient_accumulation: true
```

### 9.2 吞吐假设

对 300M-360M 级长上下文 SSM DNA 模型，保守估计：

```text
2 x A100 40G:
  8 kb: 120k-220k tokens/s
  32 kb: 80k-150k tokens/s
  64 kb: 45k-90k tokens/s
  128 kb: 20k-45k tokens/s

4 x A100 40G:
  约为 2 卡的 1.7-1.9 倍

8 x A100 40G:
  约为 2 卡的 3.2-3.6 倍
```

### 9.3 Stage 1 时间

按中位吞吐估算：

```text
2 x A100 40G:
  8 kb 30B: 2-4 天
  32 kb 60B: 5-9 天
  64 kb 60B: 8-15 天
  128 kb 30B: 8-18 天
  Stage 1 合计: 23-46 天

4 x A100 40G:
  Stage 1 合计: 13-27 天

8 x A100 40G:
  Stage 1 合计: 7-15 天
```

### 9.4 Stage 2 时间

```text
2 x A100 40G:
  20B-40B tokens: 5-14 天

4 x A100 40G:
  20B-40B tokens: 3-8 天

8 x A100 40G:
  20B-40B tokens: 2-5 天
```

### 9.5 Stage 3 时间

```text
任务数据准备: 2-7 天
单任务微调: 4-24 小时
多任务系统评估: 3-10 天
```

### 9.6 总时间

```text
2 x A100 40G:
  数据工程: 2-4 天
  Stage 1: 23-46 天
  Stage 2: 5-14 天
  Stage 3 初版评估: 5-14 天
  总计: 35-78 天

4 x A100 40G:
  总计: 23-49 天

8 x A100 40G:
  总计: 16-34 天
```

实际建议：先用 2 张 A100 启动 Stage 0 和 Stage 1 的 8 kb/32 kb 阶段。如果 32 kb 阶段稳定吞吐低于 80k tokens/s，建议扩到 4 卡；如果 64 kb/128 kb 阶段成为主要目标，建议 8 卡。

## 10. 存储和文件规模估算

```text
raw downloaded data: 100-250 GB 量级
clean FASTA/index: 600 GB-1.2 TB
annotation interval index: 100-500 GB
training shards: 1-3 TB
checkpoints:
  单个 bf16 330M checkpoint: 1-3 GB
  optimizer states with AdamW: 4-8 GB per checkpoint or more
  保留 8-10 个关键 checkpoint: 80-200 GB
logs/metrics: 10-50 GB
建议项目可用空间: 4-8 TB
```

## 11. 验证和监控

预训练监控：

```text
train loss
validation loss
masked token accuracy
span recovery accuracy
RC consistency score
next-window contrastive accuracy
genus-wise validation loss
low-representation genus validation loss
gene-region perplexity
repeat-region perplexity
```

下游评估：

```text
splice donor/acceptor AUROC and AUPRC
exon/intron/CDS/UTR per-base F1
promoter classification AUROC
TE/repeat prediction F1
gene family retrieval top-k accuracy
functional annotation multi-label mAP
cross-genus transfer performance
Glycine holdout performance
```

必须有的 sanity checks：

```text
1. train/validation/test assembly 不重叠。
2. duplicate group 不跨 split。
3. reverse-complement 输入输出一致性正常。
4. N-rich windows 没有主导 batch。
5. Glycine token 占比不超过设定上限。
6. annotation-aware 阶段没有被少数有完整注释的属支配。
```

## 12. 预期结果

正式预期：

- 获得一个 330M 级豆科 DNA foundation model。
- 在豆科基因结构、剪接位点、启动子、TE/repeat 任务上超过从零训练基线。
- 在低样本属上优于只用单作物训练的模型。
- 对大豆下游任务提供稳定的 ref/alt 序列表征。

需要实证验证：

- 是否优于通用植物模型。
- 是否优于 DNABERT-2/Nucleotide Transformer 直接迁移。
- 对无注释小属的泛化程度。
- 对真实育种性状的增益需要结合表型、表达、GWAS 或 QTL 数据评估。

## 13. 下一步执行顺序

```text
1. 生成 clean genome index。
2. 标准化 FASTA header、碱基和 .fai。
3. 标准化 GFF3/BED/functional/gene_family/repeat 注释。
4. 生成 train/validation/test split。
5. 生成 8 kb 和 32 kb genome-only shards。
6. 实现 genus-balanced streaming dataloader。
7. 实现 DoukeGenome-330M 配置和训练脚本。
8. 启动 Stage 1 8 kb/32 kb 正式训练。
9. 根据吞吐决定是否扩展到 4-8 卡。
10. 构建 annotation-aware shards 并进入 Stage 2。
```
