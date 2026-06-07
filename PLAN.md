# DoukeGenome 豆科基因组预训练大模型正式训练方案

更新时间：2026-06-07 22:56:09 CST

## 1. 项目目标

训练一个面向豆科（Fabaceae/Leguminosae）的 DNA foundation model。当前不使用 SNP/INDEL 变异矩阵，只使用已下载的豆科基因组序列及结构注释、功能注释、TE/repeat 注释等信息。

目标不是训练一个只适合大豆的模型，而是先训练豆科通用模型，再面向大豆和其他豆科作物任务继续适配。

## 2. 当前数据

当前可用于正式训练的数据：

```text
去冗余 assembly group: 526
可训练 genome: 493
覆盖属数: 69
genome QC: 493/493 ok
```

注释覆盖：

```text
结构注释: 251 个基因组
功能注释: 128 个基因组
TE/repeat 注释: 47 个基因组
三类注释齐全: 20 个基因组
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

## 3. 数据使用策略

正式策略：**保留全部 493 个基因组，但训练时做属级平衡采样。**

不建议直接按原始数量训练，因为 Glycine 和 Medicago 会主导模型；也不建议把每个属强行削到相同数量，因为小属会被重复过度采样。

训练 batch 来源：

```text
60% 属级平衡采样
30% 原始自然分布采样
10% 注释丰富基因组增强采样
```

属级权重：

```text
weight(genus) = 1 / sqrt(number_of_genomes_in_genus)
```

## 4. 文献依据

本方案参考以下前沿方向：

- Nucleotide Transformer：证明 DNA foundation model 可以从大规模基因组预训练迁移到多种基因组任务。
- DNABERT-2：证明多物种 DNA 预训练和高效 tokenizer 是有效基线，但上下文长度偏短。
- HyenaDNA：证明单碱基分辨率下可以扩展到很长 DNA 上下文。
- Caduceus：提出双向和反向互补等变的 DNA 长程序列建模。
- PlantCaduceus：证明植物跨物种 DNA 语言模型在有限标注下能提升功能区域预测。
- Evo 2：说明前沿 DNA foundation model 正在向单碱基、长上下文、跨物种和多尺度预测发展。

结论：DoukeGenome 应采用单碱基、长上下文、双向、反向互补一致性的架构，而不是只使用短上下文 BERT。

## 5. 正式模型架构

模型名称：DoukeGenome-330M

主干：RC-equivariant bidirectional MambaDNA backbone

核心配置：

```yaml
model_name: DoukeGenome-330M
architecture: bidirectional_rc_equivariant_mamba_dna
vocab_size: 9
tokens: [A, C, G, T, N, MASK, PAD, BOS, EOS]
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
target_parameters: approximately 330M
```

选择 330M 级别的原因：

- 优先考虑 2 张 A100 40G 可以启动和持续训练的配置。
- 100M 级模型对 493 个真核基因组偏小，容量不足。
- 300M 级模型是当前资源下较稳妥的正式大模型规模。
- 如果 180B token 和 128 kb 上下文阶段吞吐不足，优先增加 GPU，而不是改成非正式小模型。

## 6. 训练阶段

### Stage 1: genome-only 预训练

输入：493 个 QC=ok 的去冗余 genome FASTA。

上下文课程：

```text
8 kb -> 32 kb -> 64 kb -> 128 kb
```

目标函数：

```text
span masked nucleotide modeling
reverse-complement consistency
next-window contrastive learning
```

推荐 token budget：

```text
8 kb: 30B tokens
32 kb: 60B tokens
64 kb: 60B tokens
128 kb: 30B tokens
total: 180B tokens
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

推荐 token budget：

```text
20B-40B tokens
```

训练策略：先冻结底部 12 层训练任务头，再全模型继续训练。

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

## 7. GPU 训练参数

资源策略：

```yaml
preferred_start: 2 x A100 40G
scale_out_if_needed: 4-8 x A100 40G or equivalent
precision: bf16
parallelism: DDP or DeepSpeed ZeRO-2
activation_checkpointing: true
gradient_accumulation: true
```

优化器：

```yaml
optimizer: AdamW
peak_lr: 2.0e-4
min_lr: 2.0e-5
weight_decay: 0.1
betas: [0.9, 0.95]
warmup_ratio: 0.02
schedule: cosine
gradient_clip: 1.0
```

批量设置：

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

保存和验证：

```yaml
checkpoint_every_tokens: 1B
validation_every_tokens: 500M
keep_last_checkpoints: 5
keep_best_validation_checkpoints: 3
```

## 8. 评估设计

预训练评估：

```text
masked token accuracy
span recovery accuracy
validation loss by genus
reverse-complement consistency score
repeat-region perplexity
gene-region perplexity
low-representation genus validation loss
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

## 9. 数据切分

推荐：

```text
train: 80% assembly groups
validation: 10% assembly groups
test: 10% assembly groups
```

规则：

```text
genus-stratified split
Glycine 内部单独 holdout
小属优先保证进入 validation/test
同一 duplicate group 不跨 split
```

## 10. 预期结果

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
