# DoukeGenome 豆科结构注释驱动基因组预训练大模型正式训练方案

更新时间：2026-06-08 09:54:55 CST

## 1. 项目目标

训练一个面向豆科（Fabaceae/Leguminosae）的 DNA foundation model。当前阶段放弃没有结构注释的基因组，只使用同时满足以下条件的数据：

```text
has_genome = yes
genome_qc_status = ok
has_structural_annotation = yes
```

目标模型不是一个只会拟合大豆序列分布的 genome-only 模型，而是一个显式学习豆科基因结构、编码区、内含子、UTR、剪接位点、启动子邻域、TE/repeat 和跨属保守序列模式的结构注释驱动模型。

最终模型名称：DoukeGenome-330M。

## 2. 设计依据

当前 DNA foundation model 的前沿方向已经很清楚：

- Nucleotide Transformer 证明大规模 DNA 预训练可以迁移到多种基因组任务。
- DNABERT-2 是有效的多物种短/中上下文基线，但上下文长度不足以覆盖豆科长 intron、TE 邻域和基因上下游调控区域。
- HyenaDNA 证明单碱基分辨率可以扩展到超长 DNA 上下文。
- Caduceus 证明 DNA 模型应考虑双向上下文和 reverse-complement 等变性。
- PlantCaduceus 证明植物跨物种 DNA 模型在单碱基功能区域预测上有价值。
- Evo 2 代表最新长上下文 genome foundation model 方向。
- DNA foundation model benchmark 显示通用 DNA 模型并不会在所有任务上自动最优，尤其表达、QTL、变异效应等任务需要领域数据和专门任务头。

因此本项目采用：

```text
单碱基 token
长上下文
bidirectional Mamba/SSM backbone
reverse-complement consistency
结构注释多任务监督
豆科属级和区域级加权采样
```

参考来源：

- https://www.nature.com/articles/s41592-024-02523-z
- https://arxiv.org/abs/2306.15006
- https://papers.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html
- https://arxiv.org/abs/2403.03234
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11185591/
- https://www.nature.com/articles/s41586-026-10176-5
- https://www.nature.com/articles/s41467-025-65823-8

## 3. 正式训练数据

原始可用 genome 是 493 个，覆盖 69 个属。但正式训练计划放弃没有结构注释的基因组，保留 251 个结构注释可用基因组。

正式训练集统计：

```text
结构注释可用 genome: 251
覆盖属数: 29
总基因组长度: 300.5 Gb
平均 genome 长度: 1.20 Gb
其中有功能注释: 128
其中有 TE/repeat 注释: 47
结构+功能+TE/repeat 均有: 20
```

主要属分布：

```text
Glycine 116
Medicago 23
Arachis 21
Vigna 21
Phaseolus 13
Cicer 9
Lupinus 6
Trifolium 6
Cajanus 5
Vicia 5
Lathyrus 3
```

主要属碱基规模：

```text
Glycine: 117.9 Gb
Arachis: 47.1 Gb
Medicago: 12.6 Gb
Vigna: 10.2 Gb
Phaseolus: 7.0 Gb
Cicer: 5.1 Gb
Vicia: 49.0 Gb
Lathyrus: 13.6 Gb
```

放弃无结构注释 genome 的原因：

```text
1. 本项目目标不是单纯 genome language modeling，而是结构注释驱动的豆科模型。
2. 没有 GFF3/GTF/BED 的 genome 无法可靠提供 CDS、exon、intron、UTR、splice site 标签。
3. 保留无注释 genome 会稀释区域监督信号，使模型更偏向 k-mer 统计而不是可解释功能区域。
4. 251 个结构注释 genome 仍有约 300.5 Gb 序列，足够进行正式预训练。
```

## 4. 数据预处理

### 4.1 输入清单

输入来自本地索引：

```text
data/metadata/legume_family_nonredundant_assemblies.tsv
data/metadata/legume_family_nonredundant_files.tsv
data/metadata/legume_family_nonredundant_duplicate_groups.tsv
```

正式筛选条件：

```text
has_genome = yes
genome_qc_status = ok
has_structural_annotation = yes
```

每个 assembly 保留字段：

```text
assembly_id
species
genus
source
accession
local_fasta_path
gff3_or_gtf_path
cds_path
protein_path
repeat_annotation_path
duplicate_group_id
total_bp
n50
n_percent
```

### 4.2 FASTA 标准化

处理步骤：

```text
1. 解压或流式读取 FASTA。
2. header 标准化为 assembly_id|seq_id。
3. 碱基统一大写。
4. 非 A/C/G/T/N 字符转为 N。
5. 长度 < 1 kb 的 contig 不进入训练。
6. 连续 N >= 1 kb 的区域作为候选断点。
7. 连续 N >= 5 kb 的区域强制作为窗口切分边界。
8. 生成 .fai、sequence length table、checksum table。
```

### 4.3 N 比例阈值决策

原计划“窗口中 N 比例 > 20% 丢弃”过高。20% N 意味着 32 kb 窗口里最多 6.5 kb 是未知碱基，64 kb 窗口里最多 13 kb 是未知碱基，这会显著污染 masked nucleotide modeling 和结构区域监督。

正式阈值改为：

```text
train:
  N <= 5%: 正常使用
  5% < N <= 10%: 只允许小属或稀缺区域救援采样，sample_weight = 0.5
  N > 10%: 丢弃

validation/test:
  N <= 5%: 使用
  N > 5%: 丢弃
```

其他规则：

```text
窗口中任意连续 N >= 1 kb: 该窗口默认丢弃，除非是稀缺小属且不用于 validation/test。
窗口中有效 A/C/G/T < 90%: train 默认低权重或丢弃，validation/test 丢弃。
所有 splice site、CDS 边界、start/stop codon 监督窗口要求 N <= 2%。
```

结论：20% 对正式训练太宽，正式默认使用 5%，最多救援到 10%。

### 4.4 结构注释标准化

GFF3/GTF/BED 统一处理：

```text
1. 坐标统一为 0-based half-open 内部格式。
2. seqid 映射到标准 FASTA header。
3. 展开 gene、mRNA、exon、intron、CDS、UTR。
4. 从 transcript 结构推断 splice donor/acceptor。
5. 从 CDS 推断 start codon、stop codon 和 reading frame。
6. TE/repeat 注释转成 interval label。
7. 坐标越界、缺 parent、孤立 transcript、CDS 不成 3 倍数的记录标为低置信，不用于主监督。
```

输出标签：

```text
intergenic
promoter proximal
5UTR
CDS
intron
3UTR
splice_donor
splice_acceptor
start_codon_window
stop_codon_window
repeat_or_TE
```

## 5. 数据切分和泄漏控制

必须保证一个数据集中的基因组片段不会同时出现在训练集和其他集合。

正式切分单位：

```text
primary split unit: duplicate_group_id
secondary split unit: assembly_id
fallback split unit: chromosome/scaffold only when assembly-level split is impossible
```

默认规则：

```text
1. 同一个 duplicate_group_id 只能进入 train、validation、test 之一。
2. 同一个 assembly_id 只能进入 train、validation、test 之一。
3. 同一个 chromosome/scaffold 默认不跨 split。
4. train/validation/test 的 genomic interval 不允许重叠。
5. 如果因小属样本太少必须按 chromosome 切分，同一 chromosome 仍不跨 split。
6. 如果极端情况下必须在同一 chromosome 上切分，切分边界两侧各排除 2 x max_context_length，即 262,144 bp blackout region。
7. 任意窗口如果与其他 split 的窗口重叠超过 1 bp，直接丢弃。
8. 任意 gene_id、transcript_id、CDS interval、splice site 不跨 split。
```

推荐比例：

```text
train: 80%
validation: 10%
test: 10%
```

额外 holdout：

```text
Glycine holdout: 留出部分 Glycine accession，仅用于大豆泛化评估。
cross-genus holdout: 至少留出 Phaseolus/Vigna/Arachis/Medicago 中的若干 assembly 做跨属迁移评估。
small-genus holdout: 小属不参与过度随机拆分，优先完整 assembly 留出。
```

## 6. 训练样本和模型输入

### 6.1 窗口长度

正式多尺度窗口：

```text
8,192 bp
32,768 bp
65,536 bp
131,072 bp
```

训练阶段优先级：

```text
32 kb: 主力窗口，覆盖完整基因、启动子邻域和局部 TE 上下文。
64 kb: 长 intron、局部基因簇、TE 邻域和调控区域。
8 kb: 高密度监督，适合 splice/CDS/promoter 任务。
128 kb: 长程继续训练，不作为早期唯一主力。
```

### 6.2 样本字段

每个样本包含：

```yaml
sample_id: string
assembly_id: string
duplicate_group_id: string
species: string
genus: string
seq_id: string
start: int
end: int
strand: + or -
split: train | validation | test
input_ids: uint8 array
mlm_mask: bool array
mlm_labels: int array
region_labels: uint8 array
splice_labels: uint8 array
frame_labels: uint8 array
repeat_labels: optional uint8 array
gene_ids: optional list
sample_weight: float
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

模型输入张量：

```text
input_ids: [batch, seq_len]
mlm_labels: [batch, seq_len]
region_labels: [batch, seq_len, num_region_labels]
splice_labels: [batch, seq_len, 2]
frame_labels: [batch, seq_len, 3]
repeat_labels: [batch, seq_len, repeat_labels]
sample_weight: [batch]
genus_id: [batch]
```

## 7. 区域加权采样

由于本项目目标是结构注释驱动模型，不能让 intergenic 或 repeat-rich 背景区域淹没 CDS、剪接位点和 UTR。

训练 batch 的区域组成建议：

```text
CDS / coding exon centered windows: 25%
splice donor/acceptor centered windows: 15%
promoter/TSS upstream windows: 15%
UTR and transcript boundary windows: 10%
intron windows: 10%
TE/repeat windows: 10%
intergenic background windows: 10%
random genome coverage windows: 5%
```

如果某个基因组缺少 TE/repeat 注释，则 TE/repeat 配额只从有 repeat 注释的 47 个基因组中采样；缺失 repeat 注释的基因组不把 intergenic 区域伪标为 non-repeat。

loss 权重：

```text
masked nucleotide loss: 1.0
region label loss: 1.0
CDS/frame loss: 1.5
splice donor/acceptor loss: 2.0
start/stop codon loss: 1.5
promoter/TSS loss: 1.2
TE/repeat loss: 1.0
RC consistency loss: 0.2
next-window contrastive loss: 0.2
```

属级采样仍需平衡：

```text
weight(genus) = 1 / sqrt(number_of_structurally_annotated_genomes_in_genus)
```

约束：

```text
Glycine token 占比不超过 30%。
任一单属 token 占比不超过 30%。
小属可上采样，但同一窗口重复率不得超过 3 次/epoch。
CDS 和 splice 不能全部来自 Glycine，必须按属做最低覆盖约束。
```

## 8. 模型架构

模型名称：DoukeGenome-330M。

主干：RC-equivariant bidirectional MambaDNA。

核心配置：

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

模型流：

```text
input_ids [B, L]
  -> nucleotide embedding [B, L, D]
  -> forward MambaDNA stream
  -> reverse-complement MambaDNA stream
  -> RC-equivariant bidirectional fusion
  -> contextual representation [B, L, D]
  -> task heads
```

任务头：

```text
mlm_head: [B, L, 9]
region_head: [B, L, region_labels]
splice_head: [B, L, donor/acceptor]
frame_head: [B, L, frame0/frame1/frame2]
repeat_head: [B, L, repeat_labels]
gene_pooling_head: [num_genes, D]
variant_effect_head: pairwise ref/alt scoring
```

## 9. 预训练输入、输出和 loss

### 9.1 预训练输入

每个 batch 的主输入是结构注释基因组的窗口化 DNA 序列，不输入功能注释或 gene family 标签。

正式输入：

```yaml
input_ids:
  shape: [B, L]
  dtype: uint8 or int64
  content: A/C/G/T/N/MASK/PAD/BOS/EOS token ids

attention_mask:
  shape: [B, L]
  dtype: bool
  content: 1 for valid token, 0 for PAD

mlm_mask:
  shape: [B, L]
  dtype: bool
  content: positions selected for span masked nucleotide prediction

mlm_labels:
  shape: [B, L]
  dtype: int64
  content: original nucleotide token id at masked positions, ignore_index elsewhere

region_labels:
  shape: [B, L]
  dtype: int64 or multi-hot
  classes: intergenic, promoter_proximal, 5UTR, CDS, intron, 3UTR, repeat_or_TE, other

splice_labels:
  shape: [B, L, 2]
  dtype: binary
  channels: splice_donor, splice_acceptor

frame_labels:
  shape: [B, L]
  dtype: int64
  classes: frame0, frame1, frame2, non_CDS

start_stop_labels:
  shape: [B, L, 2]
  dtype: binary
  channels: start_codon_window, stop_codon_window

repeat_labels:
  shape: [B, L]
  dtype: binary or int64
  content: repeat/TE interval label where repeat annotation exists; ignore_index where unavailable

rc_input_ids:
  shape: [B, L]
  content: reverse-complement version of input_ids for RC consistency

next_window_input_ids:
  shape: [B, L]
  content: adjacent genomic window for contrastive learning

sample_weight:
  shape: [B]
  content: combined genus weight, region weight, N-content weight
```

### 9.2 模型架构

主干是 bidirectional RC-equivariant MambaDNA。

```text
input_ids [B, L]
  -> token embedding [B, L, 1024]
  -> forward Mamba stream, 36 layers
  -> reverse-complement Mamba stream, shared or tied RC parameters
  -> bidirectional + RC-equivariant fusion
  -> contextual representation H [B, L, 1024]
```

输出头：

```text
mlm_logits = Linear(H) -> [B, L, 9]
region_logits = Linear(H) -> [B, L, C_region]
splice_logits = Linear(H) -> [B, L, 2]
frame_logits = Linear(H) -> [B, L, 4]
start_stop_logits = Linear(H) -> [B, L, 2]
repeat_logits = Linear(H) -> [B, L, C_repeat]
window_embedding = mean_pool_or_attention_pool(H) -> [B, 1024]
```

不设置以下预训练头：

```text
functional annotation head
gene family contrastive head
protein/domain prediction head
```

### 9.3 Loss 定义

总 loss：

```text
L_total =
  1.00 * L_mlm
  + 1.00 * L_region
  + 1.50 * L_cds_frame
  + 2.00 * L_splice
  + 1.50 * L_start_stop
  + 1.20 * L_promoter
  + 1.00 * L_repeat
  + 0.20 * L_rc
  + 0.20 * L_next_window
```

各项定义：

```text
L_mlm:
  span masked nucleotide cross entropy，只在 mlm_mask=True 的位置计算。

L_region:
  per-base region classification loss。单标签区域用 cross entropy，多标签重叠区域用 BCE。

L_cds_frame:
  CDS 区域 reading frame 分类 loss。只在 CDS 或 CDS 邻域有效位置计算。

L_splice:
  splice donor/acceptor binary focal loss 或 weighted BCE。正例极少，因此 donor/acceptor 正例权重大于背景。

L_start_stop:
  start/stop codon window binary loss。只在蛋白编码 transcript 的 start/stop 邻域和 hard-negative 区域计算。

L_promoter:
  promoter/TSS upstream window classification loss。hard negative 来自远端 intergenic、intron 和 non-promoter upstream 区域。

L_repeat:
  TE/repeat interval loss。只在有 repeat 注释的 genome/window 上计算；无 repeat 注释的窗口使用 ignore_index，不当作 non-repeat。

L_rc:
  reverse-complement consistency loss。约束原窗口 embedding 和 RC 窗口 embedding 在方向校正后接近。

L_next_window:
  adjacent-window contrastive loss。相邻窗口为正样本，同 batch 其他 genome/远距离窗口为负样本。
```

有效 loss mask：

```text
PAD 位置不计算任何 loss。
N 比例超阈值的窗口不进入 validation/test。
无 repeat 注释的窗口不计算 L_repeat。
非 CDS 区域不计算 frame 分类 loss。
没有可靠 transcript 结构的基因不计算 splice/start/stop loss。
```

### 9.4 预训练 batch 组成

每个 batch 按区域来源混合：

```text
CDS / coding exon centered windows: 25%
splice donor/acceptor centered windows: 15%
promoter/TSS upstream windows: 15%
UTR and transcript boundary windows: 10%
intron windows: 10%
TE/repeat windows: 10%
intergenic background windows: 10%
random genome coverage windows: 5%
```

每个 batch 同时满足：

```text
genus-balanced sampling
single-genus token cap <= 30%
N-content train default <= 5%
validation/test N-content <= 5%
duplicate group and assembly split isolation
```

## 10. 训练阶段

### Stage 0: 数据工程

输出：

```text
structural-annotation-only genome manifest
clean FASTA index
annotation interval index
leakage-safe split table
region-weight table
genome shards
annotation-aware shards
```

资源和时间：

```text
CPU: 32 cores
内存: 128 GB
磁盘: 2-4 TB
时间: 2-4 天
```

### Stage 1: 结构注释驱动预训练

输入：251 个结构注释 genome 的加权窗口。

目标：

```text
span masked nucleotide modeling
region label prediction
CDS/frame prediction
splice donor/acceptor prediction
promoter/TSS neighborhood prediction
reverse-complement consistency
next-window contrastive learning
```

token budget：

```text
8 kb: 20B tokens
32 kb: 50B tokens
64 kb: 40B tokens
128 kb: 20B tokens
Stage 1 total: 130B tokens
```

### Stage 2: 下游任务微调和系统评估

见第 11 节。

## 11. 下游任务设计

### 11.1 基因结构预测

任务：

```text
per-base gene/intergenic/exon/intron/CDS/UTR classification
gene boundary detection
transcript boundary detection
```

评估：

```text
per-base F1
gene-level F1
boundary F1 within 10/50/100 bp
cross-genus holdout performance
```

预计优势：

```text
相比 CNN/small Transformer: 长上下文和预训练带来明显提升。
相比 DNABERT-2: 单碱基长上下文更适合 intron/exon 边界和长基因。
相比通用 Nucleotide Transformer: 豆科结构注释预训练应在豆科 holdout 上更好。
```

最可能优于基线的指标：

```text
exon/intron/CDS per-base F1
gene boundary F1
cross-genus gene structure F1
```

### 11.2 剪接位点预测

任务：

```text
splice donor prediction
splice acceptor prediction
canonical and non-canonical splice site scoring
```

评估：

```text
AUROC
AUPRC
top-k splice site retrieval
false positive rate in intronic background
```

预计优势：

```text
相比 k-mer/SVM/CNN: 可利用上下游长上下文和 transcript 结构。
相比 DNABERT-2: 不受短上下文和 k-mer 边界限制。
相比通用模型: 豆科内含子、外显子长度分布和剪接上下文更贴近训练域。
```

最可能优于基线的指标：

```text
splice donor AUPRC
splice acceptor AUPRC
低样本属 splice site transfer
```

### 11.3 CDS、reading frame 和 start/stop codon 预测

任务：

```text
CDS per-base classification
reading frame classification
start codon neighborhood scoring
stop codon neighborhood scoring
pseudo-CDS filtering
```

评估：

```text
CDS F1
frame accuracy
start/stop codon AUPRC
protein-coding transcript consistency
```

预计优势：

```text
相比 genome-only 模型: 明确见过 CDS/frame 监督。
相比通用 DNA LM: 豆科编码区 codon usage 和基因结构更匹配。
```

最可能优于基线的指标：

```text
frame accuracy
CDS boundary F1
start/stop codon AUPRC
```

### 11.4 启动子和 TSS 邻域预测

任务：

```text
TSS upstream 2 kb/5 kb/10 kb promoter classification
core promoter candidate ranking
gene-proximal regulatory window embedding
```

评估：

```text
AUROC
AUPRC
promoter vs intergenic hard-negative discrimination
cross-genus promoter transfer
```

预计优势：

```text
相比短上下文模型: 32-64 kb 表征能同时覆盖 promoter、UTR、gene body 和邻近 TE。
相比通用模型: 豆科启动子 motif、GC 分布、TE 邻域更贴近训练数据。
```

最可能优于基线的指标：

```text
promoter hard-negative AUPRC
cross-genus promoter AUROC
```

### 11.5 TE/repeat 区域识别和 TE 邻域效应表征

任务：

```text
TE/repeat interval prediction
TE insertion neighborhood representation
TE-proximal gene window scoring
```

评估：

```text
TE/repeat per-base F1
TE boundary F1
TE-proximal promoter classification
```

预计优势：

```text
相比 DNABERT-2 和短 CNN: 长窗口更适合重复序列和 TE 边界。
相比通用模型: 豆科 TE 组成和扩增历史更贴近本项目训练集。
```

注意：只有 47 个 genome 有 repeat 注释，因此该任务预计能优于短上下文基线，但泛化范围需要谨慎验证。

### 11.6 大豆变异效应和 GWAS hit prioritization

任务：

```text
ref/alt window embedding difference
coding variant effect scoring
splice-proximal variant scoring
promoter variant scoring
GWAS/QTL candidate interval prioritization
```

评估：

```text
known causal/putative causal variant ranking
GWAS peak gene prioritization
coding vs noncoding variant separation
splice-disrupting variant AUPRC
```

预计优势：

```text
相比通用 DNA LM: 对 Glycine 和近缘豆科结构区域更贴近。
相比只用 SNP matrix 的统计模型: 能解释变异所在序列上下文和基因结构。
相比短上下文模型: 能覆盖 variant 与 splice/promoter/TE/gene body 的长程关系。
```

需要谨慎：真实农艺性状预测仍需要表型、GWAS、QTL 或表达数据，DoukeGenome 主要提供候选区域和候选变异的功能先验。

## 12. 基线模型和预期优势

基线：

```text
DNABERT-2
Nucleotide Transformer
HyenaDNA
Caduceus / PlantCaduceus if checkpoint is available
CNN supervised baseline
small Transformer supervised baseline
gene annotation tool baseline when applicable
```

DoukeGenome 预计最有优势的地方：

```text
豆科 gene structure prediction
豆科 splice donor/acceptor prediction
CDS/frame/start/stop codon prediction
cross-genus low-label transfer
promoter hard-negative classification
TE/repeat boundary prediction within annotated repeat subset
Glycine and related legumes variant prioritization with structural context
```

不保证优于基线的地方：

```text
非豆科物种零样本泛化
没有表型数据的复杂农艺性状直接预测
表达量精确预测
长距离染色质互作预测
repeat 注释极少属的 TE 亚家族分类
```

预期结果表达方式：

```text
主张“预计在豆科结构注释相关任务上优于通用 DNA LM 和短上下文基线”。
不提前承诺所有任务全面领先。
所有优势必须通过 holdout assembly、holdout genus 和外部基线比较验证。
```

## 13. 资源和时间估算

由于正式训练集从 493 个 genome 调整为 251 个结构注释 genome，并去掉功能注释、gene family 和独立 TE/repeat 子集继续预训练，总 token budget 调整为结构注释驱动 Stage 1 的 130B tokens。

数据工程：

```text
CPU: 32 cores
内存: 128 GB
磁盘: 2-4 TB
时间: 2-4 天
```

训练时间估算：

```text
2 x A100 40G:
  Stage 1 130B tokens: 18-36 天
  Stage 2 下游微调和评估: 5-14 天
  总计: 25-54 天

4 x A100 40G:
  总计: 16-33 天

8 x A100 40G:
  总计: 10-23 天
```

扩卡规则：

```text
如果 32 kb 阶段稳定吞吐 < 80k tokens/s，建议扩到 4 卡。
如果 64 kb/128 kb 是主要目标，建议 8 卡。
不通过降低到非正式小模型来解决吞吐问题。
```

### 13.1 跨服务器搬运所需磁盘空间

如果在本服务器完成数据处理，再把处理好的数据搬到其他服务器训练，建议不要搬运所有中间缓存。正式推荐是搬运“clean FASTA + annotation interval index + split table + region sampling index + 训练 shard”，mask、RC 输入、next-window pair 和多数 loss mask 在训练时在线生成。

当前正式训练集为 251 个结构注释 genome，总长度约 300.5 Gb。按单碱基 token 估算，1 byte/base 的主序列至少约 300 GB；加上 FASTA header、索引、注释 interval、窗口索引、多尺度 shard 和文件系统开销后，实际需要更高空间。

三档估算：

```text
最低可训练搬运包: 0.8-1.2 TB
  内容:
    clean FASTA 或 compact 2-bit/uint8 sequence store
    .fai / checksum / sequence length index
    structural annotation interval index
    train/validation/test split table
    region sampling table
    genus/assembly metadata
  特点:
    不搬完整预生成窗口 shard
    训练服务器在线随机取窗口、在线 mask、在线生成 RC 和 next-window pair
  适用:
    训练服务器 CPU 和 IO 足够，追求搬运体积最小

推荐搬运包: 1.5-2.5 TB
  内容:
    最低可训练搬运包
    8 kb / 32 kb / 64 kb 主力窗口 shard
    validation/test 固定窗口 shard
    区域加权采样索引
  特点:
    训练启动快，复现实验方便
    128 kb 可在训练服务器在线生成或单独补充
  适用:
    推荐方案

完整处理缓存搬运包: 3-5 TB
  内容:
    clean FASTA
    所有多尺度窗口 shard: 8 kb / 32 kb / 64 kb / 128 kb
    annotation-aware labels
    validation/test 固定 shard
    中间 interval cache
    QC 报告和统计表
  特点:
    最省训练服务器预处理时间
    搬运时间和目标服务器磁盘压力最大
  适用:
    目标服务器 IO 较弱，或需要完全复现本服务器处理结果
```

不建议搬运：

```text
每一步临时解压文件
重复 FASTA 副本
每个 epoch 的动态 mask 后样本
每个窗口的 rc_input_ids 实体副本
所有 next-window pair 的实体副本
完整 optimizer checkpoint 历史
```

训练服务器建议预留：

```text
只训练不长期保存中间产物: 至少 2 TB 可用空间
推荐稳定训练: 4 TB 可用空间
完整缓存 + 多 checkpoint: 6-8 TB 可用空间
```

搬运时间粗估：

```text
1 Gbps 网络:
  1 TB 约 2.5-3.5 小时理论值，实际常见 4-8 小时
  2.5 TB 常见 10-20 小时

10 Gbps 网络:
  1 TB 常见 0.5-1.5 小时
  2.5 TB 常见 1.5-4 小时

普通机械硬盘或共享文件系统会显著拖慢，实际以 rsync/sha256 校验速度为准。
```

最终建议：优先准备 **1.5-2.5 TB 推荐搬运包**。这样训练服务器不需要重新做完整预处理，同时不会把所有临时缓存和动态样本都搬过去。

## 14. 下一步执行顺序

```text
1. 从非冗余索引生成 structural-annotation-only manifest。
2. 标准化 251 个 genome 的 FASTA 和结构注释。
3. 生成 leakage-safe split table，确保 duplicate group、assembly、interval、gene 不跨 split。
4. 将 N 阈值从 20% 改为正式 5%，最多训练救援到 10%。
5. 生成区域权重表：CDS、splice、promoter、UTR、intron、repeat、intergenic。
6. 生成 8 kb/32 kb/64 kb/128 kb 多尺度 shards。
7. 实现 region-weighted + genus-balanced streaming dataloader。
8. 准备 DoukeGenome-330M 训练配置。
9. 启动 Stage 1 结构注释驱动预训练。
10. 完成 Stage 2 下游任务微调和基线系统评估。
```
