# LegumeGenomeFM-89M：正式模型架构

> 冻结状态：**FROZEN（已冻结）**，2026-07-19。正式实现只有这一套模型语义；GPU数量只改变DDP（数据并行）和梯度累积，不改变参数、tokenizer（词元编码器）、目标或样本分布。

## 1. 定位与冻结结论

LegumeGenomeFM-89M是面向豆科多物种基因组的单碱基基础模型。核心设计是：

1. 以共享参数覆盖1,024、4,096和16,384 bp三个尺度；
2. 使用线性复杂度的分层膨胀卷积混合器，避免标准attention（二次复杂度）在16 kb上的显存开销；
3. 在模型输出层实施精确reverse-complement symmetry（反向互补对称），而不是只靠随机RC增强；
4. 用cold-genus（整属留出）、全局近重复组和材料版本组约束后续评测泄漏。

正式定义位于`src/legumegenomefm/model.py`、`src/legumegenomefm/tokenizer.py`和`configs/pretrain_stage{1,2,3}.yaml`。不得以“更大模型”“换tokenizer”或“临时attention层”修改正式主结果；任何替代只能作为预注册消融。

## 2. Tokenizer与输入语义

- 粒度：单碱基，一个token对应一个输入碱基，不使用k-mer切分。
- 词表：17个token，即`PAD`、`MASK`、`A/C/G/T/N/R/Y/S/W/K/M/B/D/H/V`。
- 互补关系：A↔T、C↔G、R↔Y、K↔M、B↔V、D↔H，N/S/W保持自身。
- 正式预训练窗口只从2-bit store（两比特压缩序列库）的ACGT callable intervals（可采样区间）抽取；含模糊IUPAC字符的位置不会被伪装成A进入训练标签。完整IUPAC词表保留给外部推理与下游接口。
- contig（染色体/序列片段）之间不拼接；窗口不能跨contig边界。
- MLM（掩码语言模型）目标只在被掩码位置计算交叉熵。

## 3. 主干结构

| 字段 | 正式值 |
|---|---:|
| 模型类 | `ReverseComplementGenomeModel` |
| 词表 | 17 |
| hidden size `d_model` | 640 |
| block数量 | 18 |
| HierarchicalMixer卷积核 | 7 |
| dilation（膨胀率） | 1、4、16、64 |
| FFN倍数 | 3 |
| dropout | 0.0 |
| 参数共享 | 输入embedding与输出head权重绑定 |
| trainable parameters（可训练参数） | **88,946,028** |
| 最大正式上下文 | **16,384 bp** |

每个block按以下顺序工作：

1. RMSNorm（均方根归一化）；
2. 线性投影为content与gate两路；
3. 四个depthwise convolution（逐通道卷积）分支，dilation为1/4/16/64；
4. 局部平均池化分支和全序列均值上下文分支；
5. 学习到的softmax权重融合六个尺度，SiLU gate控制信息流；
6. 输出投影和残差连接；
7. 第二个RMSNorm、SwiGLU式FFN和残差连接。

主干时间与激活复杂度随序列长度近似`O(L)`增长。全局均值分支提供整窗条件，但它不是精确长距离pairwise interaction（成对交互）；因此模型的正式能力边界是16 kb窗口，不宣称已经建模Mb级染色体互作。

## 4. 精确反向互补对称

模型对输入`x`和其reverse complement `RC(x)`使用同一主干：

1. 分别计算两路logits；
2. 将RC路在位置维反转，并在词表维执行互补映射；
3. 与正向logits取平均。

因此输出满足`f(x) = RC_align(f(RC(x)))`。RTX 2080 Ti在1 kb、4 kb和16 kb三种真实forward/backward/optimizer测试中的`rc_max_abs_error`均为`0.0`。代价是主干计算约翻倍；这一成本已包含显存和吞吐验证，不允许训练时关闭、推理时再临时打开。

## 5. 多尺度共享参数

三个stage使用同一参数形状：

| Stage | 上下文 | micro-batch/GPU | global batch tokens | token预算 | 目的 |
|---|---:|---:|---:|---:|---|
| 1 | 1,024 | 8 | 262,144 | 34,999,894,016 | 局部motif、剪接和编码语法 |
| 2 | 4,096 | 2 | 524,288 | 34,999,894,016 | 启动子、外显子/内含子和基因局部结构 |
| 3 | 16,384 | 1 | 524,288 | 29,999,759,360 | 长基因结构与远距离窗口信息 |

总预算为**99,999,547,392 tokens（约100B）**。Stage 2从Stage 1最终model state初始化，Stage 3从Stage 2初始化；每个stage重新建立optimizer与scheduler，避免把不同长度阶段的优化器动量误当作无缝resume。

## 6. 训练目标与优化器

- 目标：span MLM，掩码比例0.15，平均span长度3。
- Optimizer：AdamW，`betas=(0.9, 0.95)`，`weight_decay=0.1`。
- 学习率：Stage 1为`3e-4`，Stage 2为`2e-4`，Stage 3为`1e-4`。
- Scheduler：2% linear warmup（线性预热）后cosine decay（余弦衰减），最小学习率比例0.1。
- 梯度裁剪：global norm 1.0。
- 正式精度：BF16；RTX 2080 Ti兼容/验证路径使用FP16动态loss scaling（损失缩放），两者不改变模型数学接口。
- activation checkpointing（激活重计算）：开启。

## 7. 已完成的真实硬件验证

机器证据汇总：`data_manifests/gpu_validation.summary.json`。

- RTX 2080 Ti（11 GiB）完成1 kb、4 kb、16 kb的真实forward/backward/optimizer步骤；
- 16 kb峰值allocated显存3,490,339,840 bytes，峰值reserved显存3,680,501,760 bytes；
- 16 kb在25%显存配额下按预期触发OOM边界，在35%配额下PASS；
- 单GPU checkpoint从step 1/1,024 tokens恢复到step 2/2,048 tokens，model、optimizer、scheduler、scaler、RNG和采样器状态均闭合；
- 双RTX 2080 Ti DDP在两个独立物理GPU上完成一步，`world_size=2`、`tokens_seen=2,048`；
- FP16 overflow（溢出）会重试且不计入optimizer step；
- **A100未用于上述验证，也不把A100吞吐写成实测值。**

## 8. 不允许静默改变的合同

以下任一变化都会形成新模型，而不是同一模型的工程调参：词表、层数、hidden size、卷积分支/dilation、RC平均语义、最大上下文、MLM定义、三个stage的token预算或冻结数据manifest。GPU数量、梯度累积步数和CUDA_VISIBLE_DEVICES映射可在保持global batch tokens整除时改变。

当前已知边界：架构未宣称具备标准attention的精确任意位置交互；正式最大输入为16,384 bp；验证loss来自管线smoke，不是生物学性能结论；正式效果必须由`configs/evaluation_matrix.yaml`中的冻结评测合同决定。