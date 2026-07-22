# LegumeGenomeFM-HierMamba：超长上下文模型架构

> 状态：**CANDIDATE / PROFILING_REQUIRED（唯一候选，尚未冻结）**，最后更新：2026-07-22。正式最大上下文候选为262,144 bp；模型已建立代码与静态合同。按固定Mamba-2 commit和当前共享投影语义推导的候选参数量为314,669,504，但目标环境尚无生产kernel和H20实例化receipt，因此该值不得写成运行时已冻结参数量，峰值显存、吞吐和正式训练预算也仍为空。

## 1. 名称、定位与唯一性

唯一候选名称为**LegumeGenomeFM-HierMamba**。它是面向豆科多物种基因组的单碱基、双向、反向互补对称、层次化超长上下文基础模型。

当前不存在第二个并列正式模型：

- 旧LegumeGenomeFM-89M及1K/4K/16K三阶段方案为`superseded`；
- Hyena只作为冻结前的等参数/等token global-kernel比较器；
- stride消融和无RC消融只用于解释，不是备选正式模型；
- H20 spike若否决Mamba-2，必须先更新唯一候选、代码、配置和文档，再重新执行全部门禁，不能同时保留两套正式方案。

机器合同为`configs/pretrain_h20_candidate.yaml`，实现入口为`src/legumegenomefm/hiermamba.py`。

## 2. 为什么采用层次化单碱基架构

### 2.1 设计证据

Enformer、Borzoi和AlphaGenome表明，长DNA任务可以通过局部高分辨率表示、压缩的全局表示和细粒度decoder兼顾远距离上下文与碱基级输出。HyenaDNA和Evo 2证明单碱基长序列operator可扩展；Caduceus和PlantCaduceus证明双向与reverse complement（RC，反向互补）处理对DNA模型重要。Mamba-2提供线性序列复杂度的SSD核心，但论文与代码均未证明本项目在256K和H20上的实际可用性。

官方代码审计固定于：

- Caduceus commit `0060a6d8079b6a040fc55d505e15972a327b70a6`；
- Mamba commit `f577286d052741c35d39cd43bdc3fad27120f22c`；
- 证据文件`research/ultralong_architecture_evidence.json`。

### 2.2 淘汰路线

| 路线 | 淘汰或降级原因 |
|---|---|
| 256K全分辨率标准attention | 激活和成对交互为O(L²)，2–3张H20不具备可靠训练边界 |
| 旧膨胀卷积＋全局均值 | 只在16K验证，全局均值不是精确长距离交互 |
| 全分辨率Caduceus RCPS＋BiMamba直接叠加 | 每层可能形成四次全长scan，256K计算代价过高 |
| 纯HyenaDNA | 可作强比较器，但无现成精确RC合同；必须与Mamba-2同预算实测后决定 |
| k-mer/BPE tokenizer | 降低单碱基标签对齐并引入token边界；不适合统一剪接、变异和分割接口 |
| 分别训练六个长度模型 | 破坏共享表示和单一checkpoint lineage，不符合一个正式模型的要求 |

## 3. Tokenizer与输入语义

候选词表固定为7个token：

| ID | Token | 语义 | Complement |
|---:|---|---|---|
| 0 | PAD | 批内填充；正式定长窗口通常不用 | PAD |
| 1 | MASK | MLM掩码 | MASK |
| 2 | A | 腺嘌呤 | T |
| 3 | C | 胞嘧啶 | G |
| 4 | G | 鸟嘌呤 | C |
| 5 | T | 胸腺嘧啶 | A |
| 6 | N | 未知/歧义碱基 | N |

正式训练窗口只从clean ACGT连续区间采样。外部输入中的IUPAC歧义码映射为N，N不作为MLM监督标签。窗口不能跨染色体、contig、N/歧义区或污染mask。

输入形状为`input_ids: [B, L]`，dtype为`torch.long`，其中L必须：

- 在128至262,144之间；
- 被128整除；
- 正式训练时属于`[1,024, 8,192, 32,768, 65,536, 131,072, 262,144]`。

## 4. 总体结构

模型由四部分组成：

1. base-resolution local encoder（单碱基层局部编码器）；
2. 128-bp latent pyramid（层次化压缩）；
3. bidirectional Mamba-2 global core（双向全局核心）；
4. mirrored U-Net decoder（镜像解码器）与base-resolution MLM head。

候选张量路径：

| Scale | Stride | Channels | L=256K时长度 | Encoder blocks | Decoder blocks |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 128 | 262,144 | 2 | 2 |
| 1 | 2 | 192 | 131,072 | 2 | 2 |
| 2 | 4 | 256 | 65,536 | 2 | 2 |
| 3 | 8 | 384 | 32,768 | 2 | 2 |
| 4 | 16 | 512 | 16,384 | 2 | 2 |
| 5 | 32 | 768 | 8,192 | 2 | 2 |
| 6 | 64 | 1,024 | 4,096 | 2 | 2 |
| 7 | 128 | 1,024 | 2,048 | 2 | global core后回到decoder |

所有正式上下文经过相同网络。1K输入在global core中只有8个latent；256K输入为2,048个latent。模型没有依赖固定位置embedding，因此不会为每种长度建立独立参数。

## 5. Local encoder

### 5.1 Embedding

`nn.Embedding(7, 128)`将每个碱基映射为128维。输入embedding与最终LM head权重共享。

### 5.2 ConvNeXt1D block

每个scale有2个残差block：

1. depthwise Conv1d，kernel 7，same padding；
2. 转为`[B, L, C]`后执行RMSNorm；
3. pointwise Linear：`C → 4C`；
4. GELU(tanh approximation)；
5. pointwise Linear：`4C → C`；
6. 与block输入相加。

无dropout、无随机depth，保证RC双路在相同参数和确定性kernel下可比较。降采样使用`Conv1d(kernel=4, stride=2, padding=1)`，共7次，总stride为128。

## 6. Global Mamba-2 core

Global core作用于`[B, L/128, 1024]`，候选包含24个pre-norm residual block。每个block：

1. RMSNorm(1,024)；
2. 正向Mamba-2 scan；
3. 对序列反转后执行反向Mamba-2 scan，再反转输出；
4. 两路相加；
5. 与block输入残差相加。

每个方向的Mamba-2候选参数：

| 参数 | 值 |
|---|---:|
| d_model | 1,024 |
| d_state | 128 |
| d_conv | 4 |
| expand | 2 |
| headdim | 64 |
| ngroups | 8 |
| chunk_size | 256 |
| use_mem_eff_path | official fused path默认值，必须H20实测 |

正向/反向scan共享`in_proj`和`out_proj`权重；SSM动态、depthwise causal convolution、A/D/dt与gated RMSNorm独立。这样保留方向差异，同时避免重复最大投影矩阵。该绑定来自代码中的真实Parameter共享，精确参数统计必须按唯一Parameter对象去重。

当前环境缺少`mamba-ssm`和`causal-conv1d`。`_load_mamba2_class()`会在任何大模型参数分配前抛出明确错误；不允许用Identity、普通RNN或旧卷积静默替代生产core。

## 7. Mirrored U-Net decoder

Global core输出与encoder最深层skip相加，然后执行7次：

1. `ConvTranspose1d(kernel=4, stride=2, padding=1)`上采样；
2. 与同分辨率encoder skip按channel拼接；
3. `1×1 Conv1d(2C → C)`融合；
4. 执行与encoder同定义的2个ConvNeXt1D block。

每次上采样后都检查序列长度与skip严格相等；任何off-by-one直接报错。最终得到`[B, L, 128]`，RMSNorm后用共享embedding矩阵投影到7-token logits。

局部skip保留剪接位点、codon和短motif的单碱基精度；2,048-token global core负责最长256K范围的信息传播。模型不会把128-bp latent误称为128-bp输出：MLM输出始终是一碱基一个logit。

## 8. 精确reverse-complement合同

设完整backbone输出为`g(x)`，token互补置换为C，序列反转为R。最终logits定义为：

`f(x) = 0.5 × [g(x) + C⁻¹ R g(C R x)]`。

代码实现：

1. 正向输入运行完整encoder/global/decoder；
2. 对输入反转并执行A↔T、C↔G；
3. 同一完整模型运行RC输入；
4. RC logits在位置维反转、词表维按complement map重排；
5. 与正向logits求均值。

因此模型输出目标是`f(x)=RC_align(f(RC(x)))`。RC双路增加约2倍完整backbone计算，不增加模型参数。训练与推理使用相同语义，不能训练时关闭、推理时打开。

当前已验证：7-token complement map为involution（连续执行两次恢复原序列）。完整模型RC数值误差必须在生产Mamba-2装好后通过实际forward验证；当前不得沿用旧89M模型的`rc_max_abs_error=0`。

## 9. MLM输出与分布式loss接口

`HierMambaForMaskedLM.forward()`返回：

- `logits: [B, L, 7]`；
- `loss_sum`：本rank所有有效masked token的交叉熵总和；
- `masked_token_count`：本rank有效masked token数。

模型不在内部把本地mean loss伪装成全局loss。训练器必须跨rank all-reduce分子和分母，再计算：

`global_loss = global_loss_sum / global_masked_token_count`。

labels用`-100`表示忽略。候选span MLM为15% mask、平均span 3 bp，替换比例0.8 MASK / 0.1随机ACGT / 0.1原token。

## 10. 六长度连续训练

正式候选长度及每GPU microbatch：

| Context | Micro-batch/GPU | Input tokens/GPU/microstep | 候选token占比 |
|---:|---:|---:|---:|
| 1,024 | 256 | 262,144 | 10% |
| 8,192 | 32 | 262,144 | 15% |
| 32,768 | 8 | 262,144 | 20% |
| 65,536 | 4 | 262,144 | 20% |
| 131,072 | 2 | 262,144 | 20% |
| 262,144 | 1 | 262,144 | 15% |

rank 0只在optimizer-step边界抽取长度并广播；同一step的所有rank和microstep长度相同。比例按成功更新的token计，不按样本数。所有长度共享一套模型、optimizer、scheduler、RNG lineage和checkpoint。

上述microbatch是H20 profiling候选，不是已实测值。若显存不通过，只允许减microbatch并增加梯度累积；六长度集合、最大上下文和单模型语义不能静默改变。

## 11. 参数量与实现验收

候选配置已固定模块形状。代码可独立精确计数的embedding、local encoder/decoder、上下采样、skip fusion与norm为111,470,528个参数。按固定Mamba-2 commit、`d_model=1024`、`d_state=128`、`expand=2`、`ngroups=8`、`headdim=64`，并对forward/reverse scan共享`in_proj/out_proj`去重后，24层global mixer推导为203,198,976个参数；候选总量为**314,669,504（约314.67M）**。

该值是design-formula（设计公式）结果，不是H20运行时冻结值。生产Mamba-2依赖尚未安装，不能用手算或假module冒充真实`named_parameters()`与权重receipt。

冻结步骤必须是：

1. 在隔离H20环境安装固定commit/版本；
2. 实例化`HierMambaForMaskedLM`；
3. 检查正反scan共享Parameter对象；
4. 运行`unique_trainable_parameter_count()`；
5. 另按模块汇总，并验证模块总和等于唯一Parameter总数；
6. 将结果写入配置、本文和H20 receipt；
7. 三者hash一致后将`parameter_count`状态从design-formula改为runtime-frozen整数。

在这之前，preflight要求整数参数量并拒绝启动。

## 12. 计算复杂度

令碱基长度为L、latent stride为S=128、global width为D：

- local encoder/decoder的序列复杂度近似O(L)，但不同scale channel数不同；
- global Mamba-2序列复杂度近似O((L/S)D² + (L/S)D·N)，最长latent仅2,048；
- 双向global core约增加2倍global scan；
- 整模型RC conjoin约增加2倍完整forward；
- 不存在O(L²)的256K full attention矩阵。

层次化压缩节省的是深层全局计算，不代表256K免费。全分辨率local activation、U-Net skip、RC双路和反向传播仍必须在H20上实测。

## 13. Activation checkpoint与数值语义

候选对24个global block启用non-reentrant activation checkpointing。Local encoder/decoder是否进一步checkpoint由H20峰值决定，但不能改变前向数学结果。

正式精度为BF16；RMSNorm和Mamba内部A/dt的FP32处理遵循固定生产实现。必须检查：

- loss、logits和梯度finite；
- 1K至256K各完成真实optimizer step；
- 非有限更新不增加step/tokens_seen；
- RC两路和双向scan在BF16下数值稳定；
- checkpoint前后相同输入/状态输出在定义容差内一致。

## 14. H20机器合同

目标硬件是用户提供的2–3张NVIDIA H20、每张96 GB；尚未实机probe。候选合同要求：

- world size只能为2或3；
- 目标机记录GPU型号、卡数、总/空闲显存、compute capability、驱动、CUDA、PyTorch、Triton、NCCL和互联；
- 六长度均测forward/backward/optimizer、tokens/s、allocated/reserved峰值和step walltime；
- 256K峰值后保留至少10%显存；
- 2卡和3卡均通过长度广播、global masked-token loss和checkpoint/resume；
- Mamba-2与等参数Hyena在同token、相同精度和相同硬件上完成spike；
- 结果写入`data_manifests/h20_ultralong_profile.json`。

`preflight_training.py`同时要求：

1. `contract_status: frozen`；
2. 六长度静态合同闭合；
3. 参数量为整数；
4. Mamba backend为`h20_verified`；
5. kernel selection为`frozen`；
6. 六长度、2卡、3卡和显存余量receipt全部PASS；
7. 新精简data release READY。

当前每一条运行证据门禁都保持fail-closed，故正式训练不可启动。

## 15. RTX 2080与其他硬件边界

当前RTX 2080 Ti验证只属于已撤销旧模型，不能迁移成HierMamba证据。新模型尚未在RTX 2080上实例化；Mamba-2 production kernel是否支持compute capability 7.5也未核实。

允许在RTX 2080上用同一正式参数形状和较短输入做import/forward/backward/checkpoint smoke，但不得：

- 减少layer/channel构造小模型；
- 用不同global operator替换Mamba-2；
- 把短输入smoke写成256K验证；
- 把FP16结果当作H20 BF16吞吐。

## 16. 下游输出接口

候选模型应支持：

- base logits：剪接、起止密码子和per-base segmentation；
- 任意位置hidden states：区域分类；
- masked mean / attention-free pooling：基因和区域embedding；
- reference/alternate成对forward：variant score；
- 多窗口或整256K locus embedding：调控和长距离任务；
- frozen encoder与full fine-tune共用同一backbone。

正式接口在实现后必须固定tensor shape、padding/mask语义和RC聚合方式。下游head不进入预训练参数量。

## 17. 代码入口与测试状态

- 模型：`src/legumegenomefm/hiermamba.py`；
- 候选配置：`configs/pretrain_h20_candidate.yaml`；
- 启动门禁：`scripts/preflight_training.py`；
- 模型静态/RC测试：`tests/test_hiermamba.py`；
- 门禁测试：`tests/test_training_preflight.py`。

当前真实测试：候选shape合同、128-bp stride、2,048 latent上限、RC映射involution和缺生产kernel的前置拒绝均PASS。完整模型forward、精确参数量、H20显存、吞吐和DDP仍未开始。

## 18. 架构图与训练图

最终相对路径预注册为：

- `figures/main/model_architecture.png`和`.pdf`；
- `figures/main/mixed_context_schedule.png`和`.pdf`。

在精确参数量与H20 shape实测前不生成标记“FROZEN”的图，避免图中数字早于代码冻结。

## 19. 已知限制与停止条件

具体限制是：

1. 128-bp global latent可能损失小于128 bp的跨块相位信息，需通过skip和stride消融验证；
2. RC conjoin和双向global scan提高计算约4倍于单向单路global路径；
3. U-Net skip在256K占用大，可能成为显存主项而非Mamba；
4. Mamba-2 production stack与H20驱动/CUDA兼容性未知；
5. 256K上下文只有在下游存在足够长且泄漏安全的标签时才能证明科学价值；
6. 参数规模若导致正式预算不可执行，应在冻结前调整一次并重新审计，不能训练中途缩模。

如果256K在H20上无法以≥10%显存余量完成真实optimizer step，或端到端吞吐使预算不可执行，则停止冻结，不把128K结果冒充256K正式模型。