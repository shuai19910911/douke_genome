# 临时评估: LegumeGenomeFM 多输入长度训练策略

更新时间: 2026-06-08 20:59:00 CST

## Material Passport

- 类型: 临时技术评估文档
- 项目: LegumeGenomeFM 豆科基因组基础模型
- 目标: 评估本项目应采用多长度完全混合训练、单长度阶段训练，还是渐进式扩长加 replay
- 对照文档: `zuowu_genomemodel/TEMP_CONTEXT_LENGTH_STRATEGY_EVALUATION.md`
- 结论状态: 建议采用“32 kb 起步的渐进式扩长 + 每阶段短长度 replay + token-budget sampler”
- 注意: 本文档为临时评估，只用于当前决策记录；不改动正式 `PLAN.md`

## 1. 直接结论

本项目不建议把 `8 kb / 32 kb / 64 kb / 128 kb` 从训练第一步开始完全随机等比例混合。

也不建议每个阶段只训练一个固定长度，完全不看其他长度。

最适合当前 LegumeGenomeFM 的方案是:

```text
同一个 LegumeGenomeFM-330M
先以 32 kb 为主训练
再渐进扩展到 64 kb
最后做 128 kb long-context continue pretraining
每个阶段保留少量 8 kb / 32 kb replay
按 token 数控制比例，而不是按 batch 数控制比例
```

这是一个模型连续训练，不是训练 4 个独立模型。

## 2. 为什么不能直接照搬作物项目的 8K -> 64K -> 128K -> 256K

作物项目文档的核心判断是正确的: 不应所有长度一锅混，应该渐进扩长，并在长阶段保留短长度 replay。

但豆科项目与作物项目有三个关键差异:

1. 本项目当前正式窗口是 `8,192 / 32,768 / 65,536 / 131,072 bp`，没有把 `4K / 16K / 256K` 作为正式主窗口。
2. 本项目数据是 251 个结构注释 genome，约 300.5 Gb，并且采用 compact 过滤和区域加权，不是全基因组密集滑窗。
3. 本项目第一版目标是 LegumeGenomeFM-330M，在 2 张 A100 40G 上尽量训练，128 kb 已经是显存和吞吐压力较大的长上下文阶段。

因此，本项目不应从 8 kb 作为唯一主训练起点，也不应规划 256 kb 为第一版正式目标。

## 3. 推荐训练长度课程

### Stage A: 32 kb 主训练

目的:

- 学习完整 gene body、promoter 0-5 kb、UTR、局部 TE 和 intron 结构。
- 兼顾吞吐和生物学上下文，是本项目最有性价比的起始长度。

建议 token 组成:

```text
70% tokens: 32 kb
20% tokens: 8 kb
10% tokens: 64 kb warm-up
```

8 kb replay 用于强化 CDS、splice、start/stop、短 promoter motif 等高密度监督。

### Stage B: 64 kb 继续训练

目的:

- 扩展到长 intron、局部 gene cluster、promoter-gene 上下文、TE 邻域。
- 这是本项目最重要的长上下文阶段。

建议 token 组成:

```text
70% tokens: 64 kb
20% tokens: 32 kb
10% tokens: 8 kb
```

如果资源紧张，优先把 Stage B 做扎实，而不是过早进入 128 kb。

### Stage C: 128 kb long-context continue pretraining

目的:

- 学习更远端 TE/repeat、复杂 intergenic context、长基因簇和较长调控背景。
- 作为同一个模型的长上下文继续训练，不另起模型。

建议 token 组成:

```text
75% tokens: 128 kb
20% tokens: 64 kb
5% tokens: 8 kb / 32 kb high-value replay
```

128 kb 不作为第一阶段主训练长度。若 2 张 A100 40G 吞吐太低，可在 4 卡或更多 GPU 上执行 Stage C。

## 4. 与当前 PLAN token budget 的对应关系

当前计划中 Stage 1 token budget 为:

```text
8 kb: 20B tokens
32 kb: 50B tokens
64 kb: 40B tokens
128 kb: 20B tokens
Stage 1 total: 130B tokens
```

建议保持总 token 预算基本不变，但改变执行顺序:

| 阶段 | 主长度 | token 预算 | 建议组成 |
|---|---:|---:|---|
| Stage A | 32 kb | 45B-55B | 32 kb 为主，8 kb replay，少量 64 kb warm-up |
| Stage B | 64 kb | 40B-50B | 64 kb 为主，32 kb/8 kb replay |
| Stage C | 128 kb | 20B-30B | 128 kb 为主，64 kb replay，少量 8/32 kb |
| 合计 | mixed curriculum | 120B-135B | 与当前 130B 基本一致 |

这样比“全阶段完全混合”更稳定，也比“每阶段纯单长度”更不容易遗忘短程功能任务。

## 5. Batch 和 sampler 组织方式

推荐两级采样:

```text
1. 先抽 context_bucket: 8 kb / 32 kb / 64 kb / 128 kb
2. 再在该长度内抽 region_bucket: CDS / splice / UTR / promoter / intron / TE / intergenic / random genome
```

关键要求:

- 比例按 token 数统计，不按 batch 数统计。
- 同一个 micro-batch 内只放同一长度，减少 padding 和显存波动。
- 不同长度在 gradient accumulation 层面混合。
- 每个 context bucket 内继续使用区域权重，避免长窗口被 intergenic 背景稀释。
- validation 必须分别记录 `val_loss_8kb`、`val_loss_32kb`、`val_loss_64kb`、`val_loss_128kb`。

## 6. 为什么本项目 32 kb 比 8 kb 更适合作为起始主长度

8 kb 对 splice、CDS frame、start/stop 很有价值，但它覆盖完整基因和 promoter-gene 关系的能力不足。

豆科 genome 的预训练目标不是只做短 motif 或剪接位点，而是要形成可迁移的基因组区域表征。32 kb 能同时覆盖:

- 多数单基因主体或基因片段
- promoter 0-5 kb 与 TSS 邻域
- UTR、exon、intron 的组合结构
- 局部 TE/repeat 与 gene-proximal context

因此，32 kb 应是第一阶段主长度；8 kb 是高价值 replay 和局部监督强化长度。

## 7. 为什么不建议第一版加入 256 kb

256 kb 对长程结构可能有价值，但不适合作为 LegumeGenomeFM 第一版正式目标:

- 当前 compact 窗口没有把 256 kb 作为正式长度。
- 2 张 A100 40G 上 256 kb 会显著降低吞吐，梯度累积成本高。
- 豆科当前最明确的下游任务，如 gene structure、splice、promoter、TE-proximal variant prioritization，主要收益预计来自 8-128 kb。
- 256 kb 可以作为后续 LegumeGenomeFM-long 或 midtraining 扩展，而不是第一版必要条件。

## 8. 预期效果

相比所有长度完全混合:

- 训练更稳定，tokens/s 更可控。
- 早期不会被 128 kb 长背景稀释 CDS/splice/start-stop 监督。
- 更容易判断 32 kb 到 64 kb、64 kb 到 128 kb 是否真的带来收益。

相比纯单长度阶段:

- 长阶段不容易遗忘 8 kb 短程功能任务。
- 推理时面对不同窗口长度更平滑。
- 可以持续监控每个长度 bucket 的 loss 和下游 probe。

## 9. 进入下一阶段条件

Stage A -> Stage B:

- `val_loss_32kb` 达到平台或下降明显变慢。
- splice / CDS frame / start-stop probe 稳定。
- 64 kb warm-up loss 不发散。

Stage B -> Stage C:

- `val_loss_64kb` 持续下降后进入平台。
- promoter-gene、long intron、TE-proximal probe 优于 Stage A。
- `val_loss_8kb` 和短程 probe 没有明显退化。

是否继续扩大 128 kb token:

- 只有当 128 kb validation 和长程下游 probe 明确优于 64 kb 时才继续加大预算。
- 如果 128 kb 只降低 MLM loss，但不提升 TE/proximal/intergenic 相关任务，应停止扩大 128 kb。

## 10. 最终建议

本项目采用:

```text
LegumeGenomeFM-330M
Stage A: 32 kb 主训练 + 8 kb replay + 64 kb warm-up
Stage B: 64 kb 继续训练 + 32 kb/8 kb replay
Stage C: 128 kb long-context continue pretraining + 64 kb replay
```

不采用:

```text
8/32/64/128 kb 从第一步开始等比例完全混合
```

也不采用:

```text
每个阶段只训练一个长度，完全没有 replay
```

一句话结论: 对 LegumeGenomeFM，最佳训练方式是“同一模型、32 kb 起步、渐进扩长、短长度 replay、token-budget 控制”。

## 11. 参考

- 对照评估文档: https://github.com/shuai19910911/zuowu_genomemodel/blob/main/TEMP_CONTEXT_LENGTH_STRATEGY_EVALUATION.md
- HyenaDNA: long-range genomic sequence modeling. https://arxiv.org/abs/2306.15794
- Caduceus: bidirectional and reverse-complement equivariant DNA modeling. https://arxiv.org/abs/2403.03234
- FlashAttention 工程实践: https://github.com/Dao-AILab/flash-attention
