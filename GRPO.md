## 面向 AdaIR 的 GRPO 微调

这份文档记录了我们为 AdaIR 加入的最小化 GRPO 集成、使用方法，以及后续可演进的方向（如 LoRA、更多可训练门控等）。

### 做了哪些改动（文件与关键点）

- `net/model.py`
  - 在 `FreModule` 中引入低维随机策略：
    - 频域掩码阈值比例 `(r_h, r_w) ∈ (0,1)`：用 Beta 分布头 `policy_rate`，在 `fft(...)` 中进行采样。
    - `para1/para2` 融合标量：用 Beta 分布头 `policy_fuse`，在 `forward(...)` 中进行采样。
  - 当 `stochastic=True` 时，`FreModule` 返回带采样的特征，并将每次采样的对数概率收集到列表；`AdaIR.forward(..., stochastic=True)` 返回 `(restored, total_log_prob)`，其中 `total_log_prob` 是三处 `FreModule` 的 log-prob 之和。
  - 推理阶段保持确定性，并与 `stochastic=False`（默认）完全兼容。

- `train.py`
  - 在 `training_step` 中增加 GRPO 分支（由 `--grpo` 启用）。
  - 对每个 batch，进行组采样 G 次（`--grpo_group`），收集输出与 log-prob，计算逐样本奖励，组内标准化得到优势（advantage），并最小化 GRPO 代理目标：`-E[logp * advantage]`。
  - 增加稳定项：组内最优样本的 L1-to-GT 监督与“对确定性输出的一致性”约束（由 `--grpo_lambda_sup` 与 `--grpo_lambda_consistency` 控制权重）。
  - 奖励为多指标加权：PSNR（↑）、SSIM（↑）、(1 − LPIPS)（↑）、可选 NIQE（↑，默认关闭）；LPIPS 主干改为 `alex` 以降低开销。

- `options.py`
  - 新增参数：
    - `--grpo`：启用 GRPO 微调
    - `--grpo_group`：每个输入的组采样大小（默认 4）
    - `--grpo_lambda_sup`、`--grpo_lambda_consistency`：稳定项权重
    - 奖励权重：`--grpo_w_psnr`、`--grpo_w_ssim`、`--grpo_w_lpips`、`--grpo_w_niqe`

- `README.md`
  - 增加了 GRPO 的使用示例。

### 工作原理（高层概览）

1) 低维策略：把频域掩码的阈值比例与融合强度视作“动作”。用小型卷积头参数化其分布（Beta），在训练时采样；AdaIR 主干保持确定性。

2) GRPO 目标：对每个输入，采样 K 组动作（默认 4），计算奖励，做组内归一化形成优势，最小化 `-E[logp * advantage]`，同时加上小的稳定项。

3) 推理：`stochastic=False`（默认）。模型走确定性路径，输出与改造前一致。

### 使用方法

常规（监督）训练：

```
python train.py
```

GRPO 微调（最小配置）：

```
python train.py \
  --grpo \
  --grpo_group 4 \
  --grpo_lambda_sup 0.1 \
  --grpo_lambda_consistency 0.05
```

带 GT 的奖励权重：

```
python train.py \
  --grpo \
  --grpo_w_psnr 0.4 --grpo_w_ssim 0.3 --grpo_w_lpips 0.3 --grpo_w_niqe 0.0
```

备注：
- 如果出现显存不足（OOM），先将 `--grpo_group` 降为 2，或把 `--batch_size` 降为 1。
- 也可以临时把 `--grpo_w_lpips` 或 `--grpo_w_ssim` 设为 0 以降低度量开销，稳定后再恢复。

### 性能与显存建议

- 组大小 G 的显存开销近似线性增长。首次建议 `--grpo_group 2`。
- 尝试混合精度（bf16/fp16）与 TF32。可通过 Lightning 的 `precision` 或全局 `torch.set_float32_matmul_precision('high')`。
- 限制度量成本：仅对组内“最佳样本”计算 LPIPS/SSIM，其余样本用 MSE/PSNR 近似（后续可提供开关）。
- 初期尽量冻结主干（目前只训练小策略头并配合稳定项）。如需更强表达，再逐步解冻。

### 小样本 LoRA + 最差样本微调（实践推荐）

1) 基于训练集评测 CSV 选取各任务底部样本（例如 30%，且限制 [100, 5000]）：
```
python select_worst_from_csv.py --csv path/to/*.csv --out_dir AdaIR_results/train_eval/ --percent 0.30 --min_count 100 --max_count 5000
```

2) 仅用最差样本做 GRPO 微调，并注入 LoRA，冻结大部分主干：
```
python train.py \
  --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
  --grpo --grpo_group 2 --batch_size 1 \
  --finetune_worst \
  --worst_derain AdaIR_results/train_eval/train_derain_worst.txt \
  --worst_dehaze AdaIR_results/train_eval/train_dehaze_worst.txt \
  --worst_deblur AdaIR_results/train_eval/train_deblur_worst.txt \
  --worst_enhance AdaIR_results/train_eval/train_enhance_worst.txt \
  --worst_denoise AdaIR_results/train_eval/train_denoise_worst_merged.txt \
  --lora --lora_targets attn,cross_attn --lora_r 4 --lora_alpha 4 \
  --train_policy_only \
  --grpo_w_psnr 0.4 --grpo_w_ssim 0.3 --grpo_w_lpips 0.3
```

说明：
- 这一路线显存占用低，收敛快；先稳定提升尾部样本，再按需扩大 LoRA 覆盖或解除部分冻结。

### 实现要点

- Beta 采样与 log-prob 使用 1D 形状，避免在参数维上反复 `.squeeze().unsqueeze()` 引发维度错误。
- 三处 `FreModule` 的 log-prob 汇总为 `[B]` 向量。
- 奖励在“每个输入的组内”做标准化以形成优势。

### 未来改进方向

- LoRA 微调（选择性）：
  - 在部分解码/精修或注意力层上加 LoRA，只训练 LoRA 参数，保持显存低；与当前策略头联合使用。

- 更宽的动作空间（仍保持低维）：
  - 在若干解码块（如 `decoder_level1`、`refinement`）加入标量门控作为动作；继续用 GRPO 优化；由小到大逐步扩展。

- 奖励扩展：
  - 在训练中接入批处理优化的 NIQE，以及 MUSIQ/MANIQA 等感知指标；增加“仅对组内最佳样本计算重度指标”的开关。
  - 对预 GRPO 输出加入 KL/LPIPS 等正则（当前已有一致性 L1，可扩展到特征层）。

- 数据与训练日程：
  - 在无 GT 的真实域上做 GRPO，混入少量监督 batch 以稳定训练。
  - 渐进式解冻：先训练策略头，再轻量解冻末端解码/精修模块。

- 工程优化：
  - 提供 CLI 选择“仅对组内最佳样本计算 LPIPS/SSIM”。
  - 在训练器 CLI 中加入 bf16/TF32 开关与梯度累积配置。

### 依赖

- `lpips`（感知度量，默认使用 `alex` 主干）
- `lightning.pytorch`（训练流程）
- `scikit-image`（可选评测）、`torchvision` 等

### 常见问答（FAQ）

- 问：这会改变推理阶段的行为吗？
  - 答：不会。除非显式设置 `stochastic=True`，否则推理是确定性的。

- 问：为什么在低维动作上做 GRPO，而不是对整张图引入随机性？
  - 答：低维动作更加稳定、可解释，也更省显存，同时仍能让模型围绕下游奖励自适应地调整频域门控。


