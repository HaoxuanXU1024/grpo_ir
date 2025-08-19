# GRPO 训练改进指南

## 🚀 改进概述

基于你的训练问题分析，我们对 GRPO 实现进行了全面改进，解决了以下核心问题：

### 原始问题
- **loss_pg 长时间为正且放大** → 政策梯度损失不稳定
- **loss_sup 长期不降** → 监督损失权重不当  
- **loss_cons 逐步走高** → 一致性约束过强导致特征崩塌

## 🔧 核心改进

### 1. PPO式政策损失 (train.py)
**原来**：简单 REINFORCE `-(logps * advantages).mean()`
**现在**：PPO clipped 损失，更稳定的政策更新
```python
def compute_policy_loss_ppo(self, log_probs, old_log_probs, advantages, clip_range=0.2):
    ratio = torch.exp(log_probs - old_log_probs.detach())
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

### 2. 全局奖励统计归一化
**原来**：仅批次内归一化，方差大
**现在**：可选全局统计归一化，使用指数移动平均
```python
# 全局统计更新
self.running_reward_mean = (1-α) * old_mean + α * batch_mean
self.running_reward_var = (1-α) * old_var + α * batch_var
advantages = (rewards - global_mean) / (global_std + ε)
```

### 3. 分阶段训练策略
- **阶段1 (5 epochs)**：纯监督预训练，稳定基础
- **阶段2 (20 epochs)**：渐进引入 GRPO，监督权重递减
- **阶段3 (25 epochs)**：完整 GRPO，精细调优

### 4. 梯度裁剪与正则化
- 梯度范数裁剪：防止梯度爆炸
- KL 正则化：约束策略偏移
- 优势函数裁剪：防止极端值

### 5. 自适应学习率
- GRPO 训练：使用更低学习率 (5e-6)
- 延长预热期：更平滑的训练开始
- 权重衰减：提高泛化能力

## 📊 新增监控指标

训练过程中可在 wandb 观察：
- `advantages_mean/std`：优势函数分布
- `rewards_mean/std`：奖励统计
- `lambda_sup/pg`：动态权重
- `global_reward_mean/std`：全局统计
- `loss_kl`：KL 正则化损失

## 🎯 使用方法

### 推荐方式：分阶段训练 (最稳定)
```bash
# 运行3阶段渐进训练
./train_grpo_progressive.sh
```

### 快速方式：改进单阶段训练
```bash  
# 运行改进的单阶段训练
./train_grpo_improved.sh
```

### 手动调参：
```bash
python train.py \
    --resume_ckpt /path/to/checkpoint \
    --grpo --grpo_group 2 \
    --batch_size 1 \
    --lr 5e-6 \
    --epochs 50 \
    --grpo_lambda_sup 0.5 \
    --grpo_lambda_consistency 0.1 \
    --grpo_warmup_epochs 10 \
    --grpo_global_norm \
    --grpo_max_grad_norm 1.0 \
    --grpo_clip_range 0.2 \
    --grpo_adv_clip_max 3.0 \
    --grpo_beta_kl 0.01 \
    [其他参数...]
```

## 🔍 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grpo_global_norm` | False | 启用全局奖励统计归一化 |
| `grpo_clip_range` | 0.2 | PPO裁剪范围 |
| `grpo_adv_clip_max` | 3.0 | 优势函数最大值 |
| `grpo_max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `grpo_beta_kl` | 0.01 | KL正则化权重 |
| `grpo_warmup_epochs` | 10 | 监督预训练轮数 |
| `grpo_schedule_sup` | False | 监督权重递减调度 |

## 🎛️ 调试指南

### 如果 loss_pg 仍然不降：
1. 降低学习率：`--lr 2e-6`
2. 增大梯度裁剪：`--grpo_max_grad_norm 0.5`
3. 减小clip range：`--grpo_clip_range 0.1`

### 如果 loss_sup 不降：
1. 增加监督权重：`--grpo_lambda_sup 0.8`
2. 延长预训练：`--grpo_warmup_epochs 15`
3. 减小批次大小：`--batch_size 1`

### 如果 advantages 分布异常：
1. 启用全局归一化：`--grpo_global_norm`
2. 调整奖励权重：`--grpo_w_psnr 0.6 --grpo_w_ssim 0.4`
3. 减小优势裁剪：`--grpo_adv_clip_max 2.0`

## 📈 预期改进效果

- ✅ 政策损失稳定下降
- ✅ 优势函数分布平衡 (mean≈0, std≈1)
- ✅ 训练过程更平滑
- ✅ 最终性能提升
- ✅ 避免模式崩塌

## 🚨 注意事项

1. **内存使用**：新增全局统计会略微增加内存使用
2. **训练时间**：PPO损失计算会略微增加计算开销
3. **超参敏感性**：建议从推荐参数开始，逐步调整
4. **checkpoint兼容性**：新模型与旧checkpoint兼容（`strict=False`）

使用这些改进后，GRPO训练应该会更加稳定，能够有效改善模型性能而不会出现之前的崩塌问题。 