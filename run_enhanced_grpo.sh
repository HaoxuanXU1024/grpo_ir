#!/bin/bash

# 增强版GRPO训练脚本 - 解决采样差异和奖励配置问题
# 方案: 增强采样多样性 + 优化奖励权重 + 扩大训练范围

export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR
export CUDA_VISIBLE_DEVICES=2,3

echo "🚀 增强版GRPO训练 - 解决采样差异问题"
echo "策略: 增强采样方差 + 重新配置奖励 + 部分解冻主干"

# 单阶段训练：直接GRPO，但增强策略
echo "🎯 增强版GRPO训练（单阶段，重点解决采样问题）..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12360 \
    /data2/haoxuan/AdaIR/train.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --grpo --grpo_torchrl \
    --use_advanced_rewards \
    --grpo_w_clip 0.10 \
    --grpo_w_perceptual 0.20 \
    --grpo_w_aesthetic 0.05 \
    --grpo_w_psnr_adv 0.35 \
    --grpo_w_ssim_adv 0.30 \
    --grpo_group 6 \
    --batch_size 3 \
    --epochs 50 \
    --lr 3e-5 \
    --num_gpus 2 \
    --lora \
    --lora_targets attn,cross_attn,ffn \
    --lora_r 32 \
    --lora_alpha 32 \
    --finetune_worst \
    --worst_dir AdaIR_results/worst_lists_adair5d \
    --grpo_clip_range 0.3 \
    --wblogger AdaIR-Enhanced-GRPO \
    --ckpt_dir enhanced_grpo_experiment

echo "🎉 增强版GRPO训练完成！"

# 关键改进说明：
echo "📋 关键改进:"
echo "1. ✅ 增强采样方差: 温度参数+噪声注入"
echo "2. ✅ 重新配置奖励: PSNR/SSIM占65%，先进指标35%"
echo "3. ✅ 增加GRPO组数: 6组采样增强差异"
echo "4. ✅ 扩大LoRA范围: 包含FFN层"
echo "5. ✅ 增大LoRA秩: rank=32提升容量"
echo "6. ✅ 放松clip约束: 0.3允许更大更新"
