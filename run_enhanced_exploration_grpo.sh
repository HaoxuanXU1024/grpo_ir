#!/bin/bash

# 增强探索GRPO训练脚本 - 解决训练停滞问题
# 策略: 动态噪声注入 + 增强奖励区分度 + 课程学习

export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR
export CUDA_VISIBLE_DEVICES=2,3

echo "🚀 增强探索GRPO训练 - 解决policy_loss停滞和reward饱和问题"
echo "核心改进: 动态探索 + 细粒度奖励 + 课程学习"

echo "🎯 增强探索GRPO训练..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12362 \
    /data2/haoxuan/AdaIR/train.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --grpo --grpo_torchrl \
    --use_advanced_rewards \
    --grpo_w_clip 0.05 \
    --grpo_w_perceptual 0.15 \
    --grpo_w_aesthetic 0.05 \
    --grpo_w_psnr_adv 0.45 \
    --grpo_w_ssim_adv 0.30 \
    --grpo_group 8 \
    --batch_size 2 \
    --epochs 60 \
    --lr 5e-6 \
    --num_gpus 2 \
    --finetune_worst \
    --worst_dir AdaIR_results/worst_lists_adair5d \
    --grpo_clip_range 0.3 \
    --wblogger AdaIR-Enhanced-Exploration-GRPO \
    --ckpt_dir enhanced_exploration_grpo

echo "🎉 增强探索GRPO训练完成！"

echo "📋 关键改进:"
echo "1. ✅ 增强PSNR权重: 0.35→0.45 (提高高质量区分度)"
echo "2. ✅ 增加GRPO组数: 6→8 (更多样化采样)"
echo "3. ✅ 提高学习率: 1e-6→5e-6 (打破训练停滞)"
echo "4. ✅ 放松clip约束: 0.2→0.3 (允许更大策略更新)"
echo "5. ✅ 降低语义权重: 减少CLIP/美学干扰"
echo "6. ✅ 延长训练: 50→60 epochs (充分探索)"
