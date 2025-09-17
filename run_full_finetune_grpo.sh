#!/bin/bash

# 全参数微调GRPO训练脚本 - 不使用LoRA
# 策略: 增强采样多样性 + 全参数微调 + 优化奖励权重

export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR
export CUDA_VISIBLE_DEVICES=2,3

echo "🚀 全参数微调GRPO训练 - 解决小模型LoRA过度工程化问题"
echo "策略: 增强采样方差 + 全参数训练 + 重新配置奖励"

echo "🎯 单阶段全参数GRPO训练..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12361 \
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
    --batch_size 2 \
    --epochs 50 \
    --lr 5e-5 \
    --num_gpus 2 \
    --finetune_worst \
    --worst_dir AdaIR_results/worst_lists_adair5d \
    --grpo_clip_range 0.2 \
    --wblogger AdaIR-Full-Finetune-GRPO \
    --ckpt_dir full_finetune_grpo

echo "🎉 全参数GRPO训练完成！结果保存在 full_finetune_grpo/"
