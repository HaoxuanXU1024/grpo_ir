#!/bin/bash

# TorchRL GRPO Training Script for AdaIR
echo "=== TorchRL GRPO Training for AdaIR ==="

# 安装依赖（如果需要）
if ! python -c "import torchrl" 2>/dev/null; then
    echo "Installing TorchRL..."
    ./install_torchrl.sh
fi

# 基本训练配置
echo "Starting TorchRL GRPO training with optimized parameters..."

nohup python train_grpo_torchrl.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --batch_size 1 \
    --lr 1e-5 \
    --epochs 50 \
    --finetune_worst \
    --worst_derain AdaIR_results/train_eval/train_derain_worst.txt \
    --worst_dehaze AdaIR_results/train_eval/train_dehaze_worst.txt \
    --worst_deblur AdaIR_results/train_eval/train_deblur_worst.txt \
    --worst_enhance AdaIR_results/train_eval/train_enhance_worst.txt \
    --worst_denoise AdaIR_results/train_eval/train_denoise_worst_merged.txt \
    --lora --lora_targets attn,cross_attn --lora_r 4 --lora_alpha 8 \
    --train_policy_only \
    --grpo_w_psnr 0.4 --grpo_w_ssim 0.3 --grpo_w_lpips 0.3 \
    --grpo_clip_range 0.2 \
    --grpo_max_grad_norm 1.0 \
    --ckpt_dir "torchrl_grpo_ckpts" \
    --wblogger "AdaIR-TorchRL-GRPO" \
    > torchrl_grpo.log 2>&1 &

echo "Training started with PID: $!"
echo "Monitor progress with: tail -f torchrl_grpo.log"
echo "Check wandb for detailed metrics"

# 可选：监控GPU使用情况
echo "GPU status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 