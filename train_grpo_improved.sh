#!/bin/bash

# Improved Single-Stage GRPO Training Script for AdaIR
# This script uses the improved GRPO implementation with stable parameters

echo "=== Starting Improved GRPO Training ==="
echo "Using improved loss computation, advantage normalization, and gradient clipping"
echo "========================================================================"

nohup python train.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --grpo --grpo_group 2 \
    --batch_size 1 \
    --lr 5e-6 \
    --epochs 50 \
    --finetune_worst \
    --worst_derain AdaIR_results/train_eval/train_derain_worst.txt \
    --worst_dehaze AdaIR_results/train_eval/train_dehaze_worst.txt \
    --worst_deblur AdaIR_results/train_eval/train_deblur_worst.txt \
    --worst_enhance AdaIR_results/train_eval/train_enhance_worst.txt \
    --worst_denoise AdaIR_results/train_eval/train_denoise_worst_merged.txt \
    --lora --lora_targets attn,cross_attn --lora_r 4 --lora_alpha 8 \
    --train_policy_only \
    --grpo_w_psnr 0.5 --grpo_w_ssim 0.3 --grpo_w_lpips 0.2 \
    --grpo_lambda_sup 0.5 --grpo_lambda_consistency 0.1 \
    --grpo_warmup_epochs 10 \
    --grpo_schedule_sup \
    --grpo_global_norm \
    --grpo_max_grad_norm 1.0 \
    --grpo_clip_range 0.2 \
    --grpo_adv_clip_max 3.0 \
    --grpo_beta_kl 0.01 \
    --ckpt_dir "grpo_improved" \
    --wblogger "AdaIR-GRPO-Improved" \
    > grpo_improved.log 2>&1 &

echo "Training started with PID: $!"
echo "Monitor progress with: tail -f grpo_improved.log"
echo "Check wandb for detailed metrics" 