#!/bin/bash

# Progressive GRPO Training Script for AdaIR
# This script implements a 3-stage training approach for stable GRPO fine-tuning

# Base paths and configurations
RESUME_CKPT="/data2/haoxuan/AdaIR/ckpt/adair5d.ckpt"
BASE_CMD="python train.py --resume_ckpt $RESUME_CKPT"
WORST_LISTS="--finetune_worst \
    --worst_derain AdaIR_results/train_eval/train_derain_worst.txt \
    --worst_dehaze AdaIR_results/train_eval/train_dehaze_worst.txt \
    --worst_deblur AdaIR_results/train_eval/train_deblur_worst.txt \
    --worst_enhance AdaIR_results/train_eval/train_enhance_worst.txt \
    --worst_denoise AdaIR_results/train_eval/train_denoise_worst_merged.txt"
LORA_CONFIG="--lora --lora_targets attn,cross_attn --lora_r 4 --lora_alpha 8 --train_policy_only"
GRPO_REWARDS="--grpo_w_psnr 0.5 --grpo_w_ssim 0.3 --grpo_w_lpips 0.2"

echo "=== Starting Progressive GRPO Training ==="
echo "Stage 1: Supervised Warmup (5 epochs)"
echo "Stage 2: Progressive GRPO (20 epochs)"  
echo "Stage 3: Full GRPO (25 epochs)"
echo "=========================================="

# Stage 1: Supervised Warmup Phase (5 epochs)
echo "Starting Stage 1: Supervised Warmup..."
nohup $BASE_CMD \
    --grpo --grpo_group 2 \
    --batch_size 1 \
    --lr 1e-5 \
    --epochs 5 \
    $WORST_LISTS \
    $LORA_CONFIG \
    $GRPO_REWARDS \
    --grpo_lambda_sup 1.0 \
    --grpo_lambda_consistency 0.1 \
    --grpo_warmup_epochs 5 \
    --grpo_global_norm \
    --grpo_max_grad_norm 1.0 \
    --grpo_clip_range 0.2 \
    --grpo_adv_clip_max 5.0 \
    --ckpt_dir "grpo_stage1" \
    --wblogger "AdaIR-GRPO-Stage1" \
    > stage1_warmup.log 2>&1 &

STAGE1_PID=$!
echo "Stage 1 PID: $STAGE1_PID"
wait $STAGE1_PID

echo "Stage 1 completed. Starting Stage 2..."

# Stage 2: Progressive GRPO Phase (20 epochs)
echo "Starting Stage 2: Progressive GRPO..."
STAGE1_CKPT=$(ls grpo_stage1/epoch=*.ckpt | tail -1)
nohup $BASE_CMD \
    --resume_ckpt $STAGE1_CKPT \
    --grpo --grpo_group 2 \
    --batch_size 1 \
    --lr 5e-6 \
    --epochs 20 \
    $WORST_LISTS \
    $LORA_CONFIG \
    $GRPO_REWARDS \
    --grpo_lambda_sup 0.5 \
    --grpo_lambda_consistency 0.1 \
    --grpo_warmup_epochs 0 \
    --grpo_schedule_sup \
    --grpo_global_norm \
    --grpo_max_grad_norm 1.0 \
    --grpo_clip_range 0.2 \
    --grpo_adv_clip_max 3.0 \
    --grpo_beta_kl 0.01 \
    --ckpt_dir "grpo_stage2" \
    --wblogger "AdaIR-GRPO-Stage2" \
    > stage2_progressive.log 2>&1 &

STAGE2_PID=$!
echo "Stage 2 PID: $STAGE2_PID"
wait $STAGE2_PID

echo "Stage 2 completed. Starting Stage 3..."

# Stage 3: Full GRPO Phase (25 epochs)
echo "Starting Stage 3: Full GRPO..."
STAGE2_CKPT=$(ls grpo_stage2/epoch=*.ckpt | tail -1)
nohup $BASE_CMD \
    --resume_ckpt $STAGE2_CKPT \
    --grpo --grpo_group 3 \
    --batch_size 2 \
    --lr 2e-6 \
    --epochs 25 \
    $WORST_LISTS \
    $LORA_CONFIG \
    $GRPO_REWARDS \
    --grpo_lambda_sup 0.2 \
    --grpo_lambda_consistency 0.05 \
    --grpo_warmup_epochs 0 \
    --grpo_global_norm \
    --grpo_max_grad_norm 0.5 \
    --grpo_clip_range 0.1 \
    --grpo_adv_clip_max 2.0 \
    --grpo_beta_kl 0.02 \
    --ckpt_dir "grpo_stage3" \
    --wblogger "AdaIR-GRPO-Stage3" \
    > stage3_full.log 2>&1 &

STAGE3_PID=$!
echo "Stage 3 PID: $STAGE3_PID"
wait $STAGE3_PID

echo "=== Progressive GRPO Training Completed ==="
echo "Final model saved in: grpo_stage3/"
echo "Logs available in: stage1_warmup.log, stage2_progressive.log, stage3_full.log"

# Optional: Run evaluation on final model
echo "Running evaluation on final model..."
FINAL_CKPT=$(ls grpo_stage3/epoch=*.ckpt | tail -1)
python test.py --ckpt_name $(basename $FINAL_CKPT) --mode 6 > final_evaluation.log 2>&1
echo "Evaluation completed. Results in: final_evaluation.log" 