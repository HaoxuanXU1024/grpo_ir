#!/bin/bash

# GRPO (flow-style) 训练脚本
# 使用 --grpo_flow_style 标志激活新的GRPO逻辑

# 设置工作目录和Python路径
export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR

echo "🚀 启动 Flow-Style GRPO 训练..."
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"

# 检查GPU数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "检测到 ${NUM_GPUS} 个GPU"

if [ "$NUM_GPUS" -ge 4 ]; then
    echo "🚀 启动4卡分布式 Flow-Style GRPO 训练..."
    # 确保在正确的工作目录
    cd /data2/haoxuan/AdaIR
    # 4卡分布式训练
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12356 \
        /data2/haoxuan/AdaIR/train.py \
        --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
        --grpo --grpo_flow_style \
        --grpo_group 4 \
        --batch_size 1 \
        --epochs 100 \
        --lr 5e-5 \
        --num_gpus 4 \
        --lora \
        --lora_targets attn,cross_attn \
        --lora_r 4 \
        --lora_alpha 4 \
        --train_policy_only \
        --finetune_worst \
        --worst_dir AdaIR_results/worst_lists_adair5d \
        --grpo_w_psnr 0.4 \
        --grpo_w_ssim 0.3 \
        --grpo_w_lpips 0.3 \
        --wblogger AdaIR-Flow-GRPO-4GPU \
        --ckpt_dir flow_grpo_4gpu
else
    echo "⚠️  GPU数量不足4个，使用单GPU训练..."
    # 确保在正确的工作目录
    cd /data2/haoxuan/AdaIR
    
    # 单GPU训练
    CUDA_VISIBLE_DEVICES=0 python /data2/haoxuan/AdaIR/train.py \
        --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
        --grpo --grpo_flow_style \
        --grpo_group 3 \
        --batch_size 1 \
        --epochs 50 \
        --lr 5e-5 \
        --num_gpus 1 \
        --lora \
        --lora_targets attn,cross_attn \
        --lora_r 4 \
        --lora_alpha 4 \
        --train_policy_only \
        --finetune_worst \
        --worst_dir AdaIR_results/worst_lists_adair5d \
        --grpo_w_psnr 0.4 \
        --grpo_w_ssim 0.3 \
        --grpo_w_lpips 0.3 \
        --wblogger AdaIR-Flow-GRPO-1GPU \
        --ckpt_dir flow_grpo_1gpu

fi
