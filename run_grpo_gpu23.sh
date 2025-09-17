#!/bin/bash

# Flow-Style GRPO训练脚本 - 使用GPU 2,3  
# 专门为双卡训练设计

# 设置工作目录和Python路径
export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR

echo "🚀 启动GPU 2,3双卡Flow-Style GRPO训练..."
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"

# 检查指定GPU是否可用
if ! nvidia-smi -i 2 >/dev/null 2>&1; then
    echo "❌ GPU 2 不可用，请检查GPU状态"
    exit 1
fi

if ! nvidia-smi -i 3 >/dev/null 2>&1; then
    echo "❌ GPU 3 不可用，请检查GPU状态"
    exit 1
fi

echo "✅ GPU 2,3 状态正常，启动2卡分布式Flow-Style GRPO训练..."

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=1

# 确保在正确的工作目录
cd /data2/haoxuan/AdaIR

# 2卡分布式训练
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12358 \
    /data2/haoxuan/AdaIR/train.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --grpo --grpo_flow_style \
    --grpo_group 6 \
    --batch_size 1 \
    --epochs 50 \
    --lr 5e-5 \
    --num_gpus 2 \
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
    --disable_wandb \
    --ckpt_dir flow_grpo_gpu23_9_15

echo "🎉 训练完成！结果保存在 flow_grpo_gpu23/ 目录"


