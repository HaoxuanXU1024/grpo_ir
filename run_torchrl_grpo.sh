#!/bin/bash

# 整合的TorchRL GRPO训练脚本
# 直接使用train.py，只需添加 --grpo_torchrl 参数

# 设置工作目录和Python路径
export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR

echo "🚀 启动整合的TorchRL GRPO训练..."
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"

# 检查TorchRL是否已安装
if ! python -c "import torchrl" 2>/dev/null; then
    echo "⚠️ TorchRL未安装，正在安装..."
    bash install_torchrl.sh
fi

echo "✅ 使用主训练脚本 train.py + TorchRL框架"

# 检查GPU数量
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "检测到 ${NUM_GPUS} 个GPU"

if [ "$NUM_GPUS" -ge 4 ]; then
    echo "🚀 启动4卡分布式TorchRL GRPO训练..."
    # 确保在正确的工作目录
    cd /data2/haoxuan/AdaIR
    # 4卡分布式训练 (更快)
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12355 \
        /data2/haoxuan/AdaIR/train.py \
        --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
        --grpo --grpo_torchrl \
        --grpo_group 2 \
        --batch_size 40 \
        --epochs 50 \
        --lr 5e-6 \
        --num_gpus 4 \
        --lora \
        --lora_targets attn,cross_attn \
        --lora_r 4 \
        --lora_alpha 4 \
        --train_policy_only \
        --grpo_w_psnr 0.4 \
        --grpo_w_ssim 0.3 \
        --grpo_w_lpips 0.3 \
        --wblogger AdaIR-TorchRL-GRPO-4GPU \
        --ckpt_dir torchrl_grpo_4gpu
else
    echo "⚠️  GPU数量不足4个，使用单GPU训练..."
    # 确保在正确的工作目录
    cd /data2/haoxuan/AdaIR
    
    # 单GPU训练
    CUDA_VISIBLE_DEVICES=0 python /data2/haoxuan/AdaIR/train.py \
        --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
        --grpo --grpo_torchrl \
        --grpo_group 2 \
        --batch_size 4 \
        --epochs 50 \
        --lr 5e-6 \
        --num_gpus 1 \
        --lora \
        --lora_targets attn,cross_attn \
        --lora_r 4 \
        --lora_alpha 4 \
        --train_policy_only \
        --grpo_w_psnr 0.4 \
        --grpo_w_ssim 0.3 \
        --grpo_w_lpips 0.3 \
        --wblogger AdaIR-TorchRL-GRPO-1GPU \
        --ckpt_dir torchrl_grpo_1gpu

fi
