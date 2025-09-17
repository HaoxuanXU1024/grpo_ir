#!/bin/bash

# TorchRL GRPO训练脚本 - 使用GPU 2,3
# 专门为双卡训练设计

# 设置工作目录和Python路径
export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR

echo "🚀 启动GPU 2,3双卡TorchRL GRPO训练..."
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python路径: $PYTHONPATH"

# 检查TorchRL是否已安装
if ! python -c "import torchrl" 2>/dev/null; then
    echo "⚠️ TorchRL未安装，正在安装..."
    bash install_torchrl.sh
fi

echo "✅ 使用主训练脚本 train.py + TorchRL框架"

# 检查指定GPU是否可用
if ! nvidia-smi -i 2 >/dev/null 2>&1; then
    echo "❌ GPU 2 不可用，请检查GPU状态"
    exit 1
fi

if ! nvidia-smi -i 3 >/dev/null 2>&1; then
    echo "❌ GPU 3 不可用，请检查GPU状态"  
    exit 1
fi

echo "✅ GPU 2,3 状态正常，启动2卡分布式TorchRL GRPO训练..."

# 设置可见GPU
export CUDA_VISIBLE_DEVICES=2,3

# 确保在正确的工作目录
cd /data2/haoxuan/AdaIR

# 2卡分布式训练
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12357 \
    /data2/haoxuan/AdaIR/train.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --grpo --grpo_torchrl \
    --use_advanced_rewards \
    --grpo_w_clip 0.25 \
    --grpo_w_perceptual 0.25 \
    --grpo_w_aesthetic 0.15 \
    --grpo_w_psnr_adv 0.20 \
    --grpo_w_ssim_adv 0.15 \
    --grpo_group 6 \
    --batch_size 1 \
    --epochs 50 \
    --lr 5e-5 \
    --num_gpus 2 \
    --lora \
    --lora_targets attn,cross_attn \
    --lora_r 16 \
    --lora_alpha 16 \
    --train_policy_only \
    --finetune_worst \
    --worst_dir AdaIR_results/worst_lists_adair5d \
    --ckpt_dir torchrl_grpo_gpu23_advanced_9_15
    # 纯GRPO不使用PPO剪裁参数 \
    # --disable_wandb \
    

echo "🎉 训练完成！结果保存在 torchrl_grpo_gpu23_advanced/ 目录"


