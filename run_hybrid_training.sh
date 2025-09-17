#!/bin/bash

# 混合训练策略 - 解决GRPO适用性问题
# 方案: MSE预热 → GRPO精调，降低强化学习难度

export PYTHONPATH="/data2/haoxuan/AdaIR:$PYTHONPATH"
cd /data2/haoxuan/AdaIR
export CUDA_VISIBLE_DEVICES=2,3

echo "🔄 混合训练策略 - 解决GRPO稠密预测问题"
echo "理论基础: 传统MSE为策略网络提供良好初始化，GRPO进行感知质量优化"
echo "数据策略: 使用最差30%数据进行针对性微调"

# 阶段1: 策略头MSE预热（只训练策略头）
echo "📈 阶段1: 策略头MSE预热训练（5 epochs）..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12358 \
    /data2/haoxuan/AdaIR/train.py \
    --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
    --epochs 5 \
    --lr 5e-6 \
    --batch_size 4 \
    --num_gpus 2 \
    --finetune_worst \
    --worst_dir AdaIR_results/worst_lists_adair5d \
    --wblogger AdaIR-Hybrid-Step1-MSE-FullFinetune \
    --ckpt_dir hybrid_step1_mse_full

#-- disable_wandb
# 动态找到阶段1的最后一个checkpoint
echo "🔍 查找阶段1训练的最后checkpoint..."
STEP1_CKPT=$(find hybrid_step1_mse_full/ -name "epoch=4-step=*.ckpt" | head -1)
if [ ! -f "$STEP1_CKPT" ]; then
    echo "❌ 找不到阶段1的checkpoint，请检查训练是否成功"
    exit 1
fi
echo "✅ 找到checkpoint: $STEP1_CKPT"

# 阶段2: GRPO精调感知质量
echo "🎯 阶段2: GRPO感知质量精调（45 epochs）..."
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12359 \
    /data2/haoxuan/AdaIR/train.py \
    --resume_ckpt "$STEP1_CKPT" \
    --grpo --grpo_torchrl \
    --use_advanced_rewards \
    --grpo_w_clip 0.20 \
    --grpo_w_perceptual 0.30 \
    --grpo_w_aesthetic 0.15 \
    --grpo_w_psnr_adv 0.20 \
    --grpo_w_ssim_adv 0.15 \
    --grpo_group 2 \
    --batch_size 4 \
    --epochs 45 \
    --lr 2e-6 \
    --num_gpus 2 \
    --finetune_worst \
    --worst_dir AdaIR_results/worst_lists_adair5d \
    --grpo_clip_range 0.15 \
    --wblogger AdaIR-Hybrid-GRPO-FullFinetune \
    --ckpt_dir hybrid_step2_grpo_full

echo "🎉 混合训练完成！结果保存在 hybrid_step2_grpo_full/"
