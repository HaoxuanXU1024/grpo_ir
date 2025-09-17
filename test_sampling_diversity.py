#!/usr/bin/env python3
"""
采样多样性诊断脚本
测试增强后的策略头是否真正产生了有意义的采样差异
"""

import torch
import numpy as np
from net.model import AdaIR
from net.model_torchrl import AdaIRTorchRL

def test_sampling_diversity():
    """测试采样的多样性"""
    print("🔍 测试策略头采样多样性...")
    
    # 创建模型
    model = AdaIR(decoder=True).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 创建测试数据
    batch_size = 4
    test_img = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"📊 测试数据: {test_img.shape}")
    
    # 测试确定性输出
    print("\n1️⃣ 确定性模式测试:")
    with torch.no_grad():
        det_output = model(test_img, stochastic=False)
        print(f"确定性输出形状: {det_output.shape}")
    
    # 测试随机采样的差异性
    print("\n2️⃣ 随机采样模式测试:")
    num_samples = 6  # GRPO组数
    samples = []
    log_probs = []
    
    with torch.no_grad():
        for i in range(num_samples):
            output, log_prob = model(test_img, stochastic=True)
            samples.append(output)
            log_probs.append(log_prob)
            print(f"样本 {i+1}: shape={output.shape}, log_prob_mean={log_prob.mean().item():.4f}")
    
    # 分析样本间差异
    print("\n3️⃣ 样本差异分析:")
    sample_tensor = torch.stack(samples)  # [num_samples, B, C, H, W]
    
    # 计算样本间的L2距离
    sample_std = torch.std(sample_tensor, dim=0)  # [B, C, H, W]
    mean_std = sample_std.mean().item()
    max_std = sample_std.max().item()
    
    print(f"样本间标准差 - 平均: {mean_std:.6f}, 最大: {max_std:.6f}")
    
    # 计算PSNR差异
    sample_mean = torch.mean(sample_tensor, dim=0)
    psnr_differences = []
    
    for i, sample in enumerate(samples):
        mse = torch.mean((sample - sample_mean) ** 2)
        if mse > 0:
            psnr = -10 * torch.log10(mse)
            psnr_differences.append(psnr.item())
        else:
            psnr_differences.append(float('inf'))
    
    print(f"样本PSNR差异: {[f'{p:.2f}' for p in psnr_differences[:5]]}...")
    
    # 评估多样性
    print("\n4️⃣ 多样性评估:")
    if mean_std > 1e-4:
        print("✅ 采样多样性: 足够 (标准差 > 1e-4)")
        diversity_score = "充足"
    elif mean_std > 1e-6:
        print("⚠️ 采样多样性: 有限 (1e-6 < 标准差 < 1e-4)")
        diversity_score = "有限"
    else:
        print("❌ 采样多样性: 不足 (标准差 < 1e-6)")
        diversity_score = "不足"
    
    # 分析log_prob差异
    log_prob_tensor = torch.stack(log_probs)
    log_prob_std = torch.std(log_prob_tensor, dim=0).mean().item()
    print(f"Log概率标准差: {log_prob_std:.6f}")
    
    # 总结
    print(f"\n📋 诊断总结:")
    print(f"- 样本数量: {num_samples}")
    print(f"- 输出标准差: {mean_std:.6f}")
    print(f"- 多样性评级: {diversity_score}")
    print(f"- Log概率变化: {log_prob_std:.6f}")
    
    if mean_std < 1e-5:
        print("\n🚨 建议:")
        print("1. 进一步增大temperature参数")
        print("2. 增加noise_scale")
        print("3. 考虑使用不同的采样分布")
    else:
        print("\n✅ 采样增强效果良好，可以进行GRPO训练")

def test_parameter_updates():
    """测试参数更新幅度"""
    print("\n🔧 测试策略头参数更新...")
    
    model = AdaIR(decoder=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 统计策略头参数
    policy_params = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'policy_rate' in name or 'policy_fuse' in name:
            policy_params.append((name, param.numel()))
    
    policy_param_count = sum(count for _, count in policy_params)
    
    print(f"📊 参数统计:")
    print(f"- 总参数数: {total_params:,}")
    print(f"- 策略头参数: {policy_param_count:,}")
    print(f"- 策略头占比: {policy_param_count/total_params*100:.4f}%")
    
    print(f"\n📋 策略头详情:")
    for name, count in policy_params:
        print(f"  {name}: {count}")

if __name__ == "__main__":
    test_sampling_diversity()
    test_parameter_updates()
