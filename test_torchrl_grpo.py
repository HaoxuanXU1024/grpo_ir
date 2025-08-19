#!/usr/bin/env python3
"""
测试TorchRL GRPO实现的脚本
验证环境、策略网络和训练循环是否正常工作
"""

import torch
import numpy as np
from tensordict import TensorDict

from grpo_torchrl_env import AdaIRGRPOEnv, AdaIRPolicyNetwork
from train_grpo_torchrl import SimplifiedPPOLoss, TorchRLGRPOTrainer
from net.model import AdaIR


def test_environment():
    """测试环境是否正常工作"""
    print("=== 测试AdaIR GRPO环境 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    # 创建环境
    env = AdaIRGRPOEnv(batch_size=batch_size, device=device)
    
    # 创建测试数据
    degraded = torch.randn(batch_size, 3, 128, 128, device=device)
    clean = torch.randn(batch_size, 3, 128, 128, device=device)
    degraded = torch.clamp(degraded, 0, 1)
    clean = torch.clamp(clean, 0, 1)
    
    # 设置环境数据
    env.set_data(degraded, clean)
    
    # 创建测试动作
    freq_params = torch.randn(batch_size, 3, 8, device=device) * 0.1 + 1.0
    action_dict = {
        "action": {
            "freq_params": freq_params
        }
    }
    
    # 测试环境步进
    result = env.step(action_dict)
    
    print(f"✅ 环境测试成功!")
    print(f"  奖励形状: {result['reward'].shape}")
    print(f"  奖励值: {result['reward'].mean().item():.4f}")
    print(f"  完成状态: {result['done'].all()}")
    
    return True


def test_policy_network():
    """测试策略网络"""
    print("\n=== 测试策略网络 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    # 创建策略网络
    policy = AdaIRPolicyNetwork().to(device)
    
    # 创建测试观察
    obs = TensorDict({
        "degraded_image": torch.randn(batch_size, 3, 128, 128, device=device),
        "clean_image": torch.randn(batch_size, 3, 128, 128, device=device),
    }, batch_size=(batch_size,), device=device)
    
    # 前向传播
    action = policy(obs)
    
    print(f"✅ 策略网络测试成功!")
    print(f"  动作参数形状: {action['action']['freq_params'].shape}")
    print(f"  参数范围: {action['action']['freq_params'].min().item():.3f} - {action['action']['freq_params'].max().item():.3f}")
    
    return True


def test_ppo_loss():
    """测试PPO损失函数"""
    print("\n=== 测试PPO损失函数 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    
    # 创建PPO损失
    ppo_loss = SimplifiedPPOLoss()
    
    # 创建测试数据
    log_probs = torch.randn(batch_size, device=device)
    old_log_probs = torch.randn(batch_size, device=device)
    advantages = torch.randn(batch_size, device=device)
    returns = torch.randn(batch_size, device=device)
    values = torch.randn(batch_size, device=device)
    
    # 计算损失
    loss_dict = ppo_loss(log_probs, old_log_probs, advantages, returns, values)
    
    print(f"✅ PPO损失测试成功!")
    print(f"  总损失: {loss_dict['loss_total'].item():.4f}")
    print(f"  策略损失: {loss_dict['loss_policy'].item():.4f}")
    print(f"  价值损失: {loss_dict['loss_value'].item():.4f}")
    print(f"  熵损失: {loss_dict['loss_entropy'].item():.4f}")
    
    return True


def test_full_training_step():
    """测试完整的训练步骤"""
    print("\n=== 测试完整训练步骤 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    
    # 创建环境和网络
    env = AdaIRGRPOEnv(batch_size=batch_size, device=device)
    policy = AdaIRPolicyNetwork().to(device)
    value_net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((8, 8)),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 8 * 8, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1),
    ).to(device)
    
    ppo_loss = SimplifiedPPOLoss()
    
    # 创建测试数据
    degraded = torch.randn(batch_size, 3, 128, 128, device=device)
    clean = torch.randn(batch_size, 3, 128, 128, device=device)
    degraded = torch.clamp(degraded, 0, 1)
    clean = torch.clamp(clean, 0, 1)
    
    # 设置环境
    env.set_data(degraded, clean)
    
    # 创建观察
    obs = TensorDict({
        "degraded_image": degraded,
        "clean_image": clean,
    }, batch_size=(batch_size,), device=device)
    
    # 策略采样
    action = policy(obs)
    
    # 环境步进
    env_input = TensorDict({
        "degraded_image": degraded,
        "clean_image": clean,
        "action": action["action"]
    }, batch_size=(batch_size,), device=device)
    
    result = env.step(env_input)
    
    # 价值估计
    values = value_net(degraded).squeeze(-1)
    
    # 计算简化的优势和回报
    rewards = result["reward"].squeeze(-1)
    advantages = rewards - values.detach()
    returns = rewards
    
    # log概率（简化）
    log_probs = -0.5 * torch.sum(action["action"]["freq_params"] ** 2, dim=(1, 2))
    
    # 计算损失
    loss_dict = ppo_loss(log_probs, log_probs.detach(), advantages, returns, values)
    
    print(f"✅ 完整训练步骤测试成功!")
    print(f"  奖励: {rewards.mean().item():.4f}")
    print(f"  优势: {advantages.mean().item():.4f}")
    print(f"  总损失: {loss_dict['loss_total'].item():.4f}")
    
    return True


def test_adair_model():
    """测试AdaIR模型加载"""
    print("\n=== 测试AdaIR模型 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建AdaIR模型
    model = AdaIR(decoder=True).to(device)
    model.eval()
    
    # 测试前向传播
    test_input = torch.randn(1, 3, 256, 256, device=device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ AdaIR模型测试成功!")
    print(f"  输入形状: {test_input.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: {output.min().item():.3f} - {output.max().item():.3f}")
    
    return True


def main():
    """主测试函数"""
    print("🚀 开始测试TorchRL GRPO实现...")
    
    tests = [
        ("AdaIR模型", test_adair_model),
        ("环境", test_environment),
        ("策略网络", test_policy_network),
        ("PPO损失", test_ppo_loss),
        ("完整训练步骤", test_full_training_step),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! TorchRL GRPO实现就绪!")
        print("\n下一步可以运行:")
        print("  bash run_torchrl_grpo.sh")
    else:
        print("⚠️  有测试失败，请检查实现")
    
    return passed == total


if __name__ == "__main__":
    main() 