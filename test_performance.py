#!/usr/bin/env python3
"""
性能测试脚本
用于验证优化后的奖励系统性能提升
"""

import torch
import time
import numpy as np
from rewards import create_adair_reward_fn

def test_reward_performance():
    """测试奖励系统性能"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # 创建测试数据
    batch_size = 4
    h, w = 256, 256
    
    restored_images = torch.randn(batch_size, 3, h, w).to(device)
    clean_images = torch.randn(batch_size, 3, h, w).to(device)
    
    # 配置奖励权重
    reward_config = {
        "clip_similarity": 0.25,
        "perceptual": 0.25,
        "psnr": 0.20,
        "ssim": 0.15,
        "aesthetic": 0.15,
    }
    
    print("Creating reward function...")
    reward_fn = create_adair_reward_fn(device, reward_config)
    
    # 预热
    print("Warming up...")
    for i in range(3):
        _ = reward_fn(restored_images, clean_images)
    
    # 性能测试
    print("Running performance test...")
    num_iterations = 10
    start_time = time.time()
    
    for i in range(num_iterations):
        rewards = reward_fn(restored_images, clean_images)
        if i == 0:
            print(f"Sample rewards: {rewards[:3]}")
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    print(f"\n🚀 Performance Results:")
    print(f"Average time per batch: {avg_time:.3f}s")
    print(f"Images per second: {batch_size / avg_time:.1f}")
    print(f"Expected speedup: ~10-20x compared to before optimization")
    
    # 内存使用情况
    if device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {memory_used:.2f} GB")

if __name__ == "__main__":
    test_reward_performance()
