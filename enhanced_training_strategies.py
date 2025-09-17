#!/usr/bin/env python3
"""
增强训练策略集合
解决GRPO训练停滞和奖励饱和问题
"""

import torch
import numpy as np
from typing import Dict, Tuple

class EnhancedRewardDesign:
    """增强奖励设计 - 提高区分度和动态性"""
    
    def __init__(self, base_weights: Dict[str, float]):
        self.base_weights = base_weights
        self.adaptive_threshold = 0.8  # 奖励阈值
        
    def calculate_enhanced_reward(self, restored_images, clean_images, base_reward):
        """计算增强奖励，增加区分度"""
        
        # 1. 基础奖励加权
        enhanced_reward = base_reward.clone()
        
        # 2. 添加细粒度PSNR奖励 (增强高质量区间的区分度)
        psnr_values = self._calculate_psnr(restored_images, clean_images)
        
        # 使用非线性映射增强高PSNR区间的区分度
        def psnr_to_reward_nonlinear(psnr):
            # 对于高PSNR值，使用指数函数增强区分度
            normalized_psnr = torch.clamp(psnr / 40.0, 0.0, 1.0)
            if psnr > 30.0:  # 高质量区间
                return torch.pow(normalized_psnr, 0.5)  # 根号函数，增强高值区分度
            else:
                return normalized_psnr
        
        enhanced_psnr_reward = psnr_to_reward_nonlinear(psnr_values)
        
        # 3. 添加梯度保持奖励 (鼓励保持图像细节)
        gradient_reward = self._calculate_gradient_preservation(restored_images, clean_images)
        
        # 4. 添加对比度奖励 (鼓励适当的对比度)
        contrast_reward = self._calculate_contrast_reward(restored_images, clean_images)
        
        # 5. 综合奖励 (动态权重调整)
        alpha = 0.6  # 基础奖励权重
        beta = 0.25  # 增强PSNR权重  
        gamma = 0.1  # 梯度保持权重
        delta = 0.05 # 对比度权重
        
        final_reward = (alpha * enhanced_reward + 
                       beta * enhanced_psnr_reward + 
                       gamma * gradient_reward + 
                       delta * contrast_reward)
        
        return torch.clamp(final_reward, 0.0, 1.0)
    
    def _calculate_psnr(self, restored, clean):
        """计算PSNR值"""
        mse = torch.mean((restored - clean) ** 2, dim=(1, 2, 3)) + 1e-8
        psnr = 10.0 * torch.log10(1.0 / mse)
        return psnr
    
    def _calculate_gradient_preservation(self, restored, clean):
        """计算梯度保持奖励"""
        # 使用Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        device = restored.device
        sobel_x = sobel_x.to(device).repeat(1, restored.shape[1], 1, 1)
        sobel_y = sobel_y.to(device).repeat(1, restored.shape[1], 1, 1)
        
        # 计算梯度
        grad_restored_x = torch.nn.functional.conv2d(restored, sobel_x, padding=1, groups=restored.shape[1])
        grad_restored_y = torch.nn.functional.conv2d(restored, sobel_y, padding=1, groups=restored.shape[1])
        grad_restored = torch.sqrt(grad_restored_x**2 + grad_restored_y**2 + 1e-8)
        
        grad_clean_x = torch.nn.functional.conv2d(clean, sobel_x, padding=1, groups=clean.shape[1])
        grad_clean_y = torch.nn.functional.conv2d(clean, sobel_y, padding=1, groups=clean.shape[1])
        grad_clean = torch.sqrt(grad_clean_x**2 + grad_clean_y**2 + 1e-8)
        
        # 梯度相似度
        grad_similarity = 1.0 - torch.mean(torch.abs(grad_restored - grad_clean), dim=(1, 2, 3))
        return torch.clamp(grad_similarity, 0.0, 1.0)
    
    def _calculate_contrast_reward(self, restored, clean):
        """计算对比度奖励"""
        # 计算局部标准差作为对比度指标
        def local_std(img):
            # 3x3窗口的局部标准差
            kernel = torch.ones(1, 1, 3, 3).to(img.device) / 9.0
            kernel = kernel.repeat(img.shape[1], 1, 1, 1)
            
            mean_local = torch.nn.functional.conv2d(img, kernel, padding=1, groups=img.shape[1])
            mean_sq_local = torch.nn.functional.conv2d(img**2, kernel, padding=1, groups=img.shape[1])
            var_local = mean_sq_local - mean_local**2
            std_local = torch.sqrt(var_local + 1e-8)
            return torch.mean(std_local, dim=(1, 2, 3))
        
        std_restored = local_std(restored)
        std_clean = local_std(clean)
        
        # 对比度相似度奖励
        contrast_similarity = 1.0 - torch.abs(std_restored - std_clean) / (std_clean + 1e-8)
        return torch.clamp(contrast_similarity, 0.0, 1.0)

class NoiseInjectionStrategy:
    """噪声注入策略 - 增强探索"""
    
    def __init__(self, noise_schedule: str = "adaptive"):
        self.noise_schedule = noise_schedule
        self.current_epoch = 0
        
    def get_exploration_noise(self, current_reward_mean: float, epoch: int):
        """根据当前状态动态调整探索噪声"""
        self.current_epoch = epoch
        
        if self.noise_schedule == "adaptive":
            # 自适应噪声：奖励停滞时增加噪声
            if current_reward_mean > 0.75:  # 高奖励区间，需要更多探索
                noise_scale = 0.3
            elif current_reward_mean > 0.6:
                noise_scale = 0.2
            else:
                noise_scale = 0.1
                
            # 训练后期逐渐减少噪声
            decay_factor = max(0.5, 1.0 - epoch / 100.0)
            return noise_scale * decay_factor
            
        elif self.noise_schedule == "cosine":
            # 余弦退火噪声
            max_noise = 0.4
            min_noise = 0.05
            progress = epoch / 100.0
            noise_scale = min_noise + (max_noise - min_noise) * (1 + np.cos(np.pi * progress)) / 2
            return noise_scale
            
        else:
            return 0.1  # 固定噪声

class CurriculumDataStrategy:
    """课程学习数据策略"""
    
    def __init__(self, total_epochs: int = 50):
        self.total_epochs = total_epochs
        
    def get_difficulty_schedule(self, epoch: int):
        """获取当前epoch应该使用的数据难度分布"""
        progress = epoch / self.total_epochs
        
        if progress < 0.3:  # 前30%：简单数据为主
            return {
                "easy": 0.6,    # 最好40%的数据
                "medium": 0.3,  # 中等30%的数据  
                "hard": 0.1     # 最差30%的数据
            }
        elif progress < 0.7:  # 中期：平衡分布
            return {
                "easy": 0.3,
                "medium": 0.4,
                "hard": 0.3
            }
        else:  # 后期：困难数据为主
            return {
                "easy": 0.2,
                "medium": 0.3,
                "hard": 0.5
            }

def create_enhanced_training_config(current_epoch: int, reward_mean: float):
    """创建增强训练配置"""
    
    # 奖励设计
    base_weights = {
        "psnr": 0.4,
        "ssim": 0.3, 
        "perceptual": 0.2,
        "clip": 0.1
    }
    reward_designer = EnhancedRewardDesign(base_weights)
    
    # 噪声注入
    noise_injector = NoiseInjectionStrategy("adaptive")
    exploration_noise = noise_injector.get_exploration_noise(reward_mean, current_epoch)
    
    # 课程学习
    curriculum = CurriculumDataStrategy()
    difficulty_schedule = curriculum.get_difficulty_schedule(current_epoch)
    
    return {
        "reward_designer": reward_designer,
        "exploration_noise": exploration_noise,
        "difficulty_schedule": difficulty_schedule,
        "enhanced_sampling_params": {
            "temperature": 3.0 + exploration_noise,  # 动态温度
            "noise_scale": 0.15 + exploration_noise * 0.5,  # 动态噪声
            "entropy_bonus": 0.02  # 熵奖励，鼓励探索
        }
    }

if __name__ == "__main__":
    # 测试增强策略
    print("🧪 测试增强训练策略...")
    
    # 模拟当前训练状态
    current_epoch = 25
    reward_mean = 0.8
    
    config = create_enhanced_training_config(current_epoch, reward_mean)
    
    print(f"📊 当前epoch: {current_epoch}")
    print(f"📊 当前奖励: {reward_mean}")
    print(f"🔧 探索噪声: {config['exploration_noise']:.3f}")
    print(f"📚 难度分布: {config['difficulty_schedule']}")
    print(f"⚙️ 采样参数: {config['enhanced_sampling_params']}")
