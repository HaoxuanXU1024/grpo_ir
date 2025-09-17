"""
TorchRL Environment for GRPO Training on AdaIR
将AdaIR模型包装成标准的RL环境，支持PPO训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from tensordict import TensorDict

from net.model import AdaIR
from utils.val_utils import compute_psnr_ssim
from utils.pytorch_ssim import SSIM
import lpips
from rewards import create_adair_reward_fn


class AdaIRGRPOEnv:
    """AdaIR GRPO环境，支持真正的策略参数注入"""
    
    def __init__(self, 
                 batch_size: int = 1,
                 device: str = "cuda",
                 reward_weights: Dict[str, float] = None,
                 use_advanced_rewards: bool = True):
        
        self.batch_size = batch_size
        self.device = device
        self.use_advanced_rewards = use_advanced_rewards
        
        # 初始化AdaIR模型
        # 注意: 这里创建的模型实例主要用于获取结构信息，实际训练中会被外部传入的模型替代
        from net.model_torchrl import AdaIRTorchRL
        self.adair_model = AdaIRTorchRL(decoder=True).to(device)
        self.adair_model.eval()
        
        if use_advanced_rewards:
            # 使用高级奖励系统 - 优化权重配置，包含PSNR/SSIM
            if reward_weights is None:
                reward_weights = {
                    "clip_similarity": 0.25,  # 语义相似度
                    "perceptual": 0.25,       # 感知质量 (LPIPS)  
                    "aesthetic": 0.15,        # 美学质量
                    "psnr": 0.20,            # PSNR指标
                    "ssim": 0.15,            # SSIM指标
                }
            self.reward_config = reward_weights  # 保存配置供后续使用
            self.reward_fn = create_adair_reward_fn(device, reward_weights)
            # 预缓存各设备的奖励函数，避免训练时重复创建
            self._device_reward_cache = {}
            print(f"[INFO] Using advanced reward system with balanced weights: {reward_weights}")
        else:
            # 使用原始奖励系统
            if reward_weights is None:
                reward_weights = {"psnr": 0.4, "ssim": 0.3, "lpips": 0.3}
            self.reward_weights = reward_weights
            # 奖励计算组件
            self.ssim_metric = SSIM().eval().to(device)
            self.lpips_metric = lpips.LPIPS(net='alex').eval().to(device)
            print(f"[INFO] Using original reward system with weights: {reward_weights}")
        
        # 当前状态
        self.current_degraded = None
        self.current_clean = None
        
    def reset(self, obs_dict=None):
        """重置环境"""
        if obs_dict is not None:
            self.current_degraded = obs_dict.get("degraded_image")
            self.current_clean = obs_dict.get("clean_image")
        return obs_dict
    
    def step(self, action_dict):
        """执行一步：使用策略动作运行AdaIR并计算奖励"""
        
        # 提取动作参数, 现在是采样好的动作值
        actions = action_dict["action"]["actions"]  # [B, 3, 4]
        
        # 运行AdaIR模型with策略参数
        restored_image = self._run_adair_with_policy(
            self.current_degraded, actions
        )
        
        # 计算奖励
        reward = self._compute_reward(restored_image, self.current_clean)
        
        # 对于图像修复任务，每个episode只有一步
        done = torch.ones(self.current_degraded.shape[0], 1, device=self.device, dtype=torch.bool)
        
        return TensorDict({
            "degraded_image": self.current_degraded,
            "clean_image": self.current_clean,
            "reward": reward,
            "done": done,
            "restored_image": restored_image,  # 用于调试和可视化
        }, batch_size=self.current_degraded.shape[:1], device=self.device)
    
    def _run_adair_with_policy(self, degraded: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """使用策略动作运行AdaIR模型 - 真正的动作注入版本"""
        
        # 确保输入在正确的设备上
        degraded = degraded.to(self.device)
        
        # 处理DDP包装的模型
        actual_model = self.adair_model
        if hasattr(self.adair_model, 'module'):
            # 如果模型被DDP包装，使用.module访问实际模型
            actual_model = self.adair_model.module
        
        # 确保模型在正确的设备上
        actual_model = actual_model.to(self.device)
        
        # 使用 model_torchrl 中定义的标准方法注入动作
        actual_model.inject_actions(actions)
        
        # 使用stochastic=True模式，FreModule现在会使用我们注入的动作
        # 模型现在只返回restored_image
        result = actual_model(degraded, stochastic=True)

        # 修正：处理模型在stochastic模式下返回元组的情况
        if isinstance(result, tuple):
            restored = result[0]
        else:
            restored = result
            
        return restored
    
    def _compute_reward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """计算多指标组合奖励"""
        
        # 确保输入在正确的设备上
        restored = restored.to(self.device)
        clean = clean.to(self.device)
        
        if self.use_advanced_rewards:
            # 使用高级奖励系统
            with torch.no_grad():
                try:
                    # 确保所有张量在同一设备上
                    current_device = restored.device
                    restored = restored.to(current_device)
                    clean = clean.to(current_device)
                    
                    # 使用缓存的设备特定奖励函数，避免重复创建
                    device_key = str(current_device)
                    if device_key not in self._device_reward_cache:
                        print(f"[INFO] Creating reward function for device {device_key} (one-time setup)")
                        
                        # 设置环境变量以减少HuggingFace的verbose输出
                        import os
                        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
                        
                        from rewards import create_adair_reward_fn
                        self._device_reward_cache[device_key] = create_adair_reward_fn(
                            device=device_key, 
                            reward_config=self.reward_config
                        )
                    
                    device_specific_reward_fn = self._device_reward_cache[device_key]
                    rewards = device_specific_reward_fn(restored, clean)
                    
                    # 确保奖励是张量格式
                    if not isinstance(rewards, torch.Tensor):
                        rewards = torch.tensor(rewards, device=current_device, dtype=torch.float32)
                    rewards = rewards.view(-1, 1)
                    return rewards
                except Exception as e:
                    print(f"[WARNING] Advanced reward calculation failed: {e}")
                    print("[INFO] Falling back to simple reward")
                    # 回退到简单奖励
                    return self._compute_simple_reward(restored, clean)
        else:
            # 使用原始奖励系统
            return self._compute_simple_reward(restored, clean)
    
    def _compute_simple_reward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """计算简单的多指标组合奖励（原始方法）"""
        
        # 确保指标计算器在正确的设备上
        self.ssim_metric = self.ssim_metric.to(self.device)
        self.lpips_metric = self.lpips_metric.to(self.device)
        
        batch_size = restored.shape[0]
        rewards = torch.zeros(batch_size, 1, device=self.device)
        
        with torch.no_grad():
            for i in range(batch_size):
                # PSNR
                mse = torch.mean((restored[i:i+1] - clean[i:i+1]) ** 2) + 1e-6
                psnr = 10.0 * torch.log10(1.0 / mse)
                psnr_norm = torch.clamp(psnr / 40.0, 0.0, 1.0)
                
                # SSIM  
                ssim = self.ssim_metric(restored[i:i+1], clean[i:i+1])
                ssim_norm = torch.clamp(ssim, 0.0, 1.0)
                
                # LPIPS
                restored_p = restored[i:i+1] * 2 - 1
                clean_p = clean[i:i+1] * 2 - 1
                lp = self.lpips_metric(restored_p, clean_p).squeeze()
                lp_norm = torch.clamp(lp, 0.0, 1.0)
                one_minus_lp = 1.0 - lp_norm
                
                # 组合奖励
                total_reward = (
                    self.reward_weights["psnr"] * psnr_norm +
                    self.reward_weights["ssim"] * ssim_norm +
                    self.reward_weights["lpips"] * one_minus_lp
                )
                
                rewards[i, 0] = total_reward
        
        return rewards
    
    def set_data(self, degraded: torch.Tensor, clean: torch.Tensor):
        """设置当前批次的数据"""
        self.current_degraded = degraded
        self.current_clean = clean


class AdaIRPolicyNetwork(nn.Module):
    """策略网络：输出FreModule的Beta分布参数"""
    
    def __init__(self, input_dim: int = 3*128*128, hidden_dim: int = 512):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.ReLU(),
        )
        
        # 输出Beta分布参数: 12对(alpha, beta) = 24个参数
        # 每个FreModule需要4个动作值（rate_h, rate_w, fuse_1, fuse_2）
        # 3个FreModule × 4个动作 = 12个动作值
        # 每个动作值需要(alpha, beta)参数 = 12 × 2 = 24个输出
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 12),  # 输出12个alpha参数
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 12),  # 输出12个beta参数
        )
        
        # 更保守的初始化：使用更稳定的Beta分布参数
        # 目标：Beta(5, 5) 更集中在0.5附近，方差更小
        self.alpha_head[-1].bias.data.fill_(1.6)  # log(5) ≈ 1.6，经过softplus后约为5
        self.alpha_head[-1].weight.data.normal_(0, 0.001)  # 减小权重方差，更保守
        
        self.beta_head[-1].bias.data.fill_(1.6)  # 同样初始化为约5
        self.beta_head[-1].weight.data.normal_(0, 0.001)  # 减小权重方差
        
    def forward(self, observation: TensorDict) -> TensorDict:
        """策略网络前向传播 - 输出明确的Beta分布参数"""
        
        degraded_image = observation["degraded_image"]  # [B, 3, 128, 128]
        
        # 提取特征
        features = self.feature_extractor(degraded_image)
        
        # 分别计算alpha和beta参数
        alpha_raw = self.alpha_head(features)  # [B, 12]
        beta_raw = self.beta_head(features)    # [B, 12]
        
        # 确保参数为正并且有合理的范围
        alphas = F.softplus(alpha_raw) + 1.0  # 范围[1, inf)，避免退化分布
        betas = F.softplus(beta_raw) + 1.0    # 范围[1, inf)
        
        # 组合成策略网络期望的格式 [B, 3, 8]
        # 将12对(alpha,beta)重新排列为24个参数，然后reshape为[B, 3, 8]
        alpha_beta_combined = torch.stack([alphas, betas], dim=-1)  # [B, 12, 2]
        policy_params = alpha_beta_combined.view(degraded_image.shape[0], -1)  # [B, 24]
        policy_params = policy_params.view(-1, 3, 8)  # [B, 3, 8]
        
        return TensorDict({
            "action": TensorDict({
                "freq_params": policy_params
            }, batch_size=degraded_image.shape[:1])
        }, batch_size=degraded_image.shape[:1]) 