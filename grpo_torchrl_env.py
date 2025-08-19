"""
TorchRL Environment for GRPO Training on AdaIR
将AdaIR模型包装成标准的RL环境，支持PPO训练
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
import torch.nn.functional as F

from net.model_torchrl import AdaIRTorchRL  # 使用TorchRL版本
from utils.val_utils import compute_psnr_ssim
from utils.pytorch_ssim import SSIM
import lpips


class AdaIRGRPOEnv(EnvBase):
    """TorchRL环境包装器：将AdaIR的GRPO训练包装成标准RL环境"""
    
    def __init__(self, 
                 batch_size: int = 1,
                 device: str = "cuda",
                 reward_weights: Dict[str, float] = None):
        
        if reward_weights is None:
            reward_weights = {"psnr": 0.4, "ssim": 0.3, "lpips": 0.3}
        
        self.batch_size = batch_size
        self.reward_weights = reward_weights
        
        # 初始化AdaIR模型（TorchRL版本）
        self.adair_model = AdaIRTorchRL(decoder=True).to(device)
        self.adair_model.eval()
        
        # 奖励计算组件
        self.ssim_metric = SSIM().eval().to(device)
        self.lpips_metric = lpips.LPIPS(net='alex').eval().to(device)
        
        # 定义观察空间：退化图像patch
        self.observation_spec = CompositeSpec({
            "degraded_image": UnboundedContinuousTensorSpec(
                shape=(3, 128, 128), 
                device=device,
                dtype=torch.float32
            ),
            "clean_image": UnboundedContinuousTensorSpec(
                shape=(3, 128, 128),
                device=device, 
                dtype=torch.float32
            ),
        }, shape=(batch_size,))
        
        # 定义动作空间：每个FreModule的Beta分布参数
        # 3个FreModule × 8个参数 = 24维动作空间
        self.action_spec = CompositeSpec({
            "freq_params": BoundedTensorSpec(
                low=0.1, high=10.0,  # Beta分布参数范围
                shape=(3, 8),  # 3个FreModule，每个8个参数(4个alpha+4个beta)  
                device=device,
                dtype=torch.float32
            )
        }, shape=(batch_size,))
        
        # 定义奖励空间
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), device=device, dtype=torch.float32
        )
        
        super().__init__(
            device=device,
            batch_size=[batch_size] if batch_size > 1 else [],
        )
        
        # 当前状态
        self.current_degraded = None
        self.current_clean = None
        
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        """重置环境，加载新的图像数据"""
        # 这里通常从数据加载器获取新数据
        # 为演示，我们生成随机数据
        batch_shape = tensordict.shape if tensordict is not None else (self.batch_size,)
        
        self.current_degraded = torch.randn(*batch_shape, 3, 128, 128, device=self.device)
        self.current_clean = torch.randn(*batch_shape, 3, 128, 128, device=self.device)
        
        # 规范化到[0,1]
        self.current_degraded = torch.clamp(self.current_degraded, 0, 1)
        self.current_clean = torch.clamp(self.current_clean, 0, 1)
        
        return TensorDict({
            "degraded_image": self.current_degraded,
            "clean_image": self.current_clean,
        }, batch_size=batch_shape, device=self.device)
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """执行一步：使用策略动作运行AdaIR并计算奖励"""
        
        # 提取动作参数
        freq_params = tensordict["action"]["freq_params"]  # [B, 3, 8]
        
        # 运行AdaIR模型with策略参数
        restored_image = self._run_adair_with_policy(
            self.current_degraded, freq_params
        )
        
        # 计算奖励
        reward = self._compute_reward(restored_image, self.current_clean)
        
        # 准备输出
        next_state = TensorDict({
            "degraded_image": self.current_degraded,
            "clean_image": self.current_clean,
        }, batch_size=self.current_degraded.shape[:1], device=self.device)
        
        # 对于图像修复任务，每个episode只有一步
        done = torch.ones(self.current_degraded.shape[0], 1, device=self.device, dtype=torch.bool)
        
        return TensorDict({
            **next_state,
            "reward": reward,
            "done": done,
            "restored_image": restored_image,  # 用于调试和可视化
        }, batch_size=self.current_degraded.shape[:1], device=self.device)
    
    def _run_adair_with_policy(self, degraded: torch.Tensor, freq_params: torch.Tensor) -> torch.Tensor:
        """使用策略参数运行AdaIR模型"""
        
        with torch.no_grad():
            # 使用TorchRL版本的参数注入
            restored = self.adair_model(degraded, stochastic=True, external_params=freq_params)
            if isinstance(restored, tuple):
                restored = restored[0]  # 只要输出，不要log_prob
            
        return restored
    
    def _compute_reward(self, restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """计算多指标组合奖励"""
        
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
        
        # 输出3个FreModule × 8个参数的Beta分布参数
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3 * 8),  # 24个参数
            nn.Softplus(),  # 确保输出为正
        )
        
        # 初始化：输出接近1.0，对应Beta(1,1)均匀分布
        self.policy_head[-2].bias.data.fill_(0.0)
        self.policy_head[-2].weight.data.normal_(0, 0.01)
        
    def forward(self, observation: TensorDict) -> TensorDict:
        """策略网络前向传播"""
        
        degraded_image = observation["degraded_image"]  # [B, 3, 128, 128]
        
        # 提取特征
        features = self.feature_extractor(degraded_image)
        
        # 输出策略参数
        policy_params = self.policy_head(features) + 0.1  # 确保最小值0.1
        policy_params = policy_params.view(-1, 3, 8)  # [B, 3, 8]
        
        return TensorDict({
            "action": TensorDict({
                "freq_params": policy_params
            }, batch_size=degraded_image.shape[:1])
        }, batch_size=degraded_image.shape[:1]) 