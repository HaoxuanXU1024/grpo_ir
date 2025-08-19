"""
TorchRL Environment for GRPO Training on AdaIR
将AdaIR模型包装成标准的RL环境，支持PPO训练
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from tensordict import TensorDict
import torch.nn.functional as F

from net.model import AdaIR
from utils.val_utils import compute_psnr_ssim
from utils.pytorch_ssim import SSIM
import lpips


class AdaIRGRPOEnv:
    """AdaIR GRPO环境，支持真正的策略参数注入"""
    
    def __init__(self, 
                 batch_size: int = 1,
                 device: str = "cuda",
                 reward_weights: Dict[str, float] = None):
        
        if reward_weights is None:
            reward_weights = {"psnr": 0.4, "ssim": 0.3, "lpips": 0.3}
        
        self.batch_size = batch_size
        self.reward_weights = reward_weights
        self.device = device
        
        # 初始化AdaIR模型
        self.adair_model = AdaIR(decoder=True).to(device)
        self.adair_model.eval()
        
        # 奖励计算组件
        self.ssim_metric = SSIM().eval().to(device)
        self.lpips_metric = lpips.LPIPS(net='alex').eval().to(device)
        
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
        
        # 提取动作参数
        freq_params = action_dict["action"]["freq_params"]  # [B, 3, 8]
        
        # 运行AdaIR模型with策略参数
        restored_image = self._run_adair_with_policy(
            self.current_degraded, freq_params
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
    
    def _run_adair_with_policy(self, degraded: torch.Tensor, freq_params: torch.Tensor) -> torch.Tensor:
        """使用策略参数运行AdaIR模型 - 真正的参数注入版本"""
        
        # 注入策略参数到FreModule
        self._inject_policy_to_fremodules(freq_params)
        
        with torch.no_grad():
            # 使用stochastic=True模式，这样FreModule会使用我们注入的参数
            result = self.adair_model(degraded, stochastic=True)
            if isinstance(result, tuple):
                restored = result[0]  # (output, log_prob)
            else:
                restored = result
            
        return restored
    
    def _inject_policy_to_fremodules(self, freq_params: torch.Tensor):
        """将策略参数注入到AdaIR的FreModule中"""
        
        # 获取3个FreModule
        fre_modules = [self.adair_model.fre1, self.adair_model.fre2, self.adair_model.fre3]
        
        for i, fre_module in enumerate(fre_modules):
            params = freq_params[:, i, :]  # [B, 8]
            
            # 覆盖策略头的前向传播
            # 这是一个hack，但对于RL训练是必要的
            self._override_policy_head(fre_module, params)
    
    def _override_policy_head(self, fre_module, params):
        """临时覆盖FreModule的策略头输出"""
        
        # 分离rate和fuse参数
        rate_params = params[:, :4]   # [B, 4] - alpha_h, beta_h, alpha_w, beta_w  
        fuse_params = params[:, 4:8]  # [B, 4] - alpha_1, beta_1, alpha_2, beta_2
        
        # 创建临时的forward函数来替换policy_rate的输出
        def temp_rate_forward(x):
            # x是pooled特征 [B, dim, 1, 1]
            batch_size = x.size(0)
            # 直接返回我们的策略参数，确保形状正确
            output = rate_params.view(batch_size, 4, 1, 1)
            # 应用softplus确保参数为正
            return F.softplus(output) + 1e-4
        
        def temp_fuse_forward(x):
            # x是pooled特征 [B, dim, 1, 1] 
            batch_size = x.size(0)
            # 直接返回我们的融合参数
            output = fuse_params.view(batch_size, 4, 1, 1)
            return F.softplus(output) + 1e-4
        
        # 临时替换forward方法
        fre_module.policy_rate._temp_forward = fre_module.policy_rate.forward
        fre_module.policy_rate.forward = temp_rate_forward
        
        fre_module.policy_fuse._temp_forward = fre_module.policy_fuse.forward  
        fre_module.policy_fuse.forward = temp_fuse_forward
    
    def _restore_policy_heads(self):
        """恢复原始的策略头"""
        fre_modules = [self.adair_model.fre1, self.adair_model.fre2, self.adair_model.fre3]
        
        for fre_module in fre_modules:
            if hasattr(fre_module.policy_rate, '_temp_forward'):
                fre_module.policy_rate.forward = fre_module.policy_rate._temp_forward
                delattr(fre_module.policy_rate, '_temp_forward')
                
            if hasattr(fre_module.policy_fuse, '_temp_forward'):
                fre_module.policy_fuse.forward = fre_module.policy_fuse._temp_forward
                delattr(fre_module.policy_fuse, '_temp_forward')
    
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
        
        # 恢复策略头
        self._restore_policy_heads()
        
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
        )
        
        # 初始化：输出接近1.0，对应Beta(1,1)均匀分布
        self.policy_head[-1].bias.data.fill_(1.0)  # 初始化为1.0
        self.policy_head[-1].weight.data.normal_(0, 0.01)
        
    def forward(self, observation: TensorDict) -> TensorDict:
        """策略网络前向传播"""
        
        degraded_image = observation["degraded_image"]  # [B, 3, 128, 128]
        
        # 提取特征
        features = self.feature_extractor(degraded_image)
        
        # 输出策略参数
        policy_params = self.policy_head(features)  # [B, 24]
        policy_params = torch.exp(policy_params) + 0.1  # 确保为正，使用exp而不是softplus
        policy_params = policy_params.view(-1, 3, 8)  # [B, 3, 8]
        
        return TensorDict({
            "action": TensorDict({
                "freq_params": policy_params
            }, batch_size=degraded_image.shape[:1])
        }, batch_size=degraded_image.shape[:1]) 