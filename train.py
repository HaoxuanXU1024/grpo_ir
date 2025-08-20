import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset
from net.model import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.lora import apply_lora_to_adair
from PIL import Image
import numpy as np
import wandb
from options import options as opt
from utils.val_utils import compute_psnr_ssim
from utils.pytorch_ssim import SSIM as TorchSSIM
import lpips
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F

# TorchRL imports (only when needed)
if opt.grpo_torchrl:
    try:
        from tensordict import TensorDict
        from torch.distributions import Beta
        from grpo_torchrl_env import AdaIRGRPOEnv, AdaIRPolicyNetwork
    except ImportError as e:
        print(f"TorchRL imports failed: {e}")
        print("Please install TorchRL: bash install_torchrl.sh")
        exit(1)


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        # Cache baseline output for consistency (computed on-the-fly per batch)
        self.save_hyperparameters(ignore=['net'])
        # Metrics
        self.ssim_metric = TorchSSIM().eval()
        # Use LPIPS with AlexNet backbone for faster evaluation
        self.lpips_metric = lpips.LPIPS(net='alex').eval()
        
        # GRPO global reward statistics for stable advantage normalization
        self.register_buffer('running_reward_mean', torch.tensor(0.0))
        self.register_buffer('running_reward_var', torch.tensor(1.0))
        self.register_buffer('reward_count', torch.tensor(0.0))
        
        # TorchRL GRPO components (only initialized when needed)
        if opt.grpo_torchrl:
            self.setup_torchrl_grpo()
    
    def compute_policy_loss_ppo(self, log_probs, old_log_probs, advantages, clip_range=None):
        """Compute PPO-style clipped policy loss for more stable training"""
        if clip_range is None:
            clip_range = opt.grpo_clip_range
        
        ratio = torch.exp(log_probs - old_log_probs.detach())
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    def update_global_reward_stats(self, rewards):
        """Update global reward statistics for stable advantage normalization"""
        if not opt.grpo_global_norm:
            return rewards  # Use batch normalization as before
            
        batch_size = rewards.numel()
        self.reward_count += batch_size
        
        # Exponential moving average with adaptive learning rate
        alpha = min(0.1, 1.0 / self.reward_count.item())
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        
        self.running_reward_mean = (1 - alpha) * self.running_reward_mean + alpha * batch_mean
        self.running_reward_var = (1 - alpha) * self.running_reward_var + alpha * batch_var
        
        return rewards
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        if not opt.grpo:
            restored = self.net(degrad_patch)
            loss = self.loss_fn(restored,clean_patch)
            self.log("train_loss", loss)
            return loss
        elif opt.grpo_torchrl:
            # TorchRL GRPO: Use more stable PPO implementation
            return self.torchrl_grpo_step(degrad_patch, clean_patch)
        else:
            # GRPO: group sampling
            with torch.no_grad():
                det_out = self.net(degrad_patch)  # baseline deterministic output
            G = opt.grpo_group
            # Collect stochastic samples and log-probs
            rets = []
            lps = []
            for _ in range(G):
                out, lp = self.net(degrad_patch, stochastic=True)
                rets.append(out)
                lps.append(lp)
            # Compute rewards: mix of PSNR, SSIM, (1-LPIPS), optional NIQE (omitted for speed in training)
            # Normalize terms to [0,1] approximately, combine with user weights
            rewards = []  # list of [B]
            with torch.no_grad():
                w_psnr = opt.grpo_w_psnr
                w_ssim = opt.grpo_w_ssim
                w_lpips = opt.grpo_w_lpips
                w_niqe = opt.grpo_w_niqe
                eps = 1e-6
                for out in rets:
                    # PSNR (approx, per-image). Here we compute MSE then convert to PSNR; clamp for stability
                    mse = torch.mean((out - clean_patch) ** 2, dim=(1,2,3)) + eps
                    psnr = 10.0 * torch.log10(1.0 / mse)
                    psnr_norm = torch.clamp(psnr / 40.0, 0.0, 1.0)  # assume 40dB as near-perfect

                    # SSIM (torch implementation operates per-batch; returns mean)
                    # For reward per-sample, compute in loop
                    ssim_vals = []
                    for i in range(out.size(0)):
                        ssim_vals.append(self.ssim_metric(out[i:i+1], clean_patch[i:i+1]))
                    ssim_vals = torch.stack(ssim_vals).squeeze()
                    ssim_norm = torch.clamp(ssim_vals, 0.0, 1.0)

                    # LPIPS (lower better) -> (1 - LPIPS_norm)
                    # lpips expects [-1,1]
                    out_p = out * 2 - 1
                    gt_p = clean_patch * 2 - 1
                    lp = self.lpips_metric(out_p, gt_p).squeeze()  # [B]
                    lp_norm = torch.clamp(lp, 0.0, 1.0)
                    one_minus_lp = 1.0 - lp_norm

                    # NIQE omitted in training loop for speed; if w_niqe>0, set neutral value 0.5
                    niqe_term = torch.full_like(psnr_norm, 0.5)

                    total = w_psnr * psnr_norm + w_ssim * ssim_norm + w_lpips * one_minus_lp + w_niqe * niqe_term
                    rewards.append(total)
            rewards = torch.stack(rewards, dim=1)  # [B, G]
            
            # Update global reward statistics
            self.update_global_reward_stats(rewards)
            
            # Improved advantage computation with stable normalization
            if opt.grpo_global_norm and self.reward_count > 100:  # Use global stats after enough samples
                advantages = (rewards - self.running_reward_mean) / (self.running_reward_var.sqrt() + 1e-4)
            else:
                # Batch-wise normalization with improved stability
                mean = rewards.mean(dim=1, keepdim=True)
                std = rewards.std(dim=1, keepdim=True) + 1e-4
                advantages = (rewards - mean) / std
            
            # Clip advantages for stability
            advantages = torch.clamp(advantages, -opt.grpo_adv_clip_max, opt.grpo_adv_clip_max)
            
            # Policy gradient with PPO-style clipped loss
            logps = torch.stack(lps, dim=1)  # [B, G]
            old_logps = logps.detach()  # Use current logps as "old" for first iteration
            pg_loss = self.compute_policy_loss_ppo(logps, old_logps, advantages.detach())

            # Stabilizers: supervised L1 on best sample + consistency to deterministic baseline
            # pick best per-sample by reward
            best_idx = torch.argmax(rewards, dim=1)  # [B]
            best_out = torch.stack([rets[g][i] for i, g in enumerate(best_idx.tolist())], dim=0)
            sup_loss = self.loss_fn(best_out, clean_patch)
            cons_loss = self.loss_fn(best_out, det_out)

            # KL regularization between best and deterministic outputs
            kl_loss = 0.0
            if opt.grpo_beta_kl > 0:
                kl_loss = torch.mean((best_out - det_out) ** 2)
            
            # Adaptive loss weighting for progressive training
            current_epoch = self.current_epoch
            if current_epoch < opt.grpo_warmup_epochs:
                # Supervised warmup phase
                lambda_sup = 1.0
                lambda_cons = 0.1
                lambda_pg = 0.0  # No policy gradient during warmup
            else:
                # Progressive GRPO phase
                if opt.grpo_schedule_sup:
                    # Gradually reduce supervised loss weight
                    progress = min((current_epoch - opt.grpo_warmup_epochs) / max(opt.epochs - opt.grpo_warmup_epochs, 1), 1.0)
                    lambda_sup = opt.grpo_lambda_sup * (1.0 - 0.5 * progress)  # Reduce to 50% of original
                else:
                    lambda_sup = opt.grpo_lambda_sup
                lambda_cons = opt.grpo_lambda_consistency
                lambda_pg = 1.0

            loss = lambda_pg * pg_loss + lambda_sup * sup_loss + lambda_cons * cons_loss + opt.grpo_beta_kl * kl_loss
            
            # Comprehensive logging for monitoring training stability
            self.log_dict({
                "loss_pg": pg_loss,
                "loss_sup": sup_loss,
                "loss_cons": cons_loss,
                "loss_kl": kl_loss,
                "train_loss": loss,
                "advantages_mean": advantages.mean(),
                "advantages_std": advantages.std(),
                "rewards_mean": rewards.mean(),
                "rewards_std": rewards.std(),
                "lambda_sup": lambda_sup,
                "lambda_pg": lambda_pg,
                "global_reward_mean": self.running_reward_mean,
                "global_reward_std": self.running_reward_var.sqrt(),
            })
            return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        # TorchRL GRPO: Optimize policy and value networks
        if opt.grpo_torchrl:
            params = []
            
            if opt.train_policy_only:
                # Add policy network and value network params
                params.extend(list(self.policy_network.parameters()))
                params.extend(list(self.value_network.parameters()))
                
                # Add FreModule policy heads
                for m in self.net.modules():
                    name = m.__class__.__name__
                    if name == 'FreModule':
                        for subn, p in m.named_parameters():
                            if subn.startswith('policy_rate') or subn.startswith('policy_fuse'):
                                params.append(p)
                
                # Add LoRA params if wrapped
                if opt.lora:
                    for m in self.net.modules():
                        if hasattr(m, 'down') and hasattr(m, 'up') and hasattr(m, 'base'):
                            params.extend(list(m.down.parameters()))
                            params.extend(list(m.up.parameters()))
            else:
                params = list(self.parameters())
            
            # 修复学习率：使用极保守的学习率设置以确保稳定性
            if opt.lr > 0:
                lr = opt.lr * 0.01  # 极大幅降低，如2e-4 * 0.01 = 2e-6
            else:
                lr = 2e-6  # 使用极保守的默认学习率
            
            optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
            print(f"[INFO] TorchRL GRPO optimizer: lr={lr} (极保守设置), params={len(params)}")
        
        # Original GRPO or standard training
        elif opt.train_policy_only:
            policy_params = []
            for m in self.net.modules():
                name = m.__class__.__name__
                if name == 'FreModule':
                    for subn, p in m.named_parameters():
                        if subn.startswith('policy_rate') or subn.startswith('policy_fuse'):
                            policy_params.append(p)
                # Collect LoRA params if wrapped
                if hasattr(m, 'down') and hasattr(m, 'up') and hasattr(m, 'base'):
                    for p in m.down.parameters():
                        policy_params.append(p)
                    for p in m.up.parameters():
                        policy_params.append(p)
            
            # Use lower learning rate for GRPO training
            lr = opt.lr if not opt.grpo else opt.lr * 0.1
            optimizer = optim.AdamW(policy_params, lr=lr, weight_decay=1e-4)
        else:
            lr = 2e-4 if not opt.grpo else 5e-6  # Much lower LR for GRPO
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
            
        # Adjust scheduler for GRPO training
        if opt.grpo:
            # Use linear warmup + cosine annealing with longer warmup for GRPO
            warmup_epochs = max(opt.grpo_warmup_epochs, 15)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer, 
                warmup_epochs=warmup_epochs, 
                max_epochs=opt.epochs
            )
        else:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer, 
                warmup_epochs=15, 
                max_epochs=180
            )

        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add gradient clipping for GRPO training
        if opt.grpo and opt.grpo_max_grad_norm > 0:
            # Get parameters based on training mode
            if opt.train_policy_only:
                params = []
                
                # 修复：为TorchRL GRPO添加策略网络和价值网络参数
                if opt.grpo_torchrl:
                    # 添加TorchRL组件参数
                    if hasattr(self, 'policy_network'):
                        params.extend(list(self.policy_network.parameters()))
                    if hasattr(self, 'value_network'):
                        params.extend(list(self.value_network.parameters()))
                
                # 添加FreModule策略头参数
                for m in self.net.modules():
                    name = m.__class__.__name__
                    if name == 'FreModule':
                        for subn, p in m.named_parameters():
                            if subn.startswith('policy_rate') or subn.startswith('policy_fuse'):
                                params.append(p)
                    # Collect LoRA params if wrapped
                    if hasattr(m, 'down') and hasattr(m, 'up') and hasattr(m, 'base'):
                        for p in m.down.parameters():
                            params.append(p)
                        for p in m.up.parameters():
                            params.append(p)
            else:
                params = self.parameters()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(params, opt.grpo_max_grad_norm)
        
        # Call the parent optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def setup_torchrl_grpo(self):
        """设置TorchRL GRPO组件"""
        # 获取正确的设备
        current_device = next(self.parameters()).device
        
        # 创建TorchRL环境包装器
        self.grpo_env = AdaIRGRPOEnv(
            batch_size=opt.batch_size,
            device=current_device,
            reward_weights={
                "psnr": opt.grpo_w_psnr,
                "ssim": opt.grpo_w_ssim,
                "lpips": opt.grpo_w_lpips
            }
        )
        self.grpo_env.adair_model = self.net  # 使用主模型
        
        # 创建策略网络
        self.policy_network = AdaIRPolicyNetwork().to(current_device)
        
        # 创建价值网络
        self.value_network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(current_device)
        
        print("[INFO] TorchRL GRPO components initialized")
    
    def torchrl_grpo_step(self, degrad_patch, clean_patch):
        """TorchRL GRPO训练步骤 - 修正版PPO实现"""
        # 获取正确的设备
        current_device = next(self.parameters()).device
        
        # 确保输入数据在正确的设备上
        degrad_patch = degrad_patch.to(current_device)
        clean_patch = clean_patch.to(current_device)
        
        # 优化设备同步：只在设备不匹配时才执行.to()操作
        if not hasattr(self, '_device_synced') or self._device_synced != current_device:
            self.net = self.net.to(current_device)
            self.policy_network = self.policy_network.to(current_device) 
            self.value_network = self.value_network.to(current_device)
            self._device_synced = current_device
            print(f"[INFO] Synced TorchRL components to device: {current_device}")
        
        # 更新环境中的模型引用
        self.grpo_env.adair_model = self.net
        self.grpo_env.device = current_device
        
        # 设置环境数据
        self.grpo_env.set_data(degrad_patch, clean_patch)
        
        # 创建观察
        obs = TensorDict({
            "degraded_image": degrad_patch,
            "clean_image": clean_patch,
        }, batch_size=degrad_patch.shape[:1], device=current_device)
        
        # === 策略采样与概率计算 ===
        action_td = self.policy_network(obs)
        raw_params = action_td["action"]["freq_params"]  # [B, 3, 8]
        
        # 修复维度处理：将24个参数转换为正确的Beta分布参数
        # 每个FreModule有8个参数：4个rate参数 + 4个fuse参数
        # 每4个参数对应2对(alpha,beta)：rate(alpha,beta) + rate(alpha,beta), fuse(alpha,beta) + fuse(alpha,beta)
        # 总共：3个FreModule × 4对(alpha,beta) = 12对(alpha,beta) = 24个参数 ✓
        batch_size = raw_params.shape[0]
        
        # 但实际上，根据FreModule的定义：
        # policy_rate输出4个：[alpha_h, beta_h, alpha_w, beta_w] - 2对(alpha,beta)
        # policy_fuse输出4个：[alpha_1, beta_1, alpha_2, beta_2] - 2对(alpha,beta)  
        # 每个FreModule共4对(alpha,beta)，3个FreModule共12对(alpha,beta)
        
        # 重新整形：[B, 3, 8] -> [B, 24] -> [B, 12, 2]
        reshaped_params = raw_params.view(batch_size, -1)  # [B, 24]
        alpha_beta_pairs = reshaped_params.view(batch_size, 12, 2)  # [B, 12, 2]
        
        # 确保alpha, beta > 0 (Beta分布要求)
        alphas = F.softplus(alpha_beta_pairs[..., 0]) + 1.0  # [B, 12] 
        betas = F.softplus(alpha_beta_pairs[..., 1]) + 1.0   # [B, 12]
        
        # 创建Beta分布
        policy_dist = Beta(alphas, betas)
        
        # 从分布中采样动作
        sampled_actions = policy_dist.sample()  # [B, 12]
        
        # 计算当前策略的log概率
        current_log_probs = policy_dist.log_prob(sampled_actions).sum(dim=-1)  # [B]
        
        # === 获取或初始化old_log_probs ===
        # 使用目标网络方法：维护一个缓慢更新的"旧"策略网络
        
        if not hasattr(self, '_target_policy_network'):
            # 创建目标策略网络（旧策略的拷贝）
            import copy
            self._target_policy_network = copy.deepcopy(self.policy_network)
            self._target_update_count = 0
        
        # 每N步更新一次目标网络
        self._target_update_count += 1
        if self._target_update_count % 10 == 0:  # 每10步更新一次
            # 软更新：target = 0.9 * target + 0.1 * current
            for target_param, current_param in zip(self._target_policy_network.parameters(), 
                                                   self.policy_network.parameters()):
                target_param.data = 0.9 * target_param.data + 0.1 * current_param.data
        
        # 使用目标网络计算old_log_probs
        with torch.no_grad():
            old_action_td = self._target_policy_network(obs)
            old_raw_params = old_action_td["action"]["freq_params"]
            
            # 重新计算old分布参数
            old_reshaped = old_raw_params.view(batch_size, -1)
            old_alpha_beta = old_reshaped.view(batch_size, 12, 2)
            old_alphas = F.softplus(old_alpha_beta[..., 0]) + 1.0
            old_betas = F.softplus(old_alpha_beta[..., 1]) + 1.0
            old_policy_dist = Beta(old_alphas, old_betas)
            
            # 使用相同的采样动作计算old_log_probs
            old_log_probs = old_policy_dist.log_prob(sampled_actions).sum(dim=-1)
        
        # 将采样的动作重新整形为FreModule期望的格式
        # sampled_actions: [B, 12] - 12个Beta分布采样值
        # 需要重新整形为：[B, 3, 8] - 3个FreModule，每个8个参数
        
        # 每个FreModule需要8个参数：
        # - policy_rate: 4个参数 [alpha_h, beta_h, alpha_w, beta_w] -> 采样2个值 [r_h, r_w] 
        # - policy_fuse: 4个参数 [alpha_1, beta_1, alpha_2, beta_2] -> 采样2个值 [g_1, g_2]
        # 所以每个FreModule从4对(alpha,beta)采样得到4个值
        
        # 重新整形：[B, 12] -> [B, 3, 4] -> [B, 3, 8] (重复以匹配原始格式)
        freq_params_sampled = sampled_actions.view(batch_size, 3, 4)  # [B, 3, 4]
        
        # 扩展为[B, 3, 8]格式以兼容环境接口
        # 前4个是rate相关，后4个是fuse相关
        freq_params_expanded = torch.zeros(batch_size, 3, 8, device=current_device)
        freq_params_expanded[:, :, :4] = freq_params_sampled  # rate相关的4个采样值
        freq_params_expanded[:, :, 4:8] = freq_params_sampled  # fuse相关的4个采样值 (复用)
        
        # === 环境步进 ===
        env_input = TensorDict({
            **obs, 
            "action": TensorDict({
                "freq_params": freq_params_expanded
            }, batch_size=degrad_patch.shape[:1])
        }, batch_size=degrad_patch.shape[:1], device=current_device)
        
        next_obs_td = self.grpo_env.step(env_input)
        
        # === 价值估计 ===
        values = self.value_network(degrad_patch).squeeze(-1)
        
        # === 奖励与优势计算 ===
        rewards = next_obs_td["reward"].squeeze(-1).to(current_device)
        values = values.to(current_device)
        
        # 优势计算（简化版GAE）
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === 正确的PPO损失计算 ===
        # 计算重要性采样比率
        ratio = torch.exp(current_log_probs - old_log_probs.detach())
        
        # 紧急安全检查：如果ratio过于极端，直接停止策略更新
        if torch.any(ratio > 100) or torch.any(ratio < 0.01):
            print(f"[EMERGENCY] 检测到极端ratio值: max={ratio.max():.1f}, min={ratio.min():.3f}")
            print("[EMERGENCY] 跳过此次策略更新，仅使用监督损失")
            # 返回纯监督损失，跳过策略更新
            with torch.no_grad():
                det_output = self.net(degrad_patch)
            restored_image = next_obs_td["restored_image"]
            emergency_loss = self.loss_fn(restored_image, clean_patch)
            return emergency_loss
        
        # 强制限制ratio到安全范围
        ratio = torch.clamp(ratio, 0.1, 10.0)  # 更严格的硬性限制
        
        # 添加更强的约束：如果ratio过大，进行额外的KL惩罚
        kl_divergence = current_log_probs - old_log_probs.detach()
        mean_kl = kl_divergence.mean()
        
        # PPO clipped objective with enhanced constraints
        clip_range = opt.grpo_clip_range
        
        # 根据KL散度动态调整clip_range
        if mean_kl.abs() > 1.0:  # 极大的KL散度
            clip_range = clip_range * 0.1  # 严重收紧
            print(f"[WARNING] KL散度极大({mean_kl:.3f})，严重收紧clip_range至{clip_range:.3f}")
        elif mean_kl.abs() > 0.5:
            clip_range = clip_range * 0.5
            print(f"[INFO] KL散度过大({mean_kl:.3f})，收紧clip_range至{clip_range:.3f}")
        
        # 更保守的ratio clipping
        ratio_clipped = torch.clamp(ratio, 0.5, 2.0)  # 硬性限制ratio范围
        
        surr1 = ratio_clipped * advantages
        surr2 = torch.clamp(ratio_clipped, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 添加更强的KL惩罚项
        kl_penalty = 0.1 * mean_kl.abs()  # 增加KL惩罚系数
        policy_loss = policy_loss + kl_penalty
        
        # === 价值损失 ===
        value_loss = nn.MSELoss()(values, rewards)
        
        # === 熵正则化（鼓励探索）===
        entropy = policy_dist.entropy().mean()
        entropy_loss = -0.05 * entropy  # 增加熵系数，鼓励更多探索
        
        # === 监督损失（稳定项）===
        with torch.no_grad():
            det_output = self.net(degrad_patch)
        restored_image = next_obs_td["restored_image"]
        sup_loss = self.loss_fn(restored_image, clean_patch)
        cons_loss = self.loss_fn(restored_image, det_output)
        
        # === 总损失 ===
        total_loss = (policy_loss + 
                     0.5 * value_loss +
                     entropy_loss +
                     opt.grpo_lambda_sup * sup_loss + 
                     opt.grpo_lambda_consistency * cons_loss)
        
        # PPO训练不需要手动更新old_log_probs，它们在每个batch开始时自动重置
        
        # === 验证检查 ===
        # 检查关键值是否在合理范围内
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[WARNING] Invalid total_loss detected: {total_loss}")
        
        if ratio.mean() > 2.0 or ratio.mean() < 0.5:
            print(f"[WARNING] PPO ratio异常: mean={ratio.mean():.3f}, std={ratio.std():.3f}")
        
        if rewards.mean() < 0 or rewards.mean() > 1:
            print(f"[WARNING] 奖励范围异常: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
        
        # 检查梯度是否正常
        total_grad_norm = 0.0
        for p in self.policy_network.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2) ** 2
        total_grad_norm = total_grad_norm ** 0.5
        if total_grad_norm > 10.0:
            print(f"[WARNING] 策略网络梯度过大: {total_grad_norm:.3f}")
        
        # === 详细日志记录 ===
        self.log("torchrl_policy_loss", policy_loss)
        self.log("torchrl_kl_penalty", kl_penalty)
        self.log("torchrl_kl_divergence", mean_kl.abs())
        self.log("torchrl_value_loss", value_loss)
        self.log("torchrl_entropy_loss", entropy_loss)
        self.log("torchrl_entropy", entropy)
        self.log("torchrl_sup_loss", sup_loss)
        self.log("torchrl_cons_loss", cons_loss)
        self.log("torchrl_total_loss", total_loss)
        self.log("torchrl_reward_mean", rewards.mean())
        self.log("torchrl_advantages_mean", advantages.mean())
        self.log("torchrl_advantages_std", advantages.std())
        self.log("torchrl_ratio_mean", ratio.mean())
        self.log("torchrl_ratio_std", ratio.std())
        self.log("torchrl_ratio_clipped_mean", ratio_clipped.mean())
        self.log("torchrl_current_log_probs_mean", current_log_probs.mean())
        self.log("torchrl_old_log_probs_mean", old_log_probs.mean())
        self.log("torchrl_alpha_mean", alphas.mean())
        self.log("torchrl_beta_mean", betas.mean())
        
        return total_loss


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    # Optional: restrict training to worst-case lists for GRPO finetuning
    trainset = AdaIRTrainDataset(opt)
    if opt.finetune_worst:
        # Override sample_ids to only include worst lists
        worst_paths = {
            'derain': opt.worst_derain,
            'dehaze': opt.worst_dehaze,
            'deblur': opt.worst_deblur,
            'enhance': opt.worst_enhance,
            'denoise': opt.worst_denoise,
        }
        filtered = []
        # Build maps from known training directories
        derain_root = opt.derain_dir if hasattr(opt, 'derain_dir') else 'data/Train/Derain/'
        dehaze_root = opt.dehaze_dir if hasattr(opt, 'dehaze_dir') else 'data/Train/Dehaze/'
        gopro_root = opt.gopro_dir if hasattr(opt, 'gopro_dir') else 'data/Train/Deblur/'
        enhance_root = opt.enhance_dir if hasattr(opt, 'enhance_dir') else 'data/Train/Enhance/'
        denoise_root = opt.denoise_dir if hasattr(opt, 'denoise_dir') else 'data/Train/Denoise/'

        # Helper to safely read list
        def read_list(path):
            if not os.path.exists(path):
                return []
            return [ln.strip() for ln in open(path) if ln.strip()]

        # Derain: entries like 'rainy/rain-xxx.png'
        for rel in read_list(worst_paths['derain']):
            rainy_path = os.path.join(derain_root, rel)
            if os.path.exists(rainy_path):
                filtered.append({'clean_id': rainy_path, 'de_type': 3})

        # Dehaze: entries are relative under dehaze_dir (synthetic/...)
        for rel in read_list(worst_paths['dehaze']):
            syn_path = os.path.join(dehaze_root, rel)
            if os.path.exists(syn_path):
                filtered.append({'clean_id': syn_path, 'de_type': 4})

        # Deblur: entries like 'blur/xxx.png'
        for rel in read_list(worst_paths['deblur']):
            if rel.startswith('blur/'):
                name = rel.split('blur/')[-1]
            else:
                name = rel
            blur_path = os.path.join(gopro_root, 'blur', name)
            if os.path.exists(blur_path):
                filtered.append({'clean_id': name, 'de_type': 5})

        # Enhance: entries like 'low/xxx.png'
        for rel in read_list(worst_paths['enhance']):
            if rel.startswith('low/'):
                name = rel.split('low/')[-1]
            else:
                name = rel
            low_path = os.path.join(enhance_root, 'low', name)
            if os.path.exists(low_path):
                filtered.append({'clean_id': name, 'de_type': 6})

        # Denoise: list contains filenames in denoise_dir
        for name in read_list(worst_paths['denoise']):
            cpath = os.path.join(denoise_root, name)
            if os.path.exists(cpath):
                # Use sigma 25 as middle difficulty by default; GRPO 训练中会随机 group 采样
                filtered.append({'clean_id': cpath, 'de_type': 1})

        if len(filtered) == 0:
            print('[WARN] No worst samples found; falling back to full train set')
        else:
            trainset.sample_ids = filtered
            print(f"[INFO] Using worst lists for finetune: {len(filtered)} samples")
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = AdaIRModel()

    # Optionally resume from a checkpoint (strict=False for GRPO heads compatibility)
    if opt.resume_ckpt is not None and os.path.exists(opt.resume_ckpt):
        print(f"[INFO] Resuming weights from {opt.resume_ckpt} (strict=False)")
        model = model.load_from_checkpoint(opt.resume_ckpt, strict=False)

    # Optionally inject LoRA adapters
    if opt.lora:
        wrapped = apply_lora_to_adair(model.net, targets=opt.lora_targets, rank=opt.lora_r, alpha=opt.lora_alpha, dropout=opt.lora_dropout)
        print(f"[INFO] Applied LoRA to {wrapped} layers (targets={opt.lora_targets}, r={opt.lora_r}, alpha={opt.lora_alpha})")
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()