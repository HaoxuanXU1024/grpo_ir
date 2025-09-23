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
from typing import Tuple

# TorchRL imports (only when needed)
if opt.grpo_torchrl:
    try:
        from tensordict import TensorDict
        from torch.distributions import Beta
        from grpo_torchrl_env import AdaIRGRPOEnv, AdaIRPolicyNetwork
        from net.model_torchrl import AdaIRTorchRL # 导入AdaIRTorchRL
    except ImportError as e:
        print(f"TorchRL imports failed: {e}")
        print("Please install TorchRL: bash install_torchrl.sh")
        exit(1)


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 切换到 AdaIRTorchRL 模型以支持更清晰的动作注入
        if opt.grpo_torchrl:
            self.net = AdaIRTorchRL(decoder=True)
        else:
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
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        # Since we have single-step episodes, this simplifies greatly.
        # The 'next_value' is 0 because the episode is 'done'.
        next_value = 0 
        delta = rewards + gamma * next_value * (1.0 - dones) - values
        advantages = delta + gamma * gae_lambda * last_gae_lam * (1.0 - dones) # last_gae_lam is 0
        
        returns = advantages + values
        return advantages, returns

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
        elif opt.grpo and opt.grpo_flow_style:
            # GRPO with flow-style per-image advantage normalization
            G = opt.grpo_group
            batch_size = degrad_patch.size(0)

            # 1. Repeat data for group sampling
            repeated_degrad = degrad_patch.repeat_interleave(G, dim=0)
            repeated_clean = clean_patch.repeat_interleave(G, dim=0)

            # 2. Forward pass with stochastic policy
            # out: [B*G, C, H, W], lp: [B*G]
            out, lp = self.net(repeated_degrad, stochastic=True)

            # 3. Compute rewards for each sample
            with torch.no_grad():
                w_psnr = opt.grpo_w_psnr
                w_ssim = opt.grpo_w_ssim
                w_lpips = opt.grpo_w_lpips
                eps = 1e-6
                
                # PSNR (approx, per-image)
                mse = torch.mean((out - repeated_clean) ** 2, dim=(1,2,3)) + eps
                psnr = 10.0 * torch.log10(1.0 / mse)
                psnr_norm = torch.clamp(psnr / 40.0, 0.0, 1.0)

                # SSIM (per-image)
                ssim_vals = []
                for i in range(out.size(0)):
                    ssim_vals.append(self.ssim_metric(out[i:i+1], repeated_clean[i:i+1]))
                ssim_vals = torch.stack(ssim_vals).squeeze()
                ssim_norm = torch.clamp(ssim_vals, 0.0, 1.0)
                
                # LPIPS (per-image)
                out_p = out * 2 - 1
                gt_p = repeated_clean * 2 - 1
                lpips_vals = self.lpips_metric(out_p, gt_p).squeeze()
                lp_norm = torch.clamp(lpips_vals, 0.0, 1.0)
                one_minus_lp = 1.0 - lp_norm
                
                # Total reward
                rewards_flat = w_psnr * psnr_norm + w_ssim * ssim_norm + w_lpips * one_minus_lp
            
            # 4. Per-image (group-wise) advantage normalization - THE CORE OF FLOW-GRPO
            rewards = rewards_flat.view(batch_size, G) # [B, G]
            mean_rewards = rewards.mean(dim=1, keepdim=True)
            std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards - mean_rewards) / std_rewards
            advantages_flat = advantages.view(-1) # [B*G]

            # 5. 纯GRPO策略梯度（无PPO剪裁/ratio）：REINFORCE with group-normalized advantages
            pg_loss = -(advantages_flat.detach() * lp).mean()

            # 6. Stabilization losses (optional but recommended)
            sup_loss = self.loss_fn(out, repeated_clean) # Supervised loss on all samples
            
            with torch.no_grad():
                det_out = self.net(repeated_degrad)
            cons_loss = self.loss_fn(out, det_out) # Consistency loss on all samples

            # 7. Total Loss
            loss = pg_loss + opt.grpo_lambda_sup * sup_loss + opt.grpo_lambda_consistency * cons_loss

            self.log_dict({
                "policy_loss": pg_loss,
                "total_loss": loss,
                "reward_mean": rewards.mean(),
            }, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            # GRPO: group sampling (original implementation)
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
            
            # 纯GRPO策略梯度（无PPO剪裁/ratio）
            logps = torch.stack(lps, dim=1)  # [B, G]
            pg_loss = -(advantages.detach() * logps).mean()

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
            
            # 精简日志记录 - 只保留核心指标
            self.log_dict({
                "policy_loss": pg_loss,
                "total_loss": loss,
                "reward_mean": rewards.mean(),
            }, on_step=False, on_epoch=True, prog_bar=True)
            return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        # TorchRL GRPO: 参数组提升策略相关学习率
        if opt.grpo_torchrl:
            def unique_params(param_list):
                seen = set()
                uniq = []
                for p in param_list:
                    if id(p) not in seen:
                        uniq.append(p)
                        seen.add(id(p))
                return uniq

            policy_params = list(self.policy_network.parameters()) if hasattr(self, 'policy_network') else []
            value_params = list(self.value_network.parameters()) if hasattr(self, 'value_network') else []

            # In TorchRL path we use external actions; internal GRPO heads are not used at inference.
            # For stability, when training policy-only we do NOT include backbone/LoRA/heads.
            fre_head_params = []
            lora_params = []
            norm_refine_params = []
            output_params = []

            # 去重
            policy_params = unique_params(policy_params)
            value_params = unique_params(value_params)
            fre_head_params = unique_params(fre_head_params)
            lora_params = unique_params(lora_params)
            norm_refine_params = unique_params(norm_refine_params)
            output_params = unique_params(output_params)

            base_lr = opt.lr if getattr(opt, 'lr', 0) > 0 else 5e-5
            lr_policy = max(base_lr * 8.0, 3e-4)
            lr_value = max(base_lr * 4.0, 1e-4)
            lr_heads = max(base_lr * 6.0, 1e-4)
            lr_mid = max(base_lr * 2.0, 5e-5)

            param_groups = []
            if opt.train_policy_only:
                # Freeze backbone to prevent degradation; train policy network only
                for p in self.net.parameters():
                    p.requires_grad = False
                if hasattr(self, 'value_network'):
                    for p in self.value_network.parameters():
                        p.requires_grad = False
                if len(policy_params) > 0:
                    param_groups.append({'params': policy_params, 'lr': lr_policy})
            else:
                # 全参数微调：策略相关仍然更高LR，其余使用base_lr
                all_rest = []
                used = {id(p) for group in [policy_params, value_params, fre_head_params, lora_params, norm_refine_params, output_params] for p in group}
                for p in self.parameters():
                    if id(p) not in used:
                        all_rest.append(p)
                if len(policy_params) > 0:
                    param_groups.append({'params': policy_params, 'lr': lr_policy})
                # keep the rest as-is only for full finetune
                if len(value_params) > 0:
                    param_groups.append({'params': value_params, 'lr': lr_value})
                if len(all_rest) > 0:
                    param_groups.append({'params': unique_params(all_rest), 'lr': base_lr})

            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            if opt.train_policy_only:
                print(f"[INFO] TorchRL GRPO optimizer (policy-only): policy_lr={lr_policy}; backbone frozen")
            else:
                print(f"[INFO] TorchRL GRPO optimizer with param groups: policy={lr_policy}, value={lr_value}, base={base_lr}")

        # Original GRPO or standard（非TorchRL）
        elif opt.train_policy_only:
            def unique_params(param_list):
                seen = set()
                uniq = []
                for p in param_list:
                    if id(p) not in seen:
                        uniq.append(p)
                        seen.add(id(p))
                return uniq

            fre_head_params = []
            for m in self.net.modules():
                if m.__class__.__name__ == 'FreModule':
                    for subn, p in m.named_parameters():
                        if subn.startswith('policy_rate') or subn.startswith('policy_fuse'):
                            fre_head_params.append(p)

            lora_params = []
            if opt.lora:
                for m in self.net.modules():
                    if hasattr(m, 'down') and hasattr(m, 'up') and hasattr(m, 'base'):
                        lora_params.extend(list(m.down.parameters()))
                        lora_params.extend(list(m.up.parameters()))

            norm_refine_params = []
            for name, module in self.net.named_modules():
                if any(dec_name in name for dec_name in ['decoder_level1', 'refinement']):
                    if 'norm' in name.lower() or isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        norm_refine_params.extend(list(module.parameters()))

            output_params = list(self.net.output.parameters()) if hasattr(self.net, 'output') else []

            fre_head_params = unique_params(fre_head_params)
            lora_params = unique_params(lora_params)
            norm_refine_params = unique_params(norm_refine_params)
            output_params = unique_params(output_params)

            base_lr = opt.lr if getattr(opt, 'lr', 0) > 0 else 5e-5
            lr_heads = max(base_lr * 8.0, 1e-4)
            lr_mid = max(base_lr * 2.0, 5e-5)

            param_groups = []
            if len(fre_head_params) > 0:
                param_groups.append({'params': fre_head_params, 'lr': lr_heads})
            if len(lora_params) > 0:
                param_groups.append({'params': lora_params, 'lr': lr_mid})
            if len(norm_refine_params) > 0:
                param_groups.append({'params': norm_refine_params, 'lr': lr_mid})
            if len(output_params) > 0:
                param_groups.append({'params': output_params, 'lr': lr_mid})

            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            print(f"[INFO] GRPO optimizer (policy-only) with param groups: fre_heads={lr_heads}, mid={lr_mid}")
        else:
            base_lr = opt.lr if not opt.grpo and getattr(opt, 'lr', 0) > 0 else 5e-5
            optimizer = optim.AdamW(self.parameters(), lr=base_lr, weight_decay=1e-4)

        # 调整scheduler：GRPO使用更长预热
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
                if opt.grpo_torchrl:
                    if hasattr(self, 'policy_network'):
                        params.extend(list(self.policy_network.parameters()))
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
        
        # 检查是否使用高级奖励系统
        use_advanced_rewards = getattr(opt, 'use_advanced_rewards', True)
        
        if use_advanced_rewards:
            # 使用高级奖励系统 - 平衡传统指标和先进指标
            reward_weights = {
                "clip_similarity": getattr(opt, 'grpo_w_clip', 0.25),
                "perceptual": getattr(opt, 'grpo_w_perceptual', 0.25), 
                "aesthetic": getattr(opt, 'grpo_w_aesthetic', 0.15),
                "psnr": getattr(opt, 'grpo_w_psnr_adv', 0.20),
                "ssim": getattr(opt, 'grpo_w_ssim_adv', 0.15),
            }
        else:
            # 使用原始奖励系统
            reward_weights = {
                "psnr": opt.grpo_w_psnr,
                "ssim": opt.grpo_w_ssim,
                "lpips": opt.grpo_w_lpips
            }
        
        # 创建TorchRL环境包装器
        self.grpo_env = AdaIRGRPOEnv(
            batch_size=opt.batch_size,
            device=current_device,
            reward_weights=reward_weights,
            use_advanced_rewards=use_advanced_rewards
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
        """TorchRL GRPO训练步骤 - 纯GRPO（组采样+相对优势，无PPO剪裁）"""
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
            if hasattr(self, '_target_policy_network'):
                self._target_policy_network = self._target_policy_network.to(current_device)
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
        
        # === 组采样：每个样本采样G次动作，计算组内相对优势 ===
        G = getattr(opt, 'grpo_group', 4)
        batch_size = degrad_patch.shape[0]

        all_log_probs = []  # [G, B]
        all_rewards = []    # [G, B]
        restored_images = []  # list of [B, C, H, W]

        for _ in range(G):
            action_td = self.policy_network(obs)
            raw_params = action_td["action"]["freq_params"]  # [B, 3, 8]
            cur_bs = raw_params.shape[0]
            reshaped_params = raw_params.view(cur_bs, -1)  # [B, 24]
            alpha_beta_pairs = reshaped_params.view(cur_bs, 12, 2)
            alphas = F.softplus(alpha_beta_pairs[..., 0]) + 1.0
            betas = F.softplus(alpha_beta_pairs[..., 1]) + 1.0
            policy_dist = Beta(alphas, betas)
            sampled_actions = policy_dist.rsample()  # [B, 12]
            log_prob = policy_dist.log_prob(sampled_actions).sum(dim=-1)  # [B]

            actions_reshaped = sampled_actions.view(cur_bs, 3, 4)
            env_input = TensorDict({
                **obs,
                "action": TensorDict({
                    "actions": actions_reshaped
                }, batch_size=degrad_patch.shape[:1])
            }, batch_size=degrad_patch.shape[:1], device=current_device)

            next_obs_td = self.grpo_env.step(env_input)
            reward_g = next_obs_td["reward"].squeeze(-1)  # [B]
            restored_g = next_obs_td["restored_image"]    # [B, C, H, W]

            all_log_probs.append(log_prob)
            all_rewards.append(reward_g)
            restored_images.append(restored_g)

        logps = torch.stack(all_log_probs, dim=1)   # [B, G]
        rewards = torch.stack(all_rewards, dim=1)   # [B, G]

        # 组内标准化优势（GRPO核心）
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean) / std

        # 纯GRPO策略梯度
        policy_loss = -(advantages.detach() * logps).mean()

        # 监督/一致性：使用每样本最佳reward对应的输出
        with torch.no_grad():
            det_output = self.net(degrad_patch)
        best_idx = torch.argmax(rewards, dim=1)  # [B]
        best_out = torch.stack([restored_images[g][i] for i, g in enumerate(best_idx.tolist())], dim=0)
        sup_loss = self.loss_fn(best_out, clean_patch)
        cons_loss = self.loss_fn(best_out, det_output)

        # 熵正则（鼓励探索）：用最近一次分布的平均熵近似
        entropy = 0.0
        try:
            entropy = policy_dist.entropy().mean()
        except Exception:
            pass
        entropy_loss = -0.01 * entropy if isinstance(entropy, torch.Tensor) else 0.0

        total_loss = policy_loss + entropy_loss + opt.grpo_lambda_sup * sup_loss + opt.grpo_lambda_consistency * cons_loss

        self.log_dict({
            "policy_loss": policy_loss,
            "total_loss": total_loss,
            "reward_mean": rewards.mean(),
        }, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss


def main():
    # 减少不必要的verbose输出
    import os
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("Options")
    print(opt)
    
    # Logger选择：仅rank0初始化W&B，避免多进程产生多个runs；其余rank禁用logger
    rank_str = os.environ.get("RANK", "0")
    try:
        global_rank = int(rank_str)
    except Exception:
        global_rank = 0

    if opt.disable_wandb:
        print("[INFO] Wandb disabled by --disable_wandb flag, using TensorBoard logger")
        logger = TensorBoardLogger(save_dir = "logs/")
    elif opt.wblogger is not None and global_rank == 0:
        try:
            logger = WandbLogger(
                project=opt.wblogger,
                name="AdaIR-GRPO",
                log_model=False,
                group="TorchRL-GRPO",
                job_type="train"
            )
            print(f"[INFO] Using Wandb logger (rank0) with project: {opt.wblogger}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Wandb logger: {e}")
            print("[INFO] Falling back to TensorBoard logger")
            logger = TensorBoardLogger(save_dir = "logs/")
    else:
        logger = False

    # 记录精简超参到W&B（仅rank0）
    if logger and isinstance(logger, WandbLogger):
        try:
            reward_cfg = {}
            if getattr(opt, 'use_advanced_rewards', False):
                reward_cfg = {
                    'clip': getattr(opt, 'grpo_w_clip', 0.25),
                    'perceptual': getattr(opt, 'grpo_w_perceptual', 0.25),
                    'aesthetic': getattr(opt, 'grpo_w_aesthetic', 0.15),
                    'psnr': getattr(opt, 'grpo_w_psnr_adv', 0.20),
                    'ssim': getattr(opt, 'grpo_w_ssim_adv', 0.15),
                }
            else:
                reward_cfg = {
                    'psnr': getattr(opt, 'grpo_w_psnr', 0.4),
                    'ssim': getattr(opt, 'grpo_w_ssim', 0.3),
                    'lpips': getattr(opt, 'grpo_w_lpips', 0.3),
                }
            logger.experiment.config.update({
                'lr': getattr(opt, 'lr', 5e-5),
                'epochs': getattr(opt, 'epochs', 50),
                'batch_size': getattr(opt, 'batch_size', 1),
                'grpo_group': getattr(opt, 'grpo_group', 4),
                'grpo_torchrl': getattr(opt, 'grpo_torchrl', False),
                'train_policy_only': getattr(opt, 'train_policy_only', False),
                'use_advanced_rewards': getattr(opt, 'use_advanced_rewards', False),
                'reward_cfg': reward_cfg,
                'clip_range': getattr(opt, 'grpo_clip_range', 0.2),
                'adv_clip_max': getattr(opt, 'grpo_adv_clip_max', 5.0),
                'max_grad_norm': getattr(opt, 'grpo_max_grad_norm', 0.5),
            }, allow_val_change=True)
        except Exception as e:
            print(f"[WARN] Failed to log hyperparams to WandB: {e}")

    # Optional: restrict training to worst-case lists for GRPO finetuning
    trainset = AdaIRTrainDataset(opt)
    if opt.finetune_worst:
        # Use worst_dir to construct file paths automatically
        worst_dir = opt.worst_dir
        worst_paths = {
            'derain': os.path.join(worst_dir, 'train_derain_worst.txt'),
            'dehaze': os.path.join(worst_dir, 'train_dehaze_worst.txt'),
            'deblur': os.path.join(worst_dir, 'train_deblur_worst.txt'),
            'enhance': os.path.join(worst_dir, 'train_enhance_worst.txt'),
            'denoise': os.path.join(worst_dir, 'train_denoise_worst_merged.txt'),
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
                print(f"[WARN] Worst list file not found: {path}")
                return []
            with open(path, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            print(f"[INFO] Loaded {len(lines)} samples from {path}")
            return lines

        # Derain: entries like 'rainy/rain-xxx.png' - store full path
        for rel in read_list(worst_paths['derain']):
            full_path = os.path.join(derain_root, rel)
            if os.path.exists(full_path):
                filtered.append({'clean_id': full_path, 'de_type': 3})

        # Dehaze: entries like 'synthetic/partX/xxx.jpg' - store full path
        for rel in read_list(worst_paths['dehaze']):
            full_path = os.path.join(dehaze_root, rel)
            if os.path.exists(full_path):
                filtered.append({'clean_id': full_path, 'de_type': 4})

        # Deblur: entries like 'blur/xxx.png' - extract filename only
        for rel in read_list(worst_paths['deblur']):
            if rel.startswith('blur/'):
                filename = rel.split('blur/')[-1]
            else:
                filename = rel
            blur_path = os.path.join(gopro_root, 'blur', filename)
            if os.path.exists(blur_path):
                filtered.append({'clean_id': filename, 'de_type': 5})

        # Enhance: entries like 'low/xxx.png' - extract filename only  
        for rel in read_list(worst_paths['enhance']):
            if rel.startswith('low/'):
                filename = rel.split('low/')[-1]
            else:
                filename = rel
            low_path = os.path.join(enhance_root, 'low', filename)
            if os.path.exists(low_path):
                filtered.append({'clean_id': filename, 'de_type': 6})

        # Denoise: entries are bare filenames - store full path
        for filename in read_list(worst_paths['denoise']):
            full_path = os.path.join(denoise_root, filename)
            if os.path.exists(full_path):
                # Use sigma 25 as middle difficulty by default
                filtered.append({'clean_id': full_path, 'de_type': 1})

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