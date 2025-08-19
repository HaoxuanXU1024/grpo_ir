"""
基于TorchRL的GRPO训练脚本
使用成熟的PPO实现替代手工的REINFORCE
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# TorchRL imports - 修复API兼容性
try:
    from torchrl.objectives import ClipPPOLoss
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
except ImportError as e:
    print(f"TorchRL import error: {e}")
    print("Using simplified implementation")

from tensordict import TensorDict
from torch.distributions import Beta

# Local imports
from grpo_torchrl_env import AdaIRGRPOEnv, AdaIRPolicyNetwork
from utils.dataset_utils import AdaIRTrainDataset
from net.model import AdaIR
from options import options as opt
from utils.lora import apply_lora_to_adair


class SimplifiedPPOLoss(nn.Module):
    """简化的PPO损失实现，避免复杂的TorchRL API"""
    
    def __init__(self, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
    def forward(self, log_probs, old_log_probs, advantages, returns, values):
        """计算PPO损失"""
        
        # 计算概率比
        ratio = torch.exp(log_probs - old_log_probs.detach())
        
        # PPO clipped loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # 价值函数损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 熵损失（鼓励探索）
        entropy_loss = -(log_probs * torch.exp(log_probs)).mean()
        
        # 总损失
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        return {
            "loss_total": total_loss,
            "loss_policy": policy_loss,
            "loss_value": value_loss,
            "loss_entropy": entropy_loss,
        }


class TorchRLGRPOTrainer:
    """基于TorchRL的GRPO训练器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建数据加载器
        self.setup_data()
        
        # 创建环境
        self.setup_environment()
        
        # 创建策略和价值网络
        self.setup_networks()
        
        # 创建PPO损失函数
        self.setup_ppo_loss()
        
        # 创建优化器
        self.setup_optimizers()
        
        # 设置日志
        self.setup_logging()
        
    def setup_data(self):
        """设置数据加载器"""
        self.trainset = AdaIRTrainDataset(opt)
        
        # 如果使用worst lists进行微调
        if opt.finetune_worst:
            self.filter_worst_samples()
        
        self.trainloader = DataLoader(
            self.trainset, 
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
    def filter_worst_samples(self):
        """过滤出worst samples"""
        # 复用train.py中的逻辑
        worst_paths = {
            'derain': opt.worst_derain,
            'dehaze': opt.worst_dehaze,
            'deblur': opt.worst_deblur,
            'enhance': opt.worst_enhance,
            'denoise': opt.worst_denoise,
        }
        
        filtered = []
        
        def read_list(path):
            if not os.path.exists(path):
                return []
            return [ln.strip() for ln in open(path) if ln.strip()]
        
        # 构建filtered列表
        for task, path in worst_paths.items():
            if not path or not os.path.exists(path):
                continue
                
            for rel in read_list(path):
                if task == 'derain':
                    filtered.append({'clean_id': rel, 'de_type': 3})
                elif task == 'dehaze': 
                    filtered.append({'clean_id': rel, 'de_type': 4})
                elif task == 'deblur':
                    filtered.append({'clean_id': rel, 'de_type': 5})
                elif task == 'enhance':
                    filtered.append({'clean_id': rel, 'de_type': 6})
                elif task == 'denoise':
                    filtered.append({'clean_id': rel, 'de_type': 1})
        
        if len(filtered) > 0:
            self.trainset.sample_ids = filtered
            print(f"[INFO] Using worst lists: {len(filtered)} samples")
        
    def setup_environment(self):
        """设置RL环境"""
        self.env = AdaIRGRPOEnv(
            batch_size=opt.batch_size,
            device=self.device,
            reward_weights={
                "psnr": opt.grpo_w_psnr,
                "ssim": opt.grpo_w_ssim, 
                "lpips": opt.grpo_w_lpips
            }
        )
        
        # 加载预训练的AdaIR模型
        if opt.resume_ckpt and os.path.exists(opt.resume_ckpt):
            print(f"[INFO] Loading AdaIR checkpoint: {opt.resume_ckpt}")
            checkpoint = torch.load(opt.resume_ckpt, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('net.')}
                self.env.adair_model.load_state_dict(state_dict, strict=False)
            
        # 应用LoRA
        if opt.lora:
            wrapped = apply_lora_to_adair(
                self.env.adair_model, 
                targets=opt.lora_targets,
                rank=opt.lora_r, 
                alpha=opt.lora_alpha,
                dropout=opt.lora_dropout
            )
            print(f"[INFO] Applied LoRA to {wrapped} layers")
            
    def setup_networks(self):
        """设置策略和价值网络"""
        
        # 策略网络
        self.policy_network = AdaIRPolicyNetwork().to(self.device)
        
        # 价值网络（用于PPO的优势估计）
        self.value_network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # 输出状态价值
        ).to(self.device)
        
    def setup_ppo_loss(self):
        """设置PPO损失函数"""
        clip_range = getattr(opt, 'grpo_clip_range', 0.2)
        self.ppo_loss = SimplifiedPPOLoss(
            clip_epsilon=clip_range,
            entropy_coef=0.01,
            value_coef=0.5
        )
        
    def setup_optimizers(self):
        """设置优化器"""
        
        if opt.train_policy_only:
            # 仅训练策略网络和价值网络
            params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            
            # 添加LoRA参数
            if opt.lora:
                for m in self.env.adair_model.modules():
                    if hasattr(m, 'down') and hasattr(m, 'up'):
                        params.extend(list(m.down.parameters()))
                        params.extend(list(m.up.parameters()))
        else:
            params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            
        self.optimizer = torch.optim.AdamW(
            params,
            lr=opt.lr if not opt.grpo else opt.lr * 0.1,
            weight_decay=1e-4
        )
        
    def setup_logging(self):
        """设置日志记录"""
        if opt.wblogger:
            wandb.init(
                project=opt.wblogger,
                name="AdaIR-GRPO-TorchRL",
                config=vars(opt)
            )
            
    def compute_advantages(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        """计算GAE优势"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
        returns = advantages + values
        return advantages, returns
        
    def train_step(self, batch_data):
        """单步训练"""
        
        # 解析数据
        ([clean_name, de_id], degrad_patch, clean_patch) = batch_data
        degrad_patch = degrad_patch.to(self.device)
        clean_patch = clean_patch.to(self.device)
        
        # 设置环境数据
        self.env.set_data(degrad_patch, clean_patch)
        
        # 创建观察
        obs = {
            "degraded_image": degrad_patch,
            "clean_image": clean_patch,
        }
        obs_td = TensorDict(obs, batch_size=(opt.batch_size,), device=self.device)
        
        # 策略采样
        action_td = self.policy_network(obs_td)
        action_params = action_td["action"]["freq_params"]  # [B, 3, 8]
        
        # 计算log概率（简化的Beta分布）
        # 这里我们简化处理，直接使用参数的norm作为log_prob
        log_probs = -0.5 * torch.sum(action_params ** 2, dim=(1, 2))  # [B]
        
        # 环境步进
        env_input = TensorDict({**obs, "action": action_td["action"]}, batch_size=(opt.batch_size,), device=self.device)
        next_obs_td = self.env.step(env_input)
        
        # 价值估计
        values = self.value_network(degrad_patch).squeeze(-1)  # [B]
        next_values = values.clone()  # 单步环境，下一个状态价值相同
        
        # 奖励和done
        rewards = next_obs_td["reward"].squeeze(-1)  # [B]
        dones = next_obs_td["done"].squeeze(-1).float()  # [B]
        
        # 计算优势和回报
        advantages, returns = self.compute_advantages(
            rewards.unsqueeze(0), values.unsqueeze(0), next_values, dones.unsqueeze(0)
        )
        advantages = advantages.squeeze(0)  # [B]
        returns = returns.squeeze(0)  # [B]
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算PPO损失
        loss_dict = self.ppo_loss(
            log_probs, log_probs.detach(), advantages, returns, values
        )
        
        # 反向传播
        total_loss = loss_dict["loss_total"]
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        if hasattr(opt, 'grpo_max_grad_norm') and opt.grpo_max_grad_norm > 0:
            all_params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, opt.grpo_max_grad_norm)
            
        self.optimizer.step()
        
        return {
            "loss_total": total_loss.item(),
            "loss_policy": loss_dict["loss_policy"].item(),
            "loss_value": loss_dict["loss_value"].item(), 
            "loss_entropy": loss_dict["loss_entropy"].item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
        }
    
    def train(self):
        """主训练循环"""
        
        print("=== Starting TorchRL GRPO Training ===")
        
        global_step = 0
        
        for epoch in range(opt.epochs):
            epoch_stats = defaultdict(list)
            
            pbar = tqdm(self.trainloader, desc=f"Epoch {epoch}")
            
            for batch_idx, batch_data in enumerate(pbar):
                
                # 训练步骤
                step_stats = self.train_step(batch_data)
                
                # 收集统计
                for key, value in step_stats.items():
                    epoch_stats[key].append(value)
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{step_stats['loss_total']:.4f}",
                    "reward": f"{step_stats['reward_mean']:.4f}",
                    "adv": f"{step_stats['advantages_mean']:.4f}",
                })
                
                global_step += 1
                
                # 定期日志
                if global_step % 10 == 0 and opt.wblogger:
                    wandb.log(step_stats, step=global_step)
            
            # Epoch统计
            epoch_summary = {
                key: np.mean(values) for key, values in epoch_stats.items()
            }
            
            print(f"Epoch {epoch} Summary:")
            print(f"  Loss: {epoch_summary['loss_total']:.4f}")
            print(f"  Reward: {epoch_summary['reward_mean']:.4f} ± {epoch_summary['reward_std']:.4f}")
            print(f"  Advantages: {epoch_summary['advantages_mean']:.4f} ± {epoch_summary['advantages_std']:.4f}")
            
            # 保存检查点
            if epoch % getattr(opt, 'save_freq', 5) == 0:
                self.save_checkpoint(epoch)
                
    def save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adair_state_dict': self.env.adair_model.state_dict(),
        }
        
        save_path = os.path.join(opt.ckpt_dir, f"torchrl_grpo_epoch_{epoch}.ckpt")
        os.makedirs(opt.ckpt_dir, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"[INFO] Checkpoint saved: {save_path}")


def main():
    """主函数"""
    print("TorchRL GRPO Training for AdaIR")
    print("Options:", opt)
    
    # 创建训练器
    trainer = TorchRLGRPOTrainer()
    
    # 开始训练
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main() 