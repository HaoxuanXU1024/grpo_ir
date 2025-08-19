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

# TorchRL imports
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.algorithms import PPOLoss
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tensordict import TensorDict
from torchrl.modules import ActorValueOperator, ProbabilisticActor, ValueOperator
from torchrl.envs import ParallelEnv, SerialEnv
from torch.distributions import Beta

# Local imports
from grpo_torchrl_env import AdaIRGRPOEnv, AdaIRPolicyNetwork
from utils.dataset_utils import AdaIRTrainDataset
from net.model import AdaIR
from options import options as opt
from utils.lora import apply_lora_to_adair


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
        # 这里复用train.py中的逻辑
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
        
        # 构建filtered列表（简化版本，具体实现可参考train.py）
        for task, path in worst_paths.items():
            for rel in read_list(path):
                if task == 'derain':
                    filtered.append({'clean_id': rel, 'de_type': 3})
                elif task == 'dehaze':
                    filtered.append({'clean_id': rel, 'de_type': 4})
                # ... 其他任务类似
        
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
        
        # 包装成TorchRL的Actor-Critic
        self.actor_critic = ActorValueOperator(
            actor=ProbabilisticActor(
                module=self.policy_network,
                spec=self.env.action_spec,
                distribution_class=self._create_beta_distribution,
                return_log_prob=True,
            ),
            value=ValueOperator(
                module=self.value_network,
                spec=self.env.observation_spec,
            ),
        )
        
    def _create_beta_distribution(self, logits):
        """创建Beta分布用于连续动作采样"""
        # logits shape: [B, 3, 8]
        # 重新整形为 [B, 3, 4, 2] (alpha, beta pairs)
        logits = logits.view(*logits.shape[:-1], 4, 2)
        alpha = logits[..., 0] + 0.1  # 确保 > 0
        beta = logits[..., 1] + 0.1
        return Beta(alpha, beta)
        
    def setup_ppo_loss(self):
        """设置PPO损失函数"""
        self.ppo_loss = ClipPPOLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            critic_network=self.actor_critic.get_value_operator(),
            clip_epsilon=opt.grpo_clip_range if hasattr(opt, 'grpo_clip_range') else 0.2,
            entropy_bonus=True,
            entropy_coef=0.01,
            critic_coef=0.5,
            normalize_advantage=True,
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
            params = list(self.actor_critic.parameters())
            
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
            
    def train_step(self, batch_data):
        """单步训练"""
        
        # 解析数据
        ([clean_name, de_id], degrad_patch, clean_patch) = batch_data
        degrad_patch = degrad_patch.to(self.device)
        clean_patch = clean_patch.to(self.device)
        
        # 设置环境数据
        self.env.set_data(degrad_patch, clean_patch)
        
        # 重置环境
        reset_td = TensorDict({}, batch_size=(opt.batch_size,), device=self.device)
        obs_td = self.env.reset(reset_td)
        
        # 策略采样
        with torch.no_grad():
            action_td = self.actor_critic(obs_td)
            
        # 环境步进
        next_obs_td = self.env.step(action_td)
        
        # 创建经验数据
        experience = TensorDict({
            **obs_td,
            **action_td,
            **next_obs_td,
        }, batch_size=(opt.batch_size,), device=self.device)
        
        # 计算PPO损失
        loss_dict = self.ppo_loss(experience)
        
        # 反向传播
        total_loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        if hasattr(opt, 'grpo_max_grad_norm') and opt.grpo_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), 
                opt.grpo_max_grad_norm
            )
            
        self.optimizer.step()
        
        return {
            "loss_total": total_loss.item(),
            "loss_policy": loss_dict["loss_objective"].item(),
            "loss_value": loss_dict["loss_critic"].item(),
            "loss_entropy": loss_dict["loss_entropy"].item(),
            "reward_mean": next_obs_td["reward"].mean().item(),
            "reward_std": next_obs_td["reward"].std().item(),
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
            
            # 保存检查点
            if epoch % opt.save_freq == 0:
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
    
    # 设置保存频率
    if not hasattr(opt, 'save_freq'):
        opt.save_freq = 5
        
    # 创建训练器
    trainer = TorchRLGRPOTrainer()
    
    # 开始训练
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main() 