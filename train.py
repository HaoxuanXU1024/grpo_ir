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
        # If train_policy_only is set, only optimize policy heads and LoRA params
        if opt.train_policy_only:
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