"""
AdaIR专用奖励函数
为图像恢复任务定制的奖励计算模块
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional

def aesthetic_score(device="cuda"):
    """美学质量评分 - 优化版本，复用模型实例"""
    from .aesthetic_scorer import AestheticScorer
    
    # 全局aesthetic实例字典，按设备缓存
    if not hasattr(aesthetic_score, '_aesthetic_cache'):
        aesthetic_score._aesthetic_cache = {}
    
    def _fn(restored_images, clean_images=None, **kwargs):
        try:
            if isinstance(restored_images, torch.Tensor):
                current_device = restored_images.device
            else:
                current_device = torch.device(device)
                # numpy array format
                restored_images = restored_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
                restored_images = torch.tensor(restored_images, dtype=torch.float32) / 255.0
                
            # 复用aesthetic实例，避免重复加载
            device_key = str(current_device)
            if device_key not in aesthetic_score._aesthetic_cache:
                print(f"[INFO] Initializing Aesthetic scorer for device {device_key} (one-time setup)")
                aesthetic_score._aesthetic_cache[device_key] = AestheticScorer(dtype=torch.float32, device=str(current_device))
            
            scorer = aesthetic_score._aesthetic_cache[device_key]
            restored_images = restored_images.to(current_device)
            scores = scorer(restored_images)
            return scores.cpu().tolist()
        except Exception as e:
            print(f"[WARNING] Aesthetic scorer failed: {e}")
            # 返回中性分数作为fallback
            batch_size = len(restored_images) if isinstance(restored_images, list) else restored_images.shape[0]
            return [0.5] * batch_size
    
    return _fn

def clip_similarity_score(device="cuda"):
    """CLIP图像相似度评分 - 优化版本，复用模型实例"""
    from .clip_scorer import ClipScorer
    
    # 全局CLIP实例字典，按设备缓存
    if not hasattr(clip_similarity_score, '_clip_cache'):
        clip_similarity_score._clip_cache = {}
    
    def _fn(restored_images, clean_images, **kwargs):
        try:
            # 获取当前设备
            if isinstance(restored_images, torch.Tensor):
                current_device = restored_images.device
            else:
                current_device = torch.device(device)
            
            # 复用CLIP实例，避免重复下载
            device_key = str(current_device)
            if device_key not in clip_similarity_score._clip_cache:
                print(f"[INFO] Initializing CLIP for device {device_key} (one-time setup)")
                clip_similarity_score._clip_cache[device_key] = ClipScorer(device=str(current_device))
            
            scorer = clip_similarity_score._clip_cache[device_key]
            
            # 计算恢复图像与干净图像的CLIP相似度
            scores = scorer.image_similarity(restored_images, clean_images)
            return scores.cpu().tolist()
        except Exception as e:
            print(f"[WARNING] CLIP similarity scorer failed: {e}")
            # 返回中性分数作为fallback
            batch_size = len(restored_images) if isinstance(restored_images, list) else restored_images.shape[0]
            return [0.5] * batch_size
    
    return _fn

def qwenvl_quality_score(device="cuda"):
    """QwenVL综合质量评分"""
    from .qwenvl_scorer import QwenVLScorer
    
    scorer = QwenVLScorer(device=device, dtype=torch.bfloat16)
    
    def _fn(restored_images, clean_images=None, **kwargs):
        scores = scorer(restored_images, clean_images)
        return scores
    
    return _fn

def perceptual_quality_score(device="cuda"):
    """感知质量评分 (基于现有的LPIPS) - 优化版本，复用模型实例"""
    import lpips
    
    # 全局LPIPS实例字典，按设备缓存
    if not hasattr(perceptual_quality_score, '_lpips_cache'):
        perceptual_quality_score._lpips_cache = {}
    
    def _fn(restored_images, clean_images, **kwargs):
        try:
            if isinstance(restored_images, torch.Tensor):
                restored = restored_images
                current_device = restored.device
            else:
                restored = torch.tensor(restored_images).permute(0, 3, 1, 2) / 255.0
                current_device = torch.device(device)
                
            if isinstance(clean_images, torch.Tensor):
                clean = clean_images
            else:
                clean = torch.tensor(clean_images).permute(0, 3, 1, 2) / 255.0
                
            # 确保所有张量在同一设备上
            restored = restored.to(current_device)
            clean = clean.to(current_device)
            
            # 复用LPIPS实例，避免重复加载
            device_key = str(current_device)
            if device_key not in perceptual_quality_score._lpips_cache:
                print(f"[INFO] Initializing LPIPS for device {device_key} (one-time setup)")
                perceptual_quality_score._lpips_cache[device_key] = lpips.LPIPS(net='alex', verbose=False).eval().to(current_device)
            
            lpips_metric = perceptual_quality_score._lpips_cache[device_key]
            
            # LPIPS expects [-1, 1] range
            restored_norm = restored * 2 - 1
            clean_norm = clean * 2 - 1
            
            with torch.no_grad():
                lpips_values = lpips_metric(restored_norm, clean_norm)
                # Convert to similarity score (1 - LPIPS)
                similarity_scores = 1.0 - lpips_values.squeeze()
                
            return similarity_scores.cpu().tolist()
            
        except Exception as e:
            print(f"[WARNING] Perceptual quality scorer failed: {e}")
            # 返回中性分数作为fallback
            batch_size = len(restored_images) if isinstance(restored_images, list) else restored_images.shape[0]
            return [0.5] * batch_size
    
    return _fn

def psnr_score():
    """PSNR评分"""
    def _fn(restored_images, clean_images, **kwargs):
        try:
            if isinstance(restored_images, torch.Tensor):
                restored = restored_images
            else:
                restored = torch.tensor(restored_images).permute(0, 3, 1, 2) / 255.0
                
            if isinstance(clean_images, torch.Tensor):
                clean = clean_images
            else:
                clean = torch.tensor(clean_images).permute(0, 3, 1, 2) / 255.0
            
            # 确保在同一设备上
            device = restored.device
            clean = clean.to(device)
            
            with torch.no_grad():
                # 计算MSE
                mse = torch.mean((restored - clean) ** 2, dim=(1, 2, 3)) + 1e-6
                # 计算PSNR
                psnr = 10.0 * torch.log10(1.0 / mse)
                # 归一化到[0,1]范围 (假设40dB为满分)
                psnr_norm = torch.clamp(psnr / 40.0, 0.0, 1.0)
                
            return psnr_norm.cpu().tolist()
            
        except Exception as e:
            print(f"[WARNING] PSNR scorer failed: {e}")
            batch_size = len(restored_images) if isinstance(restored_images, list) else restored_images.shape[0]
            return [0.5] * batch_size
    
    return _fn

def ssim_score():
    """SSIM评分 - 优化版本，复用模型实例"""
    from utils.pytorch_ssim import SSIM
    
    # 全局SSIM实例字典，按设备缓存
    if not hasattr(ssim_score, '_ssim_cache'):
        ssim_score._ssim_cache = {}
    
    def _fn(restored_images, clean_images, **kwargs):
        try:
            if isinstance(restored_images, torch.Tensor):
                restored = restored_images
            else:
                restored = torch.tensor(restored_images).permute(0, 3, 1, 2) / 255.0
                
            if isinstance(clean_images, torch.Tensor):
                clean = clean_images
            else:
                clean = torch.tensor(clean_images).permute(0, 3, 1, 2) / 255.0
            
            # 确保在同一设备上
            device = restored.device
            clean = clean.to(device)
            
            # 复用SSIM实例，避免重复创建
            device_key = str(device)
            if device_key not in ssim_score._ssim_cache:
                print(f"[INFO] Initializing SSIM for device {device_key} (one-time setup)")
                ssim_score._ssim_cache[device_key] = SSIM().eval().to(device)
            
            ssim_metric = ssim_score._ssim_cache[device_key]
            
            with torch.no_grad():
                batch_size = restored.shape[0]
                ssim_values = []
                
                for i in range(batch_size):
                    ssim_val = ssim_metric(restored[i:i+1], clean[i:i+1])
                    ssim_values.append(ssim_val.item())
                
            return ssim_values
            
        except Exception as e:
            print(f"[WARNING] SSIM scorer failed: {e}")
            batch_size = len(restored_images) if isinstance(restored_images, list) else restored_images.shape[0]
            return [0.5] * batch_size
    
    return _fn

def multi_reward_scorer(device="cuda", reward_weights: Dict[str, float] = None):
    """多指标组合奖励计算器"""
    
    if reward_weights is None:
        # 优化的权重配置，平衡传统指标和先进指标
        reward_weights = {
            "clip_similarity": 0.25,  # 语义相似度
            "perceptual": 0.25,       # 感知质量 (LPIPS)
            "psnr": 0.20,            # PSNR (传统重要指标)
            "ssim": 0.15,            # SSIM (传统重要指标)
            "aesthetic": 0.15,        # 美学质量
        }
    
    # 初始化各个评分器，传递设备参数
    scorers = {}
    if "clip_similarity" in reward_weights:
        scorers["clip_similarity"] = clip_similarity_score(device)
    if "aesthetic" in reward_weights:
        scorers["aesthetic"] = aesthetic_score(device)
    if "perceptual" in reward_weights:
        scorers["perceptual"] = perceptual_quality_score(device)
    if "qwenvl" in reward_weights:
        scorers["qwenvl"] = qwenvl_quality_score(device)
    if "psnr" in reward_weights:
        scorers["psnr"] = psnr_score()
    if "ssim" in reward_weights:
        scorers["ssim"] = ssim_score()
    
    def _fn(restored_images, clean_images, **kwargs):
        total_scores = None
        score_details = {}
        
        for score_name, weight in reward_weights.items():
            if score_name not in scorers:
                continue
                
            try:
                scores = scorers[score_name](restored_images, clean_images, **kwargs)
                scores = np.array(scores)
                score_details[score_name] = scores.tolist()
                
                weighted_scores = weight * scores
                
                if total_scores is None:
                    total_scores = weighted_scores
                else:
                    total_scores += weighted_scores
                    
            except Exception as e:
                print(f"Warning: {score_name} scorer failed: {e}")
                # 使用中性分数
                scores = np.ones(len(restored_images)) * 0.5
                score_details[score_name] = scores.tolist()
                weighted_scores = weight * scores
                
                if total_scores is None:
                    total_scores = weighted_scores
                else:
                    total_scores += weighted_scores
        
        score_details["total"] = total_scores.tolist()
        return total_scores.tolist(), score_details
    
    return _fn

# 为AdaIR环境提供的简化接口
def create_adair_reward_fn(device="cuda", reward_config: Dict = None):
    """为AdaIR GRPO环境创建奖励函数"""
    
    if reward_config is None:
        # 默认配置：平衡传统指标和先进指标
        reward_config = {
            "clip_similarity": 0.25,  # 语义相似度
            "perceptual": 0.25,       # 感知质量 (LPIPS)
            "psnr": 0.20,            # PSNR (传统重要指标)
            "ssim": 0.15,            # SSIM (传统重要指标) 
            "aesthetic": 0.15,        # 美学质量
        }
    
    reward_fn = multi_reward_scorer(device, reward_config)
    
    def adair_reward_wrapper(restored_tensor, clean_tensor):
        """
        AdaIR环境专用的奖励包装器
        Args:
            restored_tensor: 恢复的图像张量 [B, C, H, W]
            clean_tensor: 干净的参考图像张量 [B, C, H, W]
        Returns:
            rewards: 奖励张量 [B]
        """
        try:
            rewards, details = reward_fn(restored_tensor, clean_tensor)
            return torch.tensor(rewards, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Reward calculation failed: {e}")
            # 返回中性奖励
            batch_size = restored_tensor.shape[0]
            return torch.ones(batch_size, dtype=torch.float32) * 0.5
    
    return adair_reward_wrapper


