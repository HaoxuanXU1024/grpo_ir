"""
AdaIR Rewards Package
图像恢复任务专用的奖励函数集合
"""

from .adair_rewards import (
    aesthetic_score,
    clip_similarity_score, 
    qwenvl_quality_score,
    perceptual_quality_score,
    psnr_score,
    ssim_score,
    multi_reward_scorer,
    create_adair_reward_fn
)

__all__ = [
    'aesthetic_score',
    'clip_similarity_score', 
    'qwenvl_quality_score',
    'perceptual_quality_score',
    'psnr_score',
    'ssim_score',
    'multi_reward_scorer',
    'create_adair_reward_fn'
]


