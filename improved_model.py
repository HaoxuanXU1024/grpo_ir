"""
改进的AdaIR模型 - 增强策略随机性
解决Beta分布variance过小的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from net.model import FreModule, AdaIR

class ImprovedFreModule(FreModule):
    """改进的FreModule - 增强策略随机性"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加温度参数控制分布sharpness
        self.temperature_rate = nn.Parameter(torch.ones(1) * 2.0)  # 控制频率策略的随机性
        self.temperature_fuse = nn.Parameter(torch.ones(1) * 2.0)  # 控制融合策略的随机性
        
        # 添加探索噪声
        self.exploration_std = 0.1
        
    def forward(self, x, y, stochastic: bool = False, collector=None):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')
        
        high_feature, low_feature, log_prob = self.fft(x, stochastic=stochastic)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        # 改进的融合策略 - 增强随机性
        y_pool = F.adaptive_avg_pool2d(y, 1)
        raw = self.policy_fuse(y_pool)
        
        if stochastic:
            # GRPO模式：增强随机性
            raw = F.softplus(raw) + 1e-3
            
            # 使用温度参数调节分布锐度
            raw = raw / self.temperature_fuse.clamp(0.1, 5.0)
            
            # 确保最小variance
            raw = torch.clamp(raw, 0.1, 10.0)  # 防止过于确定
            
            a1, b1, a2, b2 = raw[:, 0, 0, 0], raw[:, 1, 0, 0], raw[:, 2, 0, 0], raw[:, 3, 0, 0]
            
            from torch.distributions import Beta
            dist1 = Beta(a1, b1)
            dist2 = Beta(a2, b2)
            
            # 添加探索噪声
            g1_flat = dist1.rsample() 
            g2_flat = dist2.rsample()
            
            # 添加轻微噪声增加探索
            if self.training:
                noise1 = torch.randn_like(g1_flat) * self.exploration_std
                noise2 = torch.randn_like(g2_flat) * self.exploration_std
                g1_flat = torch.clamp(g1_flat + noise1, 0.0, 1.0)
                g2_flat = torch.clamp(g2_flat + noise2, 0.0, 1.0)
            
            g1 = g1_flat.view(-1, 1, 1, 1)
            g2 = g2_flat.view(-1, 1, 1, 1)
            fuse_lp = dist1.log_prob(g1_flat) + dist2.log_prob(g2_flat)
        else:
            # 预训练模式：保持一定随机性，避免过度确定
            raw = F.softplus(raw) + 0.5  # 增加基础值
            a1, b1, a2, b2 = raw[:, 0, 0, 0], raw[:, 1, 0, 0], raw[:, 2, 0, 0], raw[:, 3, 0, 0]
            
            # 使用带噪声的均值，而不是纯均值
            mean1 = a1 / (a1 + b1)
            mean2 = a2 / (a2 + b2)
            
            if self.training:
                # 预训练时也添加轻微噪声保持多样性
                noise1 = torch.randn_like(mean1) * 0.05
                noise2 = torch.randn_like(mean2) * 0.05
                g1_flat = torch.clamp(mean1 + noise1, 0.0, 1.0)
                g2_flat = torch.clamp(mean2 + noise2, 0.0, 1.0)
            else:
                g1_flat = mean1
                g2_flat = mean2
            
            g1 = g1_flat.view(-1, 1, 1, 1)
            g2 = g2_flat.view(-1, 1, 1, 1)
            fuse_lp = None

        out = out * (self.para1 * g1) + y * (self.para2 * g2)

        if stochastic and collector is not None:
            total_lp = log_prob
            if fuse_lp is not None:
                total_lp = total_lp + fuse_lp
            collector.append(total_lp)

        return out

    def fft(self, x, n=128, stochastic: bool = False):
        """改进的FFT - 增强频率策略随机性"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        pooled = F.adaptive_avg_pool2d(x, 1)

        raw = self.policy_rate(pooled)
        
        if stochastic:
            # GRPO模式：增强随机性
            raw = F.softplus(raw) + 1e-3
            
            # 使用温度参数
            raw = raw / self.temperature_rate.clamp(0.1, 5.0)
            raw = torch.clamp(raw, 0.1, 10.0)
            
            a_h, b_h, a_w, b_w = raw[:, 0, 0, 0], raw[:, 1, 0, 0], raw[:, 2, 0, 0], raw[:, 3, 0, 0]
            
            from torch.distributions import Beta
            dist_h = Beta(a_h, b_h)
            dist_w = Beta(a_w, b_w)
            r_h_flat = dist_h.rsample()
            r_w_flat = dist_w.rsample()
            
            # 添加探索噪声
            if self.training:
                noise_h = torch.randn_like(r_h_flat) * self.exploration_std
                noise_w = torch.randn_like(r_w_flat) * self.exploration_std
                r_h_flat = torch.clamp(r_h_flat + noise_h, 0.0, 1.0)
                r_w_flat = torch.clamp(r_w_flat + noise_w, 0.0, 1.0)
            
            r_h = r_h_flat.view(-1, 1, 1, 1)
            r_w = r_w_flat.view(-1, 1, 1, 1)
            threshold = torch.cat([r_h, r_w], dim=1)
            log_prob = dist_h.log_prob(r_h_flat) + dist_w.log_prob(r_w_flat)
        else:
            # 预训练模式：带噪声的均值
            raw = F.softplus(raw) + 0.5
            a_h, b_h, a_w, b_w = raw[:, 0, 0, 0], raw[:, 1, 0, 0], raw[:, 2, 0, 0], raw[:, 3, 0, 0]
            
            mean_h = a_h / (a_h + b_h)
            mean_w = a_w / (a_w + b_w)
            
            if self.training:
                noise_h = torch.randn_like(mean_h) * 0.05
                noise_w = torch.randn_like(mean_w) * 0.05
                r_h_flat = torch.clamp(mean_h + noise_h, 0.0, 1.0)
                r_w_flat = torch.clamp(mean_w + noise_w, 0.0, 1.0)
            else:
                r_h_flat = mean_h
                r_w_flat = mean_w
            
            r_h = r_h_flat.view(-1, 1, 1, 1)
            r_w = r_w_flat.view(-1, 1, 1, 1)
            threshold = torch.cat([r_h, r_w], dim=1)
            log_prob = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        # FFT处理逻辑保持不变
        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()
            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)
        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low, log_prob


class ImprovedAdaIR(AdaIR):
    """改进的AdaIR模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换FreModule为改进版本
        if self.decoder:
            dim = kwargs.get('dim', 48)
            heads = kwargs.get('heads', [1, 2, 4, 8])
            bias = kwargs.get('bias', False)
            
            self.fre1 = ImprovedFreModule(dim * 2**3, num_heads=heads[2], bias=bias)
            self.fre2 = ImprovedFreModule(dim * 2**2, num_heads=heads[2], bias=bias)
            self.fre3 = ImprovedFreModule(dim * 2**1, num_heads=heads[2], bias=bias)
