"""
修改版的AdaIR模型，支持TorchRL参数注入
基于原始model.py，添加外部策略参数支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# 导入原始模型的基础组件
import sys
import os
sys.path.append(os.path.dirname(__file__))
from model import *  # 导入所有原始组件


class FreModuleTorchRL(FreModule):
    """支持TorchRL外部参数注入的FreModule"""
    
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super().__init__(dim, num_heads, bias, in_dim)
        self._external_params = None
        
    def set_external_params(self, rate_params, fuse_params):
        """设置外部策略参数"""
        self._external_params = {
            'rate_params': rate_params,  # [B, 4] - alpha_h, beta_h, alpha_w, beta_w
            'fuse_params': fuse_params,  # [B, 4] - alpha_1, beta_1, alpha_2, beta_2
        }
    
    def fft(self, x, n=128, stochastic: bool = False):
        """使用外部参数的FFT处理"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        
        if stochastic and self._external_params is not None:
            # 使用外部提供的参数
            rate_params = self._external_params['rate_params']  # [B, 4]
            
            # 提取alpha, beta参数
            alpha_h = rate_params[:, 0]  # [B]
            beta_h = rate_params[:, 1]   # [B]
            alpha_w = rate_params[:, 2]  # [B]
            beta_w = rate_params[:, 3]   # [B]
            
            # 从Beta分布采样
            dist_h = Beta(alpha_h, beta_h)
            dist_w = Beta(alpha_w, beta_w)
            r_h_flat = dist_h.rsample()  # [B]
            r_w_flat = dist_w.rsample()  # [B]
            
            # 重新形状
            r_h = r_h_flat.view(-1, 1, 1, 1)
            r_w = r_w_flat.view(-1, 1, 1, 1)
            threshold = torch.cat([r_h, r_w], dim=1)
            log_prob = dist_h.log_prob(r_h_flat) + dist_w.log_prob(r_w_flat)
        else:
            # 使用原始的确定性方法
            pooled = F.adaptive_avg_pool2d(x, 1)
            threshold = self.rate_conv(pooled).sigmoid()
            log_prob = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        # 其余逻辑保持不变
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

    def forward(self, x, y, stochastic: bool = False, collector=None):
        """使用外部参数的forward"""
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')
        
        high_feature, low_feature, log_prob = self.fft(x, stochastic=stochastic)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        # 处理融合参数
        fuse_lp = None
        if stochastic and self._external_params is not None:
            fuse_params = self._external_params['fuse_params']  # [B, 4]
            
            a1 = fuse_params[:, 0]  # [B]
            b1 = fuse_params[:, 1]  # [B] 
            a2 = fuse_params[:, 2]  # [B]
            b2 = fuse_params[:, 3]  # [B]
            
            # 采样融合权重
            dist1 = Beta(a1, b1)
            dist2 = Beta(a2, b2)
            g1_flat = dist1.rsample()
            g2_flat = dist2.rsample()
            g1 = g1_flat.view(-1, 1, 1, 1)
            g2 = g2_flat.view(-1, 1, 1, 1)
            fuse_lp = dist1.log_prob(g1_flat) + dist2.log_prob(g2_flat)
        else:
            g1 = torch.ones(y.size(0), 1, 1, 1, device=y.device, dtype=y.dtype)
            g2 = torch.ones(y.size(0), 1, 1, 1, device=y.device, dtype=y.dtype)

        out = out * (self.para1 * g1) + y * (self.para2 * g2)

        if stochastic and collector is not None:
            total_lp = log_prob
            if fuse_lp is not None:
                total_lp = total_lp + fuse_lp
            collector.append(total_lp)

        return out


class AdaIRTorchRL(AdaIR):
    """支持TorchRL的AdaIR模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换FreModule为TorchRL版本
        if self.decoder:
            self.fre1 = FreModuleTorchRL(self.fre1.dim, self.fre1.num_heads, self.fre1.bias)
            self.fre2 = FreModuleTorchRL(self.fre2.dim, self.fre2.num_heads, self.fre2.bias)
            self.fre3 = FreModuleTorchRL(self.fre3.dim, self.fre3.num_heads, self.fre3.bias)
    
    def inject_policy_params(self, freq_params):
        """注入策略参数到FreModule中"""
        if not self.decoder:
            return
            
        # freq_params shape: [B, 3, 8]
        fre_modules = [self.fre1, self.fre2, self.fre3]
        
        for i, fre_module in enumerate(fre_modules):
            params = freq_params[:, i, :]  # [B, 8]
            rate_params = params[:, :4]    # [B, 4]
            fuse_params = params[:, 4:8]   # [B, 4]
            
            fre_module.set_external_params(rate_params, fuse_params)
    
    def forward(self, inp_img, stochastic: bool = False, external_params=None):
        """支持外部参数注入的forward"""
        
        # 如果提供了外部参数，注入到FreModule
        if external_params is not None:
            self.inject_policy_params(external_params)
        
        # 调用原始forward逻辑
        return super().forward(inp_img, stochastic) 