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
    """支持TorchRL外部动作注入的FreModule"""
    
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super().__init__(dim, num_heads, bias, in_dim)
        self._external_actions = None
        
    def set_external_actions(self, rate_actions, fuse_actions):
        """设置外部采样的动作"""
        self._external_actions = {
            'rate_actions': rate_actions,  # [B, 2] - r_h, r_w
            'fuse_actions': fuse_actions,  # [B, 2] - g1, g2
        }
    
    def fft(self, x, n=128, stochastic: bool = False):
        """使用外部动作的FFT处理"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        
        if stochastic and self._external_actions is not None:
            # 使用外部提供的采样动作
            rate_actions = self._external_actions['rate_actions']  # [B, 2]
            
            # 直接使用动作值
            r_h_flat = rate_actions[:, 0]
            r_w_flat = rate_actions[:, 1]
            
            # 重新形状
            r_h = r_h_flat.view(-1, 1, 1, 1)
            r_w = r_w_flat.view(-1, 1, 1, 1)
            threshold = torch.cat([r_h, r_w], dim=1)
        else:
            # 使用原始的确定性方法
            pooled = F.adaptive_avg_pool2d(x, 1)
            threshold = self.rate_conv(pooled).sigmoid()

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

        return high, low

    def forward(self, x, y, stochastic: bool = False, collector=None):
        """使用外部动作的forward"""
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')
        
        # fft不再返回log_prob
        high_feature, low_feature = self.fft(x, stochastic=stochastic)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        # 处理融合参数
        if stochastic and self._external_actions is not None:
            fuse_actions = self._external_actions['fuse_actions']  # [B, 2]
            
            g1_flat = fuse_actions[:, 0]
            g2_flat = fuse_actions[:, 1]
            g1 = g1_flat.view(-1, 1, 1, 1)
            g2 = g2_flat.view(-1, 1, 1, 1)
        else:
            g1 = torch.ones(y.size(0), 1, 1, 1, device=y.device, dtype=y.dtype)
            g2 = torch.ones(y.size(0), 1, 1, 1, device=y.device, dtype=y.dtype)

        out = out * (self.para1 * g1) + y * (self.para2 * g2)

        # log_prob相关的逻辑已移除
        return out


class AdaIRTorchRL(AdaIR):
    """支持TorchRL的AdaIR模型"""
    
    def __init__(self, *args, **kwargs):
        # 在调用父类构造函数之前，捕获或设置用于FreModule的参数
        # 这样可以避免访问不存在的属性
        dim = kwargs.get('dim', 48)
        heads = kwargs.get('heads', [1, 2, 4, 8])
        bias = kwargs.get('bias', False)

        super().__init__(*args, **kwargs)
        
        # 替换FreModule为TorchRL版本，使用我们自己计算的参数
        if self.decoder:
            self.fre1 = FreModuleTorchRL(dim * 2**3, num_heads=heads[2], bias=bias)
            self.fre2 = FreModuleTorchRL(dim * 2**2, num_heads=heads[2], bias=bias)
            self.fre3 = FreModuleTorchRL(dim * 2**1, num_heads=heads[2], bias=bias)
    
    def inject_actions(self, actions):
        """注入采样好的动作到FreModule中"""
        # actions shape: [B, 3, 4] -> 3个FreModule，每个4个动作值(r_h, r_w, g1, g2)
        if not self.decoder:
            return
            
        fre_modules = [self.fre1, self.fre2, self.fre3]
        
        for i, fre_module in enumerate(fre_modules):
            module_actions = actions[:, i, :]  # [B, 4]
            rate_actions = module_actions[:, :2]    # [B, 2] for r_h, r_w
            fuse_actions = module_actions[:, 2:4]   # [B, 2] for g1, g2
            
            fre_module.set_external_actions(rate_actions, fuse_actions)
    
    def forward(self, inp_img, stochastic: bool = False, external_params=None):
        """支持外部动作注入的forward"""
        
        # external_params 在此实现中不再使用，动作通过 inject_actions 注入
        # 为了兼容原始的 `super().forward` 调用，我们保留它
        if external_params is not None:
            # 兼容性警告或静默忽略
            pass

        # 调用原始forward逻辑，它会调用已被替换的FreModuleTorchRL的forward
        # 这里的stochastic标志将控制FreModuleTorchRL是使用确定性路径还是使用注入的动作
        return super().forward(inp_img, stochastic=stochastic) 