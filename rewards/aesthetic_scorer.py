# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.device = device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP().to(device)
        
        # 简化：如果没有预训练权重，随机初始化MLP
        # 在实际使用中，您可能需要下载官方的aesthetic预训练权重
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        
        # 转换图像格式
        if isinstance(images, torch.Tensor):
            # [B, C, H, W] tensor -> PIL images
            images = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(img) for img in images]
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


