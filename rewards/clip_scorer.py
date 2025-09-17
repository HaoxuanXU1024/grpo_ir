# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image

def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    
def get_image_transform(processor: AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class ClipScorer(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()
    
    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)
        return pixels

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        """计算图像间的CLIP相似度，适用于图像恢复任务"""
        
        # 处理输入格式
        if isinstance(pixels, torch.Tensor):
            # 确保在[0,1]范围
            if pixels.max() > 1.0:
                pixels = pixels / 255.0
            pixels = self._process(pixels).to(self.device)
        else:
            # numpy array
            pixels = torch.tensor(pixels).permute(0, 3, 1, 2) / 255.0
            pixels = self._process(pixels).to(self.device)
            
        if isinstance(ref_pixels, torch.Tensor):
            if ref_pixels.max() > 1.0:
                ref_pixels = ref_pixels / 255.0
            ref_pixels = self._process(ref_pixels).to(self.device)
        else:
            # numpy array or PIL images
            if isinstance(ref_pixels[0], Image.Image):
                ref_pixels = [np.array(img) for img in ref_pixels]
            ref_pixels = np.array(ref_pixels)
            ref_pixels = torch.tensor(ref_pixels).permute(0, 3, 1, 2) / 255.0
            ref_pixels = self._process(ref_pixels).to(self.device)

        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        sim = pixel_embeds @ ref_embeds.T
        sim = torch.diagonal(sim, 0)
        return sim


