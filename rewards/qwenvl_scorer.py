from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not found. Install with: pip install qwen-vl-utils")
    process_vision_info = None

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)
    return scores

class QwenVLScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=self.dtype,
                device_map=None,
            ).to(self.device)
            self.model.requires_grad_(False)
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
            self.available = True
        except Exception as e:
            print(f"Warning: QwenVL model not available: {e}")
            self.available = False
            
        self.task = '''
Your role is to evaluate the quality of restored/enhanced images for image restoration tasks.
Please assess the image based on the following criteria:
1. Poor (1): Severe artifacts, poor clarity, unnatural colors, significant distortions
2. Fair (2): Noticeable artifacts or blur, somewhat unnatural appearance
3. Good (3): Good overall quality with minor artifacts, natural appearance
4. Very Good (4): High quality with minimal artifacts, excellent clarity and colors
5. Excellent (5): Perfect restoration with no visible artifacts, exceptional quality

Please first provide a detailed analysis within the <Thought> tag, then give a score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the image quality in detail here]
</Thought>
<Score>X</Score>
'''
        
    @torch.no_grad()
    def __call__(self, images, ref_images=None):
        if not self.available:
            # 返回随机分数作为fallback
            return [0.5] * len(images)
            
        # 转换图像格式
        if isinstance(images, torch.Tensor):
            images = (images * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(img) for img in images]
        
        images_base64 = [pil_image_to_base64(image) for image in images]
        messages = []
        for base64_qwen in images_base64:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": self.task},
                    ],
                },
            ])

        if process_vision_info is None:
            # Fallback: 返回随机分数
            return [0.5] * len(images)

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        rewards = extract_scores(output_texts)
        return rewards


