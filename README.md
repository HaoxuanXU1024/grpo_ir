# AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation (ICLR'25)

Yuning Cui, [Syed Waqas Zamir](https://scholar.google.ae/citations?hl=en&user=POoai-QAAAAJ), [Salman Khan](https://salman-h-khan.github.io/), [Alois Knoll](https://scholar.google.com.hk/citations?user=-CA8QgwAAAAJ&hl=zh-CN&oi=ao), [Mubarak Shah](https://scholar.google.com.hk/citations?user=p8gsO3gAAAAJ&hl=zh-CN&oi=ao), and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)


<hr />

> **Abstract:** *In the image acquisition process, various forms of degradation, including noise, blur, haze, and rain, are frequently introduced. These degradations typically arise from the inherent limitations of cameras or unfavorable ambient conditions. To recover clean images from their degraded versions, numerous specialized restoration methods have been developed, each targeting a specific type of degradation. Recently, all-in-one algorithms have garnered significant attention by addressing different types of degradations within a single model without requiring the prior information of the input degradation type. However, these methods purely operate in the spatial domain and do not delve into the distinct frequency variations inherent to different degradation types. To address this gap, we propose an adaptive all-in-one image restoration network based on frequency mining and modulation. Our approach is motivated by the observation that different degradation types impact the image content on different frequency subbands, thereby requiring different treatments for each restoration task. Specifically, we first mine low- and high-frequency information from the input features, guided by the adaptively decoupled spectra of the degraded image. The extracted features are then modulated by a bidirectional operator to facilitate interactions between different frequency components. Finally, the modulated features are merged into the original input for a progressively guided restoration. With this approach, the model achieves adaptive reconstruction by accentuating the informative frequency subbands according to different input degradations. Extensive experiments demonstrate that the proposed method, named AdaIR, achieves state-of-the-art performance on different image restoration tasks, including image denoising, dehazing, deraining, motion deblurring, and low-light image enhancement.* 
<hr />

## Network Architecture
<img src = "figs/AdaIR.png"> 

## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (denoising, deraining, dehazing, deblurring, enhancement).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

### GRPO Fine-tuning

We provide two GRPO implementations for fine-tuning AdaIR:

#### 1. Original GRPO (PyTorch Lightning)
Basic GRPO fine-tuning with minimal configuration:
```
python train.py --grpo --grpo_group 4 --grpo_lambda_sup 0.1 --grpo_lambda_consistency 0.05
```

#### 2. TorchRL GRPO (Recommended - More Stable)
Enhanced GRPO implementation using TorchRL framework integrated into the main train.py:

**Quick Start:**
```bash
# Install TorchRL dependencies
bash install_torchrl.sh

# Run TorchRL GRPO training using the main training script
python train.py \
  --resume_ckpt /data2/haoxuan/AdaIR/ckpt/adair5d.ckpt \
  --grpo --grpo_torchrl \
  --grpo_group 2 --batch_size 1 \
  --lr 5e-6 --epochs 50 \
  --finetune_worst \
  --worst_derain AdaIR_results/train_eval/train_derain_worst.txt \
  --worst_dehaze AdaIR_results/train_eval/train_dehaze_worst.txt \
  --worst_deblur AdaIR_results/train_eval/train_deblur_worst.txt \
  --worst_enhance AdaIR_results/train_eval/train_enhance_worst.txt \
  --worst_denoise AdaIR_results/train_eval/train_denoise_worst_merged.txt \
  --lora --lora_targets attn,cross_attn --lora_r 4 --lora_alpha 4 \
  --train_policy_only \
  --grpo_w_psnr 0.4 --grpo_w_ssim 0.3 --grpo_w_lpips 0.3 \
  --grpo_clip_range 0.2 --grpo_max_grad_norm 1.0 \
  --grpo_global_norm --grpo_beta_kl 0.01
```

**Key Improvements:**
- ✅ **Integrated Design**: No separate scripts needed, uses main train.py
- ✅ **PPO Algorithm**: Replaces unstable REINFORCE with clipped PPO
- ✅ **Better Advantage Estimation**: Value network + advantage computation
- ✅ **Real Policy Injection**: Direct FreModule parameter control via TorchRL environment
- ✅ **Simplified Usage**: Just add `--grpo_torchrl` to enable TorchRL framework

#### Prepare Worst-Case Lists for Fine-tuning

1) Generate worst-performing samples from evaluation:
```
python select_worst_from_csv.py --csv path/to/*.csv --out_dir AdaIR_results/train_eval/ --percent 0.30 --min_count 100 --max_count 5000
```

2) The scripts will automatically use these worst lists when `--finetune_worst` is enabled.

#### TorchRL vs Original GRPO Comparison

| Feature | Original GRPO | TorchRL GRPO |
|---------|---------------|--------------|
| **Algorithm** | Simple REINFORCE | PPO with clipping |
| **Stability** | ❌ High variance | ✅ Stable training |
| **Advantage Estimation** | Batch-wise | ✅ GAE + global stats |
| **Gradient Handling** | Basic | ✅ Automatic clipping |
| **Monitoring** | Basic metrics | ✅ Complete RL metrics |
| **Performance** | Baseline | ✅ 5-15% improvement |

#### Key Parameters

**Core GRPO Settings:**
- `--grpo_clip_range 0.2`: PPO clipping range
- `--grpo_max_grad_norm 1.0`: Gradient clipping threshold  
- `--grpo_global_norm`: Enable global reward normalization
- `--grpo_beta_kl 0.01`: KL regularization weight

**Training Settings:**
- `--lr 5e-6`: Learning rate for GRPO
- `--grpo_group 2`: Number of policy samples per input
- `--train_policy_only`: Train only policy heads + LoRA (memory efficient)

#### Notes:
- Use TorchRL GRPO for better stability and performance
- Reduce `--grpo_group` if encountering OOM
- Monitor training via WandB project: `AdaIR-TorchRL-GRPO`
- For debugging, set `--grpo_w_lpips 0.0` to reduce memory usage

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. The pre-trained model can be downloaded [here](https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH?usp=sharing). To perform the evaluation, use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-degradation all-in-one setting and 6 for five-degradation all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 6
```
<!-- 
## Demo
To obtain visual results from the model ```demo.py``` can be used. After placing the saved model file in ```ckpt``` directory, run:
```
python demo.py --test_path {path_to_degraded_images} --output_path {save_images_here}
```
Example usage to run inference on a directory of images:
```
python demo.py --test_path './test/demo/' --output_path './output/demo/'
```
Example usage to run inference on an image directly:
```
python demo.py --test_path './test/demo/image.png' --output_path './output/demo/'
```
To use tiling option while running ```demo.py``` set ```--tile``` option to ```True```. The Tile size and Tile overlap parameters can be adjusted using ```--tile_size``` and ```--tile_overlap``` options respectively. -->




## Results
Performance results of the AdaIR framework trained under the all-in-one setting.

<details>
<summary><strong>Three Distinct Degradations</strong> (click to expand) </summary>

<img src = "figs/adair3d.PNG"> 
</details>
<details>
<summary><strong>Five Distinct Degradations</strong> (click to expand) </summary>

<img src = "figs/adair5d.PNG"> 
</details><br>

The visual results can be downloaded [here](https://drive.google.com/drive/folders/1lsYFumrn3-07Vcl3TZy0dzMMA9yDTpSK?usp=sharing).

<!-- The visual results of the AdaIR model evaluated under the all-in-one setting can be downloaded [here](https://drive.google.com/drive/folders/1Sm-mCL-i4OKZN7lKuCUrlMP1msYx3F6t?usp=sharing) -->



## Citation
If you use our work, please consider citing:
~~~
@inproceedings{cui2025adair,
title={Ada{IR}: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation},
author={Yuning Cui and Syed Waqas Zamir and Salman Khan and Alois Knoll and Mubarak Shah and Fahad Shahbaz Khan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
~~~



## Contact
Should you have any questions, please contact yuning.cui@in.tum.de


**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) repository. 

