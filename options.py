import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=2,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--gopro_dir', type=str, default='data/Train/Deblur/',
                    help='where clean images of denoising saves.')
parser.add_argument('--enhance_dir', type=str, default='data/Train/Enhance/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="AdaIR",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="AdaIR",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4, help = "Number of GPUs to use for training")

# GRPO fine-tuning options
parser.add_argument('--grpo', action='store_true', help='Enable GRPO fine-tuning with stochastic low-dim actions')
parser.add_argument('--grpo_torchrl', action='store_true', help='Use TorchRL framework for GRPO (more stable PPO implementation)')
parser.add_argument('--grpo_flow_style', action='store_true', help='Use flow-grpo style per-prompt advantage normalization')
parser.add_argument('--grpo_group', type=int, default=4, help='Group size (number of stochastic samples per input)')
parser.add_argument('--grpo_lambda_sup', type=float, default=0.1, help='Weight for supervised L1 stabilization')
parser.add_argument('--grpo_lambda_consistency', type=float, default=0.05, help='Weight for consistency to deterministic output')
parser.add_argument('--grpo_reward', type=str, default='l1', choices=['l1'], help='Reward type for GRPO (default: -L1)')
parser.add_argument('--grpo_w_psnr', type=float, default=0.4, help='Weight for PSNR component in reward')
parser.add_argument('--grpo_w_ssim', type=float, default=0.3, help='Weight for SSIM component in reward')
parser.add_argument('--grpo_w_lpips', type=float, default=0.3, help='Weight for (1-LPIPS) component in reward')
parser.add_argument('--grpo_w_niqe', type=float, default=0.0, help='Weight for NIQE component in reward (uses 1/(1+niqe))')

# 新增的GRPO稳定性参数
parser.add_argument('--grpo_max_grad_norm', type=float, default=0.5, help='Max gradient norm for clipping in GRPO (reduced for stability)')
parser.add_argument('--grpo_beta_kl', type=float, default=0.01, help='KL regularization weight for GRPO')
parser.add_argument('--grpo_clip_range', type=float, default=0.2, help='PPO clip range for policy loss')
parser.add_argument('--grpo_adv_clip_max', type=float, default=5.0, help='Maximum value for advantage clipping')
parser.add_argument('--grpo_global_norm', action='store_true', help='Use global reward normalization instead of batch-wise')
parser.add_argument('--grpo_warmup_epochs', type=int, default=5, help='Number of epochs for supervised warmup before GRPO')
parser.add_argument('--grpo_schedule_sup', action='store_true', help='Gradually reduce supervised loss weight during training')

# Resume/LoRA options
parser.add_argument('--resume_ckpt', type=str, default=None, help='Path to a Lightning checkpoint to resume (strict=False)')
parser.add_argument('--lora', action='store_true', help='Enable LoRA adapters on attention/FFN 1x1 convs')
parser.add_argument('--lora_targets', type=str, default='attn,cross_attn', help='Comma-separated targets: attn,cross_attn,ffn')
parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank')
parser.add_argument('--lora_alpha', type=float, default=4.0, help='LoRA alpha (scaling)')
parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
parser.add_argument('--train_policy_only', action='store_true', help='Train only GRPO policy heads (and LoRA if enabled); freeze backbone')

# GRPO finetune on worst lists
parser.add_argument('--finetune_worst', action='store_true', help='Use worst-case file lists for GRPO finetuning')
parser.add_argument('--worst_dir', type=str, default='AdaIR_results/train_eval/', help='Directory containing worst list txts')
parser.add_argument('--worst_derain', type=str, default='AdaIR_results/train_eval/train_derain_worst.txt')
parser.add_argument('--worst_dehaze', type=str, default='AdaIR_results/train_eval/train_dehaze_worst.txt')
parser.add_argument('--worst_deblur', type=str, default='AdaIR_results/train_eval/train_deblur_worst.txt')
parser.add_argument('--worst_enhance', type=str, default='AdaIR_results/train_eval/train_enhance_worst.txt')
parser.add_argument('--worst_denoise', type=str, default='AdaIR_results/train_eval/train_denoise_worst_merged.txt')

options = parser.parse_args()

