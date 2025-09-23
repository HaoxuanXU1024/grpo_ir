import os
import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import csv
from typing import List, Tuple

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import AdaIR
from net.model_torchrl import AdaIRTorchRL
from grpo_torchrl_env import AdaIRPolicyNetwork
try:
    from tensordict import TensorDict
except ImportError:
    print("Warning: tensordict not found. Run 'pip install tensordict'")
    TensorDict = None


class AdaIRModel(pl.LightningModule):
    def __init__(self, grpo_enabled=False):
        super().__init__()
        self.grpo_enabled = grpo_enabled
        if self.grpo_enabled:
            print("INFO: Initializing model with TorchRL GRPO components for testing.")
            if TensorDict is None:
                raise ImportError("tensordict is required for GRPO evaluation. Please install it.")
            self.net = AdaIRTorchRL(decoder=True)
            self.policy_network = AdaIRPolicyNetwork()
            # A value_network is also part of the saved checkpoint, so we must define it here to load weights,
            # even if it's not used during inference.
            self.value_network = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
            self.net = AdaIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        if self.grpo_enabled:
            # Ensure networks are on the same device as input
            device = x.device
            self.net = self.net.to(device)
            self.policy_network = self.policy_network.to(device)

            # Build observation for policy network
            obs = TensorDict({"degraded_image": x}, batch_size=x.shape[:1], device=device)

            # Policy outputs Beta parameters per action; convert to action means
            with torch.no_grad():
                action_td = self.policy_network(obs)
            freq_params = action_td["action"]["freq_params"]  # [B, 3, 8] = 4 (alpha,beta) pairs per FreModule
            # Reshape to pairs: [B, 3, 4, 2] -> (..., 0)=alpha, (...,1)=beta
            params = freq_params.view(freq_params.shape[0], 3, 4, 2)
            alphas = params[..., 0]
            betas = params[..., 1]
            actions = alphas / (alphas + betas + 1e-8)  # [B, 3, 4] -> (r_h, r_w, g1, g2)
            # Clamp actions to a safe range to avoid extreme masks/fusions
            actions = torch.clamp(actions, 0.2, 0.8)

            # Inject actions and run stochastic forward (uses injected actions path)
            self.net.inject_actions(actions)
            result = self.net(x, stochastic=True)
            if isinstance(result, tuple):
                return result[0]
            return result
        else:
            # Original deterministic forward pass
            return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


def _psnr_ssim_tensor(net, degrad_img: torch.Tensor, clean_img: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        degrad_img = degrad_img.cuda()
        clean_img = clean_img.cuda()
        restored = net(degrad_img)
        temp_psnr, temp_ssim, _ = compute_psnr_ssim(restored, clean_img)
    return float(temp_psnr), float(temp_ssim)

def _write_csv(rows: List[Tuple[str, float, float]], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "psnr", "ssim"])
        for r in rows:
            writer.writerow(r)

def _select_worst(rows: List[Tuple[str, float, float]], top_k: int) -> List[str]:
    # 排序依据：先按 PSNR 升序，再按 SSIM 升序
    rows_sorted = sorted(rows, key=lambda x: (x[1], x[2]))
    worst = [r[0] for r in rows_sorted[:max(0, int(top_k))]]
    return worst

def eval_train_derain(net, root: str, csv_out: str, top_k: int = 100, worst_list_out: str = None):
    # 结构: root/rainy/*.png 与 root/gt/norain-*.png
    from utils.image_utils import crop_img
    from PIL import Image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    rainy_dir = os.path.join(root, 'rainy')
    gt_dir = os.path.join(root, 'gt')
    exts = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    imgs = [f for f in os.listdir(rainy_dir) if f.endswith(exts)]
    rows = []
    pbar = tqdm(imgs, desc='Eval Train Derain')
    for name in pbar:
        rainy_path = os.path.join(rainy_dir, name)
        gt_name = 'norain-' + name.split('rain-')[-1] if 'rain-' in name else name
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            continue
        degrad = crop_img(np.array(Image.open(rainy_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(gt_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rows.append((os.path.join('rainy', name), p, s))
        pbar.set_postfix(psnr=f"{p:.2f}", ssim=f"{s:.4f}")
    _write_csv(rows, csv_out)
    if worst_list_out is not None:
        worst = _select_worst(rows, top_k)
        with open(worst_list_out, 'w') as f:
            for w in worst:
                f.write(w + '\n')

def eval_train_dehaze(net, root: str, csv_out: str, top_k: int = 100, worst_list_out: str = None):
    # 结构: root/synthetic/*.png 与 root/original/<prefix>.png (取下划线前缀)
    from utils.image_utils import crop_img
    from PIL import Image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    syn_dir = os.path.join(root, 'synthetic')
    ori_dir = os.path.join(root, 'original')
    exts = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    imgs = [f for f in os.listdir(syn_dir) if f.endswith(exts)]
    rows = []
    pbar = tqdm(imgs, desc='Eval Train Dehaze')
    for name in pbar:
        syn_path = os.path.join(syn_dir, name)
        prefix = name.split('_')[0]
        ext = '.' + name.split('.')[-1]
        ori_path = os.path.join(ori_dir, prefix + ext)
        if not os.path.exists(ori_path):
            continue
        degrad = crop_img(np.array(Image.open(syn_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(ori_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rows.append((os.path.join('synthetic', name), p, s))
        pbar.set_postfix(psnr=f"{p:.2f}", ssim=f"{s:.4f}")
    _write_csv(rows, csv_out)
    if worst_list_out is not None:
        worst = _select_worst(rows, top_k)
        with open(worst_list_out, 'w') as f:
            for w in worst:
                f.write(w + '\n')

def eval_train_deblur(net, root: str, csv_out: str, top_k: int = 100, worst_list_out: str = None):
    # 结构: root/blur/*.png 与 root/sharp/*.png (同名)
    from utils.image_utils import crop_img
    from PIL import Image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    blur_dir = os.path.join(root, 'blur')
    sharp_dir = os.path.join(root, 'sharp')
    exts = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    imgs = [f for f in os.listdir(blur_dir) if f.endswith(exts)]
    rows = []
    pbar = tqdm(imgs, desc='Eval Train Deblur')
    for name in pbar:
        b_path = os.path.join(blur_dir, name)
        s_path = os.path.join(sharp_dir, name)
        if not os.path.exists(s_path):
            continue
        degrad = crop_img(np.array(Image.open(b_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(s_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rows.append((os.path.join('blur', name), p, s))
        pbar.set_postfix(psnr=f"{p:.2f}", ssim=f"{s:.4f}")
    _write_csv(rows, csv_out)
    if worst_list_out is not None:
        worst = _select_worst(rows, top_k)
        with open(worst_list_out, 'w') as f:
            for w in worst:
                f.write(w + '\n')

def eval_train_enhance(net, root: str, csv_out: str, top_k: int = 100, worst_list_out: str = None):
    # 结构: root/low/*.png 与 root/gt/*.png (同名)
    from utils.image_utils import crop_img
    from PIL import Image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    low_dir = os.path.join(root, 'low')
    gt_dir = os.path.join(root, 'gt')
    exts = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    imgs = [f for f in os.listdir(low_dir) if f.endswith(exts)]
    rows = []
    pbar = tqdm(imgs, desc='Eval Train Enhance')
    for name in pbar:
        l_path = os.path.join(low_dir, name)
        g_path = os.path.join(gt_dir, name)
        if not os.path.exists(g_path):
            continue
        degrad = crop_img(np.array(Image.open(l_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(g_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rows.append((os.path.join('low', name), p, s))
        pbar.set_postfix(psnr=f"{p:.2f}", ssim=f"{s:.4f}")
    _write_csv(rows, csv_out)
    if worst_list_out is not None:
        worst = _select_worst(rows, top_k)
        with open(worst_list_out, 'w') as f:
            for w in worst:
                f.write(w + '\n')

def eval_train_denoise(net, root: str, sigmas: List[int], out_dir: str):
    # 针对去噪训练的 clean 集合，在多种 sigma 下评估，分别输出 CSV
    from utils.image_utils import crop_img
    from PIL import Image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    exts = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')
    imgs = [f for f in os.listdir(root) if f.endswith(exts)]
    for sigma in sigmas:
        rows = []
        pbar = tqdm(imgs, desc=f'Eval Train Denoise sigma={sigma}')
        for name in pbar:
            c_path = os.path.join(root, name)
            clean = crop_img(np.array(Image.open(c_path).convert('RGB')), base=16)
            # 合成噪声
            noise = np.random.randn(*clean.shape)
            noisy = np.clip(clean + noise * sigma, 0, 255).astype(np.uint8)
            clean_t = to_tensor(clean).unsqueeze(0)
            noisy_t = to_tensor(noisy).unsqueeze(0)
            p, s = _psnr_ssim_tensor(net, noisy_t, clean_t)
            rows.append((name, p, s))
            pbar.set_postfix(psnr=f"{p:.2f}", ssim=f"{s:.4f}")
        csv_out = os.path.join(out_dir, f'denoise_sigma{sigma}.csv')
        _write_csv(rows, csv_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=6,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for enhance, 5 for all-in-one (three tasks), 6 for all-in-one (five tasks)')
    
    parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='save path of test hazy images')
    parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='save path of test hazy images')
    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')

    parser.add_argument('--output_path', type=str, default="AdaIR_results1/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="adair5d.ckpt", help='checkpoint save path')
    parser.add_argument('--grpo_torchrl_ckpt', action='store_true', help='Set this flag if loading a GRPO+TorchRL fine-tuned checkpoint')
    # 训练集评测与导出 CSV
    parser.add_argument('--eval_train', action='store_true', help='Evaluate training data and export CSV metrics')
    parser.add_argument('--train_task', type=str, default='all', choices=['all','derain','dehaze','deblur','enhance','denoise'], help='Which train subset to evaluate')
    parser.add_argument('--train_derain_root', type=str, default='data/Train/Derain/', help='Train derain root (contains rainy/ and gt/)')
    parser.add_argument('--train_dehaze_root', type=str, default='data/Train/Dehaze/', help='Train dehaze root (contains synthetic/ and original/)')
    parser.add_argument('--train_deblur_root', type=str, default='data/Train/Deblur/', help='Train deblur root (contains blur/ and sharp/)')
    parser.add_argument('--train_enhance_root', type=str, default='data/Train/Enhance/', help='Train enhance root (contains low/ and gt/)')
    parser.add_argument('--train_denoise_root', type=str, default='data/Train/Denoise/', help='Train denoise clean root (images only)')
    parser.add_argument('--csv_dir', type=str, default='AdaIR_results/train_eval/', help='Where to save CSVs')
    parser.add_argument('--worst_topk', type=int, default=100, help='Top-K worst images to select per task')
    parser.add_argument('--worst_list', type=str, default='AdaIR_results/train_eval/worst_list.txt', help='Path to save worst list (per task appended)')
    parser.add_argument('--eval_variants', type=str, default='p2', choices=['p0','p1','p2','all'], help='Which inference variants to evaluate (p0=base, p1=ft-no-policy, p2=ft-with-policy, all=run all three)')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    # ckpt_path = "ckpt/" + testopt.ckpt_name
    ckpt_path = "/data2/haoxuan/AdaIR/torchrl_grpo_gpu23_advanced_9_23/epoch=32-step=122760.ckpt"

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))
    # Variants to evaluate: P0 (base), P1 (ft no policy), P2 (ft with policy)
    variants_to_run = ['p2'] if testopt.eval_variants != 'all' else ['p0','p1','p2']
    if testopt.eval_variants in ('p0','p1','p2'):
        variants_to_run = [testopt.eval_variants]

    original_output_root = testopt.output_path

    for variant in variants_to_run:
        if variant == 'p0':
            v_tag = 'P0'
            v_ckpt = os.path.join('ckpt', testopt.ckpt_name)
            v_grpo = False
            print('[P0] Running base pretrained model (no policy).')
        elif variant == 'p1':
            v_tag = 'P1'
            v_ckpt = ckpt_path
            v_grpo = False
            print('[P1] Running finetuned weights (policy disabled).')
        else:
            v_tag = 'P2'
            v_ckpt = ckpt_path
            v_grpo = True
            print('[P2] Running finetuned weights with TorchRL policy (actions injected).')

        print(f'[{v_tag}] Loading checkpoint: {v_ckpt}')
        try:
            net = AdaIRModel.load_from_checkpoint(
                v_ckpt,
                strict=False,
                grpo_enabled=v_grpo
            ).cuda()
        except Exception as e:
            print(f'[{v_tag}] Failed to load checkpoint: {e}')
            continue
        net.eval()

        # Set variant-specific output root
        testopt.output_path = os.path.join(original_output_root, v_tag) + '/'
        os.makedirs(testopt.output_path, exist_ok=True)

        # Optional: evaluate training data and export CSVs
        if testopt.eval_train:
            os.makedirs(testopt.csv_dir, exist_ok=True)
            if testopt.train_task in ('all', 'derain'):
                eval_train_derain(net, testopt.train_derain_root, os.path.join(testopt.csv_dir, f'derain_{v_tag}.csv'), testopt.worst_topk, testopt.worst_list)
            if testopt.train_task in ('all', 'dehaze'):
                eval_train_dehaze(net, testopt.train_dehaze_root, os.path.join(testopt.csv_dir, f'dehaze_{v_tag}.csv'), testopt.worst_topk, testopt.worst_list)
            if testopt.train_task in ('all', 'deblur'):
                eval_train_deblur(net, testopt.train_deblur_root, os.path.join(testopt.csv_dir, f'deblur_{v_tag}.csv'), testopt.worst_topk, testopt.worst_list)
            if testopt.train_task in ('all', 'enhance'):
                eval_train_enhance(net, testopt.train_enhance_root, os.path.join(testopt.csv_dir, f'enhance_{v_tag}.csv'), testopt.worst_topk, testopt.worst_list)
            if testopt.train_task in ('all', 'denoise'):
                eval_train_denoise(net, testopt.train_denoise_root, [15,25,50], testopt.csv_dir)
            if testopt.train_task != 'all' or testopt.eval_train:
                print(f'[{v_tag}] Train evaluation done. CSV saved to {testopt.csv_dir}')

        # Task-specific evaluation
        if testopt.mode == 0:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print(f'[{v_tag}] Start {name} testing Sigma=15...')
                test_Denoise(net, testset, sigma=15)

                print(f'[{v_tag}] Start {name} testing Sigma=25...')
                test_Denoise(net, testset, sigma=25)

                print(f'[{v_tag}] Start {name} testing Sigma=50...')
                test_Denoise(net, testset, sigma=50)

        elif testopt.mode == 1:
            print(f'[{v_tag}] Start testing rain streak removal...')
            derain_base_path = testopt.derain_path
            for name in derain_splits:
                print(f'[{v_tag}] Start testing {name} rain streak removal...')
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
                test_Derain_Dehaze(net, derain_set, task="derain")

        elif testopt.mode == 2:
            print(f'[{v_tag}] Start testing SOTS...')
            derain_base_path = testopt.derain_path
            name = derain_splits[0]
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="dehaze")

        elif testopt.mode == 3:
            print(f'[{v_tag}] Start testing GOPRO...')
            deblur_base_path = testopt.gopro_path
            name = deblur_splits[0]
            testopt.gopro_path = os.path.join(deblur_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15, task='deblur')
            test_Derain_Dehaze(net, derain_set, task="deblur")

        elif testopt.mode == 4:
            print(f'[{v_tag}] Start testing LOL...')
            enhance_base_path = testopt.enhance_path
            name = derain_splits[0]
            testopt.enhance_path = os.path.join(enhance_base_path,name, task='enhance')
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="enhance")

        elif testopt.mode == 5:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print(f'[{v_tag}] Start {name} testing Sigma=15...')
                test_Denoise(net, testset, sigma=15)

                print(f'[{v_tag}] Start {name} testing Sigma=25...')
                test_Denoise(net, testset, sigma=25)

                print(f'[{v_tag}] Start {name} testing Sigma=50...')
                test_Denoise(net, testset, sigma=50)

            derain_base_path = testopt.derain_path
            print(derain_splits)
            for name in derain_splits:
                print(f'[{v_tag}] Start testing {name} rain streak removal...')
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
                test_Derain_Dehaze(net, derain_set, task="derain")

            print(f'[{v_tag}] Start testing SOTS...')
            test_Derain_Dehaze(net, derain_set, task="dehaze")

        elif testopt.mode == 6:
            for testset,name in zip(denoise_tests,denoise_splits) :
                print(f'[{v_tag}] Start {name} testing Sigma=15...')
                test_Denoise(net, testset, sigma=15)

                print(f'[{v_tag}] Start {name} testing Sigma=25...')
                test_Denoise(net, testset, sigma=25)

                print(f'[{v_tag}] Start {name} testing Sigma=50...')
                test_Denoise(net, testset, sigma=50)

            derain_base_path = testopt.derain_path
            print(derain_splits)
            for name in derain_splits:
                print(f'[{v_tag}] Start testing {name} rain streak removal...')
                testopt.derain_path = os.path.join(derain_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
                test_Derain_Dehaze(net, derain_set, task="derain")

            print(f'[{v_tag}] Start testing SOTS...')
            test_Derain_Dehaze(net, derain_set, task="dehaze")

            deblur_base_path = testopt.gopro_path
            for name in deblur_splits:
                print(f'[{v_tag}] Start testing GOPRO...')
                testopt.gopro_path = os.path.join(deblur_base_path,name)
                deblur_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='deblur')
                test_Derain_Dehaze(net, deblur_set, task="deblur")

            enhance_base_path = testopt.enhance_path
            for name in enhance_splits:
                print(f'[{v_tag}] Start testing LOL...')
                testopt.enhance_path = os.path.join(enhance_base_path,name)
                derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='enhance')
                test_Derain_Dehaze(net, derain_set, task="enhance")
