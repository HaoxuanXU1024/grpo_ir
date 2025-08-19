import os
import csv
import argparse
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

import lightning.pytorch as pl

from net.model import AdaIR
from utils.image_utils import crop_img
from utils.val_utils import compute_psnr_ssim


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
    rows_sorted = sorted(rows, key=lambda x: (x[1], x[2]))
    worst = [r[0] for r in rows_sorted[:max(0, int(top_k))]]
    return worst


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
    def forward(self, x):
        return self.net(x)


def eval_derain(args, net) -> List[Tuple[str, float, float]]:
    """Follow training: data_file_dir + rainy/rainTrain.txt; paths are relative to derain_dir.
    Pair rule: rainy/rain-xxx -> gt/norain-xxx
    """
    to_tensor = ToTensor()
    list_file = os.path.join(args.data_file_dir, 'rainy', 'rainTrain.txt')
    rows: List[Tuple[str, float, float]] = []
    if not os.path.exists(list_file):
        print(f"[WARN] {list_file} not found, skip derain")
        return rows
    with open(list_file, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for rel in tqdm(lines, desc='Eval Train Derain', ncols=100):
        rainy_path = os.path.join(args.derain_dir, rel)
        # build gt path
        gt_path = rainy_path.split('rainy')[0] + 'gt/norain-' + rainy_path.split('rain-')[-1]
        if not os.path.exists(rainy_path) or not os.path.exists(gt_path):
            continue
        degrad = crop_img(np.array(Image.open(rainy_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(gt_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rows.append((rel, p, s))
    return rows


def eval_dehaze(args, net) -> List[Tuple[str, float, float]]:
    """Follow training: data_file_dir + hazy/hazy_outside.txt; each line is a path under dehaze_dir.
    Pair rule: synthetic/<name_with_underscores> -> original/<prefix>.<ext>
    """
    to_tensor = ToTensor()
    list_file = os.path.join(args.data_file_dir, 'hazy', 'hazy_outside.txt')
    rows: List[Tuple[str, float, float]] = []
    if not os.path.exists(list_file):
        print(f"[WARN] {list_file} not found, skip dehaze")
        return rows
    with open(list_file, 'r') as f:
        paths = [os.path.join(args.dehaze_dir, ln.strip()) for ln in f if ln.strip()]
    for syn_path in tqdm(paths, desc='Eval Train Dehaze', ncols=100):
        if 'synthetic' not in syn_path:
            continue
        prefix = os.path.basename(syn_path).split('_')[0]
        ext = '.' + os.path.basename(syn_path).split('.')[-1]
        ori_path = syn_path.split('synthetic')[0] + 'original/' + prefix + ext
        if not os.path.exists(syn_path) or not os.path.exists(ori_path):
            continue
        degrad = crop_img(np.array(Image.open(syn_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(ori_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rel = os.path.relpath(syn_path, args.dehaze_dir)
        rows.append((rel, p, s))
    return rows


def eval_deblur(args, net) -> List[Tuple[str, float, float]]:
    """Training lists blur/ under gopro_dir; pair with sharp/ same name."""
    to_tensor = ToTensor()
    rows: List[Tuple[str, float, float]] = []
    blur_dir = os.path.join(args.gopro_dir, 'blur')
    sharp_dir = os.path.join(args.gopro_dir, 'sharp')
    if not os.path.isdir(blur_dir) or not os.path.isdir(sharp_dir):
        print(f"[WARN] Missing {blur_dir} or {sharp_dir}, skip deblur")
        return rows
    names = sorted([n for n in os.listdir(blur_dir) if os.path.isfile(os.path.join(blur_dir, n))])
    for name in tqdm(names, desc='Eval Train Deblur', ncols=100):
        b_path = os.path.join(blur_dir, name)
        s_path = os.path.join(sharp_dir, name)
        if not os.path.exists(s_path):
            continue
        degrad = crop_img(np.array(Image.open(b_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(s_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rel = os.path.join('blur', name)
        rows.append((rel, p, s))
    return rows


def eval_enhance(args, net) -> List[Tuple[str, float, float]]:
    """Training lists low/ under enhance_dir; pair with gt/ same name."""
    to_tensor = ToTensor()
    rows: List[Tuple[str, float, float]] = []
    low_dir = os.path.join(args.enhance_dir, 'low')
    gt_dir = os.path.join(args.enhance_dir, 'gt')
    if not os.path.isdir(low_dir) or not os.path.isdir(gt_dir):
        print(f"[WARN] Missing {low_dir} or {gt_dir}, skip enhance")
        return rows
    names = sorted([n for n in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, n))])
    for name in tqdm(names, desc='Eval Train Enhance', ncols=100):
        l_path = os.path.join(low_dir, name)
        g_path = os.path.join(gt_dir, name)
        if not os.path.exists(g_path):
            continue
        degrad = crop_img(np.array(Image.open(l_path).convert('RGB')), base=16)
        clean = crop_img(np.array(Image.open(g_path).convert('RGB')), base=16)
        degrad_t, clean_t = to_tensor(degrad).unsqueeze(0), to_tensor(clean).unsqueeze(0)
        p, s = _psnr_ssim_tensor(net, degrad_t, clean_t)
        rel = os.path.join('low', name)
        rows.append((rel, p, s))
    return rows


def eval_denoise(args, net, sigmas: List[int]) -> List[List[Tuple[str, float, float]]]:
    """Training filters denoise_dir using data_file_dir/noisy/denoise.txt"""
    to_tensor = ToTensor()
    list_file = os.path.join(args.data_file_dir, 'noisy', 'denoise.txt')
    names: List[str] = []
    if not os.path.exists(list_file):
        print(f"[WARN] {list_file} not found, skip denoise")
        return []
    with open(list_file, 'r') as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    clean_paths = [os.path.join(args.denoise_dir, n) for n in os.listdir(args.denoise_dir) if n in ids]
    results: List[List[Tuple[str, float, float]]] = []
    for sigma in sigmas:
        rows: List[Tuple[str, float, float]] = []
        for c_path in tqdm(clean_paths, desc=f'Eval Train Denoise sigma={sigma}', ncols=100):
            clean = crop_img(np.array(Image.open(c_path).convert('RGB')), base=16)
            noise = np.random.randn(*clean.shape)
            noisy = np.clip(clean + noise * sigma, 0, 255).astype(np.uint8)
            clean_t = to_tensor(clean).unsqueeze(0)
            noisy_t = to_tensor(noisy).unsqueeze(0)
            p, s = _psnr_ssim_tensor(net, noisy_t, clean_t)
            rel = os.path.basename(c_path)
            rows.append((rel, p, s))
        results.append(rows)
    return results


def _parse_exts(exts_csv: str) -> Tuple[str, ...]:
    items = [s.strip() for s in exts_csv.split(',') if s.strip()]
    items = [s if s.startswith('.') else f'.{s}' for s in items]
    return tuple(items)


def eval_denoise_all(args, net, sigmas: List[int]) -> List[List[Tuple[str, float, float]]]:
    """Evaluate all images in denoise_dir by extension filter (ignore list file)."""
    to_tensor = ToTensor()
    exts = _parse_exts(args.denoise_exts)
    names = [n for n in os.listdir(args.denoise_dir) if n.endswith(exts)]
    clean_paths = [os.path.join(args.denoise_dir, n) for n in names]
    results: List[List[Tuple[str, float, float]]] = []
    for sigma in sigmas:
        rows: List[Tuple[str, float, float]] = []
        for c_path in tqdm(clean_paths, desc=f'Eval Train Denoise(all) sigma={sigma}', ncols=100):
            clean = crop_img(np.array(Image.open(c_path).convert('RGB')), base=16)
            noise = np.random.randn(*clean.shape)
            noisy = np.clip(clean + noise * sigma, 0, 255).astype(np.uint8)
            clean_t = to_tensor(clean).unsqueeze(0)
            noisy_t = to_tensor(noisy).unsqueeze(0)
            p, s = _psnr_ssim_tensor(net, noisy_t, clean_t)
            rel = os.path.basename(c_path)
            rows.append((rel, p, s))
        results.append(rows)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Lightning checkpoint to evaluate (strict=False)')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='Lists dir as in training')
    parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/')
    parser.add_argument('--denoise_eval_mode', type=str, default='list', choices=['list','all'], help='Use list file or evaluate all images in denoise_dir')
    parser.add_argument('--denoise_exts', type=str, default='png,PNG,jpg,JPG,jpeg,JPEG,bmp,BMP', help='Extensions used when denoise_eval_mode=all')
    parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/')
    parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/')
    parser.add_argument('--gopro_dir', type=str, default='data/Train/Deblur/')
    parser.add_argument('--enhance_dir', type=str, default='data/Train/Enhance/')
    parser.add_argument('--out_dir', type=str, default='AdaIR_results/train_eval/')
    parser.add_argument('--worst_topk', type=int, default=100)
    parser.add_argument('--tasks', nargs='+', default=['derain','dehaze','deblur','enhance','denoise'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.cuda.set_device(args.cuda)
    torch.set_grad_enabled(False)
    model = AdaIRModel()
    print(f"[INFO] Load from {args.ckpt} (strict=False)", flush=True)
    model = model.load_from_checkpoint(args.ckpt, strict=False)
    # Use raw AdaIR net to avoid any Lightning hooks during eval
    net = model.net.cuda().eval()

    if 'derain' in args.tasks:
        print('[INFO] Eval derain (train)', flush=True)
        rows = eval_derain(args, net)
        _write_csv(rows, os.path.join(args.out_dir, 'train_derain.csv'))
        worst = _select_worst(rows, args.worst_topk)
        with open(os.path.join(args.out_dir, 'worst_derain.txt'), 'w') as f:
            for w in worst:
                f.write(w + '\n')

    if 'dehaze' in args.tasks:
        print('[INFO] Eval dehaze (train)', flush=True)
        rows = eval_dehaze(args, net)
        _write_csv(rows, os.path.join(args.out_dir, 'train_dehaze.csv'))
        worst = _select_worst(rows, args.worst_topk)
        with open(os.path.join(args.out_dir, 'worst_dehaze.txt'), 'w') as f:
            for w in worst:
                f.write(w + '\n')

    if 'deblur' in args.tasks:
        print('[INFO] Eval deblur (train)', flush=True)
        rows = eval_deblur(args, net)
        _write_csv(rows, os.path.join(args.out_dir, 'train_deblur.csv'))
        worst = _select_worst(rows, args.worst_topk)
        with open(os.path.join(args.out_dir, 'worst_deblur.txt'), 'w') as f:
            for w in worst:
                f.write(w + '\n')

    if 'enhance' in args.tasks:
        print('[INFO] Eval enhance (train)', flush=True)
        rows = eval_enhance(args, net)
        _write_csv(rows, os.path.join(args.out_dir, 'train_enhance.csv'))
        worst = _select_worst(rows, args.worst_topk)
        with open(os.path.join(args.out_dir, 'worst_enhance.txt'), 'w') as f:
            for w in worst:
                f.write(w + '\n')

    if 'denoise' in args.tasks:
        print('[INFO] Eval denoise (train)', flush=True)
        if args.denoise_eval_mode == 'all':
            all_rows = eval_denoise_all(args, net, [15,25,50])
        else:
            all_rows = eval_denoise(args, net, [15,25,50])
        for (sigma, rows) in zip([15,25,50], all_rows):
            _write_csv(rows, os.path.join(args.out_dir, f'train_denoise_sigma{sigma}.csv'))

    print('[DONE] Train set evaluation finished. CSVs saved to', args.out_dir, flush=True)


if __name__ == '__main__':
    main()


