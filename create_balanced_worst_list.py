#!/usr/bin/env python3
"""
创建平衡的最差样本列表
解决当前"纯最差样本"策略的问题
"""

import os
import csv
import argparse
import numpy as np
from typing import List, Tuple

def read_csv_scores(csv_path: str) -> List[Tuple[str, float, float]]:
    """读取CSV文件中的PSNR/SSIM分数"""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                try:
                    name = row[0]
                    psnr = float(row[1])
                    ssim = float(row[2])
                    rows.append((name, psnr, ssim))
                except ValueError:
                    continue
    return rows

def create_balanced_selection(rows: List[Tuple[str, float, float]], 
                            strategy: str = "mixed",
                            total_samples: int = 2000) -> List[str]:
    """
    创建平衡的样本选择策略
    
    Args:
        rows: (name, psnr, ssim) 列表
        strategy: 选择策略
            - "worst_only": 仅最差30%
            - "mixed": 40%最差 + 40%中等 + 20%较好
            - "gradient": 按质量梯度分布
        total_samples: 总样本数
    """
    
    if not rows:
        return []
    
    # 按PSNR排序 (越小越差)
    sorted_rows = sorted(rows, key=lambda x: x[1])
    n = len(sorted_rows)
    
    if strategy == "worst_only":
        # 原始策略：仅最差30%
        k = min(int(0.3 * n), total_samples)
        selected = sorted_rows[:k]
        
    elif strategy == "mixed":
        # 混合策略：平衡不同质量级别
        worst_count = min(int(0.4 * total_samples), int(0.3 * n))
        medium_start = int(0.3 * n)
        medium_end = int(0.7 * n)
        medium_count = min(int(0.4 * total_samples), medium_end - medium_start)
        good_start = int(0.7 * n)
        good_count = min(int(0.2 * total_samples), n - good_start)
        
        selected = []
        selected.extend(sorted_rows[:worst_count])  # 最差的
        selected.extend(sorted_rows[medium_start:medium_start + medium_count])  # 中等的
        selected.extend(sorted_rows[good_start:good_start + good_count])  # 较好的
        
    elif strategy == "gradient":
        # 梯度策略：按质量梯度采样
        indices = []
        # 最差50%: 60%的样本
        worst_half = int(0.5 * n)
        worst_samples = int(0.6 * total_samples)
        worst_indices = np.linspace(0, worst_half-1, worst_samples, dtype=int)
        indices.extend(worst_indices)
        
        # 较好50%: 40%的样本
        better_samples = total_samples - worst_samples
        better_indices = np.linspace(worst_half, n-1, better_samples, dtype=int)
        indices.extend(better_indices)
        
        selected = [sorted_rows[i] for i in sorted(set(indices))]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return [name for name, _, _ in selected]

def analyze_selection_quality(original_rows: List[Tuple[str, float, float]], 
                            selected_names: List[str]) -> dict:
    """分析选择的样本质量分布"""
    
    # 创建名称到分数的映射
    name_to_scores = {name: (psnr, ssim) for name, psnr, ssim in original_rows}
    
    # 获取选中样本的分数
    selected_scores = []
    for name in selected_names:
        if name in name_to_scores:
            selected_scores.append(name_to_scores[name])
    
    if not selected_scores:
        return {}
    
    psnrs = [s[0] for s in selected_scores]
    ssims = [s[1] for s in selected_scores]
    
    original_psnrs = [s[1] for s in original_rows]
    
    analysis = {
        "selected_count": len(selected_scores),
        "psnr_mean": np.mean(psnrs),
        "psnr_std": np.std(psnrs),
        "psnr_min": np.min(psnrs),
        "psnr_max": np.max(psnrs),
        "ssim_mean": np.mean(ssims),
        "coverage_percentile": len([p for p in original_psnrs if p <= np.max(psnrs)]) / len(original_psnrs) * 100
    }
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="创建平衡的最差样本列表")
    parser.add_argument('--csv_dir', type=str, required=True, help='CSV文件目录')
    parser.add_argument('--out_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--strategy', type=str, default='mixed', 
                       choices=['worst_only', 'mixed', 'gradient'],
                       help='选择策略')
    parser.add_argument('--samples_per_task', type=int, default=2000, 
                       help='每个任务的样本数')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 任务列表
    tasks = ['derain', 'dehaze', 'deblur', 'enhance']
    csv_files = {
        'derain': 'derain.csv',
        'dehaze': 'dehaze.csv', 
        'deblur': 'deblur.csv',
        'enhance': 'enhance.csv'
    }
    
    print(f"🎯 使用策略: {args.strategy}")
    print(f"📊 每任务样本数: {args.samples_per_task}")
    
    all_selected = []
    
    for task in tasks:
        csv_path = os.path.join(args.csv_dir, csv_files[task])
        if not os.path.exists(csv_path):
            print(f"⚠️ 跳过 {task}: 文件不存在 {csv_path}")
            continue
            
        print(f"\n📋 处理任务: {task}")
        rows = read_csv_scores(csv_path)
        print(f"   总样本数: {len(rows)}")
        
        if not rows:
            continue
            
        # 创建选择
        selected_names = create_balanced_selection(
            rows, args.strategy, args.samples_per_task
        )
        
        # 分析质量
        analysis = analyze_selection_quality(rows, selected_names)
        print(f"   选中样本: {analysis.get('selected_count', 0)}")
        print(f"   PSNR范围: {analysis.get('psnr_min', 0):.2f} - {analysis.get('psnr_max', 0):.2f}")
        print(f"   PSNR均值: {analysis.get('psnr_mean', 0):.2f} ± {analysis.get('psnr_std', 0):.2f}")
        print(f"   质量覆盖: {analysis.get('coverage_percentile', 0):.1f}%")
        
        # 保存文件
        out_file = os.path.join(args.out_dir, f'train_{task}_balanced.txt')
        with open(out_file, 'w') as f:
            for name in selected_names:
                f.write(name + '\n')
        
        all_selected.extend([(task, name) for name in selected_names])
        print(f"   保存到: {out_file}")
    
    # 保存组合文件
    combined_file = os.path.join(args.out_dir, 'train_all_balanced.txt')
    with open(combined_file, 'w') as f:
        for task, name in all_selected:
            f.write(f"{task},{name}\n")
    
    print(f"\n🎉 处理完成!")
    print(f"📋 总选中样本: {len(all_selected)}")
    print(f"📁 组合文件: {combined_file}")
    print(f"\n💡 建议:")
    print(f"   在训练脚本中使用 --worst_dir {args.out_dir}")
    print(f"   文件名从 *_worst.txt 改为 *_balanced.txt")

if __name__ == "__main__":
    main()
