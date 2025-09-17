"""
改进的数据选择策略
避免使用质量过差的worst samples
"""

import os
import csv
import argparse
import numpy as np
from typing import List, Tuple

def read_csv_with_validation(path: str) -> List[Tuple[str, float, float]]:
    """读取CSV并验证数据质量"""
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for r in reader:
            if not r or len(r) < 3:
                continue
            name = r[0]
            try:
                psnr = float(r[1])
                ssim = float(r[2])
                
                # 数据质量验证：过滤明显异常的样本
                if psnr < 10 or psnr > 50:  # PSNR合理范围
                    continue
                if ssim < 0.3 or ssim > 1.0:  # SSIM合理范围
                    continue
                    
                rows.append((name, psnr, ssim))
            except Exception:
                continue
    return rows

def select_challenging_samples(rows: List[Tuple[str, float, float]], 
                             strategy: str = "moderate_worst",
                             percent: float = 0.20,
                             min_count: int = 50,
                             max_count: int = 2000) -> List[str]:
    """选择有挑战性但不是最差的样本"""
    
    if not rows:
        return []
    
    n = len(rows)
    rows_sorted = sorted(rows, key=lambda x: (x[1], x[2]))  # 按PSNR, SSIM升序
    
    if strategy == "moderate_worst":
        # 选择中等偏差的样本：跳过最差的10%，选择接下来的20%
        skip_worst = int(0.10 * n)  # 跳过最差的10%
        start_idx = skip_worst
        end_idx = min(start_idx + int(percent * n), n)
        selected = rows_sorted[start_idx:end_idx]
        
    elif strategy == "diverse_range":
        # 分层采样：从不同质量层级选择
        # 将数据分为5个质量层，从每层选择样本
        layer_size = n // 5
        selected = []
        
        # 从每个质量层选择一定比例
        layer_percents = [0.05, 0.15, 0.25, 0.30, 0.25]  # 偏向中等质量
        
        for i, layer_percent in enumerate(layer_percents):
            start = i * layer_size
            end = min((i + 1) * layer_size, n)
            layer_samples = rows_sorted[start:end]
            
            layer_count = int(len(layer_samples) * layer_percent)
            # 随机选择而不是只选最差的
            np.random.shuffle(layer_samples)
            selected.extend(layer_samples[:layer_count])
            
    elif strategy == "tail_distribution":
        # 基于分布尾部选择：选择低于某个阈值但不是最极端的
        psnr_values = [r[1] for r in rows]
        ssim_values = [r[2] for r in rows]
        
        psnr_threshold = np.percentile(psnr_values, 30)  # 30分位
        ssim_threshold = np.percentile(ssim_values, 30)
        
        # 选择至少一个指标低于阈值的样本
        selected = [r for r in rows if r[1] <= psnr_threshold or r[2] <= ssim_threshold]
        
        # 如果选择太多，随机采样
        if len(selected) > int(percent * n):
            np.random.shuffle(selected)
            selected = selected[:int(percent * n)]
    
    # 应用数量限制
    k = len(selected)
    k = max(min_count, k)
    k = min(max_count, k)
    k = min(k, len(selected))
    
    if k < len(selected):
        np.random.shuffle(selected)
        selected = selected[:k]
    
    return [name for (name, _, _) in selected]

def analyze_data_distribution(csv_files: List[str], output_dir: str):
    """分析数据分布，帮助选择策略"""
    
    all_data = {}
    for csv_path in csv_files:
        task = os.path.splitext(os.path.basename(csv_path))[0]
        rows = read_csv_with_validation(csv_path)
        
        if not rows:
            continue
            
        psnr_values = [r[1] for r in rows]
        ssim_values = [r[2] for r in rows]
        
        all_data[task] = {
            'count': len(rows),
            'psnr_stats': {
                'mean': np.mean(psnr_values),
                'std': np.std(psnr_values),
                'min': np.min(psnr_values),
                'max': np.max(psnr_values),
                'p10': np.percentile(psnr_values, 10),
                'p30': np.percentile(psnr_values, 30),
                'p50': np.percentile(psnr_values, 50),
            },
            'ssim_stats': {
                'mean': np.mean(ssim_values),
                'std': np.std(ssim_values),
                'min': np.min(ssim_values),
                'max': np.max(ssim_values),
                'p10': np.percentile(ssim_values, 10),
                'p30': np.percentile(ssim_values, 30),
                'p50': np.percentile(ssim_values, 50),
            }
        }
    
    # 输出分析报告
    report_path = os.path.join(output_dir, 'data_distribution_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("数据分布分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        for task, stats in all_data.items():
            f.write(f"任务: {task}\n")
            f.write(f"样本数量: {stats['count']}\n")
            f.write(f"PSNR统计: {stats['psnr_stats']}\n")
            f.write(f"SSIM统计: {stats['ssim_stats']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"数据分布分析报告已保存到: {report_path}")
    return all_data

def main():
    parser = argparse.ArgumentParser(description="改进的数据选择策略")
    parser.add_argument('--csv', nargs='+', required=True, help='CSV文件路径')
    parser.add_argument('--out_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--strategy', type=str, default='moderate_worst', 
                       choices=['moderate_worst', 'diverse_range', 'tail_distribution'],
                       help='选择策略')
    parser.add_argument('--percent', type=float, default=0.20, help='选择百分比 (默认20%)')
    parser.add_argument('--min_count', type=int, default=50, help='每个任务最少样本数')
    parser.add_argument('--max_count', type=int, default=2000, help='每个任务最多样本数')
    parser.add_argument('--analyze_only', action='store_true', help='只分析数据分布，不选择样本')
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(42)  # 保证可重复性
    
    # 分析数据分布
    print("正在分析数据分布...")
    data_stats = analyze_data_distribution(args.csv, args.out_dir)
    
    if args.analyze_only:
        return
    
    # 选择样本
    print(f"使用策略: {args.strategy}")
    combined = []
    denoise_merged_set = set()
    
    for csv_path in args.csv:
        task = os.path.splitext(os.path.basename(csv_path))[0]
        rows = read_csv_with_validation(csv_path)
        
        if not rows:
            print(f"警告: {task} 没有有效数据")
            continue
        
        selected = select_challenging_samples(
            rows, 
            strategy=args.strategy,
            percent=args.percent,
            min_count=args.min_count,
            max_count=args.max_count
        )
        
        # 输出选择结果
        out_txt = os.path.join(args.out_dir, f"{task}_challenging.txt")
        with open(out_txt, 'w') as f:
            for name in selected:
                f.write(name + '\n')
        
        combined.extend([(task, name) for name in selected])
        
        # 统计信息
        original_count = len(rows)
        selected_count = len(selected)
        print(f"[INFO] {task}: 原始={original_count}, 选择={selected_count} "
              f"({selected_count/original_count*100:.1f}%) -> {out_txt}")
        
        # 合并denoise数据
        if task.startswith('train_denoise_sigma'):
            denoise_merged_set.update(selected)
    
    # 输出合并列表
    combined_txt = os.path.join(args.out_dir, 'combined_challenging.txt')
    with open(combined_txt, 'w') as f:
        for task, name in combined:
            f.write(f"{task},{name}\n")
    print(f"[INFO] 合并列表 -> {combined_txt} (总计 {len(combined)})")
    
    # 输出denoise合并列表
    if denoise_merged_set:
        denoise_merged_txt = os.path.join(args.out_dir, 'train_denoise_challenging_merged.txt')
        with open(denoise_merged_txt, 'w') as f:
            for name in sorted(denoise_merged_set):
                f.write(name + '\n')
        print(f"[INFO] Denoise合并列表 -> {denoise_merged_txt} (去重后 {len(denoise_merged_set)})")

if __name__ == '__main__':
    main()
