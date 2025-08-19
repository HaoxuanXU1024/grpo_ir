import os
import csv
import argparse
from typing import List, Tuple


def read_csv(path: str) -> List[Tuple[str, float, float]]:
    rows: List[Tuple[str, float, float]] = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # 期望 header: ["image","psnr","ssim"]，若没有也容错
        for r in reader:
            if not r or len(r) < 3:
                continue
            name = r[0]
            try:
                psnr = float(r[1])
                ssim = float(r[2])
            except Exception:
                continue
            rows.append((name, psnr, ssim))
    return rows


def select_worst(rows: List[Tuple[str, float, float]], percent: float, min_count: int, max_count: int) -> List[str]:
    if not rows:
        return []
    n = len(rows)
    k = int(round(percent * n))
    k = max(min_count, k)
    k = min(max_count, k)
    k = min(k, n)
    # 按 PSNR 升序，再按 SSIM 升序
    rows_sorted = sorted(rows, key=lambda x: (x[1], x[2]))
    return [name for (name, _, _) in rows_sorted[:k]]


def main():
    parser = argparse.ArgumentParser(description="Select worst samples from CSVs without re-eval")
    parser.add_argument('--csv', nargs='+', required=True, help='CSV file paths (image,psnr,ssim)')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for worst lists')
    parser.add_argument('--percent', type=float, default=0.30, help='Bottom percentage per CSV (default 0.30)')
    parser.add_argument('--min_count', type=int, default=100, help='Lower bound per CSV (default 100)')
    parser.add_argument('--max_count', type=int, default=5000, help='Upper bound per CSV (default 5000)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combined: List[Tuple[str, str]] = []  # (tag, name)
    denoise_merged_set = set()
    for csv_path in args.csv:
        tag = os.path.splitext(os.path.basename(csv_path))[0]
        rows = read_csv(csv_path)
        worst = select_worst(rows, args.percent, args.min_count, args.max_count)
        out_txt = os.path.join(args.out_dir, f"{tag}_worst.txt")
        with open(out_txt, 'w') as f:
            for name in worst:
                f.write(name + '\n')
        combined.extend([(tag, w) for w in worst])
        print(f"[INFO] {tag}: total={len(rows)} select={len(worst)} -> {out_txt}")

        # 合并 denoise 的三个列表为一个去重清单
        if tag.startswith('train_denoise_sigma'):
            for name in worst:
                denoise_merged_set.add(name)

    # 额外输出一个合并清单，含任务标签，便于下游采样
    merged_txt = os.path.join(args.out_dir, 'combined_worst.txt')
    with open(merged_txt, 'w') as f:
        for tag, name in combined:
            f.write(f"{tag},{name}\n")
    print(f"[INFO] Combined list -> {merged_txt} (total {len(combined)})")

    # 输出 denoise 合并清单（若有提供 denoise 的 CSV）
    if denoise_merged_set:
        denoise_merged_txt = os.path.join(args.out_dir, 'train_denoise_worst_merged.txt')
        with open(denoise_merged_txt, 'w') as f:
            for name in sorted(denoise_merged_set):
                f.write(name + '\n')
        print(f"[INFO] Denoise merged list -> {denoise_merged_txt} (unique {len(denoise_merged_set)})")


if __name__ == '__main__':
    main()


