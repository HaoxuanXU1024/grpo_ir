import os
import re
from typing import Dict, List, Tuple, Optional

ROOT = "/data2/haoxuan/AdaIR/data/test/derain/Rain100L"
IN_DIR = os.path.join(ROOT, "input")
TG_DIR = os.path.join(ROOT, "target")

# 只处理 png（若你有其他扩展名，可扩展此元组）
VALID_EXTS = (".png", ".PNG")

# 设置为 True 可先查看计划的重命名，不真正执行
DRY_RUN = False

def list_images(d: str) -> List[str]:
    return sorted([
        os.path.join(d, f) for f in os.listdir(d)
        if os.path.isfile(os.path.join(d, f)) and f.endswith(VALID_EXTS)
    ])

def extract_id(fname: str) -> Optional[int]:
    # 从文件名中抽取第一个连续数字，例如 rain-001.png -> 1, norain-088.png -> 88
    m = re.search(r"(\d+)", os.path.basename(fname))
    if not m:
        return None
    return int(m.group(1))

def build_id_map(files: List[str]) -> Dict[int, str]:
    id_map: Dict[int, str] = {}
    for p in files:
        n = extract_id(p)
        if n is None:
            print(f"[WARN] 跳过未包含数字的文件: {p}")
            continue
        if n in id_map:
            print(f"[WARN] 数字ID重复，后者覆盖前者: id={n}, {id_map[n]} -> {p}")
        id_map[n] = p
    return id_map

def temp_name(path: str) -> str:
    d, b = os.path.split(path)
    name, ext = os.path.splitext(b)
    return os.path.join(d, f"__tmp__{name}{ext}")

def final_name(d: str, idx: int) -> str:
    return os.path.join(d, f"{idx}.png")

def rename_safe(src: str, dst: str):
    if src == dst:
        return
    if os.path.exists(dst):
        raise FileExistsError(f"目标已存在，避免覆盖: {dst}")
    if not DRY_RUN:
        os.rename(src, dst)

def main():
    in_files = list_images(IN_DIR)
    tg_files = list_images(TG_DIR)

    in_map = build_id_map(in_files)
    tg_map = build_id_map(tg_files)

    common_ids = sorted(set(in_map.keys()) & set(tg_map.keys()))
    missing_in = sorted(set(tg_map.keys()) - set(in_map.keys()))
    missing_tg = sorted(set(in_map.keys()) - set(tg_map.keys()))

    print(f"[INFO] input 文件数: {len(in_files)}, target 文件数: {len(tg_files)}")
    print(f"[INFO] 匹配成功的成对ID数: {len(common_ids)}")
    if missing_in:
        print(f"[WARN] 以下ID仅在 target 存在: {missing_in[:10]}{' ...' if len(missing_in)>10 else ''}")
    if missing_tg:
        print(f"[WARN] 以下ID仅在 input 存在: {missing_tg[:10]}{' ...' if len(missing_tg)>10 else ''}")

    if not common_ids:
        print("[ERROR] 没有可配对的样本，检查文件命名是否包含数字。")
        return

    # 阶段1：将计划重命名的文件先改为临时名，避免冲突
    temp_pairs: List[Tuple[str, str]] = []
    for n in common_ids:
        temp_in = temp_name(in_map[n])
        temp_tg = temp_name(tg_map[n])
        print(f"[STAGE1] {in_map[n]} -> {temp_in}")
        print(f"[STAGE1] {tg_map[n]} -> {temp_tg}")
        temp_pairs.append((temp_in, temp_tg))
        if not DRY_RUN:
            os.rename(in_map[n], temp_in)
            os.rename(tg_map[n], temp_tg)

    # 阶段2：按 1..N.png 进行最终命名
    for idx, (tmp_in, tmp_tg) in enumerate(temp_pairs, start=1):
        dst_in = final_name(IN_DIR, idx)
        dst_tg = final_name(TG_DIR, idx)
        print(f"[STAGE2] {tmp_in} -> {dst_in}")
        print(f"[STAGE2] {tmp_tg} -> {dst_tg}")
        rename_safe(tmp_in, dst_in)
        rename_safe(tmp_tg, dst_tg)

    print("[DONE] 重命名完成。建议手动抽查几对是否正确。")

if __name__ == "__main__":
    main()