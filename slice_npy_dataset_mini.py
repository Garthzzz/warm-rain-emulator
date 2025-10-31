# slice_npy_dataset_mini.py
"""
为 .npy 格式的warm-rain数据集创建超小切片
原始数据:
- inputs_all.npy:  [Time, IC, Repeats, 9]
- outputs_all.npy: [Time, IC, Repeats, 4]  
- tendencies.npy:  [Time, IC, Repeats, 4]
"""
import argparse
import os
import numpy as np


def inspect_npy(path):
    """检查NPY文件"""
    print(f"\n{'='*70}")
    print(f"[INFO] Inspecting: {path}")
    print(f"{'='*70}")
    
    try:
        arr = np.load(path, allow_pickle=True)
        size_mb = arr.nbytes / (1024**2)
        
        print(f"  Shape:      {arr.shape}")
        print(f"  Dtype:      {arr.dtype}")
        print(f"  Size:       {size_mb:.2f} MB")
        print(f"  Has mask:   {isinstance(arr, np.ma.MaskedArray)}")
        
        if len(arr.shape) >= 2:
            print(f"\n  Dimensions:")
            print(f"    [0] Time:    {arr.shape[0]}")
            print(f"    [1] IC:      {arr.shape[1]}")
            if len(arr.shape) >= 3:
                print(f"    [2] Repeats: {arr.shape[2]}")
            if len(arr.shape) >= 4:
                print(f"    [3] Features:{arr.shape[3]}")
        
        # 显示数据范围
        if not isinstance(arr, np.ma.MaskedArray):
            print(f"\n  Data range: [{arr.min():.6f}, {arr.max():.6f}]")
        else:
            valid_data = arr.compressed()
            if len(valid_data) > 0:
                print(f"\n  Valid data range: [{valid_data.min():.6f}, {valid_data.max():.6f}]")
                print(f"  Masked ratio: {arr.mask.sum() / arr.size * 100:.2f}%")
        
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
    
    print(f"{'='*70}\n")


def slice_npy_ultra_mini(
    src_inputs,
    src_outputs, 
    src_tendencies,
    dst_dir,
    n_ic=5,
    n_time=50,
    n_repeats=1,
    seed=42
):
    """
    从 .npy 文件创建超小数据集
    
    参数:
        src_inputs:     inputs_all.npy 路径
        src_outputs:    outputs_all.npy 路径
        src_tendencies: tendencies.npy 路径
        dst_dir:        输出目录
        n_ic:           保留的初始条件数
        n_time:         保留的时间步数
        n_repeats:      保留的重复次数
    """
    os.makedirs(dst_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    print(f"\n{'='*70}")
    print(f"Creating ULTRA-MINI dataset")
    print(f"  Target: {n_ic} ICs × {n_time} timesteps × {n_repeats} repeats")
    print(f"{'='*70}\n")
    
    # 加载三个文件
    files = {
        'inputs': src_inputs,
        'outputs': src_outputs,
        'tendencies': src_tendencies
    }
    
    data = {}
    for key, path in files.items():
        print(f"[1/3] Loading {key} from {path} ...")
        try:
            arr = np.load(path, allow_pickle=True)
            data[key] = arr
            print(f"      ✓ Shape: {arr.shape}, Type: {type(arr).__name__}")
        except Exception as e:
            print(f"      ✗ ERROR: {e}")
            return
    
    # 检查形状一致性
    print(f"\n[2/3] Checking data consistency ...")
    shapes = [data[k].shape for k in ['inputs', 'outputs', 'tendencies']]
    
    # 前3维应该一致 [Time, IC, Repeats]
    if not all(s[:3] == shapes[0][:3] for s in shapes):
        print(f"      ✗ ERROR: Shape mismatch!")
        for k, s in zip(files.keys(), shapes):
            print(f"        {k}: {s}")
        return
    
    orig_shape = shapes[0][:3]
    print(f"      ✓ Original shape: Time={orig_shape[0]}, IC={orig_shape[1]}, Repeats={orig_shape[2]}")
    
    # 确定切片参数
    max_time = min(n_time, orig_shape[0])
    max_ic = min(n_ic, orig_shape[1])
    max_repeats = min(n_repeats, orig_shape[2])
    
    # 随机选择IC (但保持有序)
    ic_idx = rng.choice(orig_shape[1], size=max_ic, replace=False)
    ic_idx.sort()
    
    # 随机选择重复实验
    repeat_idx = rng.choice(orig_shape[2], size=max_repeats, replace=False)
    repeat_idx.sort()
    
    print(f"\n[3/3] Slicing data ...")
    print(f"      Time:    [0:{max_time}]")
    print(f"      IC:      {ic_idx}")
    print(f"      Repeats: {repeat_idx}")
    
    # 切片
    sliced = {}
    for key in ['inputs', 'outputs', 'tendencies']:
        arr = data[key]
        
        # [Time, IC, Repeats, Features]
        if arr.ndim == 4:
            sliced[key] = arr[:max_time, ic_idx, :, :][:, :, repeat_idx, :]
        # [Time, IC, Repeats] (不太可能,但以防万一)
        elif arr.ndim == 3:
            sliced[key] = arr[:max_time, ic_idx, :][:, :, repeat_idx]
        else:
            print(f"      [WARN] Unexpected ndim={arr.ndim} for {key}")
            sliced[key] = arr
        
        print(f"      ✓ {key:12s}: {arr.shape} -> {sliced[key].shape}")
    
    # 保存
    print(f"\n[4/4] Saving to {dst_dir} ...")
    for key in ['inputs', 'outputs', 'tendencies']:
        dst_path = os.path.join(dst_dir, f"{key}_all.npy")
        np.save(dst_path, sliced[key])
        print(f"      ✓ {dst_path}")
    
    # 保存元信息
    meta = {
        'original_shape': orig_shape,
        'selected_ic_idx': ic_idx,
        'selected_repeat_idx': repeat_idx,
        'n_time': max_time,
        'source_files': files
    }
    meta_path = os.path.join(dst_dir, '_metadata.npz')
    np.savez(meta_path, **meta)
    print(f"      ✓ {meta_path}")
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Mini dataset created!")
    print(f"{'='*70}\n")
    
    # 验证
    print("Verification:")
    for key in ['inputs', 'outputs', 'tendencies']:
        inspect_npy(os.path.join(dst_dir, f"{key}_all.npy"))


def main():
    ap = argparse.ArgumentParser(
        description="Create ULTRA-MINI dataset from .npy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 检查原始文件
  python slice_npy_dataset_mini.py --inspect data/inputs_all.npy
  
  # 创建超小数据集 (5 ICs, 50 timesteps, 1 repeat)
  python slice_npy_dataset_mini.py --data-dir data --output-dir data_mini --n-ic 5 --n-time 50 --n-repeats 1
  
  # 创建稍大数据集
  python slice_npy_dataset_mini.py --data-dir data --output-dir data_mini --n-ic 20 --n-time 100 --n-repeats 3
        """
    )
    
    ap.add_argument("--data-dir", type=str, default="data",
                    help="Directory containing .npy files")
    ap.add_argument("--output-dir", type=str, default="data_mini",
                    help="Output directory")
    ap.add_argument("--n-ic", type=int, default=5,
                    help="Number of ICs to keep (default: 5)")
    ap.add_argument("--n-time", type=int, default=50,
                    help="Number of timesteps to keep (default: 50)")
    ap.add_argument("--n-repeats", type=int, default=1,
                    help="Number of repeats to keep (default: 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", type=str, default=None,
                    help="Only inspect a .npy file")
    
    args = ap.parse_args()
    
    # 检查模式
    if args.inspect:
        inspect_npy(args.inspect)
        return
    
    # 检查输入文件
    inputs_path = os.path.join(args.data_dir, "inputs_all.npy")
    outputs_path = os.path.join(args.data_dir, "outputs_all.npy")
    tendencies_path = os.path.join(args.data_dir, "tendencies.npy")
    
    missing_files = []
    for path in [inputs_path, outputs_path, tendencies_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print(f"\n[ERROR] Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease check --data-dir argument (current: {args.data_dir})")
        return
    
    # 执行切片
    slice_npy_ultra_mini(
        inputs_path,
        outputs_path,
        tendencies_path,
        args.output_dir,
        n_ic=args.n_ic,
        n_time=args.n_time,
        n_repeats=args.n_repeats,
        seed=args.seed
    )


if __name__ == "__main__":
    main()