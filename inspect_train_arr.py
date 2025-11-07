"""
检查 train/train_arr.npz 的完整结构
"""
import numpy as np
import os

print("="*70)
print("检查 train_arr.npz 结构")
print("="*70)

# 检查文件是否存在
train_path = 'data/train_arr.npz'
if not os.path.exists(train_path):
    print(f"\n❌ 文件不存在: {train_path}")
    print("\n可能的位置:")
    for root, dirs, files in os.walk('train'):
        for f in files:
            if f.endswith('.npz'):
                print(f"  - {os.path.join(root, f)}")
    exit(1)

# 加载文件
print(f"\n✓ 找到文件: {train_path}")
data = np.load(train_path, allow_pickle=True)

# 显示所有keys
print(f"\n1. NPZ文件中的所有keys:")
print(f"   {list(data.keys())}")

# 检查每个key的内容
print(f"\n2. 各个key的详细信息:")
print("-"*70)

for key in data.keys():
    arr = data[key]
    print(f"\n【{key}】")
    print(f"  Type:     {type(arr)}")
    
    if isinstance(arr, np.ndarray):
        print(f"  Shape:    {arr.shape}")
        print(f"  Dtype:    {arr.dtype}")
        
        # 如果是数值数组，显示统计
        if arr.dtype in [np.float32, np.float64, np.float16]:
            print(f"  Min:      {arr.min():.6e}")
            print(f"  Max:      {arr.max():.6e}")
            print(f"  Mean:     {arr.mean():.6e}")
            print(f"  N_unique: {len(np.unique(arr))}")
            
            # 显示一小部分数据
            if arr.size <= 100:
                print(f"  Data:     {arr}")
            else:
                flat = arr.flatten()
                print(f"  Sample (前10个): {flat[:10]}")
                
        elif arr.dtype == object:
            print(f"  ⚠️  这是object类型，可能包含字符串或复杂对象")
            if arr.size > 0:
                print(f"  First element: {arr.flat[0]}")
    else:
        print(f"  Value:    {arr}")

# 特别检查是否有 'data' 或 'mask' key (MaskedArray格式)
print(f"\n3. 检查是否是MaskedArray格式:")
if 'data' in data.keys() and 'mask' in data.keys():
    print("  ✓ 这是MaskedArray格式")
    masked_arr = np.ma.MaskedArray(data=data['data'], mask=data['mask'])
    print(f"  Combined shape: {masked_arr.shape}")
    print(f"  Masked elements: {masked_arr.mask.sum()} / {masked_arr.size}")
else:
    print("  这不是标准MaskedArray格式")

print("\n" + "="*70)