"""
use_sample_as_template.py
"""
import numpy as np
import os


def main():
    print("="*70)
    print("CREATING MINI DATASET FROM SAMPLE TEMPLATE")
    print("="*70)
    
    # 1. 加载sample作为参考
    print("\n[1/4] Loading sample_arr.npz as reference...")
    sample = np.load('data/sample_arr.npz')
    sample_data = np.ma.MaskedArray(**sample)
    
    print(f"  Sample shape: {sample_data.shape}")
    print(f"  Sample has {sample_data.shape[-1]} features")
    
    # 2. 加载原始数据
    print("\n[2/4] Loading raw data...")
    inputs_raw = np.load('data/inputs_all.npy')
    outputs_raw = np.load('data/outputs_all.npy')
    tend_raw = np.load('data/tendencies.npy')
    
    print(f"  Raw shapes: {inputs_raw.shape}")
    
    # 3. 提取前N列（根据sample的特征数）
    n_features_inputs = 9  # inputs需要9个特征
    n_features_outputs = 4  # outputs和tendencies需要4个特征
    
    print(f"\n[3/4] Extracting first {n_features_inputs} columns for inputs...")
    print(f"         Extracting first {n_features_outputs} columns for outputs/tend...")
    
    # 从100列中提取前N列
    inputs_extracted = inputs_raw[:, :, :, :n_features_inputs]
    outputs_extracted = outputs_raw[:, :, :, :n_features_outputs]
    tend_extracted = tend_raw[:, :, :, :n_features_outputs]
    
    print(f"  After extraction:")
    print(f"    inputs:  {inputs_extracted.shape}")
    print(f"    outputs: {outputs_extracted.shape}")
    print(f"    tend:    {tend_extracted.shape}")
    
    # 4. 转置维度: (Time, Repeats, ICs, Features) → (Time, ICs, Repeats, Features)
    print(f"\n[4/4] Transposing to match expected format...")
    inputs_t = np.transpose(inputs_extracted, (0, 2, 1, 3))
    outputs_t = np.transpose(outputs_extracted, (0, 2, 1, 3))
    tend_t = np.transpose(tend_extracted, (0, 2, 1, 3))
    
    print(f"  After transpose:")
    print(f"    inputs:  {inputs_t.shape}")
    print(f"    outputs: {outputs_t.shape}")
    print(f"    tend:    {tend_t.shape}")
    
    # 5. 切片到mini size
    print(f"\n[5/6] Slicing to mini dataset (50×5×1)...")
    
    n_time = 50
    n_ic = 5
    n_repeats = 1
    
    rng = np.random.default_rng(42)
    
    # 随机选择ICs
    ic_idx = rng.choice(inputs_t.shape[1], size=min(n_ic, inputs_t.shape[1]), replace=False)
    ic_idx.sort()
    
    # 随机选择repeats
    repeat_idx = rng.choice(inputs_t.shape[2], size=min(n_repeats, inputs_t.shape[2]), replace=False)
    repeat_idx.sort()
    
    # 执行切片
    max_time = min(n_time, inputs_t.shape[0])
    
    inputs_mini = inputs_t[:max_time, ic_idx, :, :][:, :, repeat_idx, :]
    outputs_mini = outputs_t[:max_time, ic_idx, :, :][:, :, repeat_idx, :]
    tend_mini = tend_t[:max_time, ic_idx, :, :][:, :, repeat_idx, :]
    
    print(f"  Final shapes:")
    print(f"    inputs:  {inputs_mini.shape}")
    print(f"    outputs: {outputs_mini.shape}")
    print(f"    tend:    {tend_mini.shape}")
    
    # 6. 保存
    print(f"\n[6/6] Saving to data_mini/...")
    os.makedirs('data_mini', exist_ok=True)
    
    np.save('data_mini/inputs.npy', inputs_mini)
    np.save('data_mini/outputs.npy', outputs_mini)
    np.save('data_mini/tendencies.npy', tend_mini)
    
    np.savez('data_mini/_metadata.npz',
             ic_idx=ic_idx,
             repeat_idx=repeat_idx)
    
    print("  ✓ Saved!")
    
    # 验证
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    for name in ["inputs", "outputs", "tendencies"]:
        arr = np.load(f'data_mini/{name}.npy')
        print(f"\n{name}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Sample (time=0, ic=0, repeat=0):")
        print(f"    {arr[0, 0, 0, :]}")
        print(f"  Data range: [{arr.min():.2e}, {arr.max():.2e}]")
        
        # 检查是否全是相同值
        unique_count = len(np.unique(arr))
        if unique_count <= 5:
            print(f"  ⚠️  WARNING: Only {unique_count} unique values!")
        else:
            print(f"  ✓ Has {unique_count} unique values")
    
    print("\n" + "="*70)
    print("[SUCCESS] Mini dataset created!")
    print("="*70)


if __name__ == "__main__":
    main()