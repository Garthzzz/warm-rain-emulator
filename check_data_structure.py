# check_data_structure.py
import numpy as np

inputs = np.load("data/inputs_all.npy")
print("Shape:", inputs.shape)
print("\nFirst sample (time=0, repeat=0, ic=0):")
print(inputs[0, 0, 0, :])
print("\nFeature statistics:")
for i in range(min(20, inputs.shape[3])):  # 打印前20列
    print(f"  Feature {i:2d}: min={inputs[0,0,0,i]:.6f}, max={inputs[:,:,:,i].max():.6f}")