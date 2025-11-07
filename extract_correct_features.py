"""
Extract correct features from inputs_all.npy using verified indices
"""
import numpy as np
import os

print("="*70)
print("EXTRACTING DATA WITH CORRECT FEATURE INDICES")
print("="*70)

# Load raw data
raw = np.load('data/inputs_all.npy')
print(f"\nRaw data shape: {raw.shape}")
print(f"Dimensions: (Time={raw.shape[0]}, Ensemble={raw.shape[1]}, ICs={raw.shape[2]}, Features={raw.shape[3]})")

# 🔑 Correct feature indices (verified from colleague's code)
FEATURE_INDICES = {
    'Lc': 1,   # Cloud water mixing ratio
    'Nc': 2,   # Cloud droplet number density
    'Lr': 3,   # Rain water mixing ratio
    'Nr': 4,   # Rain drop number density
    'L0': 14,  # Initial total liquid water
    'r0': 15,  # Separation radius
    'nu': 16,  # Shape parameter
}

print(f"\nFeature index mapping:")
for name, idx in FEATURE_INDICES.items():
    print(f"  {name:4s} → Index {idx}")

# Extract 7 direct features
print(f"\nExtracting 7 direct features...")
extracted_7 = raw[:, :, :, [
    FEATURE_INDICES['Lc'],
    FEATURE_INDICES['Nc'],
    FEATURE_INDICES['Lr'],
    FEATURE_INDICES['Nr'],
    FEATURE_INDICES['L0'],
    FEATURE_INDICES['r0'],
    FEATURE_INDICES['nu'],
]]  # (3599, 18, 719, 7)

print(f"  Extracted shape: {extracted_7.shape}")

# Verify data ranges
print(f"\nVerifying extracted data:")
feature_names = ['Lc', 'Nc', 'Lr', 'Nr', 'L0', 'r0', 'nu']
for i, name in enumerate(feature_names):
    col = extracted_7[:, :, :, i].flatten()
    print(f"  {name:4s}: Min={col.min():.6e}, Max={col.max():.6e}, Mean={col.mean():.6e}")

# Compute derived quantities: tau and xc
print(f"\nComputing derived quantities tau and xc...")

Lc = extracted_7[:, :, :, 0]
Nc = extracted_7[:, :, :, 1]
Lr = extracted_7[:, :, :, 2]

tau = Lr / (Lr + Lc + 1e-20)  # Avoid division by zero
xc = Lc / (Nc + 1e-20)

print(f"  tau: Min={tau.min():.6e}, Max={tau.max():.6e}, Mean={tau.mean():.6e}")
print(f"  xc:  Min={xc.min():.6e}, Max={xc.max():.6e}, Mean={xc.mean():.6e}")

# Combine into 9 features
print(f"\nCombining into 9 features...")
inputs_9 = np.stack([
    Lc,                          # 0
    Nc,                          # 1
    Lr,                          # 2
    extracted_7[:, :, :, 3],     # Nr - 3
    tau,                         # 4
    xc,                          # 5
    extracted_7[:, :, :, 5],     # r0 - 6
    extracted_7[:, :, :, 6],     # nu - 7
    extracted_7[:, :, :, 4],     # L0 - 8
], axis=3)

print(f"  Final shape: {inputs_9.shape}")

# Transpose dimensions: (Time, Ensemble, ICs, Features) → (Time, ICs, Ensemble, Features)
print(f"\nTransposing dimensions...")
inputs_9_transposed = np.transpose(inputs_9, (0, 2, 1, 3))
print(f"  After transpose: {inputs_9_transposed.shape} → (Time, ICs, Ensemble, Features)")

# Slice to mini dataset
print(f"\nSlicing to mini dataset...")
n_time = 50
n_ic = 5
n_repeat = 1

# Randomly select ICs and repeats
rng = np.random.default_rng(42)
ic_idx = rng.choice(inputs_9_transposed.shape[1], size=n_ic, replace=False)
repeat_idx = rng.choice(inputs_9_transposed.shape[2], size=n_repeat, replace=False)

ic_idx.sort()
repeat_idx.sort()

inputs_mini = inputs_9_transposed[:n_time, ic_idx, :, :][:, :, repeat_idx, :]
print(f"  Mini data shape: {inputs_mini.shape}")

# Process outputs and tendencies similarly
print(f"\nProcessing outputs and tendencies...")

# Assuming outputs and tendencies have same structure
outputs_raw = np.load('data/outputs_all.npy')
tend_raw = np.load('data/tendencies.npy')

outputs_4 = outputs_raw[:, :, :, [
    FEATURE_INDICES['Lc'],
    FEATURE_INDICES['Nc'],
    FEATURE_INDICES['Lr'],
    FEATURE_INDICES['Nr'],
]]

tend_4 = tend_raw[:, :, :, [
    FEATURE_INDICES['Lc'],
    FEATURE_INDICES['Nc'],
    FEATURE_INDICES['Lr'],
    FEATURE_INDICES['Nr'],
]]

# Transpose
outputs_4 = np.transpose(outputs_4, (0, 2, 1, 3))
tend_4 = np.transpose(tend_4, (0, 2, 1, 3))

# Slice
outputs_mini = outputs_4[:n_time, ic_idx, :, :][:, :, repeat_idx, :]
tend_mini = tend_4[:n_time, ic_idx, :, :][:, :, repeat_idx, :]

print(f"  Outputs shape: {outputs_mini.shape}")
print(f"  Tend shape: {tend_mini.shape}")

# Save
print(f"\nSaving to data_mini_correct/...")
os.makedirs('data_mini_correct', exist_ok=True)

np.save('data_mini_correct/inputs.npy', inputs_mini)
np.save('data_mini_correct/outputs.npy', outputs_mini)
np.save('data_mini_correct/tendencies.npy', tend_mini)

np.savez('data_mini_correct/_metadata.npz',
         ic_idx=ic_idx,
         repeat_idx=repeat_idx,
         feature_indices=FEATURE_INDICES)

print("  ✓ Saved!")

# Final verification
print("\n" + "="*70)
print("FINAL VERIFICATION")
print("="*70)

for name in ['inputs', 'outputs', 'tendencies']:
    arr = np.load(f'data_mini_correct/{name}.npy')
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Range: [{arr.min():.6e}, {arr.max():.6e}]")
    print(f"  Mean:  {arr.mean():.6e}")
    
    # Check for duplicate values
    n_unique = len(np.unique(arr))
    if n_unique < 100:
        print(f"  ⚠️  Only {n_unique} unique values")
    else:
        print(f"  ✓ {n_unique} unique values")

# Physical relationship verification
print(f"\nPhysical relationship verification (first 3 timesteps):")
for t in range(3):
    ic = 0
    Lc = inputs_mini[t, ic, 0, 0]
    Nc = inputs_mini[t, ic, 0, 1]
    Lr = inputs_mini[t, ic, 0, 2]
    Nr = inputs_mini[t, ic, 0, 3]
    tau = inputs_mini[t, ic, 0, 4]
    xc = inputs_mini[t, ic, 0, 5]
    L0 = inputs_mini[t, ic, 0, 8]
    
    tau_expected = Lr / (Lr + Lc + 1e-20)
    xc_expected = Lc / (Nc + 1e-20)
    L0_expected = Lc + Lr
    
    print(f"\nTimestep {t}:")
    print(f"  Lc={Lc:.6e}, Nc={Nc:.6e}, Lr={Lr:.6e}, Nr={Nr:.6e}")
    print(f"  tau={tau:.6f}, expected={tau_expected:.6f}, error={abs(tau-tau_expected)/tau_expected*100:.2f}%")
    print(f"  xc={xc:.6e}, expected={xc_expected:.6e}, error={abs(xc-xc_expected)/xc_expected*100:.2f}%")
    print(f"  L0={L0:.6e}, expected={L0_expected:.6e}, error={abs(L0-L0_expected)/L0_expected*100:.2f}%")

print("\n" + "="*70)
print("DATA EXTRACTION COMPLETE!")
print("Replace data_mini/ with data_mini_correct/")
print("="*70)