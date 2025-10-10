# view_raw_slice.py
import os
import numpy as np

# Use test first (smaller); switch to train if needed
NPZ_PATH = r"E:/warm rain/warm-rain-emulator/data/test_arr.npz"

# Slice ranges for (time, feature, IC, sims). Right bounds are inclusive.
t0, t1 = 0, 2     # time indices [0..2]
f0, f1 = 0, 3     # feature indices [0..3]
ic0, ic1 = 0, 2   # IC/level indices [0..2]
s0, s1 = 0, 1     # sims indices [0..1]

with np.load(NPZ_PATH) as z:
    data = z["data"]  # ndarray
    mask = z["mask"]  # boolean mask
    print("[FILE]", NPZ_PATH)
    print("full shape:", data.shape, "(time, feature, IC, sims)")

    # Build slice (inclusive right bounds → add +1)
    sl = (slice(t0, t1+1), slice(f0, f1+1), slice(ic0, ic1+1), slice(s0, s1+1))
    a = data[sl]
    m = mask[sl]
    print("slice shape:", a.shape)

    # Reconstruct masked array to compute stats on valid values only
    ma = np.ma.MaskedArray(a, mask=m)
    print("valid count:", ma.count(),
          " min/mean/max:", ma.min(), ma.mean(), ma.max())

    # Show one tiny feature vector at (time=0, IC=0, sims=0)
    np.set_printoptions(precision=5, suppress=True)
    print("[one feature vector] time=0, IC=0, sims=0 →",
          ma[0, :, 0, 0])
