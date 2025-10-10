# view_mini_slice.py
import os
import numpy as np

DATA_DIR = r"E:/warm rain/warm-rain-emulator/data_mini"  # change if needed
paths = {
    "inputs":     os.path.join(DATA_DIR, "inputs_all.npy"),
    "outputs":    os.path.join(DATA_DIR, "outputs_all.npy"),
    "tendencies": os.path.join(DATA_DIR, "tendencies.npy"),
}

# Slice ranges for (time, IC, sims, feature). Right bounds are inclusive.
t0, t1 = 0, 2
ic0, ic1 = 0, 2
s0, s1 = 0, 1
f0, f1 = 0, 3

np.set_printoptions(precision=5, suppress=True)

for name, p in paths.items():
    arr = np.load(p, mmap_mode="r")  # memory-mapped read (no full load)
    print(f"\n[{name}] file:", p)
    print("full shape:", arr.shape, "(time, IC, sims, feature)")

    sl = (slice(t0, t1+1), slice(ic0, ic1+1), slice(s0, s1+1), slice(f0, f1+1))
    a = arr[sl]
    print("slice shape:", a.shape)

    # Show one tiny feature vector at (time=0, IC=0, sims=0)
    vec = arr[0, 0, 0, f0:f1+1]
    print("[one feature vector] time=0, IC=0, sims=0 →", vec)
