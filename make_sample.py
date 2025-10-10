# make_sample.py
import os
import numpy as np

SRC = r"E:/warm rain/warm-rain-emulator/data/test_arr.npz"  # or train_arr.npz
DST = r"E:/warm rain/warm-rain-emulator/data/sample_arr.npz"

# Small window for (time, feature, IC, sims)
t_win, ic_win, sims_win = 50, 50, 5
with np.load(SRC) as z:
    A, M = z['data'], z['mask']
    t, f, ic, n = A.shape
    sl = (slice(0, min(t_win, t)),
          slice(0, f),                     # keep all features
          slice(0, min(ic_win, ic)),
          slice(0, min(sims_win, n)))
    a_small = A[sl]
    m_small = M[sl]
    np.savez(DST, data=a_small, mask=m_small)
    print("saved:", DST, "| shape:", a_small.shape,
          "| ~%.1f MB" % (a_small.nbytes/1024/1024))
