# peek_zip.py
import os
import zipfile

# Change to train_arr.npz if needed
NPZ_PATH = r"E:/warm rain/warm-rain-emulator/data/test_arr.npz"

with zipfile.ZipFile(NPZ_PATH) as zf:
    print("NPZ file:", NPZ_PATH)
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f" - {name:20s} | uncompressed={info.file_size/1024/1024:.2f} MB | "
              f"compressed={info.compress_size/1024/1024:.2f} MB")
