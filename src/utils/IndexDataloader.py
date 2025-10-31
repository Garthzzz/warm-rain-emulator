# src/utils/IndexDataloader.py (完整修复版)
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import os


class my_dataset(Dataset):
    def __init__(self, inputdata, tend, outputs, index_arr, step_size, moment_scheme):
        # 处理MaskedArray和普通ndarray
        if isinstance(inputdata, np.ma.MaskedArray):
            self.inputdata = inputdata.data
            self.tend = tend.data
            self.outputs = outputs.data
        else:
            self.inputdata = inputdata
            self.tend = tend
            self.outputs = outputs
        
        self.index_arr = index_arr
        self.step_size = step_size
        self.moment_scheme = moment_scheme

    def __getitem__(self, index):
        i_time, i_ic, i_repeat = self.index_arr[index]

        tend_multistep = np.empty((self.moment_scheme * 2, self.step_size))
        outputs_multistep = np.empty((self.moment_scheme * 2, self.step_size))
        
        for i_step in range(self.step_size):
            tend_multistep[:, i_step] = self.tend[i_time + i_step, i_ic, i_repeat]
            outputs_multistep[:, i_step] = self.outputs[i_time + i_step, i_ic, i_repeat]

        return (
            torch.from_numpy(self.inputdata[i_time, i_ic, i_repeat]).view(-1, 1).float(),
            torch.from_numpy(tend_multistep).view(-1, self.step_size).float(),
            torch.from_numpy(outputs_multistep).view(-1, self.step_size).float(),
        )

    def __len__(self):
        return self.index_arr.shape[0]


def normalize_data(x, flag=None):
    """标准化: (x - mean) / std"""
    x_ = x.reshape(-1, x.shape[-1])
    m = x_.mean(axis=0)
    s = x_.std(axis=0)
    s[s == 0] = 1  # 避免除0
    return (x - m) / s, m, s


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="/gpfs/work/sharmas/mc-snow-data/",
        batch_size: int = 256,
        num_workers: int = 10,
        tot_len=719,
        sim_num=98,
        load_from_memory=True,
        moment_scheme=2,
        step_size=1,
        train_size=0.9,
        single_sim_num=None,
        avg_dataloader=False,
        lo_norm=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.moment_scheme = moment_scheme
        self.tot_len = tot_len
        self.sim_num = sim_num
        self.step_size = step_size
        self.load_simulations = load_from_memory
        self.train_size = train_size
        self.single_sim_num = single_sim_num
        self.avg_dataloader = avg_dataloader
        self.lo_norm = lo_norm
        
        if self.load_simulations:
            self._load_data()

    def _load_data(self):
        """加载数据，兼容.npz和.npy格式"""
        print(f"\n[DataModule] Loading data from {self.data_dir}")
        
        loaded = False
        
        # 尝试1: .npz with _all (MaskedArray格式)
        if not loaded:
            try:
                with np.load(os.path.join(self.data_dir, "inputs_all.npz")) as npz:
                    self.inputs_arr = np.ma.MaskedArray(**npz)
                with np.load(os.path.join(self.data_dir, "outputs_all.npz")) as npz:
                    self.outputs_arr = np.ma.MaskedArray(**npz)
                with np.load(os.path.join(self.data_dir, "tendencies.npz")) as npz:
                    self.tend_arr = np.ma.MaskedArray(**npz)
                
                print("  ✓ Loaded .npz files (MaskedArray)")
                self.is_masked = True
                loaded = True
            except (FileNotFoundError, KeyError):
                pass
        
        # 尝试2: .npy without _all (新mini数据集格式)
        if not loaded:
            try:
                self.inputs_arr = np.load(os.path.join(self.data_dir, "inputs.npy"))
                self.outputs_arr = np.load(os.path.join(self.data_dir, "outputs.npy"))
                self.tend_arr = np.load(os.path.join(self.data_dir, "tendencies.npy"))
                
                print("  ✓ Loaded .npy files (mini dataset format)")
                self.is_masked = False
                loaded = True
            except FileNotFoundError:
                pass
        
        # 尝试3: .npy with _all (旧完整数据集格式)
        if not loaded:
            try:
                self.inputs_arr = np.load(os.path.join(self.data_dir, "inputs_all.npy"))
                self.outputs_arr = np.load(os.path.join(self.data_dir, "outputs_all.npy"))
                self.tend_arr = np.load(os.path.join(self.data_dir, "tendencies.npy"))
                
                print("  ✓ Loaded .npy files (full dataset format)")
                self.is_masked = False
                loaded = True
            except FileNotFoundError:
                pass
        
        if not loaded:
            raise FileNotFoundError(
                f"\n[ERROR] Could not find data files in '{self.data_dir}'.\n"
                f"Tried the following:\n"
                f"  1. inputs_all.npz, outputs_all.npz, tendencies.npz\n"
                f"  2. inputs.npy, outputs.npy, tendencies.npy\n"
                f"  3. inputs_all.npy, outputs_all.npy, tendencies.npy\n"
            )
        
        # 打印数据形状 (调试用)
        print(f"  - inputs_arr:  {self.inputs_arr.shape}")
        print(f"  - outputs_arr: {self.outputs_arr.shape}")
        print(f"  - tend_arr:    {self.tend_arr.shape}")
        
        # 修正 L0 = Lc + Lr (索引-3是L0位置)
        sim_lo = self.inputs_arr[:, :, :, 0] + self.inputs_arr[:, :, :, 2]
        self.inputs_arr[:, :, :, -3] = sim_lo
        print("  ✓ Modified L0 = Lc + Lr")
        
        # 动态检测实际的IC数和repeat数
        self.actual_tot_len = self.inputs_arr.shape[1]
        self.actual_sim_num = self.inputs_arr.shape[2]
        
        if self.actual_tot_len != self.tot_len:
            print(f"  [WARN] Config tot_len={self.tot_len}, but data has {self.actual_tot_len} ICs")
            print(f"         Using actual value: {self.actual_tot_len}")
            self.tot_len = self.actual_tot_len
        
        if self.actual_sim_num != self.sim_num:
            print(f"  [WARN] Config sim_num={self.sim_num}, but data has {self.actual_sim_num} repeats")
            print(f"         Using actual value: {self.actual_sim_num}")
            self.sim_num = self.actual_sim_num

    def setup(self, stage=None):
        print("\n[DataModule] Setting up...")
        self.calc_norm()
        self.calc_index_array()
        self.test_train()
        print("[DataModule] Setup complete!\n")

    def _get_valid_timesteps(self, ic_idx):
        """
        获取某个IC的有效时间步数
        对MaskedArray使用compress_rows，对普通array直接计算
        """
        if self.is_masked:
            # MaskedArray: 使用compress_rows去除masked行
            valid_data = np.ma.compress_rows(self.tend_arr[:, ic_idx, 0, :])
            return valid_data.shape[0]
        else:
            # 普通ndarray: 检查是否有NaN或全0行
            data_slice = self.tend_arr[:, ic_idx, 0, :]
            
            # 方法1: 检查NaN
            valid_mask = ~np.isnan(data_slice).any(axis=1)
            
            # 方法2: 或者检查全0 (如果数据用0填充无效值)
            # valid_mask = ~(data_slice == 0).all(axis=1)
            
            return valid_mask.sum()

    def calc_index_array_size(self):
        """计算索引数组总长度"""
        l_in = 0
        for i in range(self.tot_len):
            valid_steps = self._get_valid_timesteps(i)
            l = max(0, valid_steps - self.step_size + 1)
            l_in += l
        
        print(f"  - Total valid samples: {l_in}")
        return l_in

    def calc_index_array(self):
        """创建索引数组: [time, ic, repeat]"""
        l_in = self.calc_index_array_size()
        
        # 修复: 使用int而不是np.int (NumPy 1.20+已弃用)
        self.indices_arr = np.empty((l_in * self.sim_num, 3), dtype=int)
        lo = 0

        sim_nums = np.arange(self.sim_num)
        
        for i in range(self.tot_len):
            valid_steps = self._get_valid_timesteps(i)
            l = max(0, valid_steps - self.step_size + 1)
            
            if l == 0:
                continue
            
            time_points = np.arange(l)
            unique_sim_num = np.full(shape=l, fill_value=i, dtype=int)
            new_arr = np.concatenate(
                (time_points.reshape(-1, 1), unique_sim_num.reshape(-1, 1)), axis=1
            )
            new_arr = np.vstack([new_arr] * self.sim_num)
            
            sim_num_axis = np.repeat(sim_nums, l, axis=0)
            
            indices_sim = np.concatenate((new_arr, sim_num_axis.reshape(-1, 1)), axis=1)
            self.indices_arr[lo : lo + indices_sim.shape[0], :] = indices_sim
            lo += indices_sim.shape[0]
        
        print(f"  - Index array shape: {self.indices_arr.shape}")

    def calc_norm(self):
        """标准化数据"""
        print("  - Normalizing data...")
        
        if self.lo_norm:
            lo = self.inputs_arr[:, :, :, -3].reshape(
                self.inputs_arr.shape[0],
                self.inputs_arr.shape[1],
                self.inputs_arr.shape[2],
                1,
            )
            self.inputs_arr[:, :, :, :4] = self.inputs_arr[:, :, :, :4] / lo
            self.outputs_arr[:, :, :, :4] = self.outputs_arr[:, :, :, :4] / lo

        self.inputs_arr, self.inputs_mean, self.inputs_std = normalize_data(self.inputs_arr)
        self.outputs_arr, self.outputs_mean, self.outputs_std = normalize_data(self.outputs_arr)
        self.tend_arr, self.updates_mean, self.updates_std = normalize_data(self.tend_arr)
        
        print(f"    ✓ inputs normalized (mean: {self.inputs_mean[:4]})")
        print(f"    ✓ outputs normalized")
        print(f"    ✓ tendencies normalized")

        if self.avg_dataloader:
            print("  - Averaging over repeats...")
            self.inputs_arr = np.expand_dims(np.mean(self.inputs_arr, axis=2), axis=2)
            self.outputs_arr = np.expand_dims(np.mean(self.outputs_arr, axis=2), axis=2)
            self.tend_arr = np.expand_dims(np.mean(self.tend_arr, axis=2), axis=2)
            self.sim_num = 1

    def test_train(self):
        """划分训练/验证集"""
        self.dataset = my_dataset(
            self.inputs_arr,
            self.tend_arr,
            self.outputs_arr,
            self.indices_arr,
            self.step_size,
            self.moment_scheme,
        )

        train_size = int(self.train_size * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        print(f"  - Train samples: {train_size}")
        print(f"  - Val samples:   {val_size}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )