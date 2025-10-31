"""
符号回归版本的LightningModel
关键修改：
1. 用SR模型替代NN
2. 离线训练SR (在__init__中)
3. PyTorch Lightning只用于推理和评估
"""
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

from src.helpers.normalizer import normalizer
from src.models.sr_model import SymbolicRegressionModel


class LightningModelSR(pl.LightningModule):
    """
    符号回归版本的Lightning模型
    主要用于评估和推理，不进行梯度优化
    """
    
    def __init__(
        self,
        updates_mean,
        updates_std,
        inputs_mean,
        inputs_std,
        save_dir="outputs/sr_run",
        batch_size=256,
        beta=0.35,
        learning_rate=2e-4,  # SR不需要，但保留接口兼容性
        loss_func=None,
        depth=9,
        mass_cons_updates=True,
        loss_absolute=True,
        mass_cons_moments=True,
        hard_constraints_updates=True,
        hard_constraints_moments=False,
        multi_step=False,
        step_size=1,
        moment_scheme=2,
        # SR特定参数
        sr_config=None,
        train_sr_now=True,  # 是否立即训练SR
        data_module=None,   # 传入DataModule用于训练
        **kwargs
    ):
        super().__init__()
        
        self.moment_scheme = moment_scheme
        self.out_features = moment_scheme * 2
        self.save_dir = save_dir
        self.lr = learning_rate
        self.loss_func = loss_func
        self.beta = beta
        self.batch_size = batch_size
        self.loss_absolute = loss_absolute
        
        self.hard_constraints_updates = hard_constraints_updates
        self.hard_constraints_moments = hard_constraints_moments
        self.mass_cons_updates = mass_cons_updates
        self.mass_cons_moments = mass_cons_moments
        
        self.multi_step = multi_step
        self.step_size = step_size
        
        self.save_hyperparameters(ignore=['data_module'])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 标准化参数
        self.updates_std = torch.from_numpy(updates_std).float().to(device)
        self.updates_mean = torch.from_numpy(updates_mean).float().to(device)
        self.inputs_mean = torch.from_numpy(inputs_mean).float().to(device)
        self.inputs_std = torch.from_numpy(inputs_std).float().to(device)
        
        # 创建SR模型
        sr_config = sr_config or {}
        self.model = SymbolicRegressionModel(
            out_features=self.out_features,
            depth=depth,
            save_dir=os.path.join(save_dir, "sr_models"),
            **sr_config
        )
        
        # 离线训练SR
        if train_sr_now:
            if data_module is None:
                raise ValueError("data_module is required for SR training")
            self._train_sr_offline(data_module)
    
    def _train_sr_offline(self, data_module):
        """
        在PyTorch Lightning训练之前，用所有训练数据训练SR
        """
        print("\n" + "="*70)
        print("COLLECTING TRAINING DATA FOR SR")
        print("="*70)
        
        X_all, y_all = [], []
        
        # 从训练集收集数据
        train_loader = data_module.train_dataloader()
        print(f"Collecting from {len(train_loader)} batches...")
        
        for batch_idx, batch in enumerate(train_loader):
            x, updates, _ = batch
            
            # x: [batch, 9, 1] -> [batch, 9]
            # updates: [batch, 4, step_size] -> [batch, 4] (取第一步)
            x_np = x.squeeze(-1).numpy()
            y_np = updates[:, :, 0].numpy()
            
            X_all.append(x_np)
            y_all.append(y_np)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
        
        X_all = np.vstack(X_all)
        y_all = np.vstack(y_all)
        
        print(f"\n[SR] Collected {X_all.shape[0]} training samples")
        print(f"[SR] Input shape: {X_all.shape}")
        print(f"[SR] Output shape: {y_all.shape}")
        
        # 训练SR模型
        self.model.fit(X_all, y_all)
        
        print("\n[SR] Training complete. Model ready for inference.")
    
    def forward(self):
        """前向传播"""
        # SR模型预测
        self.updates = self.model(self.x)
        
        # 应用物理约束和反标准化
        self.norm_obj = normalizer(
            self.updates,
            self.x,
            self.y,
            self.updates_mean,
            self.updates_std,
            self.inputs_mean,
            self.inputs_std,
            self.device,
            self.hard_constraints_updates,
            self.hard_constraints_moments,
        )
        
        (
            self.real_x,
            self.real_y,
            self.pred_moment,
            self.pred_moment_norm,
            self.lo,
            self.ro,
        ) = self.norm_obj.calc_preds()
        
        self.pred_moment, self.pred_moment_norm = self.norm_obj.set_constraints()
    
    def loss_function(self, updates, y, k=None):
        """计算损失"""
        if self.loss_func == "mse":
            criterion = torch.nn.MSELoss()
        elif self.loss_func == "mae":
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.SmoothL1Loss(reduction="mean", beta=self.beta)
        
        if self.loss_absolute:
            pred_loss = criterion(self.pred_moment_norm, y)
        else:
            pred_loss = criterion(updates, self.updates)
        
        return pred_loss
    
    def configure_optimizers(self):
        """
        SR不需要优化器（已经训练完毕）
        但PyTorch Lightning要求这个方法存在
        返回一个dummy optimizer
        """
        return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤 - SR模型已训练，这里只是走个形式
        实际上可以跳过，但为了兼容Lightning框架保留
        """
        self.x, updates, y = batch
        self.x = self.x.squeeze()
        
        loss = torch.tensor(0.0, device=self.device)
        
        for k in range(self.step_size):
            self.y = y[:, :, k].squeeze()
            self.forward()
            
            step_loss = self.loss_function(updates[:, :, k], y[:, :, k], k)
            loss = loss + step_loss
            
            self.log(f"Train_loss_{k+1}", step_loss)
        
        self.log("train_loss", loss)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """验证步骤 - 评估SR模型性能"""
        self.x, updates, y = batch
        self.x = self.x.squeeze()
        
        loss = torch.tensor(0.0, device=self.device)
        
        for k in range(self.step_size):
            self.y = y[:, :, k].squeeze()
            self.forward()
            
            step_loss = self.loss_function(updates[:, :, k], y[:, :, k], k)
            loss = loss + step_loss
        
        self.log("tot_val_loss", loss)
        self.log("last_val_loss", step_loss)
    
    def test_step(self, initial_moments):
        """测试步骤"""
        with torch.no_grad():
            preds = self.model(initial_moments.float())
        return preds