# train_save.py (完整版，已优化)
import os
import sys
import torch
import numpy as np

# 兼容旧NumPy版本
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models.plModel import LightningModel
from src.utils.IndexDataloader import DataModule


CONFIG_PATH = "confs/"


def load_config(config_name: str):
    """读取YAML配置"""
    path = os.path.join(CONFIG_PATH, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as file:
        config = OmegaConf.load(file)
    return config


def cli_main(config):
    """主训练流程"""
    pl.seed_everything(42)
    
    print("\n" + "="*70)
    print("WARM-RAIN EMULATOR TRAINING")
    print("="*70)
    print(f"Config:")
    print(f"  - data_dir:   {config.data_dir}")
    print(f"  - tot_len:    {config.tot_len}")
    print(f"  - sim_num:    {config.sim_num}")
    print(f"  - batch_size: {config.batch_size}")
    print(f"  - step_size:  {config.step_size}")
    print(f"  - max_epochs: {config.max_epochs}")
    print("="*70 + "\n")
    
    # -------- DataModule --------
    data_module = DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        tot_len=config.tot_len,
        sim_num=config.get('sim_num', 1),  # 默认1
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        single_sim_num=config.get('single_sim_num'),
        avg_dataloader=config.get('avg_dataloader', False),
        lo_norm=False,
    )
    data_module.setup()
    
    # -------- LightningModel --------
    act = config.act
    if isinstance(act, str):
        act = eval(act)
    
    pl_model = LightningModel(
        save_dir=config.save_dir,
        updates_mean=data_module.updates_mean,
        updates_std=data_module.updates_std,
        inputs_mean=data_module.inputs_mean,
        inputs_std=data_module.inputs_std,
        batch_size=config.batch_size,
        beta=config.beta,
        learning_rate=config.learning_rate,
        act=act,
        loss_func=config.get('loss_func'),
        depth=config.depth,
        p=config.p,
        n_layers=config.n_layers,
        ns=config.ns,
        loss_absolute=config.loss_absolute,
        mass_cons_updates=config.mass_cons_updates,
        mass_cons_moments=config.mass_cons_moments,
        hard_constraints_updates=config.hard_constraints_updates,
        hard_constraints_moments=config.hard_constraints_moments,
        multi_step=config.get('multi_step', False),
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        use_batch_norm=config.use_batch_norm,
        use_dropout=config.use_dropout,
        single_sim_num=config.get('single_sim_num'),
        avg_dataloader=config.get('avg_dataloader', False),
        pretrained_path=config.get('pretrained_dir'),
        lo_norm=False,
        ro_norm=config.get('ro_norm', False),
    )
    
    # -------- Callbacks & Trainer --------
    ckpt_dir = os.path.join(config.save_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="last_val_loss",  # ← 改为监控val loss
        mode="min",
        filename="wrn-epoch{epoch:02d}",
        auto_insert_metric_name=False
    )
    
    early_stop = EarlyStopping(
        monitor="last_val_loss",
        mode="min",
        patience=50,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=32,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    
    print("\n[INFO] Starting training...")
    print("="*70 + "\n")
    
    try:
        trainer.fit(pl_model, data_module)
        print("\n" + "="*70)
        print("[SUCCESS] Training completed!")
        print("="*70 + "\n")
    except Exception as e:
        print("\n" + "="*70)
        print(f"[ERROR] Training failed: {e}")
        print("="*70 + "\n")
        raise
    
    return data_module, pl_model, trainer


if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "mini_test"  # ← 默认用mini配置
    
    if not config_name.endswith((".yaml", ".yml")):
        config_name += ".yaml"
    
    print(f"\n[INFO] Loading config: {config_name}")
    
    try:
        config = load_config(config_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"Available configs in {CONFIG_PATH}:")
        for f in os.listdir(CONFIG_PATH):
            if f.endswith(('.yaml', '.yml')):
                print(f"  - {f}")
        sys.exit(1)
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 运行训练
    cli_main(config)