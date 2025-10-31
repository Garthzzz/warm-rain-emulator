"""
符号回归训练脚本
使用方式: python train_sr.py sr_config.yaml
"""
import os
import sys
import torch
import numpy as np

# NumPy兼容性
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.models.plModel_SR import LightningModelSR
from src.utils.IndexDataloader import DataModule


CONFIG_PATH = "confs/"


def load_config(config_name: str):
    """读取配置文件"""
    path = os.path.join(CONFIG_PATH, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = OmegaConf.load(f)
    return config


def main(config):
    """主函数"""
    pl.seed_everything(42)
    
    print("\n" + "="*70)
    print("SYMBOLIC REGRESSION FOR WARM-RAIN MICROPHYSICS")
    print("="*70)
    print("Config:")
    print(f"  data_dir:      {config.data_dir}")
    print(f"  tot_len:       {config.tot_len}")
    print(f"  sim_num:       {config.sim_num}")
    print(f"  batch_size:    {config.batch_size}")
    print(f"  SR iterations: {config.sr_config.niterations}")
    print("="*70 + "\n")
    
    # 1. 加载数据
    print("[1/3] Loading data...")
    data_module = DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        tot_len=config.tot_len,
        sim_num=config.get('sim_num', 1),
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        avg_dataloader=config.get('avg_dataloader', False),
        lo_norm=False,
    )
    data_module.setup()
    print("  ✓ Data loaded\n")
    
    # 2. 创建SR模型并训练
    print("[2/3] Creating and training SR model...")
    
    # 准备SR配置
    sr_config = OmegaConf.to_container(config.sr_config, resolve=True)
    
    # 创建模型 (会自动训练SR)
    sr_model = LightningModelSR(
        save_dir=config.save_dir,
        updates_mean=data_module.updates_mean,
        updates_std=data_module.updates_std,
        inputs_mean=data_module.inputs_mean,
        inputs_std=data_module.inputs_std,
        batch_size=config.batch_size,
        beta=config.beta,
        loss_func=config.get('loss_func'),
        depth=config.depth,
        mass_cons_updates=config.mass_cons_updates,
        mass_cons_moments=config.mass_cons_moments,
        hard_constraints_updates=config.hard_constraints_updates,
        hard_constraints_moments=config.hard_constraints_moments,
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        sr_config=sr_config,
        train_sr_now=True,
        data_module=data_module,
    )
    print("  ✓ SR model trained\n")
    
    # 3. 评估模型 (可选)
    print("[3/3] Evaluating SR model...")
    
    # 创建Lightning Trainer用于评估
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=1,  # 只评估，不训练
        enable_progress_bar=True,
        logger=False,
    )
    
    # 在验证集上评估
    val_results = trainer.validate(sr_model, data_module)
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    for key, val in val_results[0].items():
        print(f"  {key}: {val:.6f}")
    print("="*70 + "\n")
    
    # 打印发现的公式
    print("\n" + "="*70)
    print("DISCOVERED EQUATIONS")
    print("="*70)
    equations = sr_model.model.get_equations()
    for name, eq_info in equations.items():
        print(f"\n{name}:")
        if 'error' in eq_info:
            print(f"  Error: {eq_info['error']}")
        else:
            print(f"  Equation: {eq_info['equation']}")
            print(f"  Complexity: {eq_info['complexity']}")
            print(f"  Loss: {eq_info['loss']:.6f}")
    print("\n" + "="*70 + "\n")
    
    # 保存位置
    equations_file = os.path.join(config.save_dir, "sr_models", "discovered_equations.txt")
    print(f"[INFO] Full equations saved to: {equations_file}")
    print(f"[INFO] Models saved to: {os.path.join(config.save_dir, 'sr_models')}")
    
    print("\n[SUCCESS] SR training complete!")
    
    return sr_model, data_module


if __name__ == "__main__":
    # 解析命令行
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "sr_mini_test"
    
    if not config_name.endswith((".yaml", ".yml")):
        config_name += ".yaml"
    
    print(f"[INFO] Loading config: {config_name}")
    
    try:
        config = load_config(config_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 运行
    main(config)