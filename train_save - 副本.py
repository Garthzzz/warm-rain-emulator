import os
import sys
import torch
import numpy as np  # ← 新增

# --- 兼容旧写法（临时补丁）---
if not hasattr(np, "int"):
    np.int = int          # noqa
if not hasattr(np, "bool"):
    np.bool = bool        # noqa
if not hasattr(np, "float"):
    np.float = float      # noqa
# --------------------------------

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models.plModel import LightningModel
from src.utils.IndexDataloader import DataModule



CONFIG_PATH = "confs/"


def load_config(config_name: str):
    """读取 YAML 配置（强制 UTF-8，避免 Windows gbk 编码问题）"""
    path = os.path.join(CONFIG_PATH, config_name)
    with open(path, "r", encoding="utf-8") as file:
        config = OmegaConf.load(file)
    return config


def cli_main(config):
    """构建 DataModule / LightningModel / Trainer 并开训"""
    pl.seed_everything(42)
    N_EPOCHS = config.max_epochs

    # -------- DataModule --------
    data_module = DataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        tot_len=config.tot_len,
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        single_sim_num=config.single_sim_num,
        avg_dataloader=config.avg_dataloader,
        lo_norm=False,
    )
    data_module.setup()  # 为后续 mean/std 做准备

    # -------- LightningModel --------
    act = config.act
    if isinstance(act, str):
        act = eval(act)  # 允许 "torch.nn.ReLU()"

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
        loss_func=config.loss_func,
        depth=config.depth,
        p=config.p,
        n_layers=config.n_layers,
        ns=config.ns,
        loss_absolute=config.loss_absolute,
        mass_cons_updates=config.mass_cons_updates,
        mass_cons_moments=config.mass_cons_moments,
        hard_constraints_updates=config.hard_constraints_updates,
        hard_constraints_moments=config.hard_constraints_moments,
        multi_step=config.multi_step,
        step_size=config.step_size,
        moment_scheme=config.moment_scheme,
        use_batch_norm=config.use_batch_norm,
        use_dropout=config.use_dropout,
        single_sim_num=config.single_sim_num,
        avg_dataloader=config.avg_dataloader,
        pretrained_path=config.pretrained_dir,
        lo_norm=False,
        ro_norm=False,
    )

    # -------- Callbacks & Trainer --------
    ckpt_dir = os.path.join(config.save_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="loss",
        mode="min",
        filename="wrn-epoch{epoch:02d}",   # 不用 {loss:.4f}
        auto_insert_metric_name=False
    )

    early_stop = EarlyStopping(monitor="last_val_loss", mode="min", patience=50, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        accelerator="gpu",
        devices=1,                 # 或 [0]
        precision=32,              # 显存紧张再改成 "16-mixed"（你的 3070 支持）
        max_epochs=N_EPOCHS,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    print("[INFO] Starting Trainer.fit ...")
    trainer.fit(pl_model, data_module)
    print("[INFO] Training finished.")
    return data_module, pl_model, trainer


if __name__ == "__main__":
    # 允许两种写法：example_config / example_config.yaml
    file_name = sys.argv[1] if len(sys.argv) > 1 else "example_config"
    if file_name.endswith((".yaml", ".yml")):
        file_name = os.path.splitext(file_name)[0]
    cfg_path = file_name + ".yaml"

    print(f"[INFO] Loading config: {cfg_path}")
    config = load_config(cfg_path)

    os.makedirs(config.save_dir, exist_ok=True)
    print(f"[INFO] save_dir: {config.save_dir}")
    print(f"[INFO] max_epochs: {config.max_epochs}")

    # 跑起来
    cli_main(config)
