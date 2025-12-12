"""
Training Script for Distributional Symbolic Regression
Inspired by Wasserstein Distributional Learning

Usage: python train_sr_distributional.py sr_distributional_test

Author: Zhengze
Date: 2025-11-10
"""
import os
import sys
import torch
import numpy as np

# NumPy compatibility fixes
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "float"):
    np.float = float

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path

# Import custom modules (adjust path as needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.IndexDataloader import DataModule
from src.models.sr_model_distributional import DistributionalSRModel


CONFIG_PATH = "confs/"


def load_config(config_name: str):
    """Load configuration file"""
    path = os.path.join(CONFIG_PATH, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = OmegaConf.load(f)
    return config


def collect_training_data(data_module, batch_size=32):
    """
    Collect training data from DataModule
    
    Returns:
        X: [N, 9] input features
        y: [N, 4] target tendencies
    """
    print("\n" + "="*70)
    print("COLLECTING TRAINING DATA FOR DISTRIBUTIONAL SR")
    print("="*70 + "\n")
    
    train_loader = data_module.train_dataloader()
    
    X_list = []
    y_list = []
    
    print(f"Collecting from {len(train_loader)} batches...")
    for i, batch in enumerate(train_loader):
        # DataModule returns: (inputs, outputs, tendencies) or more
        # We need inputs and tendencies (the derivatives we want to predict)
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 3:
            inputs, outputs, targets = batch  # tendencies are the 3rd element
        elif len(batch) >= 4:
            inputs, outputs, targets = batch[0], batch[1], batch[2]
        else:
            raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
        
        # Convert to numpy first to check shapes
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # DEBUG: Print first batch shape
        if i == 0:
            print(f"\n[DEBUG] First batch shapes:")
            print(f"  inputs:  {inputs_np.shape}")
            print(f"  targets: {targets_np.shape}")
        
        # Handle actual shape: [batch, features, 1]
        # Need to squeeze or transpose to get [batch, features]
        if inputs_np.ndim == 3:
            # Check which dimension is 1
            if inputs_np.shape[-1] == 1:
                # Shape is [batch, features, 1] → squeeze last dim
                inputs_flat = inputs_np.squeeze(-1)  # [batch, features]
            elif inputs_np.shape[1] == 1:
                # Shape is [batch, 1, features] → squeeze middle dim
                inputs_flat = inputs_np.squeeze(1)  # [batch, features]
            else:
                # Shape is [batch, tot_len, features] → flatten
                inputs_flat = inputs_np.reshape(-1, inputs_np.shape[-1])
        elif inputs_np.ndim == 4:
            # Shape is [batch, tot_len, 1, features]
            batch_size, tot_len, _, n_features = inputs_np.shape
            inputs_flat = inputs_np.reshape(batch_size * tot_len, n_features)
        elif inputs_np.ndim == 2:
            # Already correct shape [batch, features]
            inputs_flat = inputs_np
        else:
            raise ValueError(f"Unexpected inputs shape: {inputs_np.shape}")
        
        if targets_np.ndim == 3:
            if targets_np.shape[-1] == 1:
                targets_flat = targets_np.squeeze(-1)  # [batch, outputs]
            elif targets_np.shape[1] == 1:
                targets_flat = targets_np.squeeze(1)  # [batch, outputs]
            else:
                targets_flat = targets_np.reshape(-1, targets_np.shape[-1])
        elif targets_np.ndim == 4:
            batch_size, tot_len, _, n_outputs = targets_np.shape
            targets_flat = targets_np.reshape(batch_size * tot_len, n_outputs)
        elif targets_np.ndim == 2:
            targets_flat = targets_np
        else:
            raise ValueError(f"Unexpected targets shape: {targets_np.shape}")
        
        if i == 0:
            print(f"  After flatten:")
            print(f"    inputs:  {inputs_flat.shape}  (should be [N, 9])")
            print(f"    targets: {targets_flat.shape}  (should be [N, 4])\n")
        
        # Verify shapes are correct
        assert inputs_flat.ndim == 2 and inputs_flat.shape[1] == 9, \
            f"Wrong inputs shape: {inputs_flat.shape}, expected [N, 9]"
        assert targets_flat.ndim == 2 and targets_flat.shape[1] == 4, \
            f"Wrong targets shape: {targets_flat.shape}, expected [N, 4]"
        assert inputs_flat.shape[0] == targets_flat.shape[0], \
            f"Sample mismatch: inputs {inputs_flat.shape[0]} vs targets {targets_flat.shape[0]}"
        
        X_list.append(inputs_flat)
        y_list.append(targets_flat)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            print(f"  Processed {i+1}/{len(train_loader)} batches", end='\r')
    
    print(f"\n  Processed {len(train_loader)}/{len(train_loader)} batches")
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"\n[SR] Collected {X.shape[0]} training samples")
    print(f"[SR] Input shape: {X.shape}")
    print(f"[SR] Output shape: {y.shape}")
    print("\n" + "="*70 + "\n")
    
    return X, y


def evaluate_model(model, data_module):
    """
    Evaluate distributional model on validation set
    
    Returns:
        metrics: dict with MSE, NLL, and coverage statistics
    """
    print("\n" + "="*70)
    print("EVALUATING DISTRIBUTIONAL SR MODEL")
    print("="*70 + "\n")
    
    val_loader = data_module.val_dataloader()
    
    all_mu_pred = []
    all_sigma_pred = []
    all_y_true = []
    
    print("Computing predictions on validation set...")
    for i, batch in enumerate(val_loader):
        # Same unpacking logic as in collect_training_data
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 3:
            inputs, outputs, targets = batch
        elif len(batch) >= 4:
            inputs, outputs, targets = batch[0], batch[1], batch[2]
        else:
            raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
        
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Handle shape [batch, features, 1]
        if inputs_np.ndim == 3:
            if inputs_np.shape[-1] == 1:
                inputs_flat = inputs_np.squeeze(-1)
            elif inputs_np.shape[1] == 1:
                inputs_flat = inputs_np.squeeze(1)
            else:
                inputs_flat = inputs_np.reshape(-1, inputs_np.shape[-1])
        elif inputs_np.ndim == 4:
            batch_size, tot_len, _, n_features = inputs_np.shape
            inputs_flat = inputs_np.reshape(batch_size * tot_len, n_features)
        elif inputs_np.ndim == 2:
            inputs_flat = inputs_np
        else:
            raise ValueError(f"Unexpected inputs shape: {inputs_np.shape}")
        
        if targets_np.ndim == 3:
            if targets_np.shape[-1] == 1:
                targets_flat = targets_np.squeeze(-1)
            elif targets_np.shape[1] == 1:
                targets_flat = targets_np.squeeze(1)
            else:
                targets_flat = targets_np.reshape(-1, targets_np.shape[-1])
        elif targets_np.ndim == 4:
            batch_size, tot_len, _, n_outputs = targets_np.shape
            targets_flat = targets_np.reshape(batch_size * tot_len, n_outputs)
        elif targets_np.ndim == 2:
            targets_flat = targets_np
        else:
            raise ValueError(f"Unexpected targets shape: {targets_np.shape}")
        
        mu_pred, sigma_pred = model.predict_distribution(inputs_flat)
        
        all_mu_pred.append(mu_pred)
        all_sigma_pred.append(sigma_pred)
        all_y_true.append(targets_flat)
    
    all_mu_pred = np.concatenate(all_mu_pred, axis=0)
    all_sigma_pred = np.concatenate(all_sigma_pred, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0)
    
    print(f"✓ Evaluated {all_y_true.shape[0]} validation samples\n")
    
    # Compute metrics
    metrics = {}
    output_names = ["dLc_dt", "dNc_dt", "dLr_dt", "dNr_dt"]
    
    print("="*70)
    print("VALIDATION METRICS")
    print("="*70 + "\n")
    
    for i, name in enumerate(output_names):
        y_true = all_y_true[:, i]
        mu_pred = all_mu_pred[:, i]
        sigma_pred = all_sigma_pred[:, i]
        
        # MSE
        mse = np.mean((y_true - mu_pred) ** 2)
        
        # MAE
        mae = np.mean(np.abs(y_true - mu_pred))
        
        # Negative Log-Likelihood
        nll = 0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma_pred) + 
                     ((y_true - mu_pred) / sigma_pred) ** 2)
        nll = nll.mean()
        
        # Coverage (% of points within 1σ, 2σ, 3σ)
        residual = np.abs(y_true - mu_pred)
        coverage_1sigma = np.mean(residual <= sigma_pred) * 100
        coverage_2sigma = np.mean(residual <= 2 * sigma_pred) * 100
        coverage_3sigma = np.mean(residual <= 3 * sigma_pred) * 100
        
        # Mean predicted uncertainty
        mean_sigma = sigma_pred.mean()
        std_sigma = sigma_pred.std()
        
        print(f"{name}:")
        print(f"  MSE:             {mse:.6e}")
        print(f"  MAE:             {mae:.6e}")
        print(f"  NLL:             {nll:.6e}")
        print(f"  Coverage 1σ:     {coverage_1sigma:.1f}% (expected: 68.3%)")
        print(f"  Coverage 2σ:     {coverage_2sigma:.1f}% (expected: 95.4%)")
        print(f"  Coverage 3σ:     {coverage_3sigma:.1f}% (expected: 99.7%)")
        print(f"  Mean σ:          {mean_sigma:.6e}")
        print(f"  Std σ:           {std_sigma:.6e}")
        print()
        
        metrics[name] = {
            'mse': mse,
            'mae': mae,
            'nll': nll,
            'coverage_1sigma': coverage_1sigma,
            'coverage_2sigma': coverage_2sigma,
            'coverage_3sigma': coverage_3sigma,
            'mean_sigma': mean_sigma,
            'std_sigma': std_sigma,
        }
    
    # Overall metrics
    overall_mse = np.mean([m['mse'] for m in metrics.values()])
    overall_nll = np.mean([m['nll'] for m in metrics.values()])
    overall_coverage_1sigma = np.mean([m['coverage_1sigma'] for m in metrics.values()])
    
    print("="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"  Average MSE:         {overall_mse:.6e}")
    print(f"  Average NLL:         {overall_nll:.6e}")
    print(f"  Average Coverage 1σ: {overall_coverage_1sigma:.1f}%")
    print("="*70 + "\n")
    
    return metrics


def visualize_uncertainty(model, data_module, save_dir):
    """
    Visualize prediction uncertainty (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping visualization")
        return
    
    print("\n" + "="*70)
    print("GENERATING UNCERTAINTY VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Get a batch from validation set
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    
    # Same unpacking logic
    if len(batch) == 2:
        inputs, targets = batch
    elif len(batch) == 3:
        inputs, outputs, targets = batch
    elif len(batch) >= 4:
        inputs, outputs, targets = batch[0], batch[1], batch[2]
    else:
        raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
    
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Handle shape [batch, features, 1]
    if inputs_np.ndim == 3:
        if inputs_np.shape[-1] == 1:
            inputs_flat = inputs_np.squeeze(-1)
        elif inputs_np.shape[1] == 1:
            inputs_flat = inputs_np.squeeze(1)
        else:
            inputs_flat = inputs_np.reshape(-1, inputs_np.shape[-1])
    elif inputs_np.ndim == 4:
        batch_size, tot_len, _, n_features = inputs_np.shape
        inputs_flat = inputs_np.reshape(batch_size * tot_len, n_features)
    elif inputs_np.ndim == 2:
        inputs_flat = inputs_np
    else:
        raise ValueError(f"Unexpected inputs shape: {inputs_np.shape}")
    
    if targets_np.ndim == 3:
        if targets_np.shape[-1] == 1:
            targets_flat = targets_np.squeeze(-1)
        elif targets_np.shape[1] == 1:
            targets_flat = targets_np.squeeze(1)
        else:
            targets_flat = targets_np.reshape(-1, targets_np.shape[-1])
    elif targets_np.ndim == 4:
        batch_size, tot_len, _, n_outputs = targets_np.shape
        targets_flat = targets_np.reshape(batch_size * tot_len, n_outputs)
    elif targets_np.ndim == 2:
        targets_flat = targets_np
    else:
        raise ValueError(f"Unexpected targets shape: {targets_np.shape}")
    
    # Take first 100 samples for visualization
    n_viz = min(100, inputs_flat.shape[0])
    inputs_viz = inputs_flat[:n_viz]
    targets_viz = targets_flat[:n_viz]
    
    # Predict
    mu_pred, sigma_pred = model.predict_distribution(inputs_viz)
    
    # Create plots
    output_names = ["dLc_dt", "dNc_dt", "dLr_dt", "dNr_dt"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, output_names)):
        y_true = targets_viz[:, i]
        mu = mu_pred[:, i]
        sigma = sigma_pred[:, i]
        
        # Sort by true value for better visualization
        sort_idx = np.argsort(y_true)
        y_true_sorted = y_true[sort_idx]
        mu_sorted = mu[sort_idx]
        sigma_sorted = sigma[sort_idx]
        
        x = np.arange(len(y_true_sorted))
        
        # Plot true values
        ax.scatter(x, y_true_sorted, s=20, alpha=0.5, label='True', color='black')
        
        # Plot predictions with uncertainty bands
        ax.plot(x, mu_sorted, 'r-', linewidth=2, label='Predicted μ')
        ax.fill_between(x, 
                        mu_sorted - sigma_sorted, 
                        mu_sorted + sigma_sorted,
                        alpha=0.3, color='red', label='±1σ')
        ax.fill_between(x, 
                        mu_sorted - 2*sigma_sorted, 
                        mu_sorted + 2*sigma_sorted,
                        alpha=0.15, color='red', label='±2σ')
        
        ax.set_xlabel('Sample Index (sorted)', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(f'{name} - Uncertainty Visualization', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_path = Path(save_dir) / "uncertainty_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Uncertainty visualization saved to: {viz_path}\n")


def main(config):
    """Main training function"""
    pl.seed_everything(42)
    
    print("\n" + "="*70)
    print("DISTRIBUTIONAL SYMBOLIC REGRESSION FOR WARM-RAIN")
    print("="*70)
    print("Config:")
    print(f"  data_dir:      {config.data_dir}")
    print(f"  tot_len:       {config.tot_len}")
    print(f"  sim_num:       {config.sim_num}")
    print(f"  batch_size:    {config.batch_size}")
    print(f"  SR iterations: {config.sr_config.niterations}")
    print("="*70 + "\n")
    
    # ============================================================
    # Step 1: Load Data
    # ============================================================
    print("[1/4] Loading data...")
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
    
    # ============================================================
    # Step 2: Collect Training Data
    # ============================================================
    print("[2/4] Collecting training data for SR...")
    X_train, y_train = collect_training_data(data_module, config.batch_size)
    print("  ✓ Training data collected\n")
    
    # ============================================================
    # Step 3: Train Distributional SR Model
    # ============================================================
    print("[3/4] Training distributional SR model...")
    
    # Extract SR config
    sr_config = OmegaConf.to_container(config.sr_config, resolve=True)
    
    # Create and train model
    model = DistributionalSRModel(
        out_features=4,
        depth=9,
        save_dir=config.save_dir,
        **sr_config
    )
    
    # Train
    model.fit(X_train, y_train)
    print("  ✓ Distributional SR model trained\n")
    
    # ============================================================
    # Step 4: Evaluate Model
    # ============================================================
    print("[4/4] Evaluating model...")
    metrics = evaluate_model(model, data_module)
    print("  ✓ Evaluation complete\n")
    
    # ============================================================
    # Optional: Visualize Uncertainty
    # ============================================================
    try:
        visualize_uncertainty(model, data_module, config.save_dir)
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")
    
    # ============================================================
    # Print Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("DISCOVERED EQUATIONS SUMMARY")
    print("="*70 + "\n")
    
    equations = model.get_equations()
    for name, eq_info in equations.items():
        print(f"{name}:")
        if 'error' in eq_info:
            print(f"  Error: {eq_info['error']}")
        else:
            print(f"  μ equation:     {eq_info['mu']['equation']}")
            print(f"  μ complexity:   {eq_info['mu']['complexity']}")
            print(f"  log_σ equation: {eq_info['log_sigma']['equation']}")
            print(f"  log_σ complexity: {eq_info['log_sigma']['complexity']}")
        print()
    
    print("="*70 + "\n")
    
    # Print file locations
    equations_file = Path(config.save_dir) / "discovered_equations.txt"
    print(f"[INFO] Full equations saved to: {equations_file}")
    print(f"[INFO] Models saved to: {Path(config.save_dir)}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Distributional SR training complete!")
    print("="*70 + "\n")
    
    return model, data_module, metrics


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "sr_distributional_test"
    
    if not config_name.endswith((".yaml", ".yml")):
        config_name += ".yaml"
    
    print(f"[INFO] Loading config: {config_name}")
    
    try:
        config = load_config(config_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Run training
    main(config)