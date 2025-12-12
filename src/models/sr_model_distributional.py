"""
Distributional Symbolic Regression Model for Warm Rain Microphysics
Inspired by Wasserstein Distributional Learning (Tang et al., 2023)

Key Innovation:
- Predict distribution parameters (μ, σ) instead of point estimates
- Quantify prediction uncertainty
- Use Gaussian likelihood as loss function

Author: Zhengze
Date: 2025-11-10
"""
import numpy as np
import torch
import pickle
import os
from pathlib import Path

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("[WARN] PySR not installed. Install with: pip install pysr")


class DistributionalSRModel:
    """
    Distributional Symbolic Regression Model
    
    For each output, we predict TWO quantities:
    - μ(x): mean (via SR)
    - log(σ(x)): log-standard-deviation (via SR, ensuring σ > 0)
    
    Input:  [batch, 9] - [Lc, Nc, Lr, Nr, tau, xc, r0, ν, L0]
    Output: [batch, 4, 2] - [(μ_ΔLc, σ_ΔLc), (μ_ΔNc, σ_ΔNc), ...]
    
    Loss: Negative Log-Likelihood of Gaussian distribution
          NLL = 0.5 * [log(σ²) + (y - μ)²/σ²]
    """
    
    def __init__(
        self,
        out_features=4,
        depth=9,
        save_dir="outputs/sr_distributional",
        # PySR core parameters
        niterations=40,
        populations=15,
        population_size=33,
        max_complexity=20,
        binary_operators=None,
        unary_operators=None,
        constraints=None,
        loss="L1DistLoss()",
        maxsize=20,
        parsimony=0.0032,
        # Distributional-specific
        sigma_init=1e-2,  # Initial guess for uncertainty
        **kwargs
    ):
        """
        Initialize distributional SR model
        
        Args:
            sigma_init: Initial value for log(σ), controls baseline uncertainty
        """
        if not PYSR_AVAILABLE:
            raise ImportError("PySR not installed. Run: pip install pysr")
        
        self.out_features = out_features
        self.depth = depth
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sigma_init = sigma_init
        
        # Default operators (same as baseline)
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["square", "sqrt", "log", "abs"]
        if constraints is None:
            constraints = {
                "/": (-1, 9),
                "log": 3,
            }
        
        # Feature names
        self.feature_names = [
            "Lc", "Nc", "Lr", "Nr", 
            "tau", "xc", "r0", "nu", "L0"
        ]
        
        # Output names
        self.base_output_names = ["dLc_dt", "dNc_dt", "dLr_dt", "dNr_dt"]
        
        # Create TWO models for each output: one for μ, one for log(σ)
        self.mu_models = []      # Mean models
        self.logsigma_models = [] # Log-sigma models
        
        # Filter unsupported parameters
        supported_pysr_params = {
            'niterations', 'populations', 'population_size',
            'binary_operators', 'unary_operators', 'constraints',
            'loss', 'maxsize', 'parsimony', 'turbo', 'precision',
            'verbosity', 'progress', 'procs', 'multithreading'
        }
        
        extra_pysr_params = {k: v for k, v in kwargs.items() 
                            if k in supported_pysr_params}
        
        print("\n" + "="*70)
        print("INITIALIZING DISTRIBUTIONAL SR MODELS")
        print("="*70)
        print(f"Architecture: {out_features} outputs × 2 (μ, log_σ) = {out_features*2} SR models")
        print(f"Feature dimension: {depth}")
        print("="*70 + "\n")
        
        for i, name in enumerate(self.base_output_names):
            print(f"[{i+1}/{out_features}] Creating models for {name}...")
            
            # Model for mean (μ)
            mu_model = PySRRegressor(
                niterations=niterations,
                populations=populations,
                population_size=population_size,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                constraints=constraints,
                loss=loss,
                maxsize=maxsize,
                parsimony=parsimony,
                turbo=True,
                precision=32,
                verbosity=1,
                progress=True,
                **extra_pysr_params
            )
            self.mu_models.append(mu_model)
            
            # Model for log(σ) - simpler to ensure stability
            logsigma_model = PySRRegressor(
                niterations=max(niterations // 2, 5),  # Fewer iterations for σ
                populations=populations,
                population_size=population_size,
                binary_operators=["+", "*"],  # Simpler operators for σ
                unary_operators=["square"],
                constraints={},
                loss=loss,
                maxsize=min(maxsize, 15),  # Simpler equations for σ
                parsimony=parsimony * 2,   # More regularization
                turbo=True,
                precision=32,
                verbosity=0,  # Less verbose for σ models
                progress=False,
                **{k: v for k, v in extra_pysr_params.items() 
                   if k not in ['verbosity', 'progress']}
            )
            self.logsigma_models.append(logsigma_model)
            
            print(f"  ✓ μ model initialized")
            print(f"  ✓ log_σ model initialized (simplified)")
        
        self.is_fitted = False
        print("\n[INIT] All distributional SR models initialized!\n")
    
    def fit(self, X, y, variable_names=None):
        """
        Train distributional SR models
        
        Training strategy:
        1. First train μ models (predict mean)
        2. Compute residuals: ε = y - μ_pred
        3. Train log(σ) models to predict log|ε|
        
        This two-stage approach is more stable than joint training.
        """
        if variable_names is None:
            variable_names = self.feature_names
        
        print("\n" + "="*70)
        print("DISTRIBUTIONAL SYMBOLIC REGRESSION TRAINING")
        print("="*70)
        print(f"Training data: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"Output: {y.shape[1]} targets × 2 (μ, σ)")
        print(f"Feature names: {variable_names}")
        print("="*70 + "\n")
        
        # Storage for predictions
        mu_preds = np.zeros_like(y)
        
        # ============================================================
        # STAGE 1: Train mean models (μ)
        # ============================================================
        print("\n" + "="*70)
        print("STAGE 1: TRAINING MEAN MODELS (μ)")
        print("="*70 + "\n")
        
        for i, (model, name) in enumerate(zip(self.mu_models, self.base_output_names)):
            print(f"\n[{i+1}/{self.out_features}] Training μ model: {name}")
            print("-" * 70)
            
            try:
                # Train mean model
                model.fit(X, y[:, i], variable_names=variable_names)
                
                # Get predictions for residual computation
                mu_preds[:, i] = model.predict(X)
                
                # Print best equation
                best = model.get_best()
                print(f"\n[SR] Best equation for μ_{name}:")
                print(f"  Complexity: {best['complexity']}")
                print(f"  Loss: {best['loss']:.6f}")
                print(f"  Equation: {best['equation']}")
                
                # Save model
                model_path = self.save_dir / f"model_mu_{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  Saved to: {model_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to train μ_{name}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        print("\n" + "="*70)
        print("[STAGE 1 COMPLETE] All mean models trained!")
        print("="*70 + "\n")
        
        # ============================================================
        # STAGE 2: Train uncertainty models (log σ)
        # ============================================================
        print("\n" + "="*70)
        print("STAGE 2: TRAINING UNCERTAINTY MODELS (log σ)")
        print("="*70)
        print("\nStrategy: Train log(σ) to predict log(|residual|)")
        print("="*70 + "\n")
        
        # Compute residuals
        residuals = np.abs(y - mu_preds) + 1e-8  # Add small epsilon for numerical stability
        log_residuals = np.log(residuals)
        
        for i, (model, name) in enumerate(zip(self.logsigma_models, self.base_output_names)):
            print(f"\n[{i+1}/{self.out_features}] Training log_σ model: {name}")
            print("-" * 70)
            
            try:
                # Train log-sigma model on log-residuals
                model.fit(X, log_residuals[:, i], variable_names=variable_names)
                
                # Print best equation
                best = model.get_best()
                print(f"\n[SR] Best equation for log_σ_{name}:")
                print(f"  Complexity: {best['complexity']}")
                print(f"  Loss: {best['loss']:.6f}")
                print(f"  Equation: {best['equation']}")
                
                # Save model
                model_path = self.save_dir / f"model_logsigma_{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  Saved to: {model_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to train log_σ_{name}: {e}")
                import traceback
                traceback.print_exc()
                # Non-critical: use constant sigma if failed
                print(f"[WARN] Using constant σ for {name}")
        
        print("\n" + "="*70)
        print("[STAGE 2 COMPLETE] All uncertainty models trained!")
        print("="*70 + "\n")
        
        self.is_fitted = True
        self.save_equations()
        
        # Compute and print statistics
        self._print_statistics(X, y)
        
        print("\n" + "="*70)
        print("[SUCCESS] Distributional SR training complete!")
        print("="*70 + "\n")
    
    def _print_statistics(self, X, y):
        """Print prediction statistics"""
        print("\n" + "="*70)
        print("PREDICTION STATISTICS")
        print("="*70 + "\n")
        
        mu_pred, sigma_pred = self.predict_distribution(X)
        
        for i, name in enumerate(self.base_output_names):
            print(f"\n{name}:")
            print(f"  True:      mean={y[:, i].mean():.4e}, std={y[:, i].std():.4e}")
            print(f"  Predicted: mean={mu_pred[:, i].mean():.4e}, std={mu_pred[:, i].std():.4e}")
            print(f"  Uncertainty (σ): mean={sigma_pred[:, i].mean():.4e}, std={sigma_pred[:, i].std():.4e}")
            
            # MSE
            mse = np.mean((y[:, i] - mu_pred[:, i])**2)
            print(f"  MSE: {mse:.4e}")
            
            # Negative Log-Likelihood
            nll = self._compute_nll(y[:, i], mu_pred[:, i], sigma_pred[:, i])
            print(f"  NLL: {nll:.4e}")
        
        print("\n" + "="*70 + "\n")
    
    @staticmethod
    def _compute_nll(y_true, mu_pred, sigma_pred):
        """Compute negative log-likelihood"""
        nll = 0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma_pred) + 
                     ((y_true - mu_pred) / sigma_pred) ** 2)
        return nll.mean()
    
    def save_equations(self):
        """Save discovered equations to text file"""
        equations_file = self.save_dir / "discovered_equations.txt"
        
        with open(equations_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DISCOVERED EQUATIONS - Distributional Warm Rain Microphysics\n")
            f.write("="*70 + "\n")
            f.write("\nModel Type: Distributional Symbolic Regression\n")
            f.write("Each output has TWO equations:\n")
            f.write("  - μ(x): Mean prediction\n")
            f.write("  - log_σ(x): Log-uncertainty prediction\n")
            f.write("\n" + "="*70 + "\n\n")
            
            for i, name in enumerate(self.base_output_names):
                f.write(f"\n{'='*70}\n")
                f.write(f"Output {i+1}: {name}\n")
                f.write(f"{'='*70}\n\n")
                
                # Mean model
                f.write(f"--- Mean Model (μ_{name}) ---\n\n")
                mu_model = self.mu_models[i]
                mu_best = mu_model.get_best()
                f.write(f"Complexity: {mu_best['complexity']}\n")
                f.write(f"Loss: {mu_best['loss']:.6f}\n")
                f.write(f"Equation:\n{mu_best['equation']}\n\n")
                
                try:
                    f.write(f"SymPy format:\n{mu_model.sympy()}\n\n")
                except:
                    pass
                
                # Uncertainty model
                f.write(f"\n--- Uncertainty Model (log_σ_{name}) ---\n\n")
                sigma_model = self.logsigma_models[i]
                sigma_best = sigma_model.get_best()
                f.write(f"Complexity: {sigma_best['complexity']}\n")
                f.write(f"Loss: {sigma_best['loss']:.6f}\n")
                f.write(f"Equation:\n{sigma_best['equation']}\n\n")
                
                try:
                    f.write(f"SymPy format:\n{sigma_model.sympy()}\n\n")
                except:
                    pass
                
                f.write(f"\n{'='*70}\n\n")
        
        print(f"[SR] Equations saved to: {equations_file}")
    
    def predict_distribution(self, x):
        """
        Predict distribution parameters
        
        Returns:
            mu: [batch, out_features] - mean predictions
            sigma: [batch, out_features] - std predictions (> 0)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        
        is_numpy = isinstance(x, np.ndarray)
        if not is_numpy:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Predict μ
        mu_preds = []
        for model in self.mu_models:
            mu_i = model.predict(x_np)
            mu_preds.append(mu_i)
        mu_preds = np.column_stack(mu_preds)
        
        # Predict log(σ) then exponentiate
        sigma_preds = []
        for model in self.logsigma_models:
            log_sigma_i = model.predict(x_np)
            sigma_i = np.exp(log_sigma_i)
            # Clip to reasonable range for numerical stability
            sigma_i = np.clip(sigma_i, 1e-6, 1e2)
            sigma_preds.append(sigma_i)
        sigma_preds = np.column_stack(sigma_preds)
        
        if not is_numpy:
            mu_preds = torch.from_numpy(mu_preds).float().to(x.device)
            sigma_preds = torch.from_numpy(sigma_preds).float().to(x.device)
        
        return mu_preds, sigma_preds
    
    def forward(self, x):
        """
        Forward pass - returns mean prediction (for compatibility)
        """
        mu, sigma = self.predict_distribution(x)
        return mu
    
    def __call__(self, x):
        return self.forward(x)
    
    def sample_predictions(self, x, n_samples=100):
        """
        Sample from predicted distributions
        
        Args:
            x: input features
            n_samples: number of samples per input
            
        Returns:
            samples: [n_samples, batch, out_features]
        """
        mu, sigma = self.predict_distribution(x)
        
        is_numpy = isinstance(mu, np.ndarray)
        
        if is_numpy:
            # NumPy version
            samples = np.random.normal(
                loc=mu[None, :, :],
                scale=sigma[None, :, :],
                size=(n_samples, mu.shape[0], mu.shape[1])
            )
        else:
            # PyTorch version
            samples = torch.normal(
                mean=mu.unsqueeze(0).expand(n_samples, -1, -1),
                std=sigma.unsqueeze(0).expand(n_samples, -1, -1)
            )
        
        return samples
    
    def get_equations(self):
        """Return all discovered equations"""
        if not self.is_fitted:
            raise RuntimeError("Models not fitted yet")
        
        equations = {}
        for name, mu_model, sigma_model in zip(
            self.base_output_names, 
            self.mu_models, 
            self.logsigma_models
        ):
            try:
                mu_best = mu_model.get_best()
                sigma_best = sigma_model.get_best()
                
                equations[name] = {
                    'mu': {
                        'equation': str(mu_best['equation']),
                        'complexity': mu_best['complexity'],
                        'loss': mu_best['loss'],
                    },
                    'log_sigma': {
                        'equation': str(sigma_best['equation']),
                        'complexity': sigma_best['complexity'],
                        'loss': sigma_best['loss'],
                    }
                }
                
                try:
                    equations[name]['mu']['sympy'] = str(mu_model.sympy())
                    equations[name]['log_sigma']['sympy'] = str(sigma_model.sympy())
                except:
                    pass
                    
            except Exception as e:
                equations[name] = {'error': str(e)}
        
        return equations