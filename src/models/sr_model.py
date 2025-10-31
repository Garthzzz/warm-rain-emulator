"""
符号回归模型 - 完全修复版
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


class SymbolicRegressionModel:
    """
    符号回归模型 - 单步预测版本
    
    输入: [batch, 9] - [Lc, Nc, Lr, Nr, tau, xc, r0, ν, L0]
    输出: [batch, 4] - [ΔLc, ΔNc, ΔLr, ΔNr]
    """
    
    def __init__(
        self,
        out_features=4,
        depth=9,
        save_dir="outputs/sr_models",
        # PySR核心参数
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
        # 可选参数（用**kwargs捕获其他参数）
        **kwargs
    ):
        if not PYSR_AVAILABLE:
            raise ImportError("PySR not installed. Run: pip install pysr")
        
        self.out_features = out_features
        self.depth = depth
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认操作符
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["square", "sqrt", "log", "abs"]
        if constraints is None:
            constraints = {
                "/": (-1, 9),
                "log": 3,
            }
        
        # 特征名称
        self.feature_names = [
            "Lc", "Nc", "Lr", "Nr", 
            "tau", "xc", "r0", "nu", "L0"
        ]
        
        # 为每个输出创建独立模型
        self.models = []
        self.output_names = ["dLc_dt", "dNc_dt", "dLr_dt", "dNr_dt"]
        
        # 过滤掉不支持的参数
        supported_pysr_params = {
            'niterations', 'populations', 'population_size',
            'binary_operators', 'unary_operators', 'constraints',
            'loss', 'maxsize', 'parsimony', 'turbo', 'precision',
            'verbosity', 'progress'
        }
        
        # 从kwargs中提取PySR支持的参数
        extra_pysr_params = {k: v for k, v in kwargs.items() 
                            if k in supported_pysr_params}
        
        for i, name in enumerate(self.output_names):
            print(f"[SR] Initializing model for {name}")
            
            model = PySRRegressor(
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
                **extra_pysr_params  # 添加其他支持的参数
            )
            self.models.append(model)
        
        self.is_fitted = False
    
    def fit(self, X, y, variable_names=None):
        """训练SR模型"""
        if variable_names is None:
            variable_names = self.feature_names
        
        print("\n" + "="*70)
        print("SYMBOLIC REGRESSION TRAINING")
        print("="*70)
        print(f"Training data: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"Output: {y.shape[1]} targets")
        print(f"Feature names: {variable_names}")
        print("="*70 + "\n")
        
        for i, (model, name) in enumerate(zip(self.models, self.output_names)):
            print(f"\n{'='*70}")
            print(f"Training SR model {i+1}/4: {name}")
            print(f"{'='*70}")
            
            try:
                # 训练模型，传入特征名称
                model.fit(X, y[:, i], variable_names=variable_names)
                
                # 打印最佳公式
                best = model.get_best()
                print(f"\n[SR] Best equation for {name}:")
                print(f"  Complexity: {best['complexity']}")
                print(f"  Loss: {best['loss']:.6f}")
                print(f"  Equation: {best['equation']}")
                
                # 保存模型
                model_path = self.save_dir / f"model_{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  Saved to: {model_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to train {name}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        self.is_fitted = True
        self.save_equations()
        
        print(f"\n{'='*70}")
        print("[SUCCESS] All SR models trained!")
        print(f"{'='*70}\n")
    
    def save_equations(self):
        """保存公式到文本文件"""
        equations_file = self.save_dir / "discovered_equations.txt"
        
        with open(equations_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DISCOVERED EQUATIONS - Warm Rain Microphysics\n")
            f.write("="*70 + "\n\n")
            
            for i, (model, name) in enumerate(zip(self.models, self.output_names)):
                f.write(f"\n{'-'*70}\n")
                f.write(f"Output {i+1}: {name}\n")
                f.write(f"{'-'*70}\n\n")
                
                best = model.get_best()
                f.write(f"Complexity: {best['complexity']}\n")
                f.write(f"Loss: {best['loss']:.6f}\n")
                f.write(f"Equation:\n{best['equation']}\n\n")
                
                try:
                    f.write(f"SymPy format:\n{model.sympy()}\n\n")
                except:
                    pass
                
                try:
                    f.write(f"LaTeX format:\n{model.latex()}\n\n")
                except:
                    pass
        
        print(f"[SR] Equations saved to: {equations_file}")
    
    def forward(self, x):
        """前向传播"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        
        is_numpy = isinstance(x, np.ndarray)
        if not is_numpy:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # 预测
        preds = []
        for model in self.models:
            pred_i = model.predict(x_np)
            preds.append(pred_i)
        
        preds = np.column_stack(preds)
        
        if not is_numpy:
            return torch.from_numpy(preds).float().to(x.device)
        else:
            return preds
    
    def __call__(self, x):
        return self.forward(x)
    
    def get_equations(self):
        """返回所有公式"""
        if not self.is_fitted:
            raise RuntimeError("Models not fitted yet")
        
        equations = {}
        for name, model in zip(self.output_names, self.models):
            try:
                best = model.get_best()
                equations[name] = {
                    'equation': str(best['equation']),
                    'complexity': best['complexity'],
                    'loss': best['loss'],
                }
                try:
                    equations[name]['sympy'] = str(model.sympy())
                except:
                    pass
            except Exception as e:
                equations[name] = {'error': str(e)}
        
        return equations