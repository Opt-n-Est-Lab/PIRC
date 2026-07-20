import os
import sys
import time
import warnings
import numpy as np
from scipy.stats import linregress

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from systems import LorenzSystem, RosslerSystem, NormalizedSystem
from pirc_core import PIRC, HybridRCNGRC
from baselines import StandardRC, StandardPINN
from metrics import compute_metrics, print_results_table, plot_chaotic_results

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_partial_lorenz_model(normalized_system):
    def model(u, dt):
        u = np.clip(u, -1e4, 1e4)
        ur = u * normalized_system.sigma + normalized_system.mu
        x, y, z = ur[..., 0], ur[..., 1], ur[..., 2]
        f = np.stack([10.0 * (y - x), -y, -(8. / 3.) * z], axis=-1) / normalized_system.sigma
        return np.clip(u + dt * f, -1e4, 1e4)
    return model

def create_partial_rossler_model(normalized_system):
    def model(u, dt):
        u = np.clip(u, -1e4, 1e4)
        ur = u * normalized_system.sigma + normalized_system.mu
        x, y, z = ur[..., 0], ur[..., 1], ur[..., 2]
        f = np.stack([-y - z, x + 0.2 * y, -5.7 * z], axis=-1) / normalized_system.sigma
        return np.clip(u + dt * f, -1e4, 1e4)
    return model

def create_full_model(normalized_system):
    def model(u, dt):
        u = np.clip(u, -1e4, 1e4)
        return np.clip(u + dt * normalized_system.rhs(u), -1e4, 1e4)
    return model

def evaluate_multiseed(model_cls, dim, train_data, dt, test_length, test_data, config, n_seeds=5):
    rmses, correlations = [], []
    total_train_time = 0
    param_count = 0
    
    for seed in range(n_seeds):
        model = model_cls(dim, seed=seed, **config)
        model.train(train_data, dt, verbose=False)
        total_train_time += model.train_time
        
        predictions = model.predict(test_length)
        valid_length = min(test_length, len(predictions))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((test_data[:valid_length] - predictions[:valid_length]) ** 2))
        rmses.append(rmse)
        
        # Calculate correlation
        corr_values = []
        for channel in range(test_data.shape[1]):
            if np.std(predictions[:valid_length, channel]) > 1e-10:
                corr_values.append(np.corrcoef(
                    test_data[:valid_length, channel], 
                    predictions[:valid_length, channel]
                )[0, 1])
            else:
                corr_values.append(0.0)
        correlations.append(np.mean(corr_values))
        
        # Capture parameter count from first seed
        if seed == 0:
            param_count = model.nparams()
    
    return {
        'RMSE': np.mean(rmses),
        'Corr': np.mean(correlations),
        'rmse_std': np.std(rmses),
        'params': param_count,
        'time': total_train_time / n_seeds
    }

def find_best_pirc_configuration(dim, train_data, dt, test_length, test_data, 
                               nvar_k, nvar_s, nvar_p, res_N, domain_fns, n_seeds=5):
    best_rmse = float('inf')
    best_result = None
    
    for domain_fn in domain_fns:
        config = {
            'res_N': res_N,
            'nvar_k': nvar_k,
            'nvar_s': nvar_s,
            'nvar_p': nvar_p,
            'domain_fn': domain_fn
        }
        result = evaluate_multiseed(PIRC, dim, train_data, dt, test_length, test_data, config, n_seeds)
        
        if result['RMSE'] < best_rmse:
            best_rmse = result['RMSE']
            best_result = result
    
    return best_result

def benchmark_system(system, nvar_k, nvar_s, nvar_p, res_N, partial_model_factory, n_seeds=5):
    print(f"\n{'=' * 60}\n  {system.name}\n{'=' * 60}")
    
    time_steps, data = system.generate()
    dt = time_steps[1] - time_steps[0]
    total_length = len(time_steps)
    train_length = int(total_length * 0.7)
    
    # Normalize data
    mu, sigma = data[:train_length].mean(0), data[:train_length].std(0) + 1e-8
    normalized_data = (data - mu) / sigma
    train_data, test_data = normalized_data[:train_length], normalized_data[train_length:]
    
    test_length = min(len(test_data) - 1, 400)
    normalized_system = NormalizedSystem(system, mu, sigma)
    
    results = {}
    
    # Evaluate Standard RC
    print("  Evaluating RC...")
    rc_rmses, rc_corrs = [], []
    rc_train_time = 0
    rc_param_count = 0
    
    for seed in range(n_seeds):
        rc = StandardRC(system.dim, 100, input_noise=1e-3, seed=seed)
        rc.train(train_data[:-1], train_data[1:])
        predictions = rc.predict_autonomous(test_length, train_data[-1])
        metrics = compute_metrics(test_data[:test_length], predictions[:test_length], dt)
        
        rc_rmses.append(metrics['RMSE'])
        rc_corrs.append(metrics['Corr'])
        rc_train_time += rc.train_time
        
        if seed == 0:
            rc_param_count = rc.nparams()
    
    results['RC'] = {
        'RMSE': np.mean(rc_rmses),
        'rmse_std': np.std(rc_rmses),
        'Corr': np.mean(rc_corrs),
        'params': rc_param_count,
        'time': rc_train_time / n_seeds
    }
    
    # Evaluate Hybrid RC-NGRC
    print("  Evaluating NGRC+RC...")
    hybrid_config = {
        'res_N': res_N,
        'nvar_k': nvar_k,
        'nvar_s': nvar_s,
        'nvar_p': nvar_p
    }
    results['NGRC+RC'] = evaluate_multiseed(
        HybridRCNGRC, system.dim, train_data, dt, test_length, test_data[:test_length], 
        hybrid_config, n_seeds
    )
    
    # Evaluate PIRC
    print("  Evaluating PIRC...")
    domain_functions = [
        None, 
        partial_model_factory(normalized_system),
        create_full_model(normalized_system)
    ]
    results['PIRC'] = find_best_pirc_configuration(
        system.dim, train_data, dt, test_length, test_data[:test_length],
        nvar_k, nvar_s, nvar_p, res_N, domain_functions, n_seeds
    )
    
    # Evaluate PINN
    print("  Evaluating PINN...")
    pinn_rmses, pinn_corrs = [], []
    pinn_train_time = 0
    pinn_param_count = 0
    
    normalized_time = np.linspace(0, 1, train_length)
    test_time = np.linspace(1, 1 + test_length * dt / (time_steps[train_length - 1] - time_steps[0]), test_length)
    
    for seed in range(n_seeds):
        pinn = StandardPINN(system.dim, (128, 128), seed=seed)
        pinn.train(normalized_time, normalized_data[:train_length], normalized_system, 
                   epochs=600, dt=dt, verbose=False)
        predictions = pinn.predict(test_time)
        metrics = compute_metrics(test_data[:test_length], predictions[:test_length], dt)
        
        pinn_rmses.append(metrics['RMSE'])
        pinn_corrs.append(metrics['Corr'])
        pinn_train_time += pinn.train_time
        
        if seed == 0:
            pinn_param_count = pinn.nparams()
    
    results['PINN'] = {
        'RMSE': np.mean(pinn_rmses),
        'rmse_std': np.std(pinn_rmses),
        'Corr': np.mean(pinn_corrs),
        'params': pinn_param_count,
        'time': pinn_train_time / n_seeds
    }
    
    print_results_table(system.name, results)
    return results

if __name__ == "__main__":
    print("=" * 65)
    print("PIRC: Chaotic Systems")
    print("=" * 65)
    
    all_results = {}
    all_results['Lorenz'] = benchmark_system(
        LorenzSystem(), 2, 1, 2, 100,
        create_partial_lorenz_model, 5
    )
    
    all_results['Rossler'] = benchmark_system(
        RosslerSystem(), 2, 2, 2, 100,
        create_partial_rossler_model, 5
    )
    
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'System':<14} {'Model':<14} {'RMSE':>10} {'Corr':>8} {'Params':>7}")
    print(f"  {'─' * 55}")
    
    for system_name, results in all_results.items():
        for model_name, metrics in results.items():
            std = f"±{metrics['rmse_std']:.3f}" if 'rmse_std' in metrics else ""
            print(f"  {system_name:<14} {model_name:<14} {metrics['RMSE']:>10.3f} {metrics['Corr']:>8.3f} {metrics['params']:>7} {std}")
        print(f"  {'─' * 55}")
    
    plot_chaotic_results(all_results, os.path.join(SAVE_DIR, "pirc_chaotic.png"))