import time
import warnings
import numpy as np
from itertools import combinations_with_replacement
from reservoir import Reservoir, nonlinear_transform, build_nvar

CLIP = 1e4

# ==============================================================================
# Helper Functions
# ==============================================================================
def _rollout(seed_data, n_steps, dim, reservoir, weights, nvar_k, nvar_s, nvar_p, use_reservoir):
    history = list(seed_data[-((nvar_k - 1) * nvar_s + 1):])
    h = None
    if use_reservoir and reservoir is not None:
        _, h = reservoir.run(seed_data)
        
    preds = []
    for _ in range(n_steps):
        u = np.clip(np.array(history[-1]), -CLIP, CLIP)
        if use_reservoir and reservoir is not None:
            h = reservoir.step(u, h)
            h_g = h.copy()
            h_g[1::2] = h[1::2] ** 2
        else:
            h_g = None
            
        cur = np.array(history[-((nvar_k - 1) * nvar_s + 1):])
        O = []
        for i in range(nvar_k):
            idx = len(cur) - 1 - i * nvar_s
            if idx < 0: idx = 0
            O.extend(cur[idx].tolist())
        O = np.clip(np.array(O), -CLIP, CLIP)
        
        polys = []
        for deg in range(2, nvar_p + 1):
            for combo in combinations_with_replacement(range(len(O)), deg):
                v = 1.0
                for ci in combo:
                    v *= O[ci]
                    if abs(v) > CLIP:
                        v = np.sign(v) * CLIP
                        break
                polys.append(v)
                
        phi = np.concatenate([[1.0], O, polys])
        xi = np.concatenate([h_g, phi]) if use_reservoir else phi
        y = xi @ weights
        
        if np.any(np.isnan(y)) or np.any(np.abs(y) > CLIP):
            y = np.clip(np.nan_to_num(y, nan=0.0), -CLIP, CLIP)
            
        preds.append(y)
        history.append(y)
        
    return np.array(preds)


def _validation_error(seed_data, val_data, dim, reservoir, weights, nvar_k, nvar_s, nvar_p, use_reservoir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = _rollout(seed_data, len(val_data), dim, reservoir, weights, nvar_k, nvar_s, nvar_p, use_reservoir)
        nc = min(len(preds), len(val_data))
        if nc < 5:
            return float('inf')
        diff = preds[:nc] - val_data[:nc]
        if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
            return float('inf')
        mse = np.mean(diff ** 2)
        return mse if np.isfinite(mse) else float('inf')


def _solve_ridge_regression(features, targets, beta, physics_weight=0.0, physics_targets=None):
    q = features.shape[1]
    if physics_weight > 0 and physics_targets is not None:
        gram = (1 + physics_weight) * features.T @ features + beta * np.eye(q)
        rhs = features.T @ (targets + physics_weight * physics_targets)
    else:
        gram = features.T @ features + beta * np.eye(q)
        rhs = features.T @ targets
    return np.linalg.solve(gram, rhs)


# ==============================================================================
# Model Definitions
# ==============================================================================
class HybridRCNGRC:
    def __init__(self, dim, res_N=100, nvar_k=2, nvar_s=1, nvar_p=2,
                 input_noise=1e-3, seed=42):
        self.dim = dim
        self.res_N = res_N
        self.nvar_k = nvar_k
        self.nvar_s = nvar_s
        self.nvar_p = nvar_p
        self.input_noise = input_noise
        self.seed = seed

    def train(self, data, dt, verbose=True):
        t0 = time.time()
        T, d = data.shape
        val_len = max(20, int(T * 0.05))
        rng = np.random.RandomState(self.seed + 7777)
        data_noisy = data + self.input_noise * rng.randn(T, d) if self.input_noise > 0 else data
        
        self.reservoir = Reservoir(d, self.res_N, seed=self.seed)
        states, _ = self.reservoir.run(data_noisy)
        res_transformed = nonlinear_transform(states)
        
        nvar_features, offset = build_nvar(data, self.nvar_k, self.nvar_s, self.nvar_p)
        nvar_features = nvar_features[:-1]
        
        washout = min(100, len(nvar_features) // 5)
        features = np.column_stack([res_transformed[offset + washout:-1], nvar_features[washout:]])
        targets = data[offset + 1 + washout:]
        
        split_idx = len(targets) - val_len
        best_error, best_beta = float('inf'), 1e-6
        
        for lb in range(-8, -1):
            beta = 10.0 ** lb
            try:
                W_temp = _solve_ridge_regression(features[:split_idx], targets[:split_idx], beta)
                error = _validation_error(data[:-val_len], data[-val_len:], d, self.reservoir, W_temp,
                                          self.nvar_k, self.nvar_s, self.nvar_p, True)
                if error < best_error:
                    best_error, best_beta = error, beta
            except Exception:
                pass
                
        self.weights = _solve_ridge_regression(features, targets, best_beta)
        self._data = data
        if verbose:
            print(f"  Hybrid: {features.shape[1]} feats, beta={best_beta:.0e}, val={best_error:.6f}")
        self.train_time = time.time() - t0

    def predict(self, n_steps):
        return _rollout(self._data, n_steps, self.dim, self.reservoir, self.weights,
                        self.nvar_k, self.nvar_s, self.nvar_p, True)

    def nparams(self):
        return self.weights.size


class PIRC:
    def __init__(self, dim, res_N=100, nvar_k=2, nvar_s=1, nvar_p=2,
                 domain_fn=None, input_noise=1e-3, seed=42):
        self.dim = dim
        self.res_N = res_N
        self.nvar_k = nvar_k
        self.nvar_s = nvar_s
        self.nvar_p = nvar_p
        self.domain_fn = domain_fn
        self.input_noise = input_noise
        self.seed = seed

    def train(self, data, dt, verbose=True):
        t0 = time.time()
        T, d = data.shape
        val_len = max(20, int(T * 0.05))
        has_physics = self.domain_fn is not None
        rng = np.random.RandomState(self.seed + 7777)
        
        nvar_features, offset = build_nvar(data, self.nvar_k, self.nvar_s, self.nvar_p)
        nvar_features = nvar_features[:-1]
        
        data_noisy = data + self.input_noise * rng.randn(T, d) if self.input_noise > 0 else data
        self.reservoir = Reservoir(d, self.res_N, seed=self.seed)
        states, _ = self.reservoir.run(data_noisy)
        res_transformed = nonlinear_transform(states)
        
        washout = min(100, len(nvar_features) // 5)
        features_nvar = nvar_features
        features_hybrid = np.column_stack([res_transformed[offset + washout:-1], nvar_features[washout:]])
        targets_nvar = data[offset + 1:]
        targets_hybrid = data[offset + 1 + washout:]
        
        split_nvar = len(targets_nvar) - val_len
        split_hybrid = len(targets_hybrid) - val_len
        
        physics_targets_nvar, physics_targets_hybrid = None, None
        if has_physics:
            physics_pred = np.clip(self.domain_fn(data, dt), -CLIP, CLIP)
            if physics_pred.shape == data.shape:
                physics_targets_nvar = physics_pred[offset:-1]
                physics_targets_hybrid = physics_pred[offset + washout:-1]
                
        physics_weights = [0.0] + ([0.05, 0.1, 0.5, 1.0, 2.0, 5.0] if has_physics else [])
        best_error, best_config = float('inf'), None
        
        for lb in range(-8, -2):
            beta = 10.0 ** lb
            for lp in physics_weights:
                try:
                    pt_nvar = physics_targets_nvar[:split_nvar] if has_physics else None
                    W_temp = _solve_ridge_regression(features_nvar[:split_nvar], targets_nvar[:split_nvar], beta, lp, pt_nvar)
                    error = _validation_error(data[:-val_len], data[-val_len:], d, self.reservoir, W_temp,
                                              self.nvar_k, self.nvar_s, self.nvar_p, False)
                    if error < best_error:
                        best_error, best_config = error, ('nvar', beta, lp, False)
                except Exception:
                    pass
                    
                try:
                    pt_hybrid = physics_targets_hybrid[:split_hybrid] if has_physics else None
                    W_temp = _solve_ridge_regression(features_hybrid[:split_hybrid], targets_hybrid[:split_hybrid], beta, lp, pt_hybrid)
                    error = _validation_error(data[:-val_len], data[-val_len:], d, self.reservoir, W_temp,
                                              self.nvar_k, self.nvar_s, self.nvar_p, True)
                    if error < best_error:
                        best_error, best_config = error, ('hybrid', beta, lp, True)
                except Exception:
                    pass
                    
        if best_config is None:
            best_config = ('nvar', 1e-5, 0.0, False)
            
        tier, beta, lp, use_reservoir = best_config
        if tier == 'nvar':
            self.weights = _solve_ridge_regression(features_nvar, targets_nvar, beta, lp, physics_targets_nvar)
        else:
            self.weights = _solve_ridge_regression(features_hybrid, targets_hybrid, beta, lp, physics_targets_hybrid)
            
        self._data = data
        self._use_reservoir = use_reservoir
        self._tier = tier
        self._lp = lp
        self._beta = beta
        
        if verbose:
            lp_str = f"lp={lp:.2f}, " if has_physics else ""
            print(f"  PIRC [{tier}]: {lp_str}beta={beta:.0e}, val={best_error:.6f}")
        self.train_time = time.time() - t0

    def predict(self, n_steps):
        return _rollout(self._data, n_steps, self.dim, self.reservoir, self.weights,
                        self.nvar_k, self.nvar_s, self.nvar_p, self._use_reservoir)

    def nparams(self):
        return self.weights.size