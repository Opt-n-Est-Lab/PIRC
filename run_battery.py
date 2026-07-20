import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from scipy.stats import linregress

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from reservoir import Reservoir, nonlinear_transform, build_nvar

# ==============================================================================
# Configuration & Constants
# ==============================================================================
DATA_ROOT = "5_Battery_Data_Set"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

ELECTRICAL_FEATURES = [
    'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'current_mean', 'current_std', 'discharge_time',
    'voltage_at_50pct', 'voltage_slope',
]

THERMAL_FEATURES = [
    'temp_mean', 'temp_std', 'temp_min', 'temp_max',
    'temp_rise', 'temp_at_50pct', 'temp_slope',
]

IMPEDANCE_FEATURES = [
    'imp_re', 'imp_rct', 'imp_real_mean', 'imp_real_std',
    'imp_imag_mean', 'imp_imag_std', 'imp_mag_low_freq',
    'imp_phase_low_freq',
]

SENSOR_FEATURES = ELECTRICAL_FEATURES + THERMAL_FEATURES + IMPEDANCE_FEATURES
TARGET = 'capacity'
N_CAP_LAGS = 3


# ==============================================================================
# Data Processing
# ==============================================================================
def extract_electrical_features(data):
    try:
        v = data['Voltage_measured'][0, 0].flatten()
        c = data['Current_measured'][0, 0].flatten()
        t = data['Time'][0, 0].flatten()
        mi = np.argmin(np.abs(t - t[-1] / 2))
        sl = linregress(t, v).slope if len(t) > 2 else 0.0
        return {
            'voltage_mean': np.mean(v), 'voltage_std': np.std(v),
            'voltage_min': np.min(v), 'voltage_max': np.max(v),
            'current_mean': np.mean(c), 'current_std': np.std(c),
            'discharge_time': t[-1], 'voltage_at_50pct': v[mi], 'voltage_slope': sl
        }
    except Exception:
        return None

def extract_thermal_features(data):
    try:
        temp = data['Temperature_measured'][0, 0].flatten()
        t = data['Time'][0, 0].flatten()
        mi = np.argmin(np.abs(t - t[-1] / 2))
        sl = linregress(t, temp).slope if len(t) > 2 else 0.0
        return {
            'temp_mean': np.mean(temp), 'temp_std': np.std(temp),
            'temp_min': np.min(temp), 'temp_max': np.max(temp),
            'temp_rise': np.max(temp) - temp[0], 'temp_at_50pct': temp[mi], 'temp_slope': sl
        }
    except Exception:
        return None

def extract_impedance_features(data):
    try:
        re_v = float(data['Re'][0, 0][0, 0])
        rct = float(data['Rct'][0, 0][0, 0])
        z = data['Rectified_Impedance'][0, 0].flatten()
        zr, zi = np.real(z), np.imag(z)
        return {
            'imp_re': re_v, 'imp_rct': rct,
            'imp_real_mean': np.mean(zr), 'imp_real_std': np.std(zr),
            'imp_imag_mean': np.mean(zi), 'imp_imag_std': np.std(zi),
            'imp_mag_low_freq': np.abs(z[-1]), 'imp_phase_low_freq': np.angle(z[-1])
        }
    except Exception:
        return None

def load_nasa_battery_data(data_root):
    print(f"\nLoading data from: {data_root}")
    folders = {}
    for d in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, d)
        if os.path.isdir(full):
            mats = [f for f in os.listdir(full) if f.endswith('.mat')]
            if mats:
                folders[d] = full
    print(f"Found {len(folders)} data folders")

    all_samples, seen = [], {}
    for fn, fp in sorted(folders.items()):
        for mf in sorted(os.listdir(fp)):
            if not mf.endswith('.mat'):
                continue
            bid = mf.replace('.mat', '')
            try:
                mat = loadmat(os.path.join(fp, mf))
                cycles = mat[bid][0, 0]['cycle'][0]
            except Exception:
                continue

            if bid in seen and len(cycles) <= seen[bid]:
                continue
            seen[bid] = len(cycles)
            all_samples = [s for s in all_samples if s['battery_id'] != bid]

            imp_idx, imp_data = [], []
            for i in range(len(cycles)):
                try:
                    if str(cycles[i]['type'][0]) == 'impedance':
                        imp_idx.append(i)
                        imp_data.append(cycles[i]['data'])
                except Exception:
                    continue

            if not imp_idx:
                continue

            dc = 0
            for i in range(len(cycles)):
                try:
                    if str(cycles[i]['type'][0]) != 'discharge':
                        continue
                except Exception:
                    continue

                cd = cycles[i]['data']
                try:
                    cap = float(cd['Capacity'][0, 0][0, 0])
                except Exception:
                    continue

                if cap <= 0 or cap > 3.0:
                    continue

                elec = extract_electrical_features(cd)
                therm = extract_thermal_features(cd)
                if elec is None or therm is None:
                    continue

                ii = np.argmin(np.abs(np.array(imp_idx) - i))
                imp = extract_impedance_features(imp_data[ii])
                if imp is None:
                    continue

                dc += 1
                try:
                    amb = float(cycles[i]['ambient_temperature'][0][0])
                except Exception:
                    amb = 24.0

                all_samples.append({
                    'battery_id': bid, 'cycle_index': dc, 'ambient_temp': amb,
                    'capacity': cap, **elec, **therm, **imp
                })

    df = pd.DataFrame(all_samples)
    print(f"Dataset: {df.shape[0]} samples, {df['battery_id'].nunique()} batteries")
    return df

def add_capacity_lags(df, n_lags=3):
    for lag in range(1, n_lags + 1):
        df[f'cap_lag_{lag}'] = df.groupby('battery_id')['capacity'].shift(lag)
    return df.dropna().reset_index(drop=True)


# ==============================================================================
# Math & Metric Helpers
# ==============================================================================
def _solve(Xi, Y, beta, lp=0.0, Yp=None):
    q = Xi.shape[1]
    if lp > 0 and Yp is not None:
        gram = (1 + lp) * Xi.T @ Xi + beta * np.eye(q)
        rhs = Xi.T @ (Y + lp * Yp)
    else:
        gram = Xi.T @ Xi + beta * np.eye(q)
        rhs = Xi.T @ Y
    return np.linalg.solve(gram, rhs)

def _expert_trend(y, off):
    T = len(y)
    expert = np.zeros((T - off - 1, 1))
    for i in range(len(expert)):
        idx = off + i
        if idx > 5:
            w = min(20, idx)
            expert[i] = y[idx] + (y[idx] - y[idx - w]) / w
        else:
            expert[i] = y[idx]
    return expert

def battery_metrics(yt, yp):
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'corr': np.nan}
    return {
        'rmse': np.sqrt(np.mean((yt - yp) ** 2)),
        'mae': np.mean(np.abs(yt - yp)),
        'mape': np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-6))) * 100,
        'corr': np.corrcoef(yt, yp)[0, 1] if np.std(yp) > 1e-10 else 0.0
    }

def evaluate_predictions(y_test, pred, offset=0):
    y_al = y_test[offset + 1:]
    nc = min(len(y_al), len(pred))
    return np.sqrt(np.mean((y_al[:nc] - pred[:nc]) ** 2))

def _measure_inference_time(predict_fn, n_repeats=50):
    predict_fn()
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times)


# ==============================================================================
# Model Definitions
# ==============================================================================
class BatteryRC:
    def __init__(self, feat_dim, N=200, seed=42):
        self.res = Reservoir(feat_dim, N, input_scaling=0.3, seed=seed)
        self._N = N

    def train(self, X, y):
        t0 = time.time()
        S, _ = self.res.run(X)
        Sg = nonlinear_transform(S)
        wo = min(20, len(X) // 5)
        Sg_al = Sg[wo:]
        Y_al = y[wo:]
        A = np.column_stack([Sg_al, np.ones(len(Sg_al))])
        
        n_total = len(A)
        n_val = max(10, int(n_total * 0.15))
        n_tr = n_total - n_val
        
        best_mse, best_alpha = float('inf'), 1e-4
        for lb in range(-8, -1):
            alpha = 10.0 ** lb
            try:
                W_temp = np.linalg.solve(A[:n_tr].T @ A[:n_tr] + alpha * np.eye(A.shape[1]), A[:n_tr].T @ Y_al[:n_tr])
                preds = A[n_tr:] @ W_temp
                mse = np.mean((preds - Y_al[n_tr:]) ** 2)
                if mse < best_mse:
                    best_mse, best_alpha = mse, alpha
            except Exception:
                pass
                
        W = np.linalg.solve(A.T @ A + best_alpha * np.eye(A.shape[1]), A.T @ Y_al)
        self.Wo, self.bo = W[:-1], W[-1]
        self.train_time = time.time() - t0

    def predict(self, X):
        S, _ = self.res.run(X)
        return nonlinear_transform(S) @ self.Wo + self.bo

    def nparams(self):
        return self._N + 1


class BatteryNGRCRC:
    def __init__(self, feat_dim, res_N=100, nvar_k=3, nvar_s=1, nvar_p=2, seed=42):
        self.feat_dim, self.res_N = feat_dim, res_N
        self.nvar_k, self.nvar_s, self.nvar_p = nvar_k, nvar_s, nvar_p
        self.seed = seed

    def train(self, X, y, verbose=True, force_beta=None):
        t0 = time.time()
        self.res = Reservoir(self.feat_dim, self.res_N, input_scaling=0.3, seed=self.seed)
        cap = y.reshape(-1, 1)
        nvar_feat, off = build_nvar(cap, self.nvar_k, self.nvar_s, self.nvar_p)
        nvar_feat = nvar_feat[:-1]
        states, _ = self.res.run(X)
        res_g = nonlinear_transform(states)
        res_aligned = res_g[off:-1]
        Xi = np.column_stack([res_aligned, nvar_feat])
        targets = cap[off + 1:]
        
        self._off = off
        self._n_feat = Xi.shape[1]
        
        if force_beta is not None:
            best_b = force_beta
        else:
            n_total = len(targets)
            n_val = max(10, int(n_total * 0.15))
            n_tr = n_total - n_val
            best_val, best_b = float('inf'), 1e-5
            for lb in range(-8, -2):
                b = 10.0 ** lb
                try:
                    W = _solve(Xi[:n_tr], targets[:n_tr], b)
                    val_err = np.mean((Xi[n_tr:] @ W - targets[n_tr:]) ** 2)
                    if val_err < best_val:
                        best_val, best_b = val_err, b
                except Exception:
                    pass
                    
        self.Wout = _solve(Xi, targets, best_b)
        self._beta = best_b
        if verbose:
            print(f"Hybrid: {self._n_feat} feats, beta={best_b:.0e}")
        self.train_time = time.time() - t0

    def predict(self, X, y_prev):
        cap = y_prev.reshape(-1, 1)
        nvar_feat, off = build_nvar(cap, self.nvar_k, self.nvar_s, self.nvar_p)
        nvar_feat = nvar_feat[:-1]
        states, _ = self.res.run(X)
        res_g = nonlinear_transform(states)
        res_aligned = res_g[off:-1]
        H = np.column_stack([res_aligned, nvar_feat])
        return H @ self.Wout, off

    def nparams(self):
        return self._n_feat


class BatteryPIRC:
    def __init__(self, feat_dim, res_N=100, nvar_k=3, nvar_s=1, nvar_p=2, seed=42):
        self.feat_dim, self.res_N = feat_dim, res_N
        self.nvar_k, self.nvar_s, self.nvar_p = nvar_k, nvar_s, nvar_p
        self.seed = seed

    def _build_features(self, X_sensor, y_cap):
        cap = y_cap.reshape(-1, 1)
        nvar_feat, off = build_nvar(cap, self.nvar_k, self.nvar_s, self.nvar_p)
        nvar_feat = nvar_feat[:-1]
        states, _ = self.res.run(X_sensor)
        res_g = nonlinear_transform(states)
        res_aligned = res_g[off:-1]
        targets = cap[off + 1:]
        Xi_nvar = nvar_feat
        Xi_hybrid = np.column_stack([res_aligned, nvar_feat])
        return Xi_nvar, Xi_hybrid, targets, off

    def train(self, X, y, verbose=True, force_cfg=None):
        t0 = time.time()
        self.res = Reservoir(self.feat_dim, self.res_N, input_scaling=0.3, seed=self.seed)
        Xi_nvar, Xi_hybrid, targets, off = self._build_features(X, y)
        self._off = off
        Y_phys = _expert_trend(y, off)

        if force_cfg is not None:
            tier, beta, lp, use_rc, nf = force_cfg
            best_cfg = (tier, beta, lp, nf, use_rc)
        else:
            n_total = len(targets)
            n_val = max(10, int(n_total * 0.15))
            n_tr = n_total - n_val
            Xi_nvar_tr, Xi_nvar_val = Xi_nvar[:n_tr], Xi_nvar[n_tr:]
            Xi_hyb_tr, Xi_hyb_val = Xi_hybrid[:n_tr], Xi_hybrid[n_tr:]
            Y_tr, Y_val = targets[:n_tr], targets[n_tr:]
            Yp_tr = Y_phys[:n_tr]
            
            best_val_err, best_cfg = float('inf'), None
            for lb in range(-8, -2):
                b = 10.0 ** lb
                for lp in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
                    try:
                        W = _solve(Xi_nvar_tr, Y_tr, b, lp, Yp_tr)
                        val_err = np.mean((Xi_nvar_val @ W - Y_val) ** 2)
                        if val_err < best_val_err:
                            best_val_err = val_err
                            best_cfg = ('nvar', b, lp, Xi_nvar.shape[1], False)
                    except Exception:
                        pass
                    try:
                        W = _solve(Xi_hyb_tr, Y_tr, b, lp, Yp_tr)
                        val_err = np.mean((Xi_hyb_val @ W - Y_val) ** 2)
                        if val_err < best_val_err:
                            best_val_err = val_err
                            best_cfg = ('hybrid', b, lp, Xi_hybrid.shape[1], True)
                    except Exception:
                        pass
                        
            if best_cfg is None:
                best_cfg = ('nvar', 1e-4, 0.0, Xi_nvar.shape[1], False)

        tier, beta, lp, nf, use_rc = best_cfg
        Xi_full = Xi_hybrid if use_rc else Xi_nvar
        self.Wout = _solve(Xi_full, targets, beta, lp, Y_phys)
        
        self._use_rc = use_rc
        self._tier = tier
        self._lp, self._beta = lp, beta
        self._n_feat = nf
        
        if verbose:
            print(f"PIRC [{tier}]: {nf} feats, lp={lp:.1f}, beta={beta:.0e}")
        self.train_time = time.time() - t0

    def predict(self, X, y_prev):
        cap = y_prev.reshape(-1, 1)
        nvar_feat, off = build_nvar(cap, self.nvar_k, self.nvar_s, self.nvar_p)
        nvar_feat = nvar_feat[:-1]
        if self._use_rc:
            states, _ = self.res.run(X)
            res_g = nonlinear_transform(states)
            res_aligned = res_g[off:-1]
            H = np.column_stack([res_aligned, nvar_feat])
        else:
            H = nvar_feat
        return H @ self.Wout, off

    def nparams(self):
        return self._n_feat


class BatteryPINN:
    def __init__(self, feat_dim, hidden=(64, 32), lr=5e-4, lam_phys=0.5, seed=42):
        self.feat_dim, self.lr, self.lam = feat_dim, lr, lam_phys
        rng = np.random.RandomState(seed)
        sizes = [feat_dim] + list(hidden) + [1]
        self.W, self.B, self.nL = [], [], len(sizes) - 1
        for i in range(self.nL):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.B.append(np.zeros(sizes[i + 1]))

    def _fwd(self, X):
        h = X
        for i in range(self.nL - 1):
            h = np.tanh(h @ self.W[i] + self.B[i])
        return h @ self.W[-1] + self.B[-1]

    def train(self, X, y, epochs=200):
        t0 = time.time()
        Y = y.reshape(-1, 1)
        Y_phys = np.zeros_like(Y)
        for i in range(len(y)):
            if i > 5:
                w = min(20, i)
                Y_phys[i, 0] = y[i] + (y[i] - y[i - w]) / w
            else:
                Y_phys[i, 0] = y[i]

        for ep in range(epochs):
            pred = self._fwd(X)
            dl = np.mean((pred - Y) ** 2)
            if np.isnan(dl) or dl > 1e6:
                break
                
            dd_data = 2 * (pred - Y) / pred.size
            dd_phys = 2 * (pred - Y_phys) / pred.size
            d = dd_data + self.lam * dd_phys
            n = d.shape[0]
            
            acts, pres = [X], []
            h = X
            for i in range(self.nL - 1):
                z = h @ self.W[i] + self.B[i]
                pres.append(z)
                h = np.tanh(z)
                acts.append(h)
            z = h @ self.W[-1] + self.B[-1]
            pres.append(z)
            acts.append(z)
            
            dW = [None] * self.nL
            dB = [None] * self.nL
            dW[-1] = acts[-2].T @ d / n
            dB[-1] = d.mean(0)
            
            for i in range(self.nL - 2, -1, -1):
                d = (d @ self.W[i + 1].T) * (1 - np.tanh(pres[i]) ** 2)
                dW[i] = acts[i].T @ d / n
                dB[i] = d.mean(0)
                
            for i in range(self.nL):
                self.W[i] -= self.lr * dW[i]
                self.B[i] -= self.lr * dB[i]
                
        self.train_time = time.time() - t0

    def predict(self, X):
        pred = self._fwd(X).ravel()
        if np.any(np.isnan(pred)) or np.max(np.abs(pred)) > 100:
            return np.full(len(X), np.nan)
        return pred

    def nparams(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.B))


# ==============================================================================
# Plotting Utilities
# ==============================================================================
def plot_capacity_curves(test_bids, all_results, df, save_path):
    n = len(test_bids)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    colors = {'RC': '#3498DB', 'NGRC+RC': '#E67E22', 'PIRC': '#2ECC71', 'PINN': '#E74C3C'}

    for i, bid in enumerate(test_bids):
        ax = axes[i]
        r = all_results[bid]
        bdf = df[df['battery_id'] == bid].sort_values('cycle_index')
        cycles = bdf['cycle_index'].values
        y_true = bdf['capacity'].values

        ax.plot(cycles, y_true, 'k-o', ms=3, lw=3.5, label='Ground truth', zorder=5)
        for model, c in colors.items():
            if model not in r or 'pred' not in r[model]:
                continue
            yp = r[model]['pred']
            off = r[model].get('offset', 0)
            start_idx = off + 1
            end_idx = start_idx + len(yp)
            if end_idx > len(cycles):
                end_idx = len(cycles)
            yp = yp[:end_idx - start_idx]
            cx = cycles[start_idx:end_idx]
            nc = min(len(cx), len(yp))
            ax.plot(cx[:nc], yp[:nc], '-', color=c, lw=3.2, label=f'{model} (RMSE={r[model]["rmse"]:.3f})')

        temp = bdf['ambient_temp'].iloc[0]
        ax.set_title(f'Battery {bid} — {len(y_true)} cycles, T={temp:.0f}°C', fontweight='bold', fontsize=13)
        ax.set_ylabel('Capacity (Ah)', fontsize=14)
        ax.legend(fontsize=14, loc='best')
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Cycle', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_combined_figure(test_bids, all_results, overall, df, save_path):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.20)
    models = ['RC', 'NGRC+RC', 'PIRC', 'PINN']
    colors = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C']

    def draw_break(ax, x, y_base, width, cap_val):
        bw = width * 0.35
        offset = cap_val * 0.015
        ax.plot([x - bw, x + bw], [y_base, y_base + offset], color='black', lw=1.5, zorder=5)
        ax.plot([x - bw, x + bw], [y_base + offset*2, y_base + offset*3], color='black', lw=1.5, zorder=5)

    ax = fig.add_subplot(gs[0, 0])
    ax.grid(True, alpha=0.3)
    means = [np.mean(overall[m]) for m in models]
    cap_rmse = 0.30
    ax.set_ylim(0, cap_rmse * 1.15)
    for i, (m, c) in enumerate(zip(models, colors)):
        val = means[i]
        if val > cap_rmse:
            ax.bar(i, cap_rmse * 0.85, color=c, alpha=0.85)
            draw_break(ax, i, cap_rmse * 0.82, 0.8, cap_rmse)
            ax.text(i, cap_rmse * 0.95, f'{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        else:
            ax.bar(i, val, color=c, alpha=0.85)
            ax.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=14)
    ax.set_title('Mean RMSE (Ah)', fontweight='bold', fontsize=14)
    ax.set_ylabel('RMSE (Ah)', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    ax = fig.add_subplot(gs[0, 1])
    ax.grid(True, alpha=0.3)
    params = [all_results[test_bids[0]][m]['params'] for m in models]
    cap_params = 250
    ax.set_ylim(0, cap_params * 1.15)
    for i, (m, c) in enumerate(zip(models, colors)):
        val = params[i]
        if val > cap_params:
            ax.bar(i, cap_params * 0.85, color=c, alpha=0.85)
            draw_break(ax, i, cap_params * 0.82, 0.8, cap_params)
            ax.text(i, cap_params * 0.95, f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        else:
            ax.bar(i, val, color=c, alpha=0.85)
            ax.text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=14)
    ax.set_title('Trainable Parameters', fontweight='bold', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    ax = fig.add_subplot(gs[1, 0])
    ax.grid(True, alpha=0.3)
    ratios_rc = [all_results[b]['RC']['rmse'] / max(all_results[b]['PIRC']['rmse'], 1e-6) for b in test_bids]
    ratios_hyb = [all_results[b]['NGRC+RC']['rmse'] / max(all_results[b]['PIRC']['rmse'], 1e-6) for b in test_bids]
    ratios_pinn = [all_results[b]['PINN']['rmse'] / max(all_results[b]['PIRC']['rmse'], 1e-6) for b in test_bids]
    x = np.arange(len(test_bids))
    w = 0.23
    cap_ratio = 10.0
    ax.set_ylim(0, cap_ratio * 1.15)
    ax.set_yticks([0, 5, 10])
    all_ratios = [ratios_rc, ratios_hyb, ratios_pinn]
    labels = ['RC/PIRC', 'Hybrid/PIRC', 'PINN/PIRC']
    bar_colors = ['#3498DB', '#E67E22', '#E74C3C']
    for j, (ratios, label, c) in enumerate(zip(all_ratios, labels, bar_colors)):
        for i, val in enumerate(ratios):
            if val > cap_ratio:
                ax.bar(x[i] + (j - 1) * w, cap_ratio * 0.85, w, color=c, alpha=0.85)
                draw_break(ax, x[i] + (j - 1) * w, cap_ratio * 0.82, w, cap_ratio)
                ax.text(x[i] + (j - 1) * w, cap_ratio * 0.95, f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            else:
                ax.bar(x[i] + (j - 1) * w, val, w, color=c, alpha=0.85, label=label if i == 0 else "")
    ax.set_xticks(x)
    ax.set_xticklabels(test_bids, fontsize=14)
    ax.set_title('PIRC Improvement Ratio', fontweight='bold', fontsize=14)
    ax.set_ylabel('RMSE Ratio', fontsize=14)
    ax.axhline(y=1, color='gray', lw=0.5, ls='--')
    ax.legend(fontsize=17, loc='best', framealpha=0.9)
    ax.tick_params(axis='y', labelsize=14)

    ax = fig.add_subplot(gs[1, 1])
    for j, (m, c) in enumerate(zip(models, colors)):
        rmses = [(df[df['battery_id'] == b]['ambient_temp'].iloc[0], all_results[b][m]['rmse']) for b in test_bids]
        rmses.sort()
        ax.plot([r[0] for r in rmses], [r[1] for r in rmses], 'o-', color=c, label=m, ms=6, lw=3.9)
    ax.set_xlim(2, 45)
    ax.set_xlabel('Temperature (°C)', fontsize=14)
    ax.set_ylabel('RMSE (Ah)', fontsize=14)
    ax.set_title('RMSE vs Ambient Temp', fontweight='bold', fontsize=14)
    ax.legend(fontsize=17, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main Benchmark Execution
# ==============================================================================
def run_battery_benchmark(df):
    print(f"\n{'=' * 65}")
    print("PIRC: BATTERY SOH BENCHMARK")
    print(f"{'=' * 65}")

    df = add_capacity_lags(df, N_CAP_LAGS)
    cap_lag_cols = [f'cap_lag_{i}' for i in range(1, N_CAP_LAGS + 1)]
    ALL_FEATURES = SENSOR_FEATURES + cap_lag_cols
    feat_dim = len(ALL_FEATURES)

    battery_ids = sorted(df['battery_id'].unique())
    temp_groups = df.groupby(df['ambient_temp'].round())

    test_bids, val_bids = [], []
    for temp, group in temp_groups:
        bids = sorted(group['battery_id'].unique().tolist())
        if len(bids) > 1:
            test_bids.append(bids[-1])
        if len(bids) > 2:
            val_bids.append(bids[-2])

    test_bids = list(dict.fromkeys(test_bids))
    val_bids = list(dict.fromkeys(val_bids))
    train_bids = [b for b in battery_ids if b not in test_bids and b not in val_bids]

    if not val_bids and len(train_bids) > 0:
        val_bids = [train_bids.pop()]

    train_df = df[df['battery_id'].isin(train_bids)].sort_values(['battery_id', 'cycle_index'])
    X_train = train_df[ALL_FEATURES].values
    y_train = train_df[TARGET].values

    mu_x, sig_x = X_train.mean(0), X_train.std(0) + 1e-8
    mu_y, sig_y = y_train.mean(), y_train.std() + 1e-8
    X_train_n = (X_train - mu_x) / sig_x
    y_train_n = (y_train - mu_y) / sig_y

    def evaluate_on_bids(model, bids):
        total_rmse = 0
        for bid in bids:
            tdf = df[df['battery_id'] == bid].sort_values('cycle_index')
            X_eval_n = (tdf[ALL_FEATURES].values - mu_x) / sig_x
            y_eval = tdf[TARGET].values
            y_eval_n = (y_eval - mu_y) / sig_y
            pred_n, off = model.predict(X_eval_n, y_eval_n)
            pred = pred_n.ravel() * sig_y + mu_y
            total_rmse += evaluate_predictions(y_eval, pred, off)
        return total_rmse

    print(f"Train: {len(train_bids)} batteries, {len(train_df)} samples")
    print(f"Val:   {len(val_bids)} batteries: {val_bids}")
    print(f"Test:  {len(test_bids)} batteries: {test_bids}")
    print(f"Features: {feat_dim} ({len(SENSOR_FEATURES)} sensor + {N_CAP_LAGS} cap lags)")

    print("\nTuning PIRC on validation batteries...")
    best_pirc_val_rmse = float('inf')
    best_pirc_config = None
    for res_N in [50, 100, 200]:
        for seed in range(5):
            pirc = BatteryPIRC(feat_dim, res_N=res_N, nvar_k=3, nvar_s=1, nvar_p=2, seed=seed)
            pirc.train(X_train_n, y_train_n, verbose=False)
            val_rmse = evaluate_on_bids(pirc, val_bids)
            if val_rmse < best_pirc_val_rmse:
                best_pirc_val_rmse = val_rmse
                best_pirc_config = (res_N, seed, (pirc._tier, pirc._beta, pirc._lp, pirc._use_rc, pirc._n_feat))

    print("Tuning NGRC+RC on validation batteries...")
    best_ngrcrc_val_rmse = float('inf')
    best_ngrcrc_config = None
    for res_N in [50, 100, 200]:
        for seed in range(5):
            ngrcrc = BatteryNGRCRC(feat_dim, res_N=res_N, nvar_k=3, nvar_s=1, nvar_p=2, seed=seed)
            ngrcrc.train(X_train_n, y_train_n, verbose=False)
            val_rmse = evaluate_on_bids(ngrcrc, val_bids)
            if val_rmse < best_ngrcrc_val_rmse:
                best_ngrcrc_val_rmse = val_rmse
                best_ngrcrc_config = (res_N, seed, ngrcrc._beta)

    train_val_bids = train_bids + val_bids
    tv_df = df[df['battery_id'].isin(train_val_bids)].sort_values(['battery_id', 'cycle_index'])
    X_tv = tv_df[ALL_FEATURES].values
    y_tv = tv_df[TARGET].values
    X_tv_n = (X_tv - mu_x) / sig_x
    y_tv_n = (y_tv - mu_y) / sig_y

    rN, sd, f_cfg = best_pirc_config
    best_pirc_model = BatteryPIRC(feat_dim, res_N=rN, nvar_k=3, nvar_s=1, nvar_p=2, seed=sd)
    best_pirc_model.train(X_tv_n, y_tv_n, verbose=False, force_cfg=f_cfg)
    print(f"Best PIRC: res_N={rN}, seed={sd}, tier={f_cfg[0]}, lp={f_cfg[2]:.1f}, params={f_cfg[4]}")

    rN_c, sd_c, best_b = best_ngrcrc_config
    best_ngrcrc_model = BatteryNGRCRC(feat_dim, res_N=rN_c, nvar_k=3, nvar_s=1, nvar_p=2, seed=sd_c)
    best_ngrcrc_model.train(X_tv_n, y_tv_n, verbose=False, force_beta=best_b)
    print(f"Best NGRC+RC: res_N={rN_c}, seed={sd_c}, params={best_ngrcrc_model.nparams()}")

    print("\nMeasuring training times...")
    train_times = {}
    
    rc_timer = BatteryRC(feat_dim, N=200)
    rc_timer.train(X_tv_n, y_tv_n)
    train_times['RC'] = rc_timer.train_time
    
    ngrcrc_timer = BatteryNGRCRC(feat_dim, res_N=rN_c, nvar_k=3, nvar_s=1, nvar_p=2, seed=sd_c)
    ngrcrc_timer.train(X_tv_n, y_tv_n, verbose=False, force_beta=best_b)
    train_times['NGRC+RC'] = ngrcrc_timer.train_time
    
    pirc_timer = BatteryPIRC(feat_dim, res_N=rN, nvar_k=3, nvar_s=1, nvar_p=2, seed=sd)
    pirc_timer.train(X_tv_n, y_tv_n, verbose=False, force_cfg=f_cfg)
    train_times['PIRC'] = pirc_timer.train_time
    
    pinn_timer = BatteryPINN(feat_dim, (64, 32))
    pinn_timer.train(X_tv_n, y_tv_n)
    train_times['PINN'] = pinn_timer.train_time
    
    for m_name, tt in train_times.items():
        print(f"  {m_name:<8} train time: {tt:.4f}s")

    all_results = {}
    overall = {'RC': [], 'NGRC+RC': [], 'PIRC': [], 'PINN': []}

    for bid in test_bids:
        tdf = df[df['battery_id'] == bid].sort_values('cycle_index')
        X_test = tdf[ALL_FEATURES].values
        y_test = tdf[TARGET].values
        X_test_n = (X_test - mu_x) / sig_x
        y_test_n = (y_test - mu_y) / sig_y
        temp = tdf['ambient_temp'].iloc[0]
        n_cyc = len(y_test)
        n_samples = len(y_test)

        print(f"\n{'─' * 55}")
        print(f"Battery {bid}: {n_cyc} cycles, T={temp:.0f}°C, C: {y_test.min():.3f}→{y_test.max():.3f} Ah")
        res = {}

        # RC
        rc = BatteryRC(feat_dim, N=200)
        rc.train(X_tv_n, y_tv_n)
        infer_time_rc = _measure_inference_time(lambda: rc.predict(X_test_n))
        rc_pred = rc.predict(X_test_n) * sig_y + mu_y
        m = battery_metrics(y_test, rc_pred)
        m.update({'params': rc.nparams(), 'pred': rc_pred, 'offset': 0, 'temp': temp, 'cycles': n_cyc, 
                  'train_time': train_times['RC'], 'infer_time': infer_time_rc, 'infer_per_sample': infer_time_rc / n_samples})
        res['RC'] = m
        overall['RC'].append(m['rmse'])

        # NGRC+RC
        infer_time_ngrcrc = _measure_inference_time(lambda: best_ngrcrc_model.predict(X_test_n, y_test_n))
        ngrcrc_pred_n, ngrcrc_off = best_ngrcrc_model.predict(X_test_n, y_test_n)
        ngrcrc_pred = ngrcrc_pred_n.ravel() * sig_y + mu_y
        y_ngrcrc_al = y_test[ngrcrc_off + 1:]
        nc_nr = min(len(y_ngrcrc_al), len(ngrcrc_pred))
        m = battery_metrics(y_ngrcrc_al[:nc_nr], ngrcrc_pred[:nc_nr])
        m.update({'params': best_ngrcrc_model.nparams(), 'pred': ngrcrc_pred, 'offset': ngrcrc_off, 'temp': temp, 'cycles': n_cyc,
                  'train_time': train_times['NGRC+RC'], 'infer_time': infer_time_ngrcrc, 'infer_per_sample': infer_time_ngrcrc / n_samples})
        res['NGRC+RC'] = m
        overall['NGRC+RC'].append(m['rmse'])

        # PIRC
        infer_time_pirc = _measure_inference_time(lambda: best_pirc_model.predict(X_test_n, y_test_n))
        pirc_pred_n, pirc_off = best_pirc_model.predict(X_test_n, y_test_n)
        pirc_pred = pirc_pred_n.ravel() * sig_y + mu_y
        y_aligned = y_test[pirc_off + 1:]
        nc = min(len(y_aligned), len(pirc_pred))
        m = battery_metrics(y_aligned[:nc], pirc_pred[:nc])
        m.update({'params': best_pirc_model.nparams(), 'pred': pirc_pred, 'offset': pirc_off, 'temp': temp, 'cycles': n_cyc,
                  'train_time': train_times['PIRC'], 'infer_time': infer_time_pirc, 'infer_per_sample': infer_time_pirc / n_samples})
        res['PIRC'] = m
        overall['PIRC'].append(m['rmse'])

        # PINN
        pinn = BatteryPINN(feat_dim, (64, 32))
        pinn.train(X_tv_n, y_tv_n)
        infer_time_pinn = _measure_inference_time(lambda: pinn.predict(X_test_n))
        pinn_pred = pinn.predict(X_test_n) * sig_y + mu_y
        if np.any(np.isnan(pinn_pred)):
            A = np.column_stack([X_tv_n, np.ones(len(X_tv_n))])
            w = np.linalg.lstsq(A, y_tv_n, rcond=None)[0]
            pinn_pred = (np.column_stack([X_test_n, np.ones(len(X_test_n))]) @ w) * sig_y + mu_y
        m = battery_metrics(y_test, pinn_pred)
        m.update({'params': pinn.nparams(), 'pred': pinn_pred, 'offset': 0, 'temp': temp, 'cycles': n_cyc,
                  'train_time': train_times['PINN'], 'infer_time': infer_time_pinn, 'infer_per_sample': infer_time_pinn / n_samples})
        res['PINN'] = m
        overall['PINN'].append(m['rmse'])

        for model in ['RC', 'NGRC+RC', 'PIRC', 'PINN']:
            r = res[model]
            print(f"  {model:<7} RMSE={r['rmse']:.4f} MAE={r['mae']:.4f} MAPE={min(r['mape'], 999):.1f}% "
                  f"Corr={r['corr']:.3f} Params={r['params']} Infer={r['infer_time']*1000:.3f}ms ({r['infer_per_sample']*1e6:.1f}µs/sample)")
        all_results[bid] = res

    print(f"\n{'=' * 65}")
    print(f"OVERALL (mean ± std across {len(test_bids)} test batteries)")
    print(f"{'=' * 65}")
    print(f"{'Model':<8} {'Mean RMSE':>12} {'Mean MAE':>12} {'Mean Corr':>12} {'Params':>8} {'Train(s)':>10} {'Infer(ms)':>10} {'µs/sample':>10}")
    print(f"{'─' * 86}")
    
    for model in ['RC', 'NGRC+RC', 'PIRC', 'PINN']:
        rmses = [all_results[b][model]['rmse'] for b in test_bids]
        maes = [all_results[b][model]['mae'] for b in test_bids]
        corrs = [all_results[b][model]['corr'] for b in test_bids]
        p = all_results[test_bids[0]][model]['params']
        tt = train_times[model]
        infer_ms = np.mean([all_results[b][model]['infer_time'] * 1000 for b in test_bids])
        us_per_sample = np.mean([all_results[b][model]['infer_per_sample'] * 1e6 for b in test_bids])
        print(f"{model:<8} {np.mean(rmses):.4f}±{np.std(rmses):.4f} {np.mean(maes):.4f}±{np.std(maes):.4f} {np.mean(corrs):>8.4f} {p:>8} {tt:>9.4f}s {infer_ms:>9.3f}ms {us_per_sample:>9.1f}µs")

    print("\nGenerating plots...")
    plot_capacity_curves(test_bids, all_results, df, os.path.join(SAVE_DIR, "battery_capacity_curves.png"))
    plot_combined_figure(test_bids, all_results, overall, df, os.path.join(SAVE_DIR, "battery_combined_figure.png"))
    
    return all_results, overall


if __name__ == "__main__":
    print("=" * 65)
    print("PIRC: Battery SOH - NASA PCoE Dataset")
    print("=" * 65)

    if not os.path.isdir(DATA_ROOT):
        print(f"\nDATA_ROOT = '{DATA_ROOT}' not found.")
        print("Please place the NASA PCoE battery dataset folder here.")
        sys.exit(1)

    df = load_nasa_battery_data(DATA_ROOT)
    run_battery_benchmark(df)