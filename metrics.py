import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred, dt=0.02):
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    nrmse = rmse / (np.std(yt) + 1e-10)
    
    corrs = []
    for d in range(yt.shape[1]):
        if np.std(yp[:, d]) > 1e-10:
            corrs.append(np.corrcoef(yt[:, d], yp[:, d])[0, 1])
        else:
            corrs.append(0.0)
            
    return {'RMSE': rmse, 'NRMSE': nrmse, 'Corr': np.mean(corrs)}


def print_results_table(system_name, results):
    print(f"\n  {'Model':<18} {'RMSE':>10} {'Corr':>8} {'Params':>8} {'Time':>8}  ")
    print(f"  {'─' * 56}  ")
    for name, m in results.items():
        print(f"  {name:<18} {m['RMSE']:>10.3f} {m['Corr']:>8.3f}   "
              f"{m['params']:>8} {m['time']:>7.3f}s  ")


def _smart_bar_plot(ax, models, vals, colors, title, ylabel, fmt='.4f',
                    outlier_thresh=5.0, fontsize_label=11, fontsize_tick=11):
    vals_arr = np.array(vals, dtype=float)
    positive_vals = vals_arr[vals_arr > 0]
    
    has_outlier = False
    cap = None
    
    if len(positive_vals) >= 3:
        sorted_v = np.sort(positive_vals)
        non_max = sorted_v[:-1]
        if sorted_v[-1] > outlier_thresh * np.median(non_max) and np.median(non_max) > 0:
            has_outlier = True
            cap = np.max(non_max) * 1.8
    
    bars = ax.bar(models, vals, color=colors, alpha=0.85)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    if has_outlier:
        ax.set_ylim(0, cap)
        for b, v in zip(bars, vals):
            bx = b.get_x() + b.get_width() / 2
            if v > cap * 0.95:
                b.set_height(cap * 0.88)
                brk_y = cap * 0.86
                bw = b.get_width()
                for dy in [0, cap * 0.04]:
                    ax.plot([bx - bw * 0.35, bx + bw * 0.35],
                            [brk_y + dy - cap * 0.01, brk_y + dy + cap * 0.01],
                            color='white', lw=2.5, zorder=5)
                    ax.plot([bx - bw * 0.35, bx + bw * 0.35],
                            [brk_y + dy - cap * 0.01, brk_y + dy + cap * 0.01],
                            color='#333', lw=1.0, zorder=6)
                ax.text(bx, cap * 0.94, f'{v:{fmt}}',
                        ha='center', va='bottom', fontsize=fontsize_label,
                        fontweight='bold', color='#333')
            else:
                ax.text(bx, v + cap * 0.02, f'{v:{fmt}}',
                        ha='center', va='bottom', fontsize=fontsize_label)
    else:
        ymax = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, ymax * 1.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02,
                    f'{v:{fmt}}', ha='center', va='bottom',
                    fontsize=fontsize_label)
    
    ax.tick_params(axis='x', labelsize=fontsize_tick, rotation=25)
    ax.tick_params(axis='y', labelsize=9)


def plot_chaotic_results(all_results, save_path):
    colors = {
        'RC': '#3498DB', 'NVAR-only': '#9B59B6',
        'NGRC+RC': '#E67E22', 'PIRC': '#2ECC71',
        'PIRC (zero)': '#2ECC71', 'PIRC (partial)': '#27AE60',
        'PIRC (full)': '#1ABC9C', 'PINN': '#E74C3C',
    }
    
    n_sys = len(all_results)
    fig, axes = plt.subplots(n_sys, 3, figsize=(16, 4.5 * n_sys))
    if n_sys == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('PIRC: Benchmark Results', fontsize=15, fontweight='bold', y=1.02)
    
    for row, (sname, res) in enumerate(all_results.items()):
        models = list(res.keys())
        c = [colors.get(m, '#888') for m in models]
        
        vals = [res[m]['RMSE'] for m in models]
        _smart_bar_plot(axes[row, 0], models, vals, c,
                        f'{sname}: RMSE', 'RMSE', fmt='.4f')
        
        ax = axes[row, 1]
        vals = [res[m]['Corr'] for m in models]
        bars = ax.bar(models, vals, color=c, alpha=0.85)
        ax.set_title(f'{sname}: Correlation', fontweight='bold', fontsize=13)
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_ylim(min(min(vals) - 0.15, -0.3), 1.18)
        ax.tick_params(axis='x', labelsize=11, rotation=25)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        vals = [res[m]['params'] for m in models]
        _smart_bar_plot(axes[row, 2], models, vals, c,
                        f'{sname}: Parameters', 'Parameters', fmt=',')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()