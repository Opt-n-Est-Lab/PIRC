import numpy as np
from reservoir import Reservoir, nonlinear_transform
import time


class StandardRC:
    def __init__(self, dim, N=300, input_noise=1e-3, seed=42):
        self.reservoir = Reservoir(dim, N, seed=seed)
        self.dim, self.N = dim, N
        self.input_noise = input_noise
        self.seed = seed

    def train(self, inputs, targets, washout=200):
        t0 = time.time()
        T, d = inputs.shape
        rng = np.random.RandomState(self.seed + 7777)
        inputs_noisy = inputs + self.input_noise * rng.randn(T, d) if self.input_noise > 0 else inputs
        
        states, self._last_state = self.reservoir.run(inputs_noisy)
        states_transformed = nonlinear_transform(states)
        
        S_train = states_transformed[washout:]
        Y_train = targets[washout:]
        
        n_total = len(S_train)
        n_val = max(10, int(n_total * 0.15))
        n_tr = n_total - n_val
        
        S_tr, Y_tr = S_train[:n_tr], Y_train[:n_tr]
        S_val, Y_val = S_train[n_tr:], Y_train[n_tr:]
        
        A_tr = np.column_stack([S_tr, np.ones(len(S_tr))])
        A_val = np.column_stack([S_val, np.ones(len(S_val))])
        A_full = np.column_stack([S_train, np.ones(len(S_train))])
        
        best_mse = float('inf')
        best_alpha = 1e-4
        
        for lb in range(-8, -1):
            alpha = 10.0 ** lb
            try:
                W_temp = np.linalg.solve(A_tr.T @ A_tr + alpha * np.eye(A_tr.shape[1]), A_tr.T @ Y_tr)
                preds = A_val @ W_temp
                mse = np.mean((preds - Y_val) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            except np.linalg.LinAlgError:
                pass
                
        W = np.linalg.solve(A_full.T @ A_full + best_alpha * np.eye(A_full.shape[1]), A_full.T @ Y_train)
        self.W_out, self.b_out = W[:-1], W[-1]
        self.train_time = time.time() - t0

    def predict_autonomous(self, n_steps, u0):
        u, h = u0.copy(), self._last_state.copy()
        preds = []
        for _ in range(n_steps):
            h = self.reservoir.step(u, h)
            h_transformed = h.copy()
            h_transformed[1::2] = h[1::2] ** 2
            y = h_transformed @ self.W_out + self.b_out
            preds.append(y)
            u = y
        return np.array(preds)

    def nparams(self):
        return self.N * self.dim + self.dim


class StandardPINN:
    def __init__(self, dim, hidden=(128, 128), lam_phys=0.5, lr=5e-4, seed=42):
        self.dim, self.lam_phys, self.lr = dim, lam_phys, lr
        rng = np.random.RandomState(seed)
        sizes = [1] + list(hidden) + [dim]
        self.weights, self.biases = [], []
        self.n_layers = len(sizes) - 1
        
        for i in range(self.n_layers):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.weights.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.biases.append(np.zeros(sizes[i + 1]))

    def _forward(self, X):
        self.activations, self.pre_activations = [X], []
        h = X
        for i in range(self.n_layers - 1):
            z = h @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            h = np.tanh(z)
            self.activations.append(h)
        z = h @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(z)
        self.activations.append(z)
        return z

    def _backward(self, d_out):
        n = d_out.shape[0]
        dW, dB = [None] * self.n_layers, [None] * self.n_layers
        d = d_out
        
        dW[-1] = self.activations[-2].T @ d / n
        dB[-1] = d.mean(0)
        
        for i in range(self.n_layers - 2, -1, -1):
            d = (d @ self.weights[i + 1].T) * (1 - np.tanh(self.pre_activations[i]) ** 2)
            dW[i] = self.activations[i].T @ d / n
            dB[i] = d.mean(0)
            
        return dW, dB

    def train(self, t_data, y_data, norm_system, epochs=600, dt=0.02, verbose=True):
        t0 = time.time()
        X = t_data.reshape(-1, 1)
        
        for ep in range(epochs):
            y_pred = self._forward(X)
            loss_data = np.mean((y_pred - y_data) ** 2)
            
            dy = np.zeros_like(y_pred)
            dy[1:-1] = (y_pred[2:] - y_pred[:-2]) / (2 * dt)
            dy[0] = (y_pred[1] - y_pred[0]) / dt
            dy[-1] = (y_pred[-1] - y_pred[-2]) / dt
            
            loss_phys = np.mean((dy - norm_system.rhs(y_pred)) ** 2)
            
            dd_data = 2 * (y_pred - y_data) / y_pred.size
            
            eps = 1e-5
            dd_phys = np.zeros_like(y_pred)
            for d_idx in range(self.dim):
                y_plus = y_pred.copy()
                y_plus[:, d_idx] += eps
                dy_plus = np.zeros_like(y_plus)
                dy_plus[1:-1] = (y_plus[2:] - y_plus[:-2]) / (2 * dt)
                dy_plus[0] = (y_plus[1] - y_plus[0]) / dt
                dy_plus[-1] = (y_plus[-1] - y_plus[-2]) / dt
                loss_plus = np.mean((dy_plus - norm_system.rhs(y_plus)) ** 2)
                
                y_minus = y_pred.copy()
                y_minus[:, d_idx] -= eps
                dy_minus = np.zeros_like(y_minus)
                dy_minus[1:-1] = (y_minus[2:] - y_minus[:-2]) / (2 * dt)
                dy_minus[0] = (y_minus[1] - y_minus[0]) / dt
                dy_minus[-1] = (y_minus[-1] - y_minus[-2]) / dt
                loss_minus = np.mean((dy_minus - norm_system.rhs(y_minus)) ** 2)
                
                dd_phys[:, d_idx] = (loss_plus - loss_minus) / (2 * eps)
            
            dW, dB = self._backward(dd_data + self.lam_phys * dd_phys)
            
            for i in range(self.n_layers):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * dB[i]
                
            if verbose and (ep % 300 == 0 or ep == epochs - 1):
                print(f"    PINN Ep {ep:4d} | Data: {loss_data:.6f} | Phys: {loss_phys:.6f}")
                
        self.train_time = time.time() - t0

    def predict(self, X):
        return self._forward(X.reshape(-1, 1) if X.ndim == 1 else X)

    def nparams(self):
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))