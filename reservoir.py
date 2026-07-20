import numpy as np
from itertools import combinations_with_replacement


class Reservoir:
    def __init__(self, input_dim, n_nodes=100, spectral_radius=0.9,
                 input_scaling=0.5, leak_rate=0.3, seed=42):
        self.n_nodes = n_nodes
        self.leak_rate = leak_rate
        
        rng = np.random.RandomState(seed)
        
        # Input weights (sparse random projection)
        self.W_in = np.zeros((n_nodes, input_dim))
        for i in range(n_nodes):
            j = rng.randint(0, input_dim)
            self.W_in[i, j] = rng.uniform(-input_scaling, input_scaling)
            
        # Reservoir weights (sparse Erdos-Renyi graph)
        W = rng.randn(n_nodes, n_nodes) * (rng.rand(n_nodes, n_nodes) < 3.0 / n_nodes)
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0:
            W *= spectral_radius / eig_max
        self.W_res = W
        
        self.bias = rng.uniform(-0.1, 0.1, n_nodes)

    def run(self, inputs):
        T = len(inputs)
        states = np.zeros((T, self.n_nodes))
        h = np.zeros(self.n_nodes)
        
        for t in range(T):
            h = (1 - self.leak_rate) * h + self.leak_rate * np.tanh(
                self.W_in @ inputs[t] + self.W_res @ h + self.bias
            )
            states[t] = h
            
        return states, h

    def step(self, u, h):
        return (1 - self.leak_rate) * h + self.leak_rate * np.tanh(
            self.W_in @ u + self.W_res @ h + self.bias
        )


def nonlinear_transform(states):
    out = states.copy()
    out[..., 1::2] = states[..., 1::2] ** 2
    return out


def build_nvar(data, k=2, s=1, p=2):
    T, d = data.shape
    offset = (k - 1) * s
    
    linear_features = np.zeros((T - offset, k * d))
    for i in range(k):
        start = offset - i * s
        end = T - i * s if i * s > 0 else T
        linear_features[:, i * d:(i + 1) * d] = data[start:end]
        
    polynomial_features = []
    for deg in range(2, p + 1):
        for combo in combinations_with_replacement(range(linear_features.shape[1]), deg):
            col = np.ones(len(linear_features))
            for idx in combo:
                col *= linear_features[:, idx]
            polynomial_features.append(col)
            
    parts = [np.ones((len(linear_features), 1)), linear_features]
    if polynomial_features:
        parts.append(np.column_stack(polynomial_features))
        
    return np.column_stack(parts), offset