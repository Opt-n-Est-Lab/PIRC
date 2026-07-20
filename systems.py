import numpy as np
from scipy.integrate import solve_ivp


class LorenzSystem:
    dim = 3
    name = "Lorenz"

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def rhs(self, u):
        u = np.clip(u, -1e6, 1e6)
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        return np.column_stack([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ])

    def generate(self, T=30, dt=0.02, transient=500):
        t_eval = np.arange(0, T, dt)
        f = lambda t, u: self.rhs(u.reshape(1, -1)).ravel()
        sol = solve_ivp(f, [0, T], [1., 1., 1.], t_eval=t_eval,
                        rtol=1e-10, atol=1e-12)
        return sol.t[transient:], sol.y[:, transient:].T


class RosslerSystem:
    dim = 3
    name = "Rossler"

    def __init__(self, a=0.2, b=0.2, c=5.7):
        self.a = a
        self.b = b
        self.c = c

    def rhs(self, u):
        u = np.clip(u, -1e6, 1e6)
        x, y, z = u[:, 0], u[:, 1], u[:, 2]
        return np.column_stack([
            -y - z,
            x + self.a * y,
            self.b + z * (x - self.c)
        ])

    def generate(self, T=200, dt=0.05, transient=400):
        t_eval = np.arange(0, T, dt)
        f = lambda t, u: self.rhs(u.reshape(1, -1)).ravel()
        sol = solve_ivp(f, [0, T], [1., 1., 0.], t_eval=t_eval,
                        rtol=1e-10, atol=1e-12)
        return sol.t[transient:], sol.y[:, transient:].T


class MackeyGlassSystem:
    dim = 1
    name = "Mackey-Glass"

    def __init__(self, beta=0.2, gamma=0.1, n=10, tau=17):
        self.beta = beta
        self.gamma = gamma
        self.n_exp = n
        self.tau = tau

    def rhs(self, u):
        return -self.gamma * np.clip(u, -1e6, 1e6)

    def generate(self, T=2000, dt=1.0, transient=500):
        tau_steps = int(self.tau / dt)
        N = int(T / dt) + transient + tau_steps
        x = np.ones(N) * 1.2
        for i in range(tau_steps, N - 1):
            xt = x[i - tau_steps]
            x[i + 1] = x[i] + dt * (self.beta * xt / (1 + xt ** self.n_exp) - self.gamma * x[i])
        start = transient + tau_steps
        return np.arange(N - start) * dt, x[start:].reshape(-1, 1)


class NormalizedSystem:
    def __init__(self, system, mu, sigma):
        self.system = system
        self.mu = mu
        self.sigma = sigma
        self.dim = system.dim
        self.name = system.name

    def rhs(self, u_norm):
        u = np.clip(u_norm, -1e4, 1e4) * self.sigma + self.mu
        return self.system.rhs(u) / self.sigma