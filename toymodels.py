import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

# Pendulum swing toy model
def pendulum_angle(X, t_target=8.0):
    """The first (0) column of X is de length of the swing
    The second column is the gravity constant and the third is the starting angle of the swing."""
    angles_at_t2 = []

    for row in X:
        L, g, theta0_deg = row
        theta0_rad = np.radians(theta0_deg)
        omega0 = 0.0  # Starting without angular velocity

        # ODE to be solved
        def pendulum_ode(t, y):
            theta, omega = y
            return [omega, -(g / L) * np.sin(theta)]

        # Using Scipy solver to find pendulum position at target T
        sol = solve_ivp(
            pendulum_ode,
            (0, t_target),
            [theta0_rad, omega0],
            t_eval=[t_target],
            method='RK45'
        )

        theta_t2_deg = np.degrees(sol.y[0][0])
        angles_at_t2.append(theta_t2_deg)

    return np.array(angles_at_t2)


def falling_velocity(X, t_target=8.0):
    """Computes analytic velocity under gravity with linear air resistance at time t_target. dv/dt = g - kv/m
    g is gravitation, m is mass of the object, k is the (linear) air resistance and v0 is the initial speed"""

    g, m, k, v0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    v_term = (m * g) / k

    return v_term + (v0 - v_term) * np.exp(- (k / m) * t_target)

import numpy as np


def simulate_ou_process(X, num_simulations=250, mu=2.0, x0=1.0, T=5.0, dt=0.01, return_analytic=False):

    """Function simulating the Ornstein-Uhlenbeck model, a common model for interest rates
    Solution is E(X_t|X_0) = X_0*exp(-theta*t) + mu*(1-exp(-theta*t)) and Var(X_t) = sigma^2/(2*theta) (verified)"""
    
    n_params = X.shape[0]
    theta_vec = X[:, 0].reshape(-1, 1, 1)  # (n_params, 1, 1)
    sigma_vec = X[:, 1].reshape(-1, 1, 1)  # (n_params, 1, 1)
       
    if return_analytic:
        final_vals = x0*np.exp(-theta_vec*T) + mu*(1-np.exp(-theta_vec*T))
        return final_vals[:,0,0]

    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)

    all_paths = np.zeros((n_params, num_simulations, n_steps))
    all_paths[:, :, 0] = x0

    # Pre-generate random noise
    random_noise = np.random.randn(n_params, num_simulations, n_steps - 1)

    for i in range(1, n_steps):
        dx = theta_vec * (mu - all_paths[:, :, i - 1:i]) * dt + \
             sigma_vec * np.sqrt(dt) * random_noise[:, :, i - 1:i]
        all_paths[:, :, i:i+1] = all_paths[:, :, i-1:i] + dx

    final_vals = all_paths[:, :, -1].mean(axis=1)
 
    return final_vals

def simulate_ou_process_analytic(X, num_simulations=1000, mu=2.0, x0=1.0, T=5.0, dt=0.01):
    return simulate_ou_process(X, num_simulations, mu, x0, T, dt, return_analytic=True)

def elastic_net_quadratic(X, n_samples=25):
    
    np.random.seed(42)
    X_data = np.random.uniform(-5, 5, (n_samples, 1))
    
    quadratic_coefs = []
    
    for params in X:
        alpha, l1_ratio = params[0], params[1]
        y_data = -0.5*X_data**0 + 0.2*X_data + 0.5*X_data**2 + 0*0.01*np.random.randn(n_samples, 1)
        
        X_features = np.hstack([X_data**0, X_data, X_data**2])
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_features, y_data)
        
        quadratic_coefs.append(model.coef_[2])
    
    np.random.default_rng()

    return np.array(quadratic_coefs)