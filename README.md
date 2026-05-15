[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

# Feedback-Coupled Safe Adaptive Control

**Multi-Model Supervisory Feedback-Coupled Safe Adaptive Control for Markov-Switched Linear Systems**

Implementation of the paper by **Stanislav Usychenko (2026)**.

## Abstract

This paper proposes a supervisory multi-model feedback-coupled architecture for safe adaptive control of discrete-time linear systems with unknown Markovian switching. All RLS estimators run in parallel and are updated continuously. A performance-based supervisor with hysteresis selects the active model. A discrete CBF safety filter enforces hard constraints while providing additional excitation when the state approaches the safe set boundary.

 Algorithm

```python
# Algorithm 1: Supervisory Feedback-Coupled Safe Controller

# Step 1: Initialize all θ̂_{r,i}, P_r ← ρ₀I
for r in range(n_regimes):
    estimator[r] = RLSRegimeEstimator()

# Step 2: Main control loop
for k in range(K):
    # Step 3: Form regressor
    phi = [x(k), u(k-1)]
    
    # Step 4: Update all RLS models
    for r in range(n_regimes):
        error[r] = estimator[r].update(phi, x_next)
    
    # Step 5: Compute J_r(k) and update r̂(k) with hysteresis
    active_regime = supervisor.update(errors)
    
    # Step 6: Compute LQR input
    u_lqr = -K_lqr[active_regime] @ x(k)
    
    # Step 7: Solve robust CBF-QP
    u(k) = cbf_filter.compute_safe_input(x(k), u_lqr)
    
    # Step 8: Apply u(k)
    system.apply(u(k))



 INSTALLATION 

git clone https://github.com/Apuoxo/FeedbackCoupled-SafeAdaptiveControl.git
cd FeedbackCoupled-SafeAdaptiveControl
pip install -r requirements.txt


 Quick Start 

from src.controller import SafeAdaptiveController
import numpy as np

controller = SafeAdaptiveController(
    n_regimes=3,
    state_dim=4,
    input_dim=2
)

K1 = np.array([[-0.5, -0.1, 0, 0], [0, 0, -0.5, -0.1]])
K2 = np.array([[-0.8, -0.2, 0, 0], [0, 0, -0.8, -0.2]])
K3 = np.array([[-1.0, -0.3, 0, 0], [0, 0, -1.0, -0.3]])

controller.set_lqr_gain(0, K1)
controller.set_lqr_gain(1, K2)
controller.set_lqr_gain(2, K3)

A = np.eye(4)
B = np.eye(4, 2) * 0.01
controller.set_system_matrices(A, B)

x_min = np.array([-5, -5, -3.14, -3.14])
x_max = np.array([5, 5, 3.14, 3.14])
controller.set_safety_bounds(x_min, x_max)

x = np.array([1.0, 0.5, 0.1, 0.0])
x_next = None

for step in range(100):
    u = controller.step(x, x_next)
    x_next = A @ x + B @ u + np.random.randn(4) * 0.01
    x = x_next

stats = controller.get_statistics()
print(f"Switches: {stats['n_switches']}, Final regime: {stats['final_regime']}")


