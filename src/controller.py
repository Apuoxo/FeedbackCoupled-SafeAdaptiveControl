"""
Supervisory Feedback-Coupled Safe Adaptive Controller.
Implementation of Algorithm 1 from the paper.

"Multi-Model Supervisory Feedback-Coupled Safe Adaptive Control"
Stanislav Usychenko, 2026
"""

import numpy as np
from typing import Optional, Tuple, List
from .rls_estimator import RLSRegimeEstimator
from .cbf_filter import CBFFilter


class SafeAdaptiveController:
    """
    Algorithm 1: Supervisory Feedback-Coupled Safe Controller
    
    Features:
    - Parallel multi-model RLS estimators (equations 2a-2c)
    - Performance-based supervisor with hysteresis (J_r(k))
    - Robust CBF safety filter with δ_safe margin
    """
    
    def __init__(self,
                 n_regimes: int,
                 state_dim: int,
                 input_dim: int,
                 forgetting_factor: float = 0.98,
                 hysteresis: float = 0.1,
                 gamma_cbf: float = 0.9,
                 initial_cov: float = 1000.0):
        """
        Args:
            n_regimes: Number of candidate regimes (R)
            state_dim: State dimension (n_x)
            input_dim: Input dimension (n_u)
            forgetting_factor: λ for RLS (0.95-0.99)
            hysteresis: δ_h for switching (typically 0.05-0.15)
            gamma_cbf: CBF decay rate
            initial_cov: ρ₀ for initial covariance
        """
        self.n_regimes = n_regimes
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Regressor dimension: [x(k); u(k-1)]
        n_theta = state_dim + input_dim
        
        # Step 1: Initialize all RLS estimators
        self.estimators = [
            RLSRegimeEstimator(n_theta, forgetting_factor, initial_cov)
            for _ in range(n_regimes)
        ]
        
        # Performance index history for each regime
        self.J_history: List[List[float]] = [[] for _ in range(n_regimes)]
        self.current_regime = 0
        self.hysteresis = hysteresis
        self.forgetting_factor = forgetting_factor
        
        # CBF filter
        self.cbf = CBFFilter(gamma=gamma_cbf)
        
        # LQR gains for each regime (to be set externally)
        self.K_lqr: List[np.ndarray] = [
            np.zeros((input_dim, state_dim)) for _ in range(n_regimes)
        ]
        
        # System matrices (for CBF)
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        
        # Previous input for regressor
        self.u_prev = np.zeros(input_dim)
        
        # History for analysis
        self.state_history: List[np.ndarray] = []
        self.input_history: List[np.ndarray] = []
        self.regime_history: List[int] = []
        self.switch_counter = 0
        
        # Barrier functions (can be overridden)
        self.h_func = self._default_h
        self.dhdx_func = self._default_dhdx
        
        # Safety bounds
        self.x_min = -np.ones(state_dim) * 5.0
        self.x_max = np.ones(state_dim) * 5.0
    
    def set_lqr_gain(self, regime: int, K: np.ndarray):
        """Set LQR gain for a specific regime."""
        self.K_lqr[regime] = K
    
    def set_system_matrices(self, A: np.ndarray, B: np.ndarray):
        """Set system matrices for CBF computation."""
        self.A = A
        self.B = B
    
    def set_barrier_functions(self, h_func, dhdx_func):
        """Set custom barrier function and its gradient."""
        self.h_func = h_func
        self.dhdx_func = dhdx_func
    
    def set_safety_bounds(self, x_min: np.ndarray, x_max: np.ndarray):
        """Set box safety bounds."""
        self.x_min = x_min
        self.x_max = x_max
        self.h_func = lambda x: self.cbf.box_barrier(x, x_min, x_max)
        self.dhdx_func = lambda x: self.cbf.box_barrier_gradient(x, x_min, x_max)
    
    def _default_h(self, x: np.ndarray) -> float:
        """Default barrier: box constraint on states."""
        return self.cbf.box_barrier(x, self.x_min, self.x_max)
    
    def _default_dhdx(self, x: np.ndarray) -> np.ndarray:
        """Default barrier gradient."""
        return self.cbf.box_barrier_gradient(x, self.x_min, self.x_max)
    
    def _compute_regressor(self, x: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        """Compute regressor φ(k) = [x(k)ᵀ, u(k-1)ᵀ]ᵀ."""
        return np.concatenate([x, u_prev])
    
    def _compute_performance_index(self, regime: int) -> float:
        """
        Compute J_r(k) with exponential weighting.
        
        J_r(k) = Σ_{j=0}^{N_w-1} λ^j ||ε_r(k-j)||²
        """
        history = self.J_history[regime]
        if not history:
            return float('inf')
        
        J = 0.0
        weight = 1.0
        for err_sq in reversed(history[-100:]):  # Limit window
            J += weight * err_sq
            weight *= self.forgetting_factor
        return J
    
    def _select_best_regime(self) -> int:
        """Select regime with smallest performance index."""
        J_values = [self._compute_performance_index(r) for r in range(self.n_regimes)]
        return int(np.argmin(J_values))
    
    def _compute_lqr_input(self, x: np.ndarray) -> np.ndarray:
        """Compute LQR input: u_lqr = -K_{r̂(k)} x(k)."""
        K = self.K_lqr[self.current_regime]
        return -K @ x
    
    def _compute_estimation_error_bound(self) -> float:
        """Compute δ_safe proportional to current estimation error."""
        P = self.estimators[self.current_regime].get_covariance()
        return np.sqrt(np.trace(P)) * 0.01
    
    def update_estimators(self, x: np.ndarray, u_prev: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """
        Step 4: Update all RLS models.
        
        Returns array of prediction error norms for each regime.
        """
        phi = self._compute_regressor(x, u_prev)
        errors = np.zeros(self.n_regimes)
        
        for r, estimator in enumerate(self.estimators):
            error, _ = estimator.update(phi, x_next)
            errors[r] = np.linalg.norm(error)
            
            # Store squared error for performance index
            self.J_history[r].append(error ** 2)
        
        return errors
    
    def update_supervisor(self, errors: np.ndarray) -> int:
        """
        Step 5: Update supervisor and return active regime.
        
        Switching condition: J_best < J_current * (1 - δ_h)
        """
        best_regime = self._select_best_regime()
        current_J = self._compute_performance_index(self.current_regime)
        best_J = self._compute_performance_index(best_regime)
        
        # Hysteresis switching
        if best_J < current_J * (1 - self.hysteresis):
            self.current_regime = best_regime
            self.switch_counter += 1
        
        return self.current_regime
    
    def step(self, x: np.ndarray, x_next: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute one control step (Algorithm 1).
        
        Args:
            x: Current state x(k)
            x_next: Next state x(k+1) (required for estimation/training)
            
        Returns:
            u: Safe control input
        """
        # Step 3: Form regressor (implicit in update_estimators)
        
        # Step 4 & 5: Update estimators and supervisor (if training data available)
        if x_next is not None:
            errors = self.update_estimators(x, self.u_prev, x_next)
            self.update_supervisor(errors)
        
        # Step 6: Compute LQR input
        u_lqr = self._compute_lqr_input(x)
        
        # Step 7: CBF safety filter
        if self.A is not None and self.B is not None:
            error_bound = self._compute_estimation_error_bound()
            u = self.cbf.compute_safe_input(
                x, u_lqr, self.A, self.B,
                self.h_func, self.dhdx_func,
                error_bound
            )
        else:
            u = u_lqr
        
        # Store for next step
        self.u_prev = u
        self.state_history.append(x.copy())
        self.input_history.append(u.copy())
        self.regime_history.append(self.current_regime)
        
        return u
    
    def get_active_regime(self) -> int:
        """Return currently active regime index."""
        return self.current_regime
    
    def get_statistics(self) -> dict:
        """Return performance statistics."""
        return {
            'n_switches': self.switch_counter,
            'final_regime': self.current_regime,
            'n_steps': len(self.state_history),
            'regime_history': self.regime_history.copy()
        }
    
    def reset(self):
        """Reset controller state."""
        for estimator in self.estimators:
            estimator.reset()
        self.J_history = [[] for _ in range(self.n_regimes)]
        self.current_regime = 0
        self.switch_counter = 0
        self.state_history = []
        self.input_history = []
        self.regime_history = []
        self.u_prev = np.zeros(self.input_dim)