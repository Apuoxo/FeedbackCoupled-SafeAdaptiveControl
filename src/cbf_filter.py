"""
Robust Discrete Control Barrier Function (CBF) safety filter.
Implements CBF constraint with robust margin δ_safe.

From the paper:
"Multi-Model Supervisory Feedback-Coupled Safe Adaptive Control"
Stanislav Usychenko, 2026
"""

import numpy as np


class CBFFilter:
    """
    Discrete CBF safety filter with robust margin.
    
    Constraint: Δh(x, u) ≥ -γ h(x) + δ_safe
    
    The filter solves a QP to find the minimal modification to
    the nominal input that satisfies the safety constraint.
    """
    
    def __init__(self, gamma: float = 0.9, robust_margin_scale: float = 1.0):
        """
        Args:
            gamma: CBF decay rate ∈ (0,1). Higher = more aggressive.
            robust_margin_scale: Scales δ_safe from estimation error
        """
        self.gamma = gamma
        self.robust_margin_scale = robust_margin_scale
    
    def compute_safe_input(self, x: np.ndarray, u_nominal: np.ndarray,
                          A: np.ndarray, B: np.ndarray,
                          h_func, dhdx_func,
                          estimation_error_bound: float = 0.1) -> np.ndarray:
        """
        Find safe input closest to nominal via QP.
        
        Minimizes: ||u - u_nominal||²
        Subject to: dhdx @ (A @ x + B @ u) ≥ -γ h(x) + δ_safe
        
        Args:
            x: Current state
            u_nominal: Desired input (e.g., from LQR)
            A, B: System matrices for current regime
            h_func: Barrier function h(x) > 0 inside safe set
            dhdx_func: Gradient of barrier function
            estimation_error_bound: δ_safe from estimation uncertainty
            
        Returns:
            u_safe: Safe control input
        """
        # Current barrier value
        h_x = h_func(x)
        
        # If deep inside safe set, no need to modify
        if h_x > 10.0 * self.robust_margin_scale * estimation_error_bound:
            return u_nominal
        
        # Gradient
        dhdx = dhdx_func(x)
        
        # Robust safety margin
        delta_safe = self.robust_margin_scale * estimation_error_bound
        
        # Compute the constraint: F @ u ≥ g
        # where F = dhdx @ B, g = -γ h(x) + δ_safe - dhdx @ (A @ x)
        
        F = dhdx @ B
        
        # If no control authority over safety, return nominal
        if np.linalg.norm(F) < 1e-6:
            return u_nominal
        
        g = -self.gamma * h_x + delta_safe - dhdx @ (A @ x)
        
        # Analytic solution for 1D constraint (most common)
        # u = u_nominal + max(0, (g - F @ u_nominal) / (F @ F)) * Fᵀ
        F_norm_sq = F @ F
        if F_norm_sq > 1e-6:
            slack = (g - F @ u_nominal) / F_norm_sq
            if slack > 0:
                u_safe = u_nominal + slack * F
                return u_safe
        
        return u_nominal
    
    @staticmethod
    def quadratic_barrier(x: np.ndarray, x_safe: np.ndarray, margin: float = 1.0) -> float:
        """
        Quadratic barrier function: h(x) = margin - ||x - x_safe||²
        Safe when h(x) > 0.
        """
        return margin - np.sum((x - x_safe) ** 2)
    
    @staticmethod
    def quadratic_barrier_gradient(x: np.ndarray, x_safe: np.ndarray) -> np.ndarray:
        """Gradient of quadratic barrier."""
        return -2 * (x - x_safe)
    
    @staticmethod
    def box_barrier(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> float:
        """
        Box constraint barrier: h(x) = min(x - x_min, x_max - x)
        Safe when all components are > 0.
        """
        return np.min(np.concatenate([x - x_min, x_max - x]))
    
    @staticmethod
    def box_barrier_gradient(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Subgradient of box barrier."""
        dist_to_min = x - x_min
        dist_to_max = x_max - x
        min_idx = np.argmin(np.concatenate([dist_to_min, dist_to_max]))
        grad = np.zeros_like(x)
        if min_idx < len(x):
            grad[min_idx] = 1.0
        else:
            grad[min_idx - len(x)] = -1.0
        return grad