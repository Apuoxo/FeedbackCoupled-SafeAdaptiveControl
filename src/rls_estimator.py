"""
Recursive Least Squares (RLS) estimator for each regime.
Implementation of equations (2a)-(2c) from the paper:
"Multi-Model Supervisory Feedback-Coupled Safe Adaptive Control"
Stanislav Usychenko, 2026
"""

import numpy as np
from typing import Tuple


class RLSRegimeEstimator:
    """
    RLS estimator for a single regime.
    
    Implements:
        ε_r⁺(k) = x_i(k+1) - φ(k)ᵀ θ̂_{r,i}(k)           (2a)
        θ̂_{r,i}(k+1) = θ̂_{r,i}(k) + L_r(k) ε_r⁺(k)      (2b)
        P_r(k+1) = λ⁻¹ (P_r(k) - L_r(k) φ(k)ᵀ P_r(k))   (2c)
    """
    
    def __init__(self, n_theta: int, forgetting_factor: float = 0.98, 
                 initial_cov: float = 1000.0):
        """
        Args:
            n_theta: Dimension of parameter vector θ
            forgetting_factor: λ ∈ (0,1], typically 0.95-0.99
            initial_cov: ρ₀ for initial covariance matrix P(0) = ρ₀I
        """
        self.lambd = forgetting_factor
        self.theta_hat = np.zeros(n_theta)
        self.P = np.eye(n_theta) * initial_cov
        self.n_theta = n_theta
    
    def update(self, phi: np.ndarray, x_next: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update estimator with new measurement.
        
        Args:
            phi: Regressor vector φ(k) = [x(k)ᵀ, u(k-1)ᵀ]ᵀ
            x_next: Next state x(k+1)
            
        Returns:
            prediction_error: ε_r⁺(k)
            innovation_cov: For performance index (scalar)
        """
        # Prediction error (2a)
        prediction_error = x_next - np.dot(phi, self.theta_hat)
        
        # Innovation covariance
        phiP = np.dot(phi, self.P)
        innovation_cov = phiP @ phi + 1.0
        
        # Kalman gain
        L = self.P @ phi / (self.lambd + phiP @ phi)
        
        # Parameter update (2b)
        self.theta_hat += L * prediction_error
        
        # Covariance update (2c)
        self.P = (self.P - np.outer(L, phiP)) / self.lambd
        
        # Ensure symmetry (numerical stability)
        self.P = (self.P + self.P.T) / 2
        
        return prediction_error, innovation_cov
    
    def get_parameters(self) -> np.ndarray:
        """Return current parameter estimate θ̂_r."""
        return self.theta_hat.copy()
    
    def get_covariance(self) -> np.ndarray:
        """Return current covariance matrix P_r."""
        return self.P.copy()
    
    def reset(self):
        """Reset estimator to initial state."""
        self.theta_hat = np.zeros(self.n_theta)
        self.P = np.eye(self.n_theta) * 1000.0