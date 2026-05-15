"""
Recursive Least Squares (RLS) estimator for each regime.
Implementation of equations (2a)-(2c) from the paper.
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
        self.lambd = forgetting_factor
        self.theta_hat = np.zeros(n_theta)
        self.P = np.eye(n_theta) * initial_cov
    
    def update(self, phi: np.ndarray, x_next: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update estimator with new measurement.
        
        Args:
            phi: Regressor vector φ(k) = [x(k)ᵀ, u(k-1)ᵀ]ᵀ
            x_next: Next state x(k+1)
            
        Returns:
            prediction_error: ε_r⁺(k)
            innovation_cov: For performance index
        """
        # Prediction error (2a)
        prediction_error = x_next - phi @ self.theta_hat
        
        # Innovation covariance (for performance index)
        innovation_cov = phi @ self.P @ phi + 1.0
        
        # Kalman gain
        L = self.P @ phi / (self.lambd + phi @ self.P @ phi)
        
        # Parameter update (2b)
        self.theta_hat += L * prediction_error
        
        # Covariance update (2c)
        self.P = (self.P - np.outer(L, phi @ self.P)) / self.lambd
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
        return prediction_error, innovation_cov