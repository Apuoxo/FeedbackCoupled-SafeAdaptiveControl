"""
Performance-based supervisor with hysteresis for regime switching.
"""

import numpy as np
from collections import deque


class Supervisor:
    """
    Supervisory logic for multi-model adaptive control.
    
    Implements:
        J_r(k) = Σ_{j=0}^{N_w-1} λ^j ||ε_r(k-j)||²    (smoothed performance)
    
    Switching condition: J_best < J_current * (1 - δ_h)
    """
    
    def __init__(self, n_regimes: int, window_size: int = 50,
                 forget_factor: float = 0.95, hysteresis: float = 0.1):
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.forget_factor = forget_factor
        self.hysteresis = hysteresis  # δ_h
        
        self.errors = [deque(maxlen=window_size) for _ in range(n_regimes)]
        self.current_regime = 0
        self.switch_counter = 0
    
    def update_performance(self, regime: int, prediction_error: float):
        """Update performance index J_r(k)."""
        self.errors[regime].append(prediction_error ** 2)
    
    def compute_performance_index(self, regime: int) -> float:
        """Compute J_r(k) with exponential weighting."""
        errors = list(self.errors[regime])
        if not errors:
            return float('inf')
        
        J = 0.0
        weight = 1.0
        for e in reversed(errors):
            J += weight * e
            weight *= self.forget_factor
        return J
    
    def select_best_regime(self) -> int:
        """Select regime with smallest performance index."""
        J_values = [self.compute_performance_index(i) for i in range(self.n_regimes)]
        return int(np.argmin(J_values))
    
    def update(self, regime_errors: np.ndarray) -> int:
        """
        Update supervisor and return active regime.
        
        Switching occurs only if best model outperforms current
        by more than δ_h (hysteresis).
        """
        for r, err in enumerate(regime_errors):
            self.update_performance(r, err)
        
        best_regime = self.select_best_regime()
        current_J = self.compute_performance_index(self.current_regime)
        best_J = self.compute_performance_index(best_regime)
        
        # Hysteresis condition
        if best_J < current_J * (1 - self.hysteresis):
            self.current_regime = best_regime
            self.switch_counter += 1
        
        return self.current_regime