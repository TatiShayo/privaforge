"""
Core Differential Privacy Mechanisms for Privaforge.

Includes:
- LaplaceMechanism: Calibrated to L1 global sensitivity.
- GaussianMechanism: Calibrated to L2 global sensitivity with classical & analytic calibrations.
- ExponentialMechanism: Discrete candidate selection with log-sum-exp stabilization.
- VectorLaplaceMechanism: Multi-dimensional L1 noise injection with optional bounds.
- VectorGaussianMechanism: Multi-dimensional L2 noise injection with classical/analytic scaling and optional bounds.
"""

import math
import numpy as np
from typing import Sequence, Any, Optional, Union, Tuple
from scipy import stats


class LaplaceMechanism:
    """
    Differentially private Laplace mechanism for scalar and 1D numerical queries.
    
    Noise scale b = sensitivity / epsilon.
    PDF: f(x) = (1 / 2b) * exp(-|x| / b)
    Variance: 2 * b^2
    """

    def __init__(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        if not isinstance(epsilon, (int, float)) or math.isnan(epsilon) or math.isinf(epsilon):
            raise ValueError(f"Epsilon must be a finite real number, got {epsilon}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive (> 0), got {epsilon}")
            
        if not isinstance(sensitivity, (int, float)) or math.isnan(sensitivity) or math.isinf(sensitivity):
            raise ValueError(f"Sensitivity must be a finite real number, got {sensitivity}")
        if sensitivity < 0:
            raise ValueError(f"Sensitivity must be non-negative (>= 0), got {sensitivity}")
            
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val})")

        self.epsilon = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None
        
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

    @property
    def scale(self) -> float:
        """Noise scale b = sensitivity / epsilon."""
        if self.sensitivity == 0.0:
            return 0.0
        return self.sensitivity / self.epsilon

    @property
    def variance(self) -> float:
        """Variance of Laplace noise: 2 * b^2."""
        return 2.0 * (self.scale ** 2)

    def randomise(self, value: float) -> float:
        """Add Laplace noise to a scalar value and clip to [min_val, max_val] if defined."""
        if not isinstance(value, (int, float, np.number)) or math.isnan(value) or math.isinf(value):
            raise ValueError(f"Input value must be a finite number, got {value}")
            
        if self.scale == 0.0:
            noised = float(value)
        else:
            noise = self.rng.laplace(loc=0.0, scale=self.scale)
            noised = float(value) + float(noise)
            
        if self.min_val is not None or self.max_val is not None:
            noised = float(np.clip(noised, self.min_val, self.max_val))
            
        return noised

    def randomise_vector(self, values: Sequence[float]) -> np.ndarray:
        """Add independent Laplace noise to an array/vector of values."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr.copy()
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Input array contains NaN or Inf values")
            
        if self.scale == 0.0:
            noised = arr.copy()
        else:
            noise = self.rng.laplace(loc=0.0, scale=self.scale, size=arr.shape)
            noised = arr + noise
            
        if self.min_val is not None or self.max_val is not None:
            noised = np.clip(noised, self.min_val, self.max_val)
            
        return noised

    def pdf(self, x: float) -> float:
        """Probability density function of the zero-mean noise at x."""
        if self.scale == 0.0:
            return float("inf") if x == 0.0 else 0.0
        return float(np.exp(-abs(x) / self.scale) / (2.0 * self.scale))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function of the zero-mean noise at x."""
        if self.scale == 0.0:
            return 1.0 if x >= 0.0 else 0.0
        if x < 0:
            return 0.5 * float(np.exp(x / self.scale))
        else:
            return 1.0 - 0.5 * float(np.exp(-x / self.scale))


class GaussianMechanism:
    """
    Differentially private Gaussian mechanism for scalar and 1D numerical queries.
    
    Guarantees (epsilon, delta)-differential privacy.
    Standard scaling: sigma = (sensitivity * sqrt(2 * ln(1.25 / delta))) / epsilon
    Supports both classical and analytic Gaussian mechanisms.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        analytic: bool = False,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        if not isinstance(epsilon, (int, float)) or math.isnan(epsilon) or math.isinf(epsilon):
            raise ValueError(f"Epsilon must be a finite real number, got {epsilon}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive (> 0), got {epsilon}")
            
        if not isinstance(delta, (int, float)) or math.isnan(delta) or math.isinf(delta):
            raise ValueError(f"Delta must be a finite real number, got {delta}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be strictly between 0 and 1 (0 < delta < 1), got {delta}")
            
        if not isinstance(sensitivity, (int, float)) or math.isnan(sensitivity) or math.isinf(sensitivity):
            raise ValueError(f"Sensitivity must be a finite real number, got {sensitivity}")
        if sensitivity < 0:
            raise ValueError(f"Sensitivity must be non-negative (>= 0), got {sensitivity}")
            
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val})")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None
        self.analytic = bool(analytic)
        
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

    @property
    def sigma(self) -> float:
        """Standard deviation of Gaussian noise."""
        if self.sensitivity == 0.0:
            return 0.0
            
        if self.analytic:
            return self._calibrate_analytic_gaussian(self.epsilon, self.delta, self.sensitivity)
        else:
            # Classical calibration: sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
            return (self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta))) / self.epsilon

    @property
    def variance(self) -> float:
        """Variance of Gaussian noise: sigma^2."""
        return self.sigma ** 2

    @staticmethod
    def _calibrate_analytic_gaussian(epsilon: float, delta: float, sensitivity: float) -> float:
        """
        Analytic Gaussian Mechanism calibration (Balle & Wang, ICML 2018).
        Calculates exact minimal sigma for (epsilon, delta)-DP with dynamic exponential bracketing.
        """
        def phi(t):
            return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))

        delta_0 = phi(0.0) - math.exp(epsilon) * phi(-math.sqrt(2.0 * epsilon))

        def b_plus(v):
            return phi(math.sqrt(epsilon * v)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (v + 2.0)))

        def b_minus(u):
            return phi(-math.sqrt(epsilon * u)) - math.exp(epsilon) * phi(-math.sqrt(epsilon * (u + 2.0)))

        if delta >= delta_0:
            low, high = 0.0, 1.0
            while b_plus(high) < delta and high < 1e15:
                high *= 2.0
            for _ in range(120):
                mid = (low + high) / 2.0
                if b_plus(mid) < delta:
                    low = mid
                else:
                    high = mid
            v_star = (low + high) / 2.0
            alpha = math.sqrt(1.0 + v_star) + math.sqrt(v_star)
        else:
            low, high = 0.0, 1.0
            while b_minus(high) > delta and high < 1e15:
                high *= 2.0
            for _ in range(120):
                mid = (low + high) / 2.0
                if b_minus(mid) > delta:
                    low = mid
                else:
                    high = mid
            u_star = (low + high) / 2.0
            alpha = math.sqrt(1.0 + u_star) + math.sqrt(u_star)

        sigma = (alpha * sensitivity) / math.sqrt(2.0 * epsilon)
        return max(1e-12, float(sigma))

    def randomise(self, value: float) -> float:
        """Add Gaussian noise to a scalar value and clip to bounds if defined."""
        if not isinstance(value, (int, float, np.number)) or math.isnan(value) or math.isinf(value):
            raise ValueError(f"Input value must be a finite number, got {value}")
            
        if self.sigma == 0.0:
            noised = float(value)
        else:
            noise = self.rng.normal(loc=0.0, scale=self.sigma)
            noised = float(value) + float(noise)
            
        if self.min_val is not None or self.max_val is not None:
            noised = float(np.clip(noised, self.min_val, self.max_val))
            
        return noised

    def randomise_vector(self, values: Sequence[float]) -> np.ndarray:
        """Add independent Gaussian noise to an array/vector of values."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr.copy()
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Input array contains NaN or Inf values")
            
        if self.sigma == 0.0:
            noised = arr.copy()
        else:
            noise = self.rng.normal(loc=0.0, scale=self.sigma, size=arr.shape)
            noised = arr + noise
            
        if self.min_val is not None or self.max_val is not None:
            noised = np.clip(noised, self.min_val, self.max_val)
            
        return noised

    def pdf(self, x: float) -> float:
        """Probability density function of the zero-mean noise at x."""
        if self.sigma == 0.0:
            return float("inf") if x == 0.0 else 0.0
        return float(stats.norm.pdf(x, loc=0.0, scale=self.sigma))

    def cdf(self, x: float) -> float:
        """Cumulative distribution function of the zero-mean noise at x."""
        if self.sigma == 0.0:
            return 1.0 if x >= 0.0 else 0.0
        return float(stats.norm.cdf(x, loc=0.0, scale=self.sigma))


class ExponentialMechanism:
    r"""
    Exponential Mechanism for discrete selection under Differential Privacy.
    
    Selects candidate r in R with probability proportional to:
    P(r) \propto exp(epsilon * u(x, r) / (2 * sensitivity))
    
    Includes log-sum-exp trick for numerical stability to prevent float overflow/underflow.
    """

    def __init__(
        self,
        epsilon: float,
        utility_sensitivity: float = 1.0,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        if not isinstance(epsilon, (int, float)) or math.isnan(epsilon) or math.isinf(epsilon):
            raise ValueError(f"Epsilon must be a finite real number, got {epsilon}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive (> 0), got {epsilon}")
            
        if not isinstance(utility_sensitivity, (int, float)) or math.isnan(utility_sensitivity) or math.isinf(utility_sensitivity):
            raise ValueError(f"Utility sensitivity must be a finite real number, got {utility_sensitivity}")
        if utility_sensitivity <= 0:
            raise ValueError(f"Utility sensitivity must be strictly positive (> 0), got {utility_sensitivity}")

        self.epsilon = float(epsilon)
        self.utility_sensitivity = float(utility_sensitivity)
        
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

    def get_probabilities(self, utility_scores: Sequence[float]) -> np.ndarray:
        """
        Calculate selection probabilities for each candidate using the log-sum-exp trick.
        """
        scores = np.asarray(utility_scores, dtype=float)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("Utility scores must be a non-empty 1D array/sequence")
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            raise ValueError("Utility scores contain NaN or Inf values")

        factor = self.epsilon / (2.0 * self.utility_sensitivity)
        scaled_scores = scores * factor
        
        max_score = np.max(scaled_scores)
        exp_scores = np.exp(scaled_scores - max_score)
        sum_exp = np.sum(exp_scores)
        
        if sum_exp <= 0 or math.isnan(sum_exp):
            probs = np.ones(len(scores)) / float(len(scores))
            return probs
            
        probs = exp_scores / sum_exp
        total_p = np.sum(probs)
        if total_p > 0:
            probs = probs / total_p
        else:
            probs = np.ones(len(scores)) / float(len(scores))
        return probs

    def select_index(self, utility_scores: Sequence[float]) -> int:
        """Select a candidate index based on DP utility score distribution."""
        probs = self.get_probabilities(utility_scores)
        return int(self.rng.choice(len(probs), p=probs))

    def select(self, candidates: Sequence[Any], utility_scores: Sequence[float]) -> Any:
        """Select a candidate item based on DP utility score distribution."""
        if len(candidates) != len(utility_scores):
            raise ValueError(f"Length mismatch: {len(candidates)} candidates vs {len(utility_scores)} utility scores")
        idx = self.select_index(utility_scores)
        return candidates[idx]


class VectorLaplaceMechanism:
    """
    Differentially private Vector Laplace mechanism for multivariate queries.
    Calibrated to L1 global vector sensitivity with optional clipping.
    """

    def __init__(
        self,
        epsilon: float,
        l1_sensitivity: float = 1.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        if not isinstance(epsilon, (int, float)) or math.isnan(epsilon) or math.isinf(epsilon):
            raise ValueError(f"Epsilon must be a finite real number, got {epsilon}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if not isinstance(l1_sensitivity, (int, float)) or math.isnan(l1_sensitivity) or math.isinf(l1_sensitivity):
            raise ValueError(f"L1 sensitivity must be a finite real number, got {l1_sensitivity}")
        if l1_sensitivity < 0:
            raise ValueError(f"L1 sensitivity must be non-negative (>= 0), got {l1_sensitivity}")
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val})")

        self.epsilon = float(epsilon)
        self.l1_sensitivity = float(l1_sensitivity)
        self.scale = (self.l1_sensitivity / self.epsilon) if self.l1_sensitivity > 0 else 0.0
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None
        
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

    def randomise(self, vector: Sequence[float]) -> np.ndarray:
        """Add independent Laplace noise to vector with scale = l1_sensitivity / epsilon."""
        arr = np.asarray(vector, dtype=float)
        if arr.size == 0:
            return arr.copy()
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Input vector contains NaN or Inf values")
        if self.scale == 0.0:
            noised = arr.copy()
        else:
            noise = self.rng.laplace(loc=0.0, scale=self.scale, size=arr.shape)
            noised = arr + noise
        if self.min_val is not None or self.max_val is not None:
            noised = np.clip(noised, self.min_val, self.max_val)
        return noised


class VectorGaussianMechanism:
    """
    Differentially private Vector Gaussian mechanism for multivariate queries.
    Calibrated to L2 global vector sensitivity with optional classical/analytic calibration and clipping.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        l2_sensitivity: float = 1.0,
        analytic: bool = False,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ):
        if not isinstance(epsilon, (int, float)) or math.isnan(epsilon) or math.isinf(epsilon):
            raise ValueError(f"Epsilon must be a finite real number, got {epsilon}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if not isinstance(delta, (int, float)) or math.isnan(delta) or math.isinf(delta):
            raise ValueError(f"Delta must be a finite real number, got {delta}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be between 0 and 1, got {delta}")
        if not isinstance(l2_sensitivity, (int, float)) or math.isnan(l2_sensitivity) or math.isinf(l2_sensitivity):
            raise ValueError(f"L2 sensitivity must be a finite real number, got {l2_sensitivity}")
        if l2_sensitivity < 0:
            raise ValueError(f"L2 sensitivity must be non-negative (>= 0), got {l2_sensitivity}")
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val})")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.l2_sensitivity = float(l2_sensitivity)
        self.analytic = bool(analytic)
        self.min_val = float(min_val) if min_val is not None else None
        self.max_val = float(max_val) if max_val is not None else None

        if self.l2_sensitivity == 0.0:
            self.sigma = 0.0
        elif self.analytic:
            self.sigma = GaussianMechanism._calibrate_analytic_gaussian(self.epsilon, self.delta, self.l2_sensitivity)
        else:
            self.sigma = (self.l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta))) / self.epsilon
        
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

    def randomise(self, vector: Sequence[float]) -> np.ndarray:
        """Add independent Gaussian noise to vector."""
        arr = np.asarray(vector, dtype=float)
        if arr.size == 0:
            return arr.copy()
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Input vector contains NaN or Inf values")
        if self.sigma == 0.0:
            noised = arr.copy()
        else:
            noise = self.rng.normal(loc=0.0, scale=self.sigma, size=arr.shape)
            noised = arr + noise
        if self.min_val is not None or self.max_val is not None:
            noised = np.clip(noised, self.min_val, self.max_val)
        return noised
