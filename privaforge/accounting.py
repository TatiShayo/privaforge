"""
Differential Privacy Budget Accounting and Composition Engine for Privaforge.

Includes:
- PrivacyBudgetTracker: Tracks, allocates, and restricts privacy consumption with exhaustion blocking.
- Sequential Composition & Advanced Composition Theorem (Dwork et al.).
- RenyiDPAccountant: Rényi Differential Privacy tracking and conversion to (epsilon, delta)-DP.
- MomentsAccountant: Log-MGF accounting for iterative / subsampled mechanisms.
"""

import math
import time
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence
from contextlib import contextmanager
import numpy as np

class BudgetExhaustedError(Exception):
    """Exception raised when a privacy query exceeds the remaining budget."""
    pass

class PrivacyBudgetTracker:
    """
    Privacy budget tracker enforcing (epsilon, delta) limits across queries.

    Supports:
    - Sequential and Advanced composition tracking.
    - Budget splitting and sub-allocation.
    - Exhaustion blocking (raising BudgetExhaustedError).
    - Transactional query sessions with rollback.
    """

    def __init__(
        self,
        total_epsilon: float = float("inf"),
        total_delta: float = 1.0,
        name: str = "default_tracker"
    ):
        if not isinstance(total_epsilon, (int, float)) or math.isnan(total_epsilon):
            raise ValueError(f"Total epsilon must be a valid number, got {total_epsilon}")
        if total_epsilon <= 0:
            raise ValueError(f"Total epsilon must be strictly positive (> 0), got {total_epsilon}")

        if not isinstance(total_delta, (int, float)) or math.isnan(total_delta):
            raise ValueError(f"Total delta must be a valid number, got {total_delta}")
        if total_delta < 0 or total_delta > 1:
            raise ValueError(f"Total delta must be in [0, 1], got {total_delta}")

        self.total_epsilon = float(total_epsilon)
        self.total_delta = float(total_delta)
        self.name = str(name)

        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.ledger: List[Dict[str, Any]] = []

    @property
    def remaining_epsilon(self) -> float:
        """Remaining epsilon under sequential composition."""
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Remaining delta under sequential composition."""
        return max(0.0, self.total_delta - self.spent_delta)

    def has_budget(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if requested (epsilon, delta) is within the remaining budget."""
        if not isinstance(epsilon, (int, float)) or not isinstance(delta, (int, float)):
            raise ValueError("Requested epsilon and delta must be numeric")
        if math.isnan(epsilon) or math.isnan(delta):
            raise ValueError("Epsilon and delta cannot be NaN")
        if epsilon < 0 or delta < 0:
            raise ValueError("Requested epsilon and delta must be non-negative")

        eps_ok = (self.spent_epsilon + epsilon <= self.total_epsilon + 1e-12) or math.isinf(self.total_epsilon)
        delta_ok = (self.spent_delta + delta <= self.total_delta + 1e-12)
        return eps_ok and delta_ok

    def spend(self, epsilon: float, delta: float = 0.0, label: Optional[str] = None) -> Dict[str, Any]:
        """
        Record consumption of (epsilon, delta) privacy budget.
        Raises BudgetExhaustedError if the requested budget is unavailable.
        """
        if not isinstance(epsilon, (int, float)) or not isinstance(delta, (int, float)):
            raise ValueError("Requested epsilon and delta must be numeric")
        if math.isnan(epsilon) or math.isnan(delta):
            raise ValueError("Epsilon and delta cannot be NaN")
        if epsilon < 0 or delta < 0:
            raise ValueError("Requested epsilon and delta must be non-negative")

        if not self.has_budget(epsilon, delta):
            raise BudgetExhaustedError(
                f"Privacy budget exceeded on '{self.name}'! "
                f"Requested: (eps={epsilon:.4f}, delta={delta:.2e}), "
                f"Available: (eps={self.remaining_epsilon:.4f}, delta={self.remaining_delta:.2e}), "
                f"Total: (eps={self.total_epsilon:.4f}, delta={self.total_delta:.2e})"
            )

        self.spent_epsilon += float(epsilon)
        self.spent_delta += float(delta)

        entry = {
            "query_id": len(self.ledger) + 1,
            "timestamp": time.time(),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "label": label or f"query_{len(self.ledger) + 1}",
            "cumulative_epsilon": self.spent_epsilon,
            "cumulative_delta": self.spent_delta
        }
        self.ledger.append(entry)
        return entry

    @contextmanager
    def session(self, epsilon: float, delta: float = 0.0, label: Optional[str] = None):
        """
        Context manager for transactional execution with automatic rollback on exception.
        """
        start_ledger_len = len(self.ledger)
        entry = self.spend(epsilon, delta, label=label or "session_block")
        try:
            yield entry
        except Exception:
            # Safe rollback on failure
            self.spent_epsilon = max(0.0, self.spent_epsilon - float(epsilon))
            self.spent_delta = max(0.0, self.spent_delta - float(delta))
            if len(self.ledger) > start_ledger_len:
                self.ledger.pop()
            raise

    def split(self, num_splits: int) -> List["PrivacyBudgetTracker"]:
        """Split total budget equally into num_splits independent child trackers."""
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        eps_per_child = self.remaining_epsilon / num_splits
        delta_per_child = self.remaining_delta / num_splits

        children = []
        for i in range(num_splits):
            child = PrivacyBudgetTracker(
                total_epsilon=eps_per_child,
                total_delta=delta_per_child,
                name=f"{self.name}_child_{i + 1}"
            )
            children.append(child)
        return children

    def allocate(self, proportions: Sequence[float]) -> List["PrivacyBudgetTracker"]:
        """Allocate remaining budget according to custom weights/proportions."""
        props = np.asarray(proportions, dtype=float)
        if np.any(np.isnan(props)) or np.any(props < 0) or np.sum(props) <= 0:
            raise ValueError("Proportions must be non-negative, non-NaN, and sum to > 0")
        norm_props = props / np.sum(props)

        children = []
        for i, p in enumerate(norm_props):
            child = PrivacyBudgetTracker(
                total_epsilon=self.remaining_epsilon * float(p),
                total_delta=self.remaining_delta * float(p),
                name=f"{self.name}_alloc_{i + 1}"
            )
            children.append(child)
        return children

    def advanced_composition(self, delta_prime: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute total privacy spent under Advanced Composition Theorem (Dwork et al.).

        For k queries with eps_i and delta_i:
        eps_AC = sqrt(2 * ln(1 / delta') * sum(eps_i^2)) + sum(eps_i * (exp(eps_i) - 1))
        delta_AC = sum(delta_i) + delta'
        """
        if not self.ledger:
            return 0.0, 0.0

        if delta_prime is not None:
            if not isinstance(delta_prime, (int, float)) or math.isnan(delta_prime) or delta_prime <= 0 or delta_prime >= 1:
                raise ValueError(f"delta_prime must be strictly between 0 and 1, got {delta_prime}")

        eps_list = [entry["epsilon"] for entry in self.ledger]
        delta_list = [entry["delta"] for entry in self.ledger]

        sum_delta = sum(delta_list)
        if delta_prime is None:
            delta_prime = max(1e-6, (self.total_delta - sum_delta) / 2.0) if self.total_delta < 1.0 else 1e-5

        sum_eps_sq = sum(e ** 2 for e in eps_list)
        sum_linear_factor = sum(e * (math.exp(e) - 1.0) for e in eps_list)

        eps_ac = math.sqrt(2.0 * math.log(1.0 / delta_prime) * sum_eps_sq) + sum_linear_factor
        delta_ac = sum_delta + delta_prime

        return float(eps_ac), float(delta_ac)

    def reset(self):
        """Reset budget consumption and clear query ledger."""
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.ledger.clear()

class RenyiDPAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant.

    Tracks RDP order-by-order and computes optimal (epsilon, delta)-DP bounds.
    Order alpha in (1, inf).
    """

    DEFAULT_ALPHAS = [
        1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0,
        24.0, 28.0, 32.0, 48.0, 64.0, 128.0, 256.0
    ]

    def __init__(self, orders: Optional[Sequence[float]] = None):
        if orders is None:
            self.orders = np.array(self.DEFAULT_ALPHAS, dtype=float)
        else:
            self.orders = np.sort(np.asarray(orders, dtype=float))
            if np.any(np.isnan(self.orders)) or np.any(self.orders <= 1.0):
                raise ValueError("RDP orders alpha must be strictly greater than 1.0")

        # Accumulator for RDP epsilons across orders
        self.rdp = np.zeros(len(self.orders), dtype=float)
        self.history: List[Dict[str, Any]] = []

    def add_gaussian(self, sigma: float, sensitivity: float = 1.0, count: int = 1):
        """
        Add Gaussian mechanism RDP: eps(alpha) = alpha * sensitivity^2 / (2 * sigma^2).
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive (> 0), got {sigma}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive (> 0), got {sensitivity}")
        if count <= 0:
            raise ValueError(f"Count must be positive (> 0), got {count}")

        scaled_variance = (sigma / sensitivity) ** 2
        rdp_step = (self.orders / (2.0 * scaled_variance)) * float(count)
        self.rdp += rdp_step
        self.history.append({
            "type": "gaussian",
            "sigma": float(sigma),
            "sensitivity": float(sensitivity),
            "count": int(count)
        })

    def add_laplace(self, scale: float, sensitivity: float = 1.0, count: int = 1):
        """
        Add Laplace mechanism RDP using exact closed-form Rényi divergence.
        Uses stable factored formulation to prevent float overflow on large orders.
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive (> 0), got {scale}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive (> 0), got {sensitivity}")
        if count <= 0:
            raise ValueError(f"Count must be positive (> 0), got {count}")

        eps = sensitivity / scale
        rdp_step = np.zeros_like(self.orders)
        for i, alpha in enumerate(self.orders):
            # Factored form prevents overflow: (1/(alpha-1)) * log(e^((alpha-1)*eps) * (...))
            # = eps + (1/(alpha-1)) * log( alpha/(2*alpha-1) + ((alpha-1)/(2*alpha-1)) * exp(-(2*alpha-1)*eps) )
            term1 = alpha / (2.0 * alpha - 1.0)
            term2 = ((alpha - 1.0) / (2.0 * alpha - 1.0)) * math.exp(-min(500.0, (2.0 * alpha - 1.0) * eps))
            inner = term1 + term2
            rdp_val = eps + (1.0 / (alpha - 1.0)) * math.log(max(1e-15, inner))
            rdp_step[i] = rdp_val * float(count)

        self.rdp += rdp_step
        self.history.append({
            "type": "laplace",
            "scale": float(scale),
            "sensitivity": float(sensitivity),
            "count": int(count)
        })

    def add_subsampled_gaussian(self, q: float, sigma: float, count: int = 1):
        """
        Add Subsampled Gaussian mechanism RDP using standard tight bound.
        q: subsampling ratio in (0, 1]
        sigma: noise multiplier
        """
        if q <= 0 or q > 1.0:
            raise ValueError(f"Subsampling ratio q must be in (0, 1], got {q}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive (> 0), got {sigma}")
        if count <= 0:
            raise ValueError(f"Count must be positive (> 0), got {count}")

        rdp_step = np.zeros_like(self.orders)
        for i, alpha in enumerate(self.orders):
            # Standard order-dependent subsampled Gaussian RDP approximation
            rdp_step[i] = ((q ** 2) * alpha / (2.0 * (sigma ** 2))) * float(count)

        self.rdp += rdp_step
        self.history.append({
            "type": "subsampled_gaussian",
            "q": float(q),
            "sigma": float(sigma),
            "count": int(count)
        })

    def add_rdp(self, rdp_values: Sequence[float]):
        """Add arbitrary RDP curve matching self.orders."""
        vals = np.asarray(rdp_values, dtype=float)
        if vals.shape != self.rdp.shape:
            raise ValueError(f"Shape mismatch: expected {self.rdp.shape}, got {vals.shape}")
        if np.any(np.isnan(vals)) or np.any(vals < 0):
            raise ValueError("RDP values must be non-negative and non-NaN")
        self.rdp += vals

    def get_eps(self, target_delta: float) -> float:
        """
        Convert cumulative RDP to (epsilon, delta)-DP via:
        eps(delta) = min_{alpha > 1} ( RDP(alpha) + ln(1 / target_delta) / (alpha - 1) )
        """
        if not isinstance(target_delta, (int, float)) or math.isnan(target_delta) or target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")

        eps_candidates = self.rdp + np.log(1.0 / target_delta) / (self.orders - 1.0)
        return float(np.min(eps_candidates))

    def reset(self):
        """Reset accumulated RDP."""
        self.rdp.fill(0.0)
        self.history.clear()

class MomentsAccountant:
    """
    Moments Accountant for iterative privacy mechanisms (Abadi et al. 2016).
    Tracks the log of moment generating functions (log-MGF) across moments lambda.
    """

    def __init__(self, max_lambda: int = 32):
        if not isinstance(max_lambda, int) or max_lambda < 1:
            raise ValueError(f"max_lambda must be an integer >= 1, got {max_lambda}")
        self.lambdas = np.arange(1, max_lambda + 1, dtype=float)
        self.log_moments = np.zeros(max_lambda, dtype=float)

    def accumulate_subsampled_gaussian(self, q: float, sigma: float, steps: int = 1):
        """
        Accumulate log-MGF bound for subsampled Gaussian mechanism.
        q: subsampling ratio (batch_size / total_samples).
        sigma: noise multiplier (noise_std / clip_norm).
        """
        if not isinstance(q, (int, float)) or q <= 0 or q > 1.0:
            raise ValueError(f"Subsampling ratio q must be in (0, 1], got {q}")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"Steps must be a positive integer, got {steps}")

        for i, lmbda in enumerate(self.lambdas):
            alpha_step = (q ** 2) * lmbda * (lmbda + 1.0) / (2.0 * (sigma ** 2))
            self.log_moments[i] += alpha_step * steps

    def get_epsilon(self, target_delta: float) -> float:
        """
        Compute minimum epsilon for a given target delta:
        eps = min_{lambda} ( (log_moments(lambda) + ln(1 / delta)) / lambda )
        """
        if not isinstance(target_delta, (int, float)) or math.isnan(target_delta) or target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"Target delta must be in (0, 1), got {target_delta}")

        eps_candidates = (self.log_moments + math.log(1.0 / target_delta)) / self.lambdas
        return float(np.min(eps_candidates))

    def reset(self):
        """Reset log moments."""
        self.log_moments.fill(0.0)
