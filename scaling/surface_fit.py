"""Power-law scaling surface fitting for AlphaScale."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def power_law(N: np.ndarray, a: float, b: float, alpha: float) -> np.ndarray:
    """Compute the power-law scaling prediction.

    Accuracy = a - b * N^(-alpha)

    This models the empirical observation that accuracy increases as a
    power law of model size, with diminishing returns.

    Args:
        N: Array of parameter counts.
        a: Asymptotic maximum accuracy.
        b: Scaling coefficient.
        alpha: Power law exponent.

    Returns:
        Predicted accuracy values.
    """
    return a - b * np.power(N.astype(float), -alpha)


def power_law_compute(
    N: np.ndarray, C: np.ndarray, a: float, b: float, alpha: float, beta: float
) -> np.ndarray:
    """Power law augmented with compute (FLOPs).

    Accuracy = a - b * N^(-alpha) * C^(-beta)

    Args:
        N: Parameter counts.
        C: Compute (FLOPs).
        a: Asymptotic accuracy.
        b: Scaling coefficient.
        alpha: Parameter exponent.
        beta: Compute exponent.

    Returns:
        Predicted accuracy values.
    """
    return a - b * np.power(N.astype(float), -alpha) * np.power(C.astype(float), -beta)


def _compute_aic(n_points: int, n_params: int, residuals: np.ndarray) -> float:
    """Compute Akaike Information Criterion.

    Args:
        n_points: Number of data points.
        n_params: Number of model parameters.
        residuals: Residual errors.

    Returns:
        AIC value.
    """
    sse = np.sum(residuals ** 2)
    if sse <= 0 or n_points <= n_params:
        return float("inf")
    sigma2 = sse / n_points
    log_likelihood = -n_points / 2 * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2)
    return 2 * n_params - 2 * log_likelihood


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R².

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        R² value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


class ScalingSurfaceFitter:
    """Fits a power-law scaling surface to empirical accuracy-vs-params data.

    Args:
        use_compute: Whether to fit the compute-augmented version.
    """

    def __init__(self, use_compute: bool = False) -> None:
        self.use_compute = use_compute
        self.params_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.r2_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.fitted_: bool = False

    def fit(
        self,
        param_counts: np.ndarray,
        accuracies: np.ndarray,
        compute: Optional[np.ndarray] = None,
    ) -> Dict:
        """Fit power-law curve to accuracy vs. parameter count data.

        Args:
            param_counts: Array of model parameter counts.
            accuracies: Array of corresponding accuracy values.
            compute: Array of FLOPs (required if use_compute=True).

        Returns:
            Dictionary with keys:
            - 'a', 'b', 'alpha': Fitted parameters.
            - 'r2': Coefficient of determination.
            - 'aic': Akaike Information Criterion.
            - 'covariance': Parameter covariance matrix.

        Raises:
            RuntimeError: If curve fitting fails.
        """
        N = np.array(param_counts, dtype=float)
        y = np.array(accuracies, dtype=float)

        if len(N) < 3:
            raise ValueError("At least 3 data points required for fitting.")

        if self.use_compute and compute is not None:
            C = np.array(compute, dtype=float)
            try:
                popt, pcov = curve_fit(
                    lambda NC, a, b, alpha, beta: power_law_compute(NC[0], NC[1], a, b, alpha, beta),
                    (N, C),
                    y,
                    p0=[1.0, 1.0, 0.5, 0.5],
                    bounds=([0, 0, 0, 0], [2, 100, 5, 5]),
                    maxfev=10000,
                )
                a, b, alpha, beta = popt
                y_pred = power_law_compute(N, C, a, b, alpha, beta)
                param_names = ["a", "b", "alpha", "beta"]
            except Exception as e:
                raise RuntimeError(f"Compute-augmented fit failed: {e}")

            result = {"a": float(a), "b": float(b), "alpha": float(alpha), "beta": float(beta)}
        else:
            try:
                # Smart initial guess for b: when accuracy is already high
                # (e.g. NLP tasks that saturate quickly), b must be large
                # because b * N^(-alpha) must equal (a - min_acc) at N_min.
                # We estimate b from the smallest scale point directly.
                a_init = float(np.max(y)) + 0.01  # ceiling slightly above max
                alpha_init = 0.5
                N_min = float(np.min(N))
                b_init = max(
                    float(np.max(y) - np.min(y)),          # spread (works for vision)
                    (a_init - float(np.min(y))) * (N_min ** alpha_init),  # NLP-aware
                )
                p0 = [a_init, b_init, alpha_init]
                popt, pcov = curve_fit(
                    power_law,
                    N,
                    y,
                    p0=p0,
                    bounds=([0, 0, 1e-6], [2.0, 1e8, 5.0]),  # b upper bound 1e8
                    maxfev=50000,
                )
            except Exception as e:
                raise RuntimeError(f"Power-law fit failed: {e}")

            a, b, alpha = popt
            y_pred = power_law(N, a, b, alpha)
            result = {"a": float(a), "b": float(b), "alpha": float(alpha)}

        residuals = y - y_pred
        self.params_ = popt
        self.covariance_ = pcov
        self.r2_ = _compute_r2(y, y_pred)
        self.aic_ = _compute_aic(len(y), len(popt), residuals)
        self.fitted_ = True

        result.update({
            "r2": round(self.r2_, 6),
            "aic": round(self.aic_, 4),
            "covariance": self.covariance_.tolist(),
        })
        return result

    def predict(self, N: np.ndarray, compute: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict accuracy for given parameter counts.

        Args:
            N: Array of parameter counts to predict for.
            compute: FLOPs (required if use_compute=True).

        Returns:
            Predicted accuracy values.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.fitted_:
            raise RuntimeError("Fit must be called before predict.")

        N = np.array(N, dtype=float)
        if self.use_compute and compute is not None and len(self.params_) == 4:
            C = np.array(compute, dtype=float)
            a, b, alpha, beta = self.params_
            return power_law_compute(N, C, a, b, alpha, beta)
        else:
            a, b, alpha = self.params_[:3]
            return power_law(N, a, b, alpha)

    def predict_optimal_n(
        self,
        target_accuracy: float,
        n_search: np.ndarray,
        compute_search: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Find minimum N achieving target accuracy.

        Args:
            target_accuracy: Desired accuracy threshold.
            n_search: Candidate parameter counts to search over.
            compute_search: Corresponding FLOPs if use_compute.

        Returns:
            Optimal parameter count, or None if target is unreachable.
        """
        if not self.fitted_:
            raise RuntimeError("Fit must be called before predict_optimal_n.")

        preds = self.predict(n_search, compute_search)
        feasible = np.where(preds >= target_accuracy)[0]

        if len(feasible) == 0:
            return None

        return float(n_search[feasible[0]])