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
    """Compute Akaike Information Criterion."""
    sse = np.sum(residuals ** 2)
    if sse <= 0 or n_points <= n_params:
        return float("inf")
    sigma2 = sse / n_points
    log_likelihood = -n_points / 2 * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2)
    return 2 * n_params - 2 * log_likelihood


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    More informative than R² when the dynamic range of accuracy values
    is narrow (e.g. tasks that saturate quickly). MAE directly reports
    average prediction error in accuracy units.

    Args:
        y_true: Ground truth accuracy values.
        y_pred: Predicted accuracy values.

    Returns:
        Mean absolute error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


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
        self.mae_: Optional[float] = None
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
            Dictionary with fitted parameters, r2, mae, aic, covariance.

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
            except Exception as e:
                raise RuntimeError(f"Compute-augmented fit failed: {e}")

            result = {"a": float(a), "b": float(b), "alpha": float(alpha), "beta": float(beta)}
        else:
            # Multi-restart fitting with fixed bounds.
            # a is capped below 1.0 (accuracy is a probability).
            # b upper bound raised to 1e8 to handle high-accuracy tasks
            # where the curve must rise steeply from a low baseline.
            # Four candidate starting points are tried; best R² is kept.
            try:
                a_upper  = 0.9999
                a_init   = min(float(np.max(y)) + 0.005, a_upper)
                N_min    = float(np.min(N))
                spread   = float(np.max(y) - np.min(y))

                best_popt, best_pcov, best_r2 = None, None, -np.inf
                candidate_inits = [
                    (a_init, max(spread, (a_init - float(np.min(y))) * N_min ** 0.5), 0.5),
                    (a_init, max(spread, (a_init - float(np.min(y))) * N_min ** 1.0), 1.0),
                    (a_init, max(spread, (a_init - float(np.min(y))) * N_min ** 1.5), 1.5),
                    (a_init, spread if spread > 0 else 0.01,                           0.3),
                ]
                for p0 in candidate_inits:
                    try:
                        popt_c, pcov_c = curve_fit(
                            power_law, N, y,
                            p0=list(p0),
                            bounds=([0.0, 0.0, 1e-6], [a_upper, 1e8, 5.0]),
                            maxfev=50000,
                        )
                        y_pred_c = power_law(N, *popt_c)
                        ss_res = float(np.sum((y - y_pred_c) ** 2))
                        ss_tot = float(np.sum((y - y.mean()) ** 2))
                        r2_c = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                        if r2_c > best_r2:
                            best_r2, best_popt, best_pcov = r2_c, popt_c, pcov_c
                    except Exception:
                        continue

                if best_popt is None:
                    raise RuntimeError("All candidate starting points failed.")
                popt, pcov = best_popt, best_pcov
            except Exception as e:
                raise RuntimeError(f"Power-law fit failed: {e}")

            a, b, alpha = popt
            y_pred = power_law(N, a, b, alpha)
            result = {"a": float(a), "b": float(b), "alpha": float(alpha)}

        residuals = y - y_pred
        self.params_ = popt
        self.covariance_ = pcov
        self.r2_ = _compute_r2(y, y_pred)
        self.mae_ = _compute_mae(y, y_pred)
        self.aic_ = _compute_aic(len(y), len(popt), residuals)
        self.fitted_ = True

        result.update({
            "r2":         round(self.r2_, 6),
            "mae":        round(self.mae_, 6),
            "aic":        round(self.aic_, 4),
            "covariance": self.covariance_.tolist(),
        })
        return result

    def predict(self, N: np.ndarray, compute: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict accuracy for given parameter counts."""
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
        """Find minimum N achieving target accuracy."""
        if not self.fitted_:
            raise RuntimeError("Fit must be called before predict_optimal_n.")

        preds = self.predict(n_search, compute_search)
        feasible = np.where(preds >= target_accuracy)[0]

        if len(feasible) == 0:
            return None

        return float(n_search[feasible[0]])
