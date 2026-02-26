"""Multi-objective optimization for AlphaScale model size selection."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from scaling.surface_fit import ScalingSurfaceFitter, power_law
from scaling.bootstrap import BootstrapUncertainty
from training.energy import estimate_energy_kwh


class ScaleOptimizer:
    """Finds the optimal model size given accuracy target and budget constraints.

    Uses the fitted scaling surface to minimize compute (FLOPs) subject
    to a predicted accuracy constraint.

    The search is performed over a grid of candidate parameter counts.

    Args:
        fitter: A fitted ScalingSurfaceFitter instance.
        bootstrap: Optional BootstrapUncertainty for CI-based risk scoring.
        gpu_wattage: GPU power for energy estimation (watts).
        reference_flops_per_param: Rough FLOPs-per-parameter estimate for energy calc.
    """

    def __init__(
        self,
        fitter: ScalingSurfaceFitter,
        bootstrap: Optional[BootstrapUncertainty] = None,
        gpu_wattage: float = 250.0,
        reference_flops_per_param: float = 6.0,
    ) -> None:
        if not fitter.fitted_:
            raise ValueError("Fitter must be fitted before optimization.")
        self.fitter = fitter
        self.bootstrap = bootstrap
        self.gpu_wattage = gpu_wattage
        self.reference_flops_per_param = reference_flops_per_param

    def _compute_to_energy_kwh(self, flops: float, throughput_tflops: float = 12.0) -> float:
        """Convert FLOPs estimate to approximate energy.

        Uses GPU throughput (default: Titan V at ~12 TFLOP/s) to estimate
        wall-clock training time, then converts to energy.

        Args:
            flops: Total FLOPs for the full training run.
            throughput_tflops: GPU throughput in TFLOP/s (Titan V default).

        Returns:
            Estimated energy in kWh.
        """
        train_seconds = flops / (throughput_tflops * 1e12)
        return estimate_energy_kwh(train_seconds, self.gpu_wattage)

    def _train_time_to_energy_kwh(self, train_time_seconds: float) -> float:
        """Convert real measured training time to energy.

        Preferred over FLOPs-based estimation when real train_time is available.

        Args:
            train_time_seconds: Measured wall-clock training time in seconds.

        Returns:
            Estimated energy in kWh.
        """
        return estimate_energy_kwh(train_time_seconds, self.gpu_wattage)

    def optimize(
        self,
        target_accuracy: float,
        observed_params: np.ndarray,
        observed_accuracies: Optional[np.ndarray] = None,
        observed_compute: Optional[np.ndarray] = None,
        observed_train_times: Optional[np.ndarray] = None,
        budget_flops: Optional[float] = None,
        n_candidates: int = 1000,
    ) -> Dict:
        """Find minimum compute model achieving target accuracy.

        Args:
            target_accuracy: Minimum required predicted accuracy (0 to 1).
            observed_params: Parameter counts from scaling experiments.
            observed_accuracies: Real observed accuracy values (used for bootstrap).
            observed_compute: Corresponding FLOPs (optional).
            observed_train_times: Real measured training times in seconds (optional).
            budget_flops: Maximum allowable FLOPs (optional constraint).
            n_candidates: Number of candidate N values in grid search.

        Returns:
            Dictionary with keys:
            - 'optimal_n': Optimal parameter count.
            - 'expected_accuracy': Predicted accuracy at optimal N.
            - 'expected_energy_kwh': Estimated energy usage.
            - 'expected_compute_flops': Estimated FLOPs.
            - 'ci_lower': Lower 95% CI on accuracy (if bootstrap provided).
            - 'ci_upper': Upper 95% CI on accuracy.
            - 'risk_score': CI width.
            - 'feasible': Whether any N meets the constraint.
            - 'baseline_compute': FLOPs of the largest observed model (for savings calc).
            - 'compute_saved_fraction': Fraction of compute saved vs. largest model.
        """
        N_min = float(observed_params.min())
        N_max = float(observed_params.max()) * 3.0  # Allow extrapolation
        n_search = np.linspace(N_min, N_max, n_candidates)

        # Apply budget constraint
        if budget_flops is not None and observed_compute is not None:
            # Estimate FLOPs proportional to params
            flops_per_param = observed_compute.mean() / observed_params.mean()
            n_budget = budget_flops / flops_per_param
            n_search = n_search[n_search <= n_budget]
            if len(n_search) == 0:
                return {
                    "feasible": False,
                    "reason": "Budget too restrictive — no candidates within budget.",
                    "optimal_n": None,
                }

        preds = self.fitter.predict(n_search)

        feasible_mask = preds >= target_accuracy
        if not feasible_mask.any():
            return {
                "feasible": False,
                "reason": f"Target accuracy {target_accuracy:.4f} unreachable within search range.",
                "optimal_n": None,
                "max_predicted_accuracy": float(preds.max()),
            }

        # Minimum compute among feasible candidates = minimum N (FLOPs ∝ N)
        feasible_idx = np.where(feasible_mask)[0]
        optimal_idx = feasible_idx[0]  # Smallest N that meets threshold
        optimal_n = float(n_search[optimal_idx])
        expected_acc = float(preds[optimal_idx])

        # Compute estimates
        if observed_compute is not None and len(observed_compute) > 0:
            flops_per_param = np.polyfit(observed_params, observed_compute, 1)[0]
            optimal_flops = max(0.0, flops_per_param * optimal_n)
            baseline_flops = float(observed_compute.max())
        else:
            optimal_flops = optimal_n * self.reference_flops_per_param
            baseline_flops = float(observed_params.max()) * self.reference_flops_per_param

        # Use real train times for energy if available, otherwise fall back to FLOPs
        if observed_train_times is not None and len(observed_train_times) > 0:
            # Interpolate expected train time at optimal_n using linear fit
            time_per_param = np.polyfit(observed_params, observed_train_times, 1)[0]
            expected_train_seconds = max(0.0, time_per_param * optimal_n)
            expected_energy = self._train_time_to_energy_kwh(expected_train_seconds)
            baseline_train_seconds = float(
                observed_train_times[np.argmax(observed_params)]
            )
            baseline_energy = self._train_time_to_energy_kwh(baseline_train_seconds)
        else:
            expected_energy = self._compute_to_energy_kwh(optimal_flops)
            baseline_energy = self._compute_to_energy_kwh(baseline_flops)

        compute_saved = max(0.0, 1.0 - optimal_flops / baseline_flops) if baseline_flops > 0 else 0.0

        result: Dict = {
            "feasible": True,
            "optimal_n": optimal_n,
            "expected_accuracy": round(expected_acc, 6),
            "expected_energy_kwh": round(expected_energy, 8),
            "baseline_energy_kwh": round(baseline_energy, 8),
            "expected_compute_flops": optimal_flops,
            "baseline_compute": baseline_flops,
            "compute_saved_fraction": round(compute_saved, 4),
            "ci_lower": None,
            "ci_upper": None,
            "risk_score": None,
        }

        # Bootstrap CI if available — MUST use real observed accuracies, not predicted
        # Passing predicted values back in collapses the CI to zero variance
        bootstrap_accuracies = observed_accuracies if observed_accuracies is not None \
            else self.fitter.predict(observed_params)

        if self.bootstrap is not None:
            try:
                bs_result = self.bootstrap.run(
                    observed_params,
                    bootstrap_accuracies,
                    target_accuracy=target_accuracy,
                    n_search=n_search,
                )
                # Get CI at optimal_n
                n_search_list = np.array(bs_result["n_search"])
                closest_idx = int(np.argmin(np.abs(n_search_list - optimal_n)))
                result["ci_lower"] = float(np.array(bs_result["ci_lower"])[closest_idx])
                result["ci_upper"] = float(np.array(bs_result["ci_upper"])[closest_idx])
                result["risk_score"] = bs_result.get("risk_score")
                result["optimal_n_ci"] = bs_result.get("optimal_n_ci")
            except Exception:
                pass  # Bootstrap failed — skip CI

        return result

    def batch_optimize(
        self,
        target_accuracies: List[float],
        observed_params: np.ndarray,
        observed_accuracies: Optional[np.ndarray] = None,
        observed_compute: Optional[np.ndarray] = None,
        observed_train_times: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Optimize for multiple target accuracy levels.

        Args:
            target_accuracies: List of accuracy targets.
            observed_params: Parameter counts from experiments.
            observed_accuracies: Real observed accuracy values.
            observed_compute: Corresponding FLOPs.
            observed_train_times: Real measured training times in seconds.

        Returns:
            List of optimization result dicts.
        """
        return [
            self.optimize(tau, observed_params, observed_accuracies,
                          observed_compute, observed_train_times)
            for tau in target_accuracies
        ]
