"""Bootstrap uncertainty quantification for AlphaScale scaling surface."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from scaling.surface_fit import ScalingSurfaceFitter


class BootstrapUncertainty:
    """Estimates uncertainty in scaling surface predictions via bootstrap resampling.

    For each resample, refits the power-law surface and records predictions
    at target parameter counts. Returns mean and 95% confidence intervals.

    Args:
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        use_compute: Whether to use compute-augmented fit.
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        seed: int = 42,
        use_compute: bool = False,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.use_compute = use_compute

    def run(
        self,
        param_counts: np.ndarray,
        accuracies: np.ndarray,
        compute: Optional[np.ndarray] = None,
        target_accuracy: Optional[float] = None,
        n_search: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run bootstrap resampling and compute uncertainty estimates.

        Args:
            param_counts: Observed parameter counts.
            accuracies: Observed accuracy values.
            compute: Observed FLOPs (optional, for compute-augmented fit).
            target_accuracy: Target accuracy for optimal N estimation.
            n_search: Search grid for optimal N estimation.

        Returns:
            Dictionary with keys:
            - 'bootstrap_predictions': (n_bootstrap, len(n_search)) array if n_search given.
            - 'mean_predictions': Mean predicted accuracy over n_search grid.
            - 'ci_lower': 2.5th percentile (lower 95% CI bound).
            - 'ci_upper': 97.5th percentile (upper 95% CI bound).
            - 'optimal_n_samples': List of optimal N per resample.
            - 'optimal_n_mean': Mean optimal N.
            - 'optimal_n_ci': [lower, upper] 95% CI for optimal N.
            - 'risk_score': CI width (measure of uncertainty).
            - 'n_failed': Number of failed fits.
        """
        rng = np.random.RandomState(self.seed)
        n_points = len(param_counts)

        N = np.array(param_counts, dtype=float)
        y = np.array(accuracies, dtype=float)
        C = np.array(compute, dtype=float) if compute is not None else None

        if n_search is None:
            n_search = np.linspace(N.min(), N.max() * 2, 200)

        bootstrap_preds: List[np.ndarray] = []
        optimal_n_samples: List[float] = []
        n_failed = 0

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n_points, size=n_points, replace=True)
            N_b = N[idx]
            y_b = y[idx]
            C_b = C[idx] if C is not None else None

            # Need at least 2 unique points
            if len(np.unique(N_b)) < 2:
                n_failed += 1
                continue

            fitter = ScalingSurfaceFitter(use_compute=self.use_compute)
            try:
                fitter.fit(N_b, y_b, compute=C_b)
            except RuntimeError:
                n_failed += 1
                continue

            preds = fitter.predict(n_search, C_b.mean() * np.ones_like(n_search) if C_b is not None else None)
            bootstrap_preds.append(preds)

            if target_accuracy is not None:
                opt_n = fitter.predict_optimal_n(target_accuracy, n_search)
                if opt_n is not None:
                    optimal_n_samples.append(opt_n)

        if len(bootstrap_preds) == 0:
            raise RuntimeError("All bootstrap fits failed. Check data quality.")

        preds_arr = np.array(bootstrap_preds)  # (n_success, len(n_search))
        mean_preds = preds_arr.mean(axis=0)
        ci_lower = np.percentile(preds_arr, 2.5, axis=0)
        ci_upper = np.percentile(preds_arr, 97.5, axis=0)

        result: Dict = {
            "n_search": n_search.tolist(),
            "mean_predictions": mean_preds.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "n_failed": n_failed,
            "n_success": len(bootstrap_preds),
        }

        if len(optimal_n_samples) > 0:
            opt_arr = np.array(optimal_n_samples)
            opt_ci_lower = float(np.percentile(opt_arr, 2.5))
            opt_ci_upper = float(np.percentile(opt_arr, 97.5))
            risk_score = opt_ci_upper - opt_ci_lower

            result.update({
                "optimal_n_samples": optimal_n_samples,
                "optimal_n_mean": float(opt_arr.mean()),
                "optimal_n_ci": [opt_ci_lower, opt_ci_upper],
                "risk_score": float(risk_score),
            })
        else:
            result.update({
                "optimal_n_samples": [],
                "optimal_n_mean": None,
                "optimal_n_ci": [None, None],
                "risk_score": None,
            })

        return result
