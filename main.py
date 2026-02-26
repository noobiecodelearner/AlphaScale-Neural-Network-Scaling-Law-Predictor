"""AlphaScale — Main CLI entry point.

Usage examples:
    python main.py --domain vision --dataset_fraction 0.5 --run_scaling
    python main.py --domain vision --dataset_fraction 1.0 --run_scaling --verbose
    python main.py --fit_surface --domain vision
    python main.py --optimize --domain vision --target_accuracy 0.85
    python main.py --domain all --run_scaling
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.seed import set_seed
from utils.logger import ExperimentLogger


def _load_domain_results(
    domain: str,
    dataset_fraction: Optional[float],
    log_path: str,
) -> pd.DataFrame:
    """Load experiment results for a given domain from the CSV log.

    Args:
        domain: Domain name.
        dataset_fraction: If given, filter to this fraction.
        log_path: CSV log path.

    Returns:
        Filtered DataFrame.
    """
    logger = ExperimentLogger(log_path)
    records = logger.load()
    if not records:
        raise FileNotFoundError(f"No results found at {log_path}. Run --run_scaling first.")

    df = pd.DataFrame(records)
    numeric_cols = ["dataset_fraction", "params", "compute", "energy",
                    "val_accuracy", "test_accuracy", "train_time", "generalization_gap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["domain"] == domain]
    if dataset_fraction is not None:
        df = df[df["dataset_fraction"] == dataset_fraction]

    return df.sort_values("params").reset_index(drop=True)


def cmd_run_scaling(args: argparse.Namespace) -> None:
    """Execute scaling experiments for the requested domain(s) and fraction.

    Args:
        args: Parsed CLI arguments.
    """
    from scaling.scaling_runner import ScalingRunner

    domains = ["vision", "nlp", "tabular"] if args.domain == "all" else [args.domain]

    for domain in domains:
        config_path = str(Path(args.config_dir) / f"{domain}.yaml")
        print(f"\n{'='*60}")
        print(f" AlphaScale | Domain: {domain.upper()} | Fraction: {args.dataset_fraction}")
        print(f"{'='*60}")

        runner = ScalingRunner(
            config_path=config_path,
            log_path=args.log_path,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

        # Override fractions if a specific one is requested
        if args.dataset_fraction is not None:
            runner.cfg["dataset_fractions"] = [args.dataset_fraction]

        results = runner.run()
        print(f"\n✅ Logged {len(results)} experiments to {args.log_path}")


def cmd_fit_surface(args: argparse.Namespace) -> None:
    """Fit a power-law scaling surface to logged results.

    Args:
        args: Parsed CLI arguments.
    """
    from scaling.surface_fit import ScalingSurfaceFitter

    df = _load_domain_results(args.domain, args.dataset_fraction, args.log_path)

    if len(df) < 3:
        print(f"[ERROR] Need at least 3 data points to fit. Found {len(df)}.")
        sys.exit(1)

    param_counts = df["params"].values.astype(float)
    accuracies = df["val_accuracy"].values.astype(float)
    compute_vals = df["compute"].values.astype(float) if "compute" in df.columns else None

    fitter = ScalingSurfaceFitter(use_compute=args.use_compute)
    try:
        result = fitter.fit(param_counts, accuracies, compute=compute_vals if args.use_compute else None)
    except RuntimeError as e:
        print(f"[ERROR] Fitting failed: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f" Scaling Surface Fit — {args.domain.upper()}")
    print(f"{'='*50}")
    print(f"  Model: Accuracy = a - b * N^(-alpha)")
    print(f"  a     = {result['a']:.6f}")
    print(f"  b     = {result['b']:.6f}")
    print(f"  alpha = {result['alpha']:.6f}")
    print(f"  R²    = {result['r2']:.6f}")
    print(f"  AIC   = {result['aic']:.4f}")

    if args.bootstrap:
        from scaling.bootstrap import BootstrapUncertainty
        bootstrapper = BootstrapUncertainty(n_bootstrap=100, seed=args.seed)
        n_grid = np.linspace(param_counts.min(), param_counts.max() * 2, 300)
        bs = bootstrapper.run(
            param_counts, accuracies,
            target_accuracy=args.target_accuracy,
            n_search=n_grid,
        )
        print(f"\n  Bootstrap (100 resamples):")
        print(f"  Success: {bs['n_success']} | Failed: {bs['n_failed']}")
        if bs["optimal_n_mean"] is not None:
            print(f"  Optimal N* mean : {bs['optimal_n_mean']:,.0f}")
            print(f"  Optimal N* 95%CI: [{bs['optimal_n_ci'][0]:,.0f}, {bs['optimal_n_ci'][1]:,.0f}]")
            print(f"  Risk score      : {bs['risk_score']:,.0f}")


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run multi-objective optimization given a target accuracy.

    Args:
        args: Parsed CLI arguments.
    """
    from scaling.surface_fit import ScalingSurfaceFitter
    from scaling.bootstrap import BootstrapUncertainty
    from optimization.optimizer import ScaleOptimizer
    from training.energy import estimate_carbon_grams

    df = _load_domain_results(args.domain, args.dataset_fraction, args.log_path)

    if len(df) < 3:
        print(f"[ERROR] Need at least 3 data points. Found {len(df)}.")
        sys.exit(1)

    param_counts = df["params"].values.astype(float)
    accuracies = df["val_accuracy"].values.astype(float)
    compute_vals = df["compute"].values.astype(float) if "compute" in df.columns else None
    train_times = df["train_time"].values.astype(float) if "train_time" in df.columns else None

    fitter = ScalingSurfaceFitter(use_compute=False)
    try:
        fitter.fit(param_counts, accuracies)
    except RuntimeError as e:
        print(f"[ERROR] Fitting failed: {e}")
        sys.exit(1)

    bootstrapper = BootstrapUncertainty(n_bootstrap=100, seed=args.seed)
    optimizer = ScaleOptimizer(fitter=fitter, bootstrap=bootstrapper)

    budget_flops = args.budget_gflops * 1e9 if args.budget_gflops else None

    result = optimizer.optimize(
        target_accuracy=args.target_accuracy,
        observed_params=param_counts,
        observed_accuracies=accuracies,       # real values for bootstrap
        observed_compute=compute_vals,
        observed_train_times=train_times,     # real times for energy
        budget_flops=budget_flops,
    )

    print(f"\n{'='*55}")
    print(f" AlphaScale Optimization — {args.domain.upper()}")
    print(f" Target accuracy: τ = {args.target_accuracy:.4f}")
    print(f"{'='*55}")

    if not result["feasible"]:
        print(f"  ❌ Not feasible: {result.get('reason')}")
        if result.get("max_predicted_accuracy"):
            print(f"  Max reachable accuracy: {result['max_predicted_accuracy']:.4f}")
    else:
        from training.energy import estimate_carbon_grams
        carbon_saved_g = estimate_carbon_grams(
            result["baseline_energy_kwh"] - result["expected_energy_kwh"]
        )
        print(f"  ✅ Optimal N*      : {result['optimal_n']:,.0f} parameters")
        print(f"  Expected accuracy  : {result['expected_accuracy']:.4f}")
        print(f"  Energy estimate    : {result['expected_energy_kwh']:.6f} kWh")
        print(f"  Energy (baseline)  : {result['baseline_energy_kwh']:.6f} kWh")
        print(f"  Carbon saved       : {carbon_saved_g:.2f} g CO₂")
        print(f"  Compute saved      : {result['compute_saved_fraction']:.2%}")
        if result["ci_lower"] is not None:
            print(f"  95% CI accuracy    : [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        if result.get("risk_score"):
            print(f"  Risk score         : {result['risk_score']:,.0f} (CI width in params)")


def cmd_generate_graphs(args: argparse.Namespace) -> None:
    """Generate all competition graphs from logged experiment results.

    Args:
        args: Parsed CLI arguments.
    """
    from scaling.surface_fit import ScalingSurfaceFitter
    from scaling.bootstrap import BootstrapUncertainty
    from optimization.optimizer import ScaleOptimizer
    from training.energy import estimate_carbon_grams
    from scaling.graphs import generate_all_graphs

    domains = ["vision", "nlp", "tabular"] if args.domain == "all" else [args.domain]
    domain_data: dict = {}

    for domain in domains:
        try:
            df = _load_domain_results(domain, args.dataset_fraction, args.log_path)
        except FileNotFoundError:
            print(f"[SKIP] No results for domain={domain}. Run --run_scaling first.")
            continue

        if len(df) < 3:
            print(f"[SKIP] Too few results for domain={domain} (need ≥3, got {len(df)}).")
            continue

        param_counts  = df["params"].values.astype(float)
        accuracies    = df["val_accuracy"].values.astype(float)
        compute_vals  = df["compute"].values.astype(float) if "compute" in df.columns else None
        train_times   = df["train_time"].values.astype(float) if "train_time" in df.columns else None
        naive_idx     = int(np.argmax(param_counts))

        fitter = ScalingSurfaceFitter(use_compute=False)
        try:
            fit_result = fitter.fit(param_counts, accuracies)
        except RuntimeError as e:
            print(f"[SKIP] Fitting failed for {domain}: {e}")
            continue

        bootstrapper = BootstrapUncertainty(n_bootstrap=100, seed=args.seed)
        n_grid = np.linspace(param_counts.min(), param_counts.max() * 2, 300)
        bs = bootstrapper.run(param_counts, accuracies,
                              target_accuracy=args.target_accuracy, n_search=n_grid)

        optimizer = ScaleOptimizer(fitter=fitter, bootstrap=bootstrapper)
        opt = optimizer.optimize(
            target_accuracy=args.target_accuracy,
            observed_params=param_counts,
            observed_accuracies=accuracies,
            observed_compute=compute_vals,
            observed_train_times=train_times,
        )

        baseline_energy = opt.get("baseline_energy_kwh", 0.0)
        expected_energy = opt.get("expected_energy_kwh", 0.0)
        carbon_saved_g  = estimate_carbon_grams(max(0.0, baseline_energy - expected_energy))

        domain_data[domain] = {
            "param_counts":          param_counts,
            "accuracies":            accuracies,
            "compute":               compute_vals if compute_vals is not None else np.array([]),
            "n_search":              np.array(bs["n_search"]),
            "mean_preds":            np.array(bs["mean_predictions"]),
            "ci_lower":              np.array(bs["ci_lower"]),
            "ci_upper":              np.array(bs["ci_upper"]),
            "predicted_at_observed": fitter.predict(param_counts),
            "target_accuracy":       args.target_accuracy,
            "optimal_n":             opt.get("optimal_n"),
            "optimal_acc":           opt.get("expected_accuracy"),
            "optimal_flops":         opt.get("expected_compute_flops"),
            "naive_n":               int(param_counts[naive_idx]),
            "naive_acc":             float(accuracies[naive_idx]),
            "baseline_flops":        opt.get("baseline_compute"),
            "baseline_energy_kwh":   baseline_energy,
            "expected_energy_kwh":   expected_energy,
            "compute_saved_fraction":opt.get("compute_saved_fraction", 0.0),
            "carbon_saved_g":        carbon_saved_g,
            "fit_params": {
                "a":     fit_result["a"],
                "b":     fit_result["b"],
                "alpha": fit_result["alpha"],
                "beta":  fit_result.get("beta", 0.0),
            },
        }

    if not domain_data:
        print("[ERROR] No domain data available to plot.")
        sys.exit(1)

    generate_all_graphs(domain_data, output_dir=args.graph_dir)


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="alphascale",
        description="AlphaScale — Neural Network Scaling Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared arguments
    parser.add_argument("--domain", type=str, default="vision",
                        choices=["vision", "nlp", "tabular", "all"],
                        help="Domain to operate on.")
    parser.add_argument("--dataset_fraction", type=float, default=None,
                        help="Dataset fraction (0.25, 0.5, 1.0).")
    parser.add_argument("--log_path", type=str, default="results/experiments.csv",
                        help="Path to experiment log CSV.")
    parser.add_argument("--config_dir", type=str, default="configs",
                        help="Directory containing domain YAML configs.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to train on: auto, cuda, or cpu (default: auto).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-epoch training progress.")

    # Command flags
    parser.add_argument("--run_scaling", action="store_true",
                        help="Run controlled scaling experiments.")
    parser.add_argument("--fit_surface", action="store_true",
                        help="Fit power-law scaling surface to logged results.")
    parser.add_argument("--optimize", action="store_true",
                        help="Optimize model size for a target accuracy.")
    parser.add_argument("--generate_graphs", action="store_true",
                        help="Generate all competition graphs from logged results.")

    # Surface fitting options
    parser.add_argument("--use_compute", action="store_true",
                        help="Use compute-augmented surface fitting.")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run bootstrap uncertainty after surface fit.")

    # Optimization options
    parser.add_argument("--target_accuracy", type=float, default=0.85,
                        help="Target accuracy for optimization.")
    parser.add_argument("--budget_gflops", type=float, default=None,
                        help="Maximum FLOPs budget in GFLOPs.")

    # Graph options
    parser.add_argument("--graph_dir", type=str, default="results/graphs",
                        help="Output directory for generated graphs.")

    args = parser.parse_args()
    set_seed(args.seed)

    if not any([args.run_scaling, args.fit_surface, args.optimize, args.generate_graphs]):
        parser.print_help()
        sys.exit(0)

    if args.run_scaling:
        cmd_run_scaling(args)

    if args.fit_surface:
        cmd_fit_surface(args)

    if args.optimize:
        cmd_optimize(args)

    if args.generate_graphs:
        cmd_generate_graphs(args)


if __name__ == "__main__":
    main()