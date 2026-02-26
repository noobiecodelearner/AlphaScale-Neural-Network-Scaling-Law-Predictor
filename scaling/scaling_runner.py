"""Orchestrates controlled scaling experiments across model sizes for AlphaScale."""

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml

from data.vision_loader import load_cifar10
from data.nlp_loader import load_yahoo
from data.tabular_loader import load_tabular
from models.cnn import ScalableCNN
from models.transformer import ScalableTransformer
from models.mlp import ScalableMLP
from training.trainer import Trainer, build_optimizer
from training.energy import estimate_energy_kwh, estimate_gpu_memory_mb
from utils.logger import ExperimentLogger
from utils.seed import set_seed
from scaling.generalization_warning import GeneralizationWarningDetector, GeneralizationReport


class ScalingRunner:
    """Runs controlled scaling experiments for a given domain.

    For each combination of (model_scale, dataset_fraction), trains a model
    with fixed hyperparameters and logs all metrics to CSV.

    Args:
        config_path: Path to a domain YAML config file.
        log_path: Path for output CSV log.
        seed: Global random seed.
        device: Torch device string (e.g. 'cuda', 'cpu').
        verbose: Whether to print training progress.
    """

    def __init__(
        self,
        config_path: str,
        log_path: str = "results/experiments.csv",
        seed: int = 42,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.seed = seed
        self.verbose = verbose
        self.logger = ExperimentLogger(log_path)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._gen_detector = GeneralizationWarningDetector()
        self._epoch_logs_store: Dict[str, Dict] = {}   # scale_id → {epoch_logs, n_params}

        print(f"[AlphaScale] Running on device: {self.device}")

    def _build_vision_model(self, width_multiplier: float) -> ScalableCNN:
        return ScalableCNN(
            num_classes=self.cfg["num_classes"],
            width_multiplier=width_multiplier,
            in_channels=self.cfg["input_channels"],
        )

    def _build_nlp_model(self, num_layers: int, d_model: int) -> ScalableTransformer:
        return ScalableTransformer(
            vocab_size=self.cfg["vocab_size"],
            num_classes=self.cfg["num_classes"],
            num_layers=num_layers,
            d_model=d_model,
            max_seq_len=self.cfg["max_seq_len"],
        )

    def _build_tabular_model(self, hidden_size: int, input_dim: int) -> ScalableMLP:
        return ScalableMLP(
            input_dim=input_dim,
            num_classes=self.cfg["num_classes"],
            hidden_size=hidden_size,
        )

    def _get_loaders(self, fraction: float):
        """Return (train_loader, val_loader, test_loader[, input_dim]) for the domain."""
        domain = self.cfg["domain"]
        batch_size = self.cfg["training"]["batch_size"]

        if domain == "vision":
            return load_cifar10(
                data_path=self.cfg["data_path"],
                dataset_fraction=fraction,
                batch_size=batch_size,
                seed=self.seed,
            )
        elif domain == "nlp":
            return load_yahoo(
                data_path=self.cfg["data_path"],
                dataset_fraction=fraction,
                batch_size=batch_size,
                max_seq_len=self.cfg["max_seq_len"],
                seed=self.seed,
            )
        elif domain == "tabular":
            return load_tabular(
                data_path=self.cfg["data_path"],
                dataset_fraction=fraction,
                batch_size=batch_size,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _run_single(
        self,
        model: nn.Module,
        scale_id: str,
        fraction: float,
        input_dim: int = None,
    ) -> Dict[str, Any]:
        """Train and evaluate a single model configuration.

        Args:
            model: Instantiated model.
            scale_id: Human-readable scale identifier string.
            fraction: Dataset fraction used.
            input_dim: Input dimension (tabular only, for FLOPs estimate).

        Returns:
            Metrics dictionary ready for logging.
        """
        set_seed(self.seed)
        model = model.to(self.device)

        loaders = self._get_loaders(fraction)
        if self.cfg["domain"] == "tabular":
            train_loader, val_loader, test_loader, _ = loaders
        else:
            train_loader, val_loader, test_loader = loaders

        optimizer = build_optimizer(
            model,
            self.cfg["training"]["optimizer"],
            self.cfg["training"]["learning_rate"],
            self.cfg["training"]["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            domain=self.cfg["domain"],
            epochs=self.cfg["training"]["epochs"],
        )

        if self.verbose:
            n_params = model.count_parameters()
            print(f"\n[Scale: {scale_id} | Fraction: {fraction} | Params: {n_params:,}]")

        fit_result = trainer.fit(train_loader, val_loader, verbose=self.verbose)
        test_metrics = trainer.evaluate(test_loader)

        n_params = model.count_parameters()
        mem_mb = estimate_gpu_memory_mb(model)
        energy = estimate_energy_kwh(
            fit_result["train_time"],
            self.cfg["energy"]["gpu_wattage"],
        )

        # Estimate FLOPs
        domain = self.cfg["domain"]
        if domain == "vision":
            flops = model.estimate_flops((self.cfg["input_channels"], self.cfg["image_size"], self.cfg["image_size"]))
        elif domain == "nlp":
            flops = model.estimate_flops(self.cfg["max_seq_len"])
        else:
            flops = model.estimate_flops()

        val_acc = fit_result["best_val_accuracy"]
        test_acc = test_metrics["accuracy"]
        train_acc = fit_result["final_train_accuracy"]
        gen_gap = train_acc - test_acc

        # Run generalization warning analysis on this scale
        gen_report = self._gen_detector.analyse(
            epoch_logs=fit_result["epoch_logs"],
            scale_id=scale_id,
            n_params=n_params,
        )
        self._epoch_logs_store[scale_id] = {
            "epoch_logs": fit_result["epoch_logs"],
            "n_params": n_params,
        }
        if self.verbose:
            risk_icon = {"low": "✅", "medium": "⚡", "high": "⚠️ "}.get(gen_report.overall_risk, "")
            print(f"  {risk_icon} Gen warning [{scale_id}]: {gen_report.overall_risk.upper()} | "
                  f"gap={gen_report.final_gen_gap:.4f} | "
                  f"best_epoch={gen_report.early_stop_epoch}")

        record = {
            "domain": self.cfg["domain"],
            "dataset_fraction": fraction,
            "scale_id": scale_id,
            "params": n_params,
            "compute": flops,
            "energy": round(energy, 6),
            "val_accuracy": round(val_acc, 6),
            "test_accuracy": round(test_acc, 6),
            "train_time": round(fit_result["train_time"], 2),
            "generalization_gap": round(gen_gap, 6),
        }
        self.logger.log(record)
        return record

    def run(self) -> List[Dict[str, Any]]:
        """Run all scaling experiments for this domain.

        Iterates over all (dataset_fraction, scale) combinations defined
        in the config and trains a model for each.

        Returns:
            List of result dictionaries, one per experiment.
        """
        domain = self.cfg["domain"]
        fractions = self.cfg["dataset_fractions"]
        results = []

        if domain == "vision":
            scales = self.cfg["model_scales"]["width_multipliers"]
            for fraction in fractions:
                for wm in scales:
                    model = self._build_vision_model(wm)
                    scale_id = f"width_{wm}"
                    r = self._run_single(model, scale_id, fraction)
                    results.append(r)

        elif domain == "nlp":
            pairs = self.cfg["model_scales"]["layer_dim_pairs"]
            for fraction in fractions:
                for layers, dim in pairs:
                    model = self._build_nlp_model(layers, dim)
                    scale_id = f"L{layers}_D{dim}"
                    r = self._run_single(model, scale_id, fraction)
                    results.append(r)

        elif domain == "tabular":
            loaders_for_dim = self._get_loaders(1.0)
            _, _, _, input_dim = loaders_for_dim

            hidden_sizes = self.cfg["model_scales"]["hidden_sizes"]
            for fraction in fractions:
                for hs in hidden_sizes:
                    model = self._build_tabular_model(hs, input_dim)
                    scale_id = f"hidden_{hs}"
                    r = self._run_single(model, scale_id, fraction, input_dim=input_dim)
                    results.append(r)

        else:
            raise ValueError(f"Unknown domain: {domain}")

        print(f"\n[AlphaScale] Completed {len(results)} experiments.")
        return results

    def gen_reports(self) -> List[GeneralizationReport]:
        """Return generalization warning reports for all scales run so far.

        Returns:
            List of GeneralizationReport sorted by n_params ascending.
        """
        return self._gen_detector.analyse_all_scales(self._epoch_logs_store)


def run_single_experiment(
    domain: str,
    dataset_fraction: float,
    config_dir: str = "configs",
    log_path: str = "results/experiments.csv",
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function to run scaling for a single domain + fraction.

    Args:
        domain: 'vision', 'nlp', or 'tabular'.
        dataset_fraction: Fraction of training data to use.
        config_dir: Directory containing YAML config files.
        log_path: Output CSV path.
        seed: Random seed.
        verbose: Print training progress.

    Returns:
        List of result records.
    """
    config_path = str(Path(config_dir) / f"{domain}.yaml")
    runner = ScalingRunner(
        config_path=config_path,
        log_path=log_path,
        seed=seed,
        verbose=verbose,
    )

    # Override fractions to only run the requested one
    runner.cfg["dataset_fractions"] = [dataset_fraction]
    return runner.run()
