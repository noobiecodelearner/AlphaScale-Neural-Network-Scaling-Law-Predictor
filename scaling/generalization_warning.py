"""Early generalization warning module for AlphaScale.

Monitors training dynamics across epochs to detect when scaling further
will no longer improve generalization — i.e. diminishing returns.

Signals monitored:
  - Validation loss curvature  (second derivative flattening)
  - Generalization gap growth  (train_acc - val_acc widening)
  - Val loss plateau           (no meaningful improvement over window)
  - Overfitting onset          (val loss rising while train loss falls)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class WarningSignal:
    """A single triggered warning with context."""
    name: str
    triggered: bool
    severity: str          # 'low' | 'medium' | 'high'
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class GeneralizationReport:
    """Full report for one (scale, domain) training run."""
    scale_id: str
    n_params: int
    signals: List[WarningSignal] = field(default_factory=list)
    overall_risk: str = "low"       # 'low' | 'medium' | 'high'
    recommendation: str = ""
    early_stop_epoch: Optional[int] = None
    final_gen_gap: float = 0.0
    val_loss_slope: float = 0.0     # negative = still improving
    curvature: float = 0.0          # d²(val_loss)/d(epoch²)

    @property
    def triggered_count(self) -> int:
        return sum(1 for s in self.signals if s.triggered)

    @property
    def triggered_signals(self) -> List[WarningSignal]:
        return [s for s in self.signals if s.triggered]


# ── Core detector ──────────────────────────────────────────────────────────────

class GeneralizationWarningDetector:
    """Analyses per-epoch training logs and emits generalization warnings.

    Designed to operate on the `epoch_logs` list returned by Trainer.fit().
    Each entry must contain: 'epoch', 'train_loss', 'val_loss',
    'train_accuracy', 'val_accuracy'.

    Args:
        plateau_window: Number of trailing epochs to check for val loss plateau.
        plateau_delta: Minimum val loss improvement to NOT be considered a plateau.
        gap_threshold: Generalization gap (train_acc - val_acc) above which
                       overfitting is flagged.
        curvature_threshold: Second derivative of val loss above which
                             the curve is considered flat (diminishing returns).
        overfit_window: Epochs over which val loss must be rising to flag overfitting.
    """

    def __init__(
        self,
        plateau_window: int = 5,
        plateau_delta: float = 0.002,
        gap_threshold: float = 0.08,
        curvature_threshold: float = 0.0005,
        overfit_window: int = 4,
    ) -> None:
        self.plateau_window = plateau_window
        self.plateau_delta = plateau_delta
        self.gap_threshold = gap_threshold
        self.curvature_threshold = curvature_threshold
        self.overfit_window = overfit_window

    def _extract_series(
        self, epoch_logs: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_loss = np.array([e["train_loss"] for e in epoch_logs])
        val_loss   = np.array([e["val_loss"]   for e in epoch_logs])
        train_acc  = np.array([e["train_accuracy"] for e in epoch_logs])
        val_acc    = np.array([e["val_accuracy"]   for e in epoch_logs])
        return train_loss, val_loss, train_acc, val_acc

    def _val_loss_curvature(self, val_loss: np.ndarray) -> float:
        """Estimate mean second derivative of val_loss over last half of training."""
        if len(val_loss) < 4:
            return 0.0
        half = max(3, len(val_loss) // 2)
        tail = val_loss[-half:]
        x = np.arange(len(tail))
        try:
            coeffs = np.polyfit(x, tail, 2)
            return float(coeffs[0])
        except np.linalg.LinAlgError:
            return 0.0

    def _plateau_check(self, val_loss: np.ndarray) -> Tuple[bool, float]:
        """Return (is_plateau, improvement_over_window)."""
        if len(val_loss) < self.plateau_window:
            return False, float("nan")
        window = val_loss[-self.plateau_window:]
        improvement = float(window[0] - window[-1])
        return improvement < self.plateau_delta, improvement

    def _overfit_onset(
        self, train_loss: np.ndarray, val_loss: np.ndarray
    ) -> Tuple[bool, Optional[int]]:
        """Detect if val_loss rising while train_loss falls over last window."""
        if len(val_loss) < self.overfit_window + 1:
            return False, None
        w = self.overfit_window
        val_tail   = val_loss[-w:]
        train_tail = train_loss[-w:]
        val_rising    = all(val_tail[i] < val_tail[i + 1]   for i in range(len(val_tail) - 1))
        train_falling = all(train_tail[i] > train_tail[i + 1] for i in range(len(train_tail) - 1))
        if val_rising and train_falling:
            onset_idx = len(val_loss) - w
            return True, onset_idx + 1
        return False, None

    def _early_stop_epoch(self, val_loss: np.ndarray) -> Optional[int]:
        """Return 1-indexed epoch with minimum val_loss."""
        if len(val_loss) == 0:
            return None
        return int(np.argmin(val_loss)) + 1

    def _val_loss_slope(self, val_loss: np.ndarray) -> float:
        """Linear slope of val_loss over last half of training."""
        if len(val_loss) < 3:
            return 0.0
        half = max(2, len(val_loss) // 2)
        tail = val_loss[-half:]
        x = np.arange(len(tail))
        try:
            return float(np.polyfit(x, tail, 1)[0])
        except np.linalg.LinAlgError:
            return 0.0

    def analyse(
        self,
        epoch_logs: List[Dict],
        scale_id: str = "unknown",
        n_params: int = 0,
    ) -> GeneralizationReport:
        """Analyse epoch logs and produce a GeneralizationReport.

        Args:
            epoch_logs: List of per-epoch dicts from Trainer.fit().
            scale_id: Human-readable scale identifier.
            n_params: Parameter count for this scale.

        Returns:
            GeneralizationReport with all signals populated.
        """
        if len(epoch_logs) < 3:
            return GeneralizationReport(
                scale_id=scale_id,
                n_params=n_params,
                recommendation="Too few epochs to analyse.",
            )

        train_loss, val_loss, train_acc, val_acc = self._extract_series(epoch_logs)
        gen_gap    = float(train_acc[-1] - val_acc[-1])
        curvature  = self._val_loss_curvature(val_loss)
        slope      = self._val_loss_slope(val_loss)
        early_stop = self._early_stop_epoch(val_loss)
        signals: List[WarningSignal] = []

        # Signal 1: Val loss plateau
        is_plateau, improvement = self._plateau_check(val_loss)
        signals.append(WarningSignal(
            name="val_loss_plateau",
            triggered=is_plateau,
            severity="medium",
            message=(
                f"Val loss improved only {improvement:.4f} over last "
                f"{self.plateau_window} epochs (threshold: {self.plateau_delta})."
                if is_plateau else
                f"Val loss still improving ({improvement:.4f} over last "
                f"{self.plateau_window} epochs)."
            ),
            value=improvement,
            threshold=self.plateau_delta,
        ))

        # Signal 2: Generalization gap
        gap_triggered = gen_gap > self.gap_threshold
        signals.append(WarningSignal(
            name="generalization_gap",
            triggered=gap_triggered,
            severity="high" if gen_gap > self.gap_threshold * 1.5 else "medium",
            message=(
                f"Large generalization gap: {gen_gap:.4f} "
                f"(threshold: {self.gap_threshold}). Model may be overfitting."
                if gap_triggered else
                f"Generalization gap nominal: {gen_gap:.4f}."
            ),
            value=gen_gap,
            threshold=self.gap_threshold,
        ))

        # Signal 3: Curvature / diminishing returns
        flat = curvature > -self.curvature_threshold
        signals.append(WarningSignal(
            name="val_loss_curvature",
            triggered=flat,
            severity="low",
            message=(
                f"Val loss curve has flattened (curvature={curvature:.6f}). "
                "Scaling further unlikely to improve generalization."
                if flat else
                f"Val loss still curving downward (curvature={curvature:.6f})."
            ),
            value=curvature,
            threshold=self.curvature_threshold,
        ))

        # Signal 4: Overfitting onset
        overfit, onset_epoch = self._overfit_onset(train_loss, val_loss)
        signals.append(WarningSignal(
            name="overfitting_onset",
            triggered=overfit,
            severity="high",
            message=(
                f"Overfitting onset at epoch {onset_epoch}: val loss rising "
                "while train loss falls."
                if overfit else "No overfitting onset detected."
            ),
            value=float(onset_epoch) if onset_epoch else None,
        ))

        # Aggregate risk
        high_count   = sum(1 for s in signals if s.triggered and s.severity == "high")
        medium_count = sum(1 for s in signals if s.triggered and s.severity == "medium")
        if high_count >= 1 or (high_count + medium_count) >= 3:
            overall_risk = "high"
        elif medium_count >= 1 or any(s.triggered for s in signals):
            overall_risk = "medium"
        else:
            overall_risk = "low"

        # Recommendation
        if overall_risk == "high":
            rec = (
                f"⚠️  High risk: Scaling beyond {n_params:,} params unlikely to help. "
                f"Best val loss at epoch {early_stop}."
            )
        elif overall_risk == "medium":
            rec = (
                f"⚡ Moderate risk: {n_params:,} params shows early saturation. "
                "Monitor next scale carefully."
            )
        else:
            rec = f"✅ Low risk: {n_params:,} params still benefiting from scale."

        return GeneralizationReport(
            scale_id=scale_id,
            n_params=n_params,
            signals=signals,
            overall_risk=overall_risk,
            recommendation=rec,
            early_stop_epoch=early_stop,
            final_gen_gap=gen_gap,
            val_loss_slope=slope,
            curvature=curvature,
        )

    def analyse_all_scales(
        self,
        scale_epoch_logs: Dict[str, Dict],
    ) -> List[GeneralizationReport]:
        """Analyse multiple scales and return sorted reports.

        Args:
            scale_epoch_logs: Dict mapping scale_id →
                {'epoch_logs': [...], 'n_params': int}
        """
        reports = []
        for scale_id, info in scale_epoch_logs.items():
            report = self.analyse(
                epoch_logs=info["epoch_logs"],
                scale_id=scale_id,
                n_params=info.get("n_params", 0),
            )
            reports.append(report)
        reports.sort(key=lambda r: r.n_params)
        return reports

    def first_high_risk_scale(
        self, reports: List[GeneralizationReport]
    ) -> Optional[GeneralizationReport]:
        """Return the smallest-N scale flagged as high risk, or None."""
        for r in sorted(reports, key=lambda x: x.n_params):
            if r.overall_risk == "high":
                return r
        return None
