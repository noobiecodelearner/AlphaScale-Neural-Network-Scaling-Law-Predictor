"""AlphaScale graph generation module.

Produces all key competition visualisations:
  1. Scaling curves with 95% CI bands — per domain
  2. Compute vs Accuracy tradeoff
  3. Predicted vs Actual optimum comparison
  4. Carbon emission: Naive vs AlphaScale
  5. 2D compute-augmented scaling surface (contour/surface)
  6. Side-by-side comparison table: Naive | AlphaScale | Delta
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# Consistent colour palette across all plots
PALETTE = {
    "vision":  "#e63946",
    "nlp":     "#457b9d",
    "tabular": "#2a9d8f",
    "naive":   "#e76f51",
    "alpha":   "#264653",
    "ci":      "#a8dadc",
    "accent":  "#f4a261",
    "grid":    "#e9ecef",
}

DOMAIN_LABELS = {
    "vision":  "Vision (CIFAR-10)",
    "nlp":     "NLP (AG News)",
    "tabular": "Tabular (Adult Income)",
}


def _savefig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Graph 1: Scaling curves with 95% CI per domain ────────────────────────────

def plot_scaling_curves(
    domain_data: Dict[str, Dict],
    output_path: str = "results/graphs/graph1_scaling_curves.png",
) -> None:
    """Plot accuracy vs parameter count with 95% CI for each domain.

    Args:
        domain_data: {domain: {
            'param_counts': np.ndarray,
            'accuracies': np.ndarray,
            'n_search': np.ndarray,
            'mean_preds': np.ndarray,
            'ci_lower': np.ndarray,
            'ci_upper': np.ndarray,
            'optimal_n': float,
            'target_accuracy': float,
        }}
        output_path: Where to save the figure.
    """
    domains = list(domain_data.keys())
    n_domains = len(domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5), sharey=False)
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        d = domain_data[domain]
        color = PALETTE.get(domain, "#333")

        ax.scatter(
            d["param_counts"], d["accuracies"],
            color=color, zorder=6, s=70, label="Observed", edgecolors="white", linewidths=0.8,
        )
        ax.plot(
            d["n_search"], d["mean_preds"],
            color=color, linewidth=2.2, label="Fitted curve",
        )
        ax.fill_between(
            d["n_search"], d["ci_lower"], d["ci_upper"],
            alpha=0.18, color=color, label="95% CI",
        )
        ax.axhline(
            d["target_accuracy"], color=PALETTE["accent"],
            linestyle="--", linewidth=1.5, label=f"Target τ={d['target_accuracy']:.2f}",
        )
        if d.get("optimal_n"):
            ax.axvline(
                d["optimal_n"], color=PALETTE["alpha"],
                linestyle=":", linewidth=1.8, label=f"N*={d['optimal_n']:,.0f}",
            )

        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight="bold")
        ax.set_xlabel("Parameter Count (N)", fontsize=11)
        ax.set_ylabel("Validation Accuracy", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(color=PALETTE["grid"], linewidth=0.8)
        ax.set_facecolor("#fafafa")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
        ))

    fig.suptitle("AlphaScale — Scaling Curves with 95% Confidence Bands", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(fig, output_path)


# ── Graph 2: Compute vs Accuracy tradeoff ────────────────────────────────────

def plot_compute_accuracy_tradeoff(
    domain_data: Dict[str, Dict],
    output_path: str = "results/graphs/graph2_compute_accuracy.png",
) -> None:
    """Scatter plot of compute (FLOPs) vs accuracy, with Pareto frontier.

    Args:
        domain_data: {domain: {
            'param_counts': np.ndarray,
            'accuracies': np.ndarray,
            'compute': np.ndarray,        # FLOPs per scale point
            'optimal_n': float,
            'optimal_acc': float,
            'optimal_flops': float,
            'baseline_flops': float,
            'baseline_acc': float,
        }}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for domain, d in domain_data.items():
        color = PALETTE.get(domain, "#333")
        label = DOMAIN_LABELS.get(domain, domain)

        if d.get("compute") is not None and len(d["compute"]) > 0:
            ax.scatter(
                d["compute"], d["accuracies"],
                color=color, s=60, alpha=0.75, label=label,
                edgecolors="white", linewidths=0.6, zorder=4,
            )
            # Mark AlphaScale optimal
            if d.get("optimal_flops") and d.get("optimal_acc"):
                ax.scatter(
                    d["optimal_flops"], d["optimal_acc"],
                    color=color, s=180, marker="*", zorder=6,
                    edgecolors=PALETTE["alpha"], linewidths=1.2,
                )
            # Mark naive baseline
            if d.get("baseline_flops") and d.get("baseline_acc"):
                ax.scatter(
                    d["baseline_flops"], d["baseline_acc"],
                    color=color, s=120, marker="X", zorder=5,
                    edgecolors="red", linewidths=1.2, alpha=0.8,
                )

    # Legend proxies
    star_patch = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
                             markersize=12, label="AlphaScale N*")
    cross_patch = plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="grey",
                              markersize=10, markeredgecolor="red", label="Naive (largest model)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [star_patch, cross_patch], fontsize=9, framealpha=0.9)

    ax.set_xlabel("Compute (FLOPs)", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Compute vs Accuracy Tradeoff — All Domains", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(color=PALETTE["grid"], linewidth=0.8, which="both")
    ax.set_facecolor("#fafafa")

    plt.tight_layout()
    _savefig(fig, output_path)


# ── Graph 3: Predicted vs Actual optimum ─────────────────────────────────────

def plot_predicted_vs_actual(
    domain_data: Dict[str, Dict],
    output_path: str = "results/graphs/graph3_predicted_vs_actual.png",
) -> None:
    """Scatter of predicted accuracy vs actual accuracy at each observed scale.

    Args:
        domain_data: {domain: {
            'accuracies': np.ndarray,
            'predicted_at_observed': np.ndarray,
        }}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    all_vals = []
    for domain, d in domain_data.items():
        actual = np.array(d["accuracies"])
        predicted = np.array(d["predicted_at_observed"])
        color = PALETTE.get(domain, "#333")
        label = DOMAIN_LABELS.get(domain, domain)

        ax.scatter(actual, predicted, color=color, s=70, label=label,
                   edgecolors="white", linewidths=0.8, zorder=4, alpha=0.85)
        all_vals.extend(actual.tolist())
        all_vals.extend(predicted.tolist())

    if all_vals:
        lo = min(all_vals) - 0.01
        hi = max(all_vals) + 0.01
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Actual Accuracy", fontsize=12)
    ax.set_ylabel("Predicted Accuracy", fontsize=12)
    ax.set_title("Predicted vs. Actual Accuracy at Observed Scales", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    ax.set_facecolor("#fafafa")
    ax.set_aspect("equal")

    plt.tight_layout()
    _savefig(fig, output_path)


# ── Graph 4: Carbon emissions — Naive vs AlphaScale ──────────────────────────

def plot_carbon_comparison(
    domain_results: Dict[str, Dict],
    output_path: str = "results/graphs/graph4_carbon_comparison.png",
) -> None:
    """Grouped bar chart: CO₂ emissions naive model vs AlphaScale per domain.

    Args:
        domain_results: {domain: {
            'baseline_energy_kwh': float,
            'expected_energy_kwh': float,
            'carbon_intensity': float,   # g CO₂ / kWh, default 475
        }}
    """
    from training.energy import estimate_carbon_grams

    domains = list(domain_results.keys())
    x = np.arange(len(domains))
    width = 0.35

    naive_co2  = []
    alpha_co2  = []
    saved_co2  = []

    for domain in domains:
        d = domain_results[domain]
        intensity = d.get("carbon_intensity", 475.0)
        b_kwh = d.get("baseline_energy_kwh", 0.0)
        a_kwh = d.get("expected_energy_kwh", 0.0)
        n_g = b_kwh * intensity
        a_g = a_kwh * intensity
        naive_co2.append(n_g)
        alpha_co2.append(a_g)
        saved_co2.append(n_g - a_g)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: grouped bars
    bars_naive = ax1.bar(x - width / 2, naive_co2, width,
                         label="Naive (largest model)", color=PALETTE["naive"],
                         edgecolor="white", linewidth=0.8)
    bars_alpha = ax1.bar(x + width / 2, alpha_co2, width,
                         label="AlphaScale N*", color=PALETTE["alpha"],
                         edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars_naive, naive_co2):
        unit = "kg" if val >= 1000 else "g"
        display = val / 1000 if val >= 1000 else val
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(naive_co2),
                 f"{display:.1f}{unit}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_alpha, alpha_co2):
        unit = "kg" if val >= 1000 else "g"
        display = val / 1000 if val >= 1000 else val
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(naive_co2),
                 f"{display:.1f}{unit}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color=PALETTE["alpha"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([DOMAIN_LABELS.get(d, d) for d in domains], fontsize=10)
    ax1.set_ylabel("CO₂ Emissions (grams)", fontsize=11)
    ax1.set_title("CO₂ Emissions: Naive vs AlphaScale", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax1.set_facecolor("#fafafa")

    # Right: CO₂ saved bars
    bars_saved = ax2.bar(
        [DOMAIN_LABELS.get(d, d) for d in domains], saved_co2,
        color=[PALETTE.get(d, "#333") for d in domains],
        edgecolor="white", linewidth=0.8,
    )
    for bar, val in zip(bars_saved, saved_co2):
        unit = "kg" if val >= 1000 else "g"
        display = val / 1000 if val >= 1000 else val
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * max(saved_co2 or [1]),
                 f"{display:.2f} {unit}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_ylabel("CO₂ Saved (grams)", fontsize=11)
    ax2.set_title("CO₂ Saved by AlphaScale", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax2.set_facecolor("#fafafa")

    fig.suptitle("Environmental Impact Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, output_path)


# ── Graph 5: 2D compute-augmented scaling surface ────────────────────────────

def plot_scaling_surface_2d(
    domain: str,
    param_counts: np.ndarray,
    compute_vals: np.ndarray,
    accuracies: np.ndarray,
    a: float,
    b: float,
    alpha: float,
    beta: float = 0.0,
    output_path: Optional[str] = None,
) -> None:
    """Contour plot of the 2D (N, C) → accuracy scaling surface.

    If beta==0 (N-only model), C axis becomes a proportional proxy.

    Args:
        domain: Domain name for title.
        param_counts: Observed N values.
        compute_vals: Observed FLOPs values.
        accuracies: Observed accuracy values.
        a, b, alpha: Fitted power law parameters.
        beta: Compute exponent (0 = N-only model).
        output_path: Where to save. Defaults to results/graphs/graph5_{domain}.png
    """
    if output_path is None:
        output_path = f"results/graphs/graph5_surface_{domain}.png"

    from scaling.surface_fit import power_law, power_law_compute

    N_range = np.linspace(param_counts.min() * 0.5, param_counts.max() * 2.0, 80)
    C_range = np.linspace(compute_vals.min() * 0.5, compute_vals.max() * 2.0, 80)
    NN, CC = np.meshgrid(N_range, C_range)

    if beta != 0.0:
        ZZ = power_law_compute(NN, CC, a, b, alpha, beta)
    else:
        # C axis is just a proxy; surface depends only on N
        ZZ = power_law(NN, a, b, alpha)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    color = PALETTE.get(domain, "#457b9d")

    # Left: contour
    contour = ax1.contourf(NN, CC, ZZ, levels=20, cmap="RdYlGn", alpha=0.85)
    cbar = fig.colorbar(contour, ax=ax1, label="Predicted Accuracy")
    ax1.scatter(param_counts, compute_vals, c=accuracies, cmap="RdYlGn",
                s=90, edgecolors="black", linewidths=0.8, zorder=5,
                vmin=ZZ.min(), vmax=ZZ.max())
    ax1.set_xlabel("Parameter Count (N)", fontsize=11)
    ax1.set_ylabel("Compute / FLOPs", fontsize=11)
    ax1.set_title(f"Scaling Surface — {DOMAIN_LABELS.get(domain, domain)}", fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e9:.1f}G" if x >= 1e9 else f"{x/1e6:.0f}M"
    ))

    # Right: 1D accuracy vs N slice with compute coloured dots
    n_dense = np.linspace(param_counts.min() * 0.5, param_counts.max() * 2.0, 300)
    acc_dense = power_law(n_dense, a, b, alpha)
    ax2.plot(n_dense, acc_dense, color=color, linewidth=2.2, label="Fitted curve")
    scatter = ax2.scatter(param_counts, accuracies, c=compute_vals, cmap="plasma",
                          s=80, edgecolors="white", linewidths=0.8, zorder=5)
    fig.colorbar(scatter, ax=ax2, label="FLOPs")
    ax2.set_xlabel("Parameter Count (N)", fontsize=11)
    ax2.set_ylabel("Validation Accuracy", fontsize=11)
    ax2.set_title("Accuracy vs N (compute-coloured)", fontsize=12, fontweight="bold")
    ax2.grid(color=PALETTE["grid"], linewidth=0.8)
    ax2.set_facecolor("#fafafa")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))

    fig.suptitle(
        f"Compute-Augmented Scaling Surface — {DOMAIN_LABELS.get(domain, domain)}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, output_path)


# ── Graph 6 / Table: Naive vs AlphaScale side-by-side ────────────────────────

def plot_comparison_table(
    domain_results: Dict[str, Dict],
    output_path: str = "results/graphs/graph6_comparison_table.png",
) -> None:
    """Render a clean comparison table as a figure.

    Args:
        domain_results: {domain: {
            'naive_n': int,
            'naive_acc': float,
            'naive_energy_kwh': float,
            'optimal_n': int,
            'optimal_acc': float,
            'expected_energy_kwh': float,
            'compute_saved_fraction': float,
            'baseline_energy_kwh': float,
        }}
    """
    domains = list(domain_results.keys())

    col_labels = [
        "Domain",
        "Naive N\n(params)",
        "Naive Acc",
        "AlphaScale N*\n(params)",
        "AlphaScale Acc",
        "Δ Acc",
        "Compute\nSaved",
        "CO₂ Naive\n(g)",
        "CO₂ Alpha\n(g)",
        "CO₂ Saved\n(g)",
    ]

    rows = []
    row_colors = []

    for domain in domains:
        d = domain_results[domain]
        intensity = d.get("carbon_intensity", 475.0)
        co2_naive = d.get("baseline_energy_kwh", 0.0) * intensity
        co2_alpha = d.get("expected_energy_kwh", 0.0) * intensity
        co2_saved = co2_naive - co2_alpha

        delta_acc = d.get("optimal_acc", 0.0) - d.get("naive_acc", 0.0)

        rows.append([
            DOMAIN_LABELS.get(domain, domain),
            f"{int(d.get('naive_n', 0)):,}",
            f"{d.get('naive_acc', 0.0):.4f}",
            f"{int(d.get('optimal_n', 0)):,}",
            f"{d.get('optimal_acc', 0.0):.4f}",
            f"{delta_acc:+.4f}",
            f"{d.get('compute_saved_fraction', 0.0):.1%}",
            f"{co2_naive:.2f}",
            f"{co2_alpha:.2f}",
            f"{co2_saved:.2f}",
        ])

        # colour rows by domain
        base = PALETTE.get(domain, "#cccccc")
        row_colors.append([base + "33"] * len(col_labels))   # hex + alpha 20%

    fig, ax = plt.subplots(figsize=(16, max(2.5, 1.2 * len(domains) + 1.5)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(PALETTE["alpha"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style data rows
    for i, rcolors in enumerate(row_colors, start=1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#f8f9fa" if i % 2 == 0 else "white")

        # Highlight savings columns green
        for j_savings in [6, 9]:   # Compute Saved, CO₂ Saved
            table[i, j_savings].set_facecolor("#d4edda")
            table[i, j_savings].set_text_props(fontweight="bold", color="#155724")

        # Highlight accuracy delta
        delta_val = float(rows[i - 1][5])
        color_delta = "#d4edda" if delta_val >= -0.02 else "#f8d7da"
        table[i, 5].set_facecolor(color_delta)

    ax.set_title(
        "Naive Largest Model  vs  AlphaScale Recommended Model",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    _savefig(fig, output_path)


# ── Convenience: generate all graphs at once ──────────────────────────────────

def generate_all_graphs(
    domain_data: Dict[str, Dict],
    output_dir: str = "results/graphs",
) -> None:
    """Generate all 6 graphs from a unified domain_data dict.

    Each domain entry should contain:
        param_counts, accuracies, compute, n_search, mean_preds,
        ci_lower, ci_upper, predicted_at_observed,
        optimal_n, optimal_acc, optimal_flops,
        naive_n, naive_acc, baseline_flops, baseline_energy_kwh,
        expected_energy_kwh, compute_saved_fraction, target_accuracy,
        fit_params: {a, b, alpha, beta}

    Args:
        domain_data: Unified dict (see above).
        output_dir: Directory for saving all graphs.
    """
    print("\n[AlphaScale] Generating graphs...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_scaling_curves(
        domain_data,
        output_path=f"{output_dir}/graph1_scaling_curves.png",
    )

    plot_compute_accuracy_tradeoff(
        domain_data,
        output_path=f"{output_dir}/graph2_compute_accuracy.png",
    )

    plot_predicted_vs_actual(
        domain_data,
        output_path=f"{output_dir}/graph3_predicted_vs_actual.png",
    )

    plot_carbon_comparison(
        domain_data,
        output_path=f"{output_dir}/graph4_carbon_comparison.png",
    )

    for domain, d in domain_data.items():
        if d.get("compute") is not None and len(d.get("compute", [])) > 0:
            fp = d.get("fit_params", {})
            plot_scaling_surface_2d(
                domain=domain,
                param_counts=np.array(d["param_counts"]),
                compute_vals=np.array(d["compute"]),
                accuracies=np.array(d["accuracies"]),
                a=fp.get("a", 1.0),
                b=fp.get("b", 1.0),
                alpha=fp.get("alpha", 0.5),
                beta=fp.get("beta", 0.0),
                output_path=f"{output_dir}/graph5_surface_{domain}.png",
            )

    plot_comparison_table(
        domain_data,
        output_path=f"{output_dir}/graph6_comparison_table.png",
    )

    print(f"[AlphaScale] All graphs saved to {output_dir}/")
