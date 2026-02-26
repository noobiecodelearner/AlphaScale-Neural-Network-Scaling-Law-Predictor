"""AlphaScale Streamlit dashboard for scaling analysis and model size recommendation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scaling.surface_fit import ScalingSurfaceFitter
from scaling.bootstrap import BootstrapUncertainty
from scaling.generalization_warning import GeneralizationWarningDetector
from optimization.optimizer import ScaleOptimizer
from training.energy import estimate_carbon_grams
from utils.logger import ExperimentLogger

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

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaScale Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ AlphaScale — Neural Network Scaling Analyzer")
st.markdown(
    "Predict the optimal model size and compute allocation using "
    "controlled scaling experiments, uncertainty modelling, and multi-objective optimisation."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")
LOG_PATH          = st.sidebar.text_input("Results CSV path", value="results/experiments.csv")
domain            = st.sidebar.selectbox("Domain", ["vision", "nlp", "tabular"])
dataset_fraction  = st.sidebar.selectbox("Dataset fraction", [0.25, 0.5, 1.0], index=2)
target_accuracy   = st.sidebar.slider("Target accuracy (τ)", 0.50, 0.99, 0.85, 0.01)
n_bootstrap       = st.sidebar.number_input("Bootstrap resamples", 20, 500, 100, 10)
budget_gflops     = st.sidebar.number_input("FLOPs budget (GFLOPs, 0 = no limit)", 0, value=0, step=100)
carbon_intensity  = st.sidebar.number_input("Grid carbon intensity (g CO₂/kWh)", 100, 1000, 475, 25,
                                             help="Global avg ≈ 475. Saudi Arabia ≈ 630. EU ≈ 255.")

DOMAIN_LABELS = {
    "vision":  "Vision (CIFAR-10)",
    "nlp":     "NLP (AG News)",
    "tabular": "Tabular (Adult Income)",
}

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_results(log_path: str) -> pd.DataFrame:
    logger = ExperimentLogger(log_path)
    records = logger.load()
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    numeric_cols = ["dataset_fraction", "params", "compute", "energy",
                    "val_accuracy", "test_accuracy", "train_time", "generalization_gap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


df_all = load_results(LOG_PATH)

if df_all.empty:
    st.warning(f"No experiment results found at `{LOG_PATH}`. Run scaling experiments first.")
    st.stop()

df = df_all[(df_all["domain"] == domain) & (df_all["dataset_fraction"] == dataset_fraction)].copy()

if df.empty:
    st.warning(f"No results for domain=**{domain}** / fraction=**{dataset_fraction}**.")
    st.stop()

df = df.sort_values("params").reset_index(drop=True)

with st.expander("📋 Raw experiment data", expanded=False):
    st.dataframe(df, use_container_width=True)

# ── Fit scaling surface ───────────────────────────────────────────────────────

st.header("📈 Scaling Surface Fit")

param_counts = df["params"].values.astype(float)
accuracies   = df["val_accuracy"].values.astype(float)
compute_vals = df["compute"].values.astype(float) if "compute" in df.columns else None
train_times  = df["train_time"].values.astype(float) if "train_time" in df.columns else None

fitter = ScalingSurfaceFitter(use_compute=False)

try:
    fit_result = fitter.fit(param_counts, accuracies)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{fit_result['r2']:.4f}")
    col2.metric("AIC", f"{fit_result['aic']:.2f}")
    col3.metric("α (power exponent)", f"{fit_result['alpha']:.4f}")
    col4.metric("Asymptote (a)", f"{fit_result['a']:.4f}")
    st.caption(
        f"Fitted: Accuracy = {fit_result['a']:.4f} − {fit_result['b']:.4f} × N^(−{fit_result['alpha']:.4f})"
    )
except RuntimeError as e:
    st.error(f"Surface fitting failed: {e}")
    st.stop()

# ── Bootstrap uncertainty ─────────────────────────────────────────────────────

st.header("🔁 Bootstrap Uncertainty (95% CI)")

bootstrapper = BootstrapUncertainty(n_bootstrap=int(n_bootstrap), seed=42)
n_grid = np.linspace(param_counts.min(), param_counts.max() * 2, 300)

with st.spinner("Running bootstrap resamples..."):
    try:
        bs_result = bootstrapper.run(
            param_counts, accuracies,
            target_accuracy=target_accuracy,
            n_search=n_grid,
        )
    except RuntimeError as e:
        st.error(f"Bootstrap failed: {e}")
        st.stop()

n_search   = np.array(bs_result["n_search"])
mean_preds = np.array(bs_result["mean_predictions"])
ci_lower   = np.array(bs_result["ci_lower"])
ci_upper   = np.array(bs_result["ci_upper"])
color      = PALETTE.get(domain, "#333")

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(param_counts, accuracies, color=color, zorder=5, label="Observed", s=65,
           edgecolors="white", linewidths=0.8)
ax.plot(n_search, mean_preds, color=color, linewidth=2.2, label="Fitted curve (mean)")
ax.fill_between(n_search, ci_lower, ci_upper, alpha=0.18, color=color, label="95% CI")
ax.axhline(target_accuracy, color=PALETTE["accent"], linestyle="--", linewidth=1.5,
           label=f"Target τ = {target_accuracy:.2f}")
ax.set_xlabel("Parameter Count (N)")
ax.set_ylabel("Validation Accuracy")
ax.set_title(f"Scaling Curve — {DOMAIN_LABELS.get(domain, domain)} (fraction={dataset_fraction})")
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
))
st.pyplot(fig)
plt.close(fig)

st.caption(
    f"Bootstrap: {bs_result['n_success']} / {n_bootstrap} resamples succeeded | "
    f"Failed: {bs_result['n_failed']}"
)

# ── 2D Scaling surface (compute vs params) ────────────────────────────────────

if compute_vals is not None and len(compute_vals) > 0:
    st.header("🗺️ 2D Compute-Augmented Scaling Surface")
    from scaling.surface_fit import power_law

    N_range = np.linspace(param_counts.min() * 0.5, param_counts.max() * 2, 60)
    C_range = np.linspace(compute_vals.min() * 0.5, compute_vals.max() * 2, 60)
    NN, CC  = np.meshgrid(N_range, C_range)
    a, b, alpha = fit_result["a"], fit_result["b"], fit_result["alpha"]
    ZZ = power_law(NN, a, b, alpha)

    fig2d, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(14, 5))

    contour = ax_c.contourf(NN, CC, ZZ, levels=20, cmap="RdYlGn", alpha=0.85)
    fig2d.colorbar(contour, ax=ax_c, label="Predicted Accuracy")
    ax_c.scatter(param_counts, compute_vals, c=accuracies, cmap="RdYlGn",
                 s=90, edgecolors="black", linewidths=0.8, zorder=5,
                 vmin=ZZ.min(), vmax=ZZ.max())
    ax_c.set_xlabel("Parameters (N)")
    ax_c.set_ylabel("FLOPs (compute)")
    ax_c.set_title("Accuracy Contour Surface")
    ax_c.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e9:.1f}G" if x >= 1e9 else f"{x/1e6:.0f}M"
    ))

    n_dense = np.linspace(param_counts.min() * 0.5, param_counts.max() * 2, 300)
    acc_dense = power_law(n_dense, a, b, alpha)
    ax_s.plot(n_dense, acc_dense, color=color, linewidth=2.2, label="Fitted curve")
    sc = ax_s.scatter(param_counts, accuracies, c=compute_vals, cmap="plasma",
                      s=80, edgecolors="white", linewidths=0.8, zorder=5)
    fig2d.colorbar(sc, ax=ax_s, label="FLOPs")
    ax_s.set_xlabel("Parameters (N)")
    ax_s.set_ylabel("Validation Accuracy")
    ax_s.set_title("Accuracy vs N (compute-coloured)")
    ax_s.grid(alpha=0.3)
    ax_s.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))

    fig2d.suptitle(f"Compute-Augmented Scaling Surface — {DOMAIN_LABELS.get(domain, domain)}",
                   fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2d)
    plt.close(fig2d)

# ── Predicted vs Actual ───────────────────────────────────────────────────────

st.header("🎯 Predicted vs Actual Accuracy")

predicted_at_observed = fitter.predict(param_counts)
fig_pva, ax_pva = plt.subplots(figsize=(6, 6))
ax_pva.scatter(accuracies, predicted_at_observed, color=color, s=70, zorder=5,
               edgecolors="white", linewidths=0.8)
lims = [
    min(accuracies.min(), predicted_at_observed.min()) - 0.01,
    max(accuracies.max(), predicted_at_observed.max()) + 0.01,
]
ax_pva.plot(lims, lims, "k--", alpha=0.5, linewidth=1.5)
ax_pva.set_xlabel("Actual Accuracy")
ax_pva.set_ylabel("Predicted Accuracy")
ax_pva.set_title(f"Predicted vs. Actual — {DOMAIN_LABELS.get(domain, domain)}")
ax_pva.grid(alpha=0.3)
ax_pva.set_xlim(lims)
ax_pva.set_ylim(lims)
st.pyplot(fig_pva)
plt.close(fig_pva)

# ── Optimisation ──────────────────────────────────────────────────────────────

st.header("🚀 Model Size Recommendation")

budget_flops = (budget_gflops * 1e9) if budget_gflops > 0 else None
optimizer    = ScaleOptimizer(fitter=fitter, bootstrap=bootstrapper, gpu_wattage=250.0)

opt_result = optimizer.optimize(
    target_accuracy=target_accuracy,
    observed_params=param_counts,
    observed_accuracies=accuracies,
    observed_compute=compute_vals,
    observed_train_times=train_times,
    budget_flops=budget_flops,
)

if not opt_result["feasible"]:
    st.error(
        f"❌ Optimisation failed: {opt_result.get('reason', 'Unknown')}  \n"
        f"Maximum predicted accuracy: {opt_result.get('max_predicted_accuracy', 'N/A')}"
    )
else:
    opt_n          = opt_result["optimal_n"]
    exp_acc        = opt_result["expected_accuracy"]
    exp_energy     = opt_result["expected_energy_kwh"]
    saved_frac     = opt_result["compute_saved_fraction"]
    baseline_energy = opt_result.get("baseline_energy_kwh", 0.0)

    co2_naive_g  = baseline_energy * carbon_intensity
    co2_alpha_g  = exp_energy * carbon_intensity
    co2_saved_g  = max(0.0, co2_naive_g - co2_alpha_g)
    co2_saved_kg = co2_saved_g / 1000.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recommended N*",    f"{opt_n:,.0f}")
    c2.metric("Predicted Accuracy", f"{exp_acc:.4f}")
    c3.metric("Compute Saved",      f"{saved_frac * 100:.1f}%")
    c4.metric("CO₂ Saved",
              f"{co2_saved_kg:.3f} kg" if co2_saved_kg >= 1 else f"{co2_saved_g:.1f} g")
    c5.metric("Energy Saved (kWh)", f"{max(0.0, baseline_energy - exp_energy):.4f}")

    if opt_result["ci_lower"] is not None:
        ci_lo = opt_result["ci_lower"]
        ci_hi = opt_result["ci_upper"]
        risk  = opt_result.get("risk_score")
        st.info(
            f"**95% CI on predicted accuracy at N*:** [{ci_lo:.4f}, {ci_hi:.4f}]  \n"
            + (f"**Risk score (CI width):** {risk:,.0f} parameters" if risk else "")
        )

    st.success(
        f"✅ **{opt_n:,.0f} parameters** predicted to achieve **{exp_acc:.2%}** accuracy — "
        f"saving **{saved_frac:.1%}** of compute and **{co2_saved_g:.1f} g CO₂** "
        f"({co2_saved_kg:.3f} kg) vs. the naive largest model."
    )

    # ── Side-by-side comparison table ─────────────────────────────────────────
    st.header("📊 Naive vs AlphaScale — Comparison Table")

    naive_idx  = int(np.argmax(param_counts))
    naive_n    = int(param_counts[naive_idx])
    naive_acc  = float(accuracies[naive_idx])
    delta_acc  = exp_acc - naive_acc
    delta_icon = "🟢" if delta_acc >= -0.02 else "🔴"

    table_data = {
        "Metric": [
            "Parameter count",
            "Predicted accuracy",
            "Compute (FLOPs)",
            "Training energy (kWh)",
            "CO₂ emitted (g)",
            "CO₂ emitted (kg)",
        ],
        "Naive (largest model)": [
            f"{naive_n:,}",
            f"{naive_acc:.4f}",
            f"{opt_result.get('baseline_compute', 0):,.0f}" if opt_result.get('baseline_compute') else "—",
            f"{baseline_energy:.4f}",
            f"{co2_naive_g:.2f}",
            f"{co2_naive_g/1000:.4f}",
        ],
        "AlphaScale N*": [
            f"{opt_n:,.0f}",
            f"{exp_acc:.4f}",
            f"{opt_result.get('expected_compute_flops', 0):,.0f}" if opt_result.get('expected_compute_flops') else "—",
            f"{exp_energy:.4f}",
            f"{co2_alpha_g:.2f}",
            f"{co2_alpha_g/1000:.4f}",
        ],
        "Delta / Saved": [
            f"−{naive_n - int(opt_n):,} params",
            f"{delta_icon} {delta_acc:+.4f}",
            f"−{saved_frac:.1%}",
            f"−{max(0.0, baseline_energy - exp_energy):.4f} kWh",
            f"−{co2_saved_g:.2f} g",
            f"−{co2_saved_kg:.4f} kg",
        ],
    }

    st.dataframe(
        pd.DataFrame(table_data).set_index("Metric"),
        use_container_width=True,
    )

    # ── Carbon bar chart ──────────────────────────────────────────────────────
    st.header("🌍 CO₂ Emissions — Naive vs AlphaScale")

    fig_co2, (ax_bar, ax_saved) = plt.subplots(1, 2, figsize=(12, 4))

    bars = ax_bar.bar(
        ["Naive (largest model)", "AlphaScale N*"],
        [co2_naive_g, co2_alpha_g],
        color=["#e76f51", "#264653"],
        edgecolor="white", linewidth=0.8, width=0.5,
    )
    for bar, val in zip(bars, [co2_naive_g, co2_alpha_g]):
        unit   = "kg" if val >= 1000 else "g"
        display = val / 1000 if val >= 1000 else val
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{display:.2f} {unit}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("CO₂ (grams)")
    ax_bar.set_title("Emissions per Training Run")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.set_facecolor("#fafafa")

    ax_saved.barh(["CO₂ saved"], [co2_saved_g], color="#2a9d8f", edgecolor="white")
    ax_saved.text(co2_saved_g * 1.01, 0,
                  f"{co2_saved_g:.1f} g  ({co2_saved_kg:.3f} kg)",
                  va="center", fontsize=10, fontweight="bold", color="#264653")
    ax_saved.set_xlabel("Grams of CO₂")
    ax_saved.set_title("CO₂ Saved by AlphaScale")
    ax_saved.grid(axis="x", alpha=0.3)
    ax_saved.set_facecolor("#fafafa")
    ax_saved.set_xlim(0, co2_saved_g * 1.4 if co2_saved_g > 0 else 1)

    plt.tight_layout()
    st.pyplot(fig_co2)
    plt.close(fig_co2)

# ── Generalization Warning Module ─────────────────────────────────────────────

st.header("⚠️ Generalization Warning Analysis")
st.markdown(
    "Monitors training dynamics to detect diminishing returns, overfitting onset, "
    "and val loss plateaus across each scale point."
)

EPOCH_LOGS_KEY = f"epoch_logs_{domain}_{dataset_fraction}"

if EPOCH_LOGS_KEY not in st.session_state:
    st.info(
        "Generalization warning analysis requires per-epoch training logs. "
        "These are stored in memory when running experiments. "
        "Re-run `--run_scaling` or load results with epoch logs to see this panel."
    )
else:
    epoch_logs_by_scale = st.session_state[EPOCH_LOGS_KEY]
    params_by_scale     = st.session_state.get(f"params_{domain}_{dataset_fraction}", {})

    detector = GeneralizationWarningDetector()
    combined = {
        sid: {"epoch_logs": logs, "n_params": params_by_scale.get(sid, 0)}
        for sid, logs in epoch_logs_by_scale.items()
    }
    reports = detector.analyse_all_scales(combined)

    for report in reports:
        risk_color = {"low": "✅", "medium": "⚡", "high": "⚠️"}.get(report.overall_risk, "")
        with st.expander(
            f"{risk_color} {report.scale_id} — {report.n_params:,} params — "
            f"Risk: {report.overall_risk.upper()}",
            expanded=(report.overall_risk != "low"),
        ):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Gen gap", f"{report.final_gen_gap:.4f}")
            col_b.metric("Val loss slope", f"{report.val_loss_slope:.5f}")
            col_c.metric("Curvature", f"{report.curvature:.6f}")
            col_d.metric("Best epoch", str(report.early_stop_epoch or "—"))

            st.markdown(f"**Recommendation:** {report.recommendation}")

            for sig in report.signals:
                icon = "🔴" if (sig.triggered and sig.severity == "high") else \
                       "🟡" if (sig.triggered and sig.severity == "medium") else \
                       "🟠" if sig.triggered else "🟢"
                st.markdown(f"{icon} **{sig.name}**: {sig.message}")

# ── Summary table ─────────────────────────────────────────────────────────────

st.header("📋 All Experiments Summary")
summary = df[[
    "scale_id", "params", "compute", "energy",
    "val_accuracy", "test_accuracy", "train_time", "generalization_gap"
]].copy()
summary = summary.rename(columns={
    "scale_id": "Scale", "params": "Params", "compute": "FLOPs",
    "energy": "Energy (kWh)", "val_accuracy": "Val Acc",
    "test_accuracy": "Test Acc", "train_time": "Train Time (s)",
    "generalization_gap": "Gen Gap",
})
st.dataframe(summary.style.format({
    "Params": "{:,.0f}", "FLOPs": "{:,.0f}", "Energy (kWh)": "{:.6f}",
    "Val Acc": "{:.4f}", "Test Acc": "{:.4f}",
    "Train Time (s)": "{:.1f}", "Gen Gap": "{:.4f}",
}), use_container_width=True)

st.caption("AlphaScale — Principled predictive scaling with uncertainty quantification.")

