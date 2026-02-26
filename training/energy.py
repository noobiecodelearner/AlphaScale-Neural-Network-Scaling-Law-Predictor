"""Energy estimation for AlphaScale training runs."""


def estimate_energy_kwh(
    train_time_seconds: float,
    gpu_wattage: float = 250.0,
) -> float:
    """Estimate energy consumption in kilowatt-hours.

    Uses a fixed GPU wattage assumption. Assumes the GPU runs at full
    power for the entire training duration.

    Args:
        train_time_seconds: Total training time in seconds.
        gpu_wattage: GPU power draw in watts (default: 250W).

    Returns:
        Energy consumed in kilowatt-hours.
    """
    hours = train_time_seconds / 3600.0
    return (gpu_wattage * hours) / 1000.0


def estimate_carbon_grams(
    energy_kwh: float,
    carbon_intensity_g_per_kwh: float = 475.0,
) -> float:
    """Estimate CO₂ emissions from energy use.

    Args:
        energy_kwh: Energy consumption in kWh.
        carbon_intensity_g_per_kwh: Grid carbon intensity (default: global avg ~475 g/kWh).

    Returns:
        Estimated CO₂ emissions in grams.
    """
    return energy_kwh * carbon_intensity_g_per_kwh


def estimate_gpu_memory_mb(model) -> float:
    """Estimate peak GPU memory usage for model parameters in MB.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        Memory used by parameters in megabytes.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)
