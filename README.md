# αScale — Neural Network Scaling Law Predictor

> **Find the minimum model size that hits your accuracy target — before you train.**

αScale automates scaling law analysis across vision, NLP, and tabular domains. It fits power-law curves to small-scale experiments, quantifies uncertainty via bootstrap resampling, and recommends the optimal model size (N*) that achieves a target accuracy at minimum compute and CO₂ cost.

---

## The Problem

Every team training a neural network faces the same question before writing a single line of training code: **how large should the model be?**

The naive answer is "as large as compute allows." The result is systematic waste — organizations routinely train models 5–15× larger than necessary for their target accuracy, burning GPU hours, electricity, and money on parameters that contribute nothing to performance.

αScale solves this problem before training begins.

---

## Results

| Domain | Dataset | R² | MAE | Compute Saved | N* vs Naive |
|---|---|---|---|---|---|
| Vision | CIFAR-10 | 0.97 | 0.0081 | **92.7%** | 342K vs 4.7M params |
| NLP | Yahoo Answers | 0.68 | 0.0061 | **91.2%** | 883K vs 11M params |
| Tabular | Covertype | 0.999 | 0.0014 | **38.4%** | 345K vs 512K params |

MAE is reported in accuracy units — αScale predictions are within **0.6–0.8% accuracy** across all three domains.

---

## How It Works

αScale is built on the empirical observation that neural network accuracy follows a power law as a function of model size:

```
Accuracy = a - b × N^(-α)
```

Where:
- **N** — number of parameters
- **a** — asymptotic accuracy ceiling
- **b** — scaling coefficient  
- **α (alpha)** — power law exponent; the rate of improvement per unit of added parameters

The pipeline has five components:

**1. Scaling Runner** — Trains models of increasing size on the same task, holding everything constant except N. Produces clean (parameter count, accuracy) pairs.

**2. Surface Fitter** — Fits the power law via nonlinear least squares with multi-restart optimization and physically constrained bounds (accuracy is a probability, so `a < 1.0`). Reports both R² and MAE — MAE is more informative when tasks saturate quickly.

**3. Bootstrap Uncertainty Quantifier** — Runs 100 resamplings of the observed data to produce a 95% confidence interval on N* and CI bands on the full curve.

**4. Multi-Objective Optimizer** — Given a target accuracy τ, finds the minimum N such that predicted accuracy ≥ τ. Simultaneously computes expected energy (kWh), CO₂ emissions, and compute savings vs the naive baseline.

**5. Generalization Warning System** — Monitors per-epoch training dynamics using four signals (generalization gap, loss plateau, overfitting onset, curvature flattening) to produce LOW/MEDIUM/HIGH risk ratings and a best_epoch recommendation.

---

## Quick Start

### Install

```bash
git clone https://github.com/noobiecodelearner/AlphaScale-Neural-Network-Scaling-Law-Predictor/
cd alphascale
pip install -r requirements.txt
```

### Run scaling experiments

```bash
# Single domain
python main.py --domain vision --run_scaling --device cuda --verbose

# All domains
python main.py --domain all --run_scaling --device cuda
```

### Fit the scaling curve

```bash
python main.py --domain vision --fit_surface --bootstrap --dataset_fraction 1.0
```

Output:
```
==================================================
 Scaling Surface Fit — VISION
==================================================
  Model: Accuracy = a - b * N^(-alpha)
  a     = 0.884201
  b     = 2.341500
  alpha = 0.312847
  R²    = 0.969800
  MAE   = 0.008100
  AIC   = -38.2100
  Bootstrap (100 resamples):
  Success: 100 | Failed: 0
  Optimal N* mean : 342,150
  Optimal N* 95%CI: [298,000, 401,000]
```

### Find optimal model size

```bash
python main.py --domain vision --optimize --target_accuracy 0.85 --bootstrap
```

Output:
```
=======================================================
 AlphaScale Optimization — VISION
 Target accuracy: τ = 0.8500
=======================================================
  ✅ Optimal N*      : 342,150 parameters
  Expected accuracy  : 0.8502
  Energy estimate    : 0.000937 kWh
  Energy (baseline)  : 0.028127 kWh
  Carbon saved       : 12.92 g CO₂
  Compute saved      : 92.74%
  95% CI accuracy    : [0.8378, 0.8598]
```

### Generate all graphs

```bash
python main.py --domain all --generate_graphs --dataset_fraction 1.0
```

Saves 6 publication-quality figures to `results/graphs/`.

---

## Project Structure

```
alphascale/
├── main.py                    # CLI entry point
├── configs/
│   ├── vision.yaml            # CIFAR-10 CNN config
│   ├── nlp.yaml               # Yahoo Answers Transformer config
│   └── tabular.yaml           # Covertype MLP config
├── models/
│   ├── cnn.py                 # Scalable CNN
│   ├── transformer.py         # Scalable Transformer
│   └── mlp.py                 # Scalable MLP
├── scaling/
│   ├── scaling_runner.py      # Controlled experiment runner
│   ├── surface_fit.py         # Power-law curve fitting (R², MAE)
│   ├── bootstrap.py           # Bootstrap uncertainty quantification
│   ├── graphs.py              # Graph generation (6 competition figures)
│   └── generalization_warning.py  # Overfitting detection
├── optimization/
│   └── optimizer.py           # Multi-objective N* optimizer
├── training/
│   ├── trainer.py             # Training loop with early stopping
│   ├── energy.py              # Energy and carbon estimation
│   └── metrics.py             # Accuracy and loss metrics
├── data/
│   ├──raw/put your datasets here in separate folders
│   ├── vision_loader.py       # CIFAR-10 loader
│   ├── nlp_loader.py          # Yahoo Answers loader (with caching)
│   └── tabular_loader.py      # Covertype loader (with caching)
└── results/
    ├── experiments.csv        # Logged experiment results
    └── graphs/                # Generated figures
```

---

## Configuration

Each domain has a YAML config in `configs/`. Example for vision:

```yaml
domain: vision
dataset: cifar10
data_path: data/raw/cifar10
num_classes: 10

dataset_fractions:
  - 0.25
  - 0.5
  - 1.0

model_scales:
  layer_channel_pairs:
    - [2, 32]
    - [2, 64]
    - [3, 96]
    - [3, 128]
    - [4, 160]
    - [4, 192]

training:
  epochs: 15
  batch_size: 128
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0001

energy:
  gpu_wattage: 250
```

---

## CLI Reference

```
python main.py [--domain {vision,nlp,tabular,all}]
               [--dataset_fraction {0.25,0.5,1.0}]
               [--run_scaling]
               [--fit_surface]
               [--optimize]
               [--generate_graphs]
               [--target_accuracy FLOAT]
               [--bootstrap]
               [--device {auto,cuda,cpu}]
               [--verbose]
               [--graph_dir PATH]
               [--log_path PATH]
```

| Flag | Description |
|---|---|
| `--run_scaling` | Run controlled scaling experiments |
| `--fit_surface` | Fit power-law curve to logged results |
| `--optimize` | Find N* for a target accuracy |
| `--generate_graphs` | Generate all 6 competition figures |
| `--bootstrap` | Add 95% CI via bootstrap resampling |
| `--dataset_fraction` | Filter results to a specific data fraction |
| `--target_accuracy` | Target accuracy τ for optimization |
| `--domain all` | Run across all three domains |

---

## Datasets

| Domain | Dataset | Samples | Classes | Source |
|---|---|---|---|---|
| Vision | CIFAR-10 | 60,000 | 10 | Auto-downloaded via torchvision |
| NLP | Yahoo Answers Topics | 1,400,000 | 10 | HuggingFace: `community-datasets/yahoo_answers_topics` |
| Tabular | Forest Covertype | 581,012 | 7 | UCI ML Repository |

Place datasets in `data/raw/{cifar10,yahoo,covertype}/` before running.

---

## Scientific Background

αScale is built on the scaling law framework from:

- Kaplan et al. (2020). *Scaling Laws for Neural Language Models*. OpenAI.
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). DeepMind.

The key finding from both papers — that performance follows `Loss = b × N^(-α)` — applies broadly to classification tasks, not just language models. αScale validates this empirically across vision, NLP, and tabular domains.

**Original contributions:**
- Multi-restart curve fitting with physically constrained bounds (`a < 1.0`)
- MAE as a complementary fit metric to R² for narrow-range tasks
- Generalization-aware scaling with per-scale overfitting detection
- Lazy tokenization caching for NLP (75% preprocessing time reduction)
- Multi-objective optimization jointly minimizing compute and CO₂

---

## Requirements

```
torch>=2.0
torchvision
transformers
scipy
numpy
pandas
matplotlib
scikit-learn
pyyaml
```

GPU recommended. Tested on CUDA 12.1, Python 3.10.

---

## License

MIT License. See `LICENSE` for details.

---

## Authors
MD Naeem Hossain, Dayanch Amanov

*If αScale saves you compute time or helps with your research, a GitHub star goes a long way.*
