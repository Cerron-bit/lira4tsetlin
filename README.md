# LiRA for Tsetlin Machines

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

LiRA for Tsetlin Machines is an open-source framework for evaluating **training-data exposure** through membership inference attacks. It adapts the Likelihood Ratio Attack (LiRA) to Tsetlin Machine classifiers and provides a repeatable workflow for training shadow models, calculating membership scores, and visualizing attack performance.

> **Research software:** Results depend on the dataset, model configuration, and experimental setup. Use this project for privacy research and benchmarking—not as a guarantee that a model is private or as a complete privacy audit.

## Features

- Likelihood Ratio Attack implementation adapted for Tsetlin Machines
- Support for MNIST, Fashion-MNIST, and IMDB
- Online, offline, fixed-variance, and global-threshold LiRA evaluations
- Reusable Nix development environment with CUDA 12
- Training, inference, scoring, and evaluation pipeline in a single shell entry point
- ROC plots and per-sample histogram generation
- Apache 2.0 licensed code and third-party notices

## Quick start

**Platform requirements:** Linux, an NVIDIA GPU with CUDA 12 support, and Nix with flakes enabled.

```bash
git clone https://github.com/Cerron-bit/lira4tsetlin.git
cd lira4tsetlin
nix develop
chmod +x run.sh
./run.sh 1
```

The default run trains 32 shadow models on MNIST, calculates the attack scores, and evaluates the result at a 1% false-positive rate.

For manual installation or systems without Nix, see [Installation](#installation).

## How it works

The framework follows a shadow-model membership inference workflow:

1. Train shadow Tsetlin Machines on controlled subsets of the selected dataset.
2. Run inference for every shadow model and collect extracted vote sums.
3. Calculate online and offline LiRA membership scores.
4. Evaluate AUC, accuracy, and true-positive rate at a fixed false-positive rate.
5. Optionally generate per-sample score histograms.

The pipeline is implemented across [`core/`](core/) and controlled through [`run.sh`](run.sh).

## Installation

### Nix (recommended)

The included flake provides an `x86_64-linux` development shell for NixOS 26.05 and CUDA 12 environments.

Prerequisites:

- Nix with flakes enabled
- NVIDIA GPU drivers and CUDA support
- The NixOS `allowUnfree` setting enabled for CUDA packages

```bash
nix develop
```

The shell hook:

- Creates a Python 3.12 virtual environment
- Installs project dependencies with Poetry
- Builds the bundled `pyTsetlinMachineParallel` native module
- Checks TensorFlow GPU visibility
- Checks that the Tsetlin Machine module loads

The first launch can take considerable time because native and CUDA-related components may be built for the host machine.

### Manual installation

For other Linux distributions, install Python 3.12, Poetry, `virtualenv`, and CUDA 12-compatible NVIDIA drivers and libraries.

```bash
virtualenv -p python3.12 tsetlin_env
source tsetlin_env/bin/activate
poetry install
```

Build the bundled `pyTsetlinMachineParallel` module:

```bash
cd pyTsetlinMachineParallel
python3 setup.py build
python3 setup.py install
rm -r build pyTsetlinMachineParallel.egg-info
cd ..
```

If the native module cannot find `libTM` at runtime, add the virtual environment's package directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$PWD/tsetlin_env/lib/python3.12/site-packages:$LD_LIBRARY_PATH"
```

## Verify the installation

Check whether TensorFlow detects the GPU:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The output should contain a GPU `PhysicalDevice`. Verify the native Tsetlin Machine module:

```bash
python3 -c "from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D; tm = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=2000, T=5000, s=5.0, patch_dim=(10, 10)); print('Reachable:', tm is not None)"
```

Expected result:

```text
Reachable: True
```

## Configuration

Review the settings near the top of [`run.sh`](run.sh) before starting an experiment.

| Setting | Purpose | Default |
| --- | --- | --- |
| `DATASET` | Keras dataset name | `mnist` |
| `DATASET_SIZE` | Number of dataset samples | `60000` |
| `DATASET_CLASSES` | Number of target classes | `10` |
| `DATA_FRACTION` | Dataset divisor; `1` uses the complete dataset | `1` |
| `NUM_EXPERIMENTS` | Number of shadow models; minimum 16, preferably a power of two | `32` |
| `EPOCHS` | Training epochs per shadow model | `30` |
| `FPR` | Fixed false-positive rate for evaluation | `0.01` |

Supported built-in dataset configurations:

| Dataset | `DATASET_SIZE` | `DATASET_CLASSES` |
| --- | ---: | ---: |
| `mnist` | 60,000 | 10 |
| `fashion_mnist` | 60,000 | 10 |
| `imdb` | 25,000 | 2 |

Shadow-model parameters are in [`core/tsetlinMachine.py`](core/tsetlinMachine.py):

```text
number_of_clauses: 2000
target_voting: 50 * 100
specificity: 5.0
weighted: True
patch_dim: (10, 10)
```

Adjust these values and the dataset settings consistently. Keep a copy of the configuration used for any published experiment, because changing the model or dataset split can change the attack results.

## Running an experiment

Start from step 1 for a complete pipeline run:

```bash
./run.sh 1
```

| Step | Action |
| ---: | --- |
| 1 | Train shadow models |
| 2 | Run model inference |
| 3 | Calculate LiRA scores |
| 4 | Evaluate and plot results |
| 5 | Generate optional histograms |

The script also accepts a starting step. For example, resume after completed score calculation:

```bash
./run.sh 4
```

Do not change `NUM_EXPERIMENTS`, dataset settings, or model parameters while reusing existing intermediate results.

### Pipeline outputs

- `logs/` — training and evaluation logs
- `exp/<DATASET>/` — shadow models and experiment artifacts
- `exp/<DATASET>/data_cache/` — cached preprocessed dataset
- `exp/<DATASET>/tsetlin_machine.pkl` — saved shadow-model artifacts
- `exp/<DATASET>/keep.npy` — membership indicators
- `exp/<DATASET>/logits/` — extracted vote sums
- `eval@fixedFPR.txt` — text metrics for the configured FPR
- `exp/<DATASET>/fprtpr.pdf` — ROC plot

Example metrics:

```text
Attack Ours (online)
 AUC 0.5363, Accuracy 0.5258, TPR@1%FPR of 0.0098
Attack Ours (offline)
 AUC 0.4992, Accuracy 0.5047, TPR@1%FPR of 0.0102
Attack Global threshold
 AUC 0.5205, Accuracy 0.5239, TPR@1%FPR of 0.0119
```

These values are illustrative and should not be considered baseline expectations.

## Optional features

### Training and test accuracy logs

Set `print_comparison` to `True` in [`core/tsetlinMachine.py`](core/tsetlinMachine.py):

```python
print_comparison: bool = True
```

The shadow-model logs will include training and test accuracy information. This can increase training time.

### Per-sample histograms

Generate histograms after completing score calculation:

```bash
./run.sh 5
```

Files are written to `exp/<DATASET>/histograms/`. The default configuration selects 10 samples per class. You can change `--num_samples` in [`run.sh`](run.sh).

This feature is experimental and has only been tested with MNIST and Fashion-MNIST. It assumes image inputs with compatible dimensions and does not support text datasets.

## Project structure

```text
.
├── core/                       # Training, inference, scoring, and plotting code
├── pyTsetlinMachineParallel/   # Bundled Tsetlin Machine native implementation
├── flake.nix                   # Reproducible Nix development environment
├── pyproject.toml              # Python dependencies and project metadata
├── run.sh                      # Main four-stage attack pipeline
├── LICENSE                     # Apache License 2.0
└── THIRD_PARTY_NOTICES.md      # Third-party notices
```

## Background

LiRA was introduced by Carlini et al. as a likelihood-ratio-based membership inference attack. This project adapts that approach to Tsetlin Machines built with `pyTsetlinMachineParallel`.

Related projects:

- [TensorFlow Privacy: LiRA reference implementation](https://github.com/tensorflow/privacy)
- [CAIR: pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine)
- [Carlini et al.: Membership Inference Attacks From First Principles](https://github.com/tensorflow/privacy)

## Contributing

Contributions are welcome, especially improvements to portability, documentation, tests, and support for additional datasets.

Good contribution areas include:

- Replacing hard-coded dataset assumptions
- Adding automated tests
- Improving CUDA and Nix portability
- Supporting additional Tsetlin Machine architectures
- Documenting reproducible experiments

When opening an issue or pull request, include the operating system, Nix/Python/CUDA versions, relevant configuration, and command output. Please keep research and privacy claims reproducible and clearly distinguish experimental observations from supported guarantees.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Third-party components and notices are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
