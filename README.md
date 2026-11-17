# LiRA for Tsetlin Machines

A framework for measuring training-data exposure in Tsetlin Machines with **membership inference attacks (MIAs)**. It adapts the Likelihood Ratio Attack (LiRA) to Tsetlin Machines and provides a repeatable benchmark for evaluating privacy leakage in trained models.

> **Research software:** This project is intended for experimentation and privacy evaluation. Attack results depend on the dataset, model configuration, and experimental setup; they should not be interpreted as a guarantee of privacy or as a complete privacy audit.

## Contents

- [Background](#background)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Nix](#nix)
  - [Other Linux distributions](#other-linux-distributions)
- [Verify the installation](#verify-the-installation)
- [Configuration](#configuration)
- [Run the attack](#run-the-attack)
- [Outputs](#outputs)
- [Optional features](#optional-features)
- [License](#license)

## Background

LiRA is a membership inference attack introduced by Carlini et al. This project adapts the attack to Tsetlin Machines, using the [pyTsetlinMachineParallel implementation](https://github.com/cair/pyTsetlinMachine) as its model foundation.

- LiRA reference implementation: [tensorflow/privacy](https://github.com/tensorflow/privacy)
- Tsetlin Machine implementation: [cair/pyTsetlinMachine](https://github.com/cair/pyTsetlinMachine)

## Requirements

- Linux with a CUDA-capable NVIDIA GPU and working CUDA support
- CUDA 12
- Python 3.12
- Poetry and `virtualenv` (for the manual setup)

The project builds `pyTsetlinMachineParallel` from source. Installation and model training can take a while, and the complete attack may require substantial GPU time and disk space.

## Installation

### Nix

The included flake provides a development environment for **x86_64-linux** and is configured for NixOS 26.05 / nixpkgs `nixos-26.05`. Enable Nix flakes and configure CUDA support for your system before proceeding; see the [NixOS CUDA documentation](https://wiki.nixos.org/wiki/CUDA) and [Nix flakes documentation](https://wiki.nixos.org/wiki/Flakes).

From the repository root, run:

```bash
nix develop
```

On first launch, the shell hook creates and activates a Python 3.12 virtual environment, installs the Poetry dependencies, builds and installs `pyTsetlinMachineParallel`, and runs GPU/module checks. The CUDA libraries and native extension are built or configured for your machine, so the first launch may take some time.

A successful check should report a GPU and `Reachable: True`, for example:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Reachable: True
```

### Other Linux distributions

Ensure the requirements above are installed and CUDA 12 is available. From the repository root, create and activate a virtual environment and install the Python dependencies:

```bash
virtualenv -p python3.12 tsetlin_env
source tsetlin_env/bin/activate
poetry install
```

Build and install the bundled Tsetlin Machine extension:

```bash
cd pyTsetlinMachineParallel
python3 setup.py build
python3 setup.py install
rm -r build pyTsetlinMachineParallel.egg-info
cd ..
```

If the compiled module cannot find its shared library at runtime, set `LD_LIBRARY_PATH` to the installed `libTM` shared object. The exact filename and environment path can vary by platform and Python build. For example, on a typical CPython 3.12 x86-64 environment:

```bash
export LD_LIBRARY_PATH="$PWD/tsetlin_env/lib/python3.12/site-packages:$LD_LIBRARY_PATH"
```

## Verify the installation

Check whether TensorFlow can see the GPU:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The output should include a GPU `PhysicalDevice`. Then verify that the compiled Tsetlin Machine module can be imported and initialized:

```bash
python3 -c "from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D; TM = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=2000, T=50 * 100, s=5.0, patch_dim=(10, 10)); print('Reachable:', TM is not None)"
```

Expected final output:

```text
Reachable: True
```

## Configuration

Before running an experiment, review the settings near the top of [`run.sh`](run.sh). Dataset names follow Keras naming. The script currently documents these datasets:

| Setting | Purpose | Defaults / supported values |
| --- | --- | --- |
| `DATASET` | Dataset to load | `mnist`, `fashion_mnist`, or `imdb` |
| `DATASET_SIZE` | Number of samples in the selected dataset | 60,000 for MNIST and Fashion-MNIST; 25,000 for IMDB |
| `DATASET_CLASSES` | Number of classes | 10 for MNIST and Fashion-MNIST; 2 for IMDB |
| `DATA_FRACTION` | Dataset divisor; `1` uses the full dataset | Default: `1` |
| `NUM_EXPERIMENTS` | Number of shadow models | Default: `32`; minimum: `16`; use a power of two |
| `EPOCHS` | Training epochs per shadow model | Default: `30` |
| `FPR` | False-positive rate used for final evaluation | Default: `0.01` |

Preprocessing is included for MNIST, Fashion-MNIST, and IMDB. Review the dataset configuration in [`run.sh`](run.sh) before selecting a dataset; keep the size and class count consistent with it.

The shadow-model parameters are defined in [`core/tsetlinMachine.py`](core/tsetlinMachine.py). The default configuration is:

```text
number_of_clauses: 2000
target_voting: 50 * 100
specificity: 5.0
weighted: True
patch_dim: (10, 10)
```

The convolution patch dimensions are relevant to image data. Change the model settings to match your experiment and data before training.

## Run the attack

Make the script executable if needed, then start the pipeline from step 1:

```bash
chmod +x run.sh
./run.sh 1
```

Steps 1–4 run sequentially when started at step 1:

1. Train the shadow Tsetlin Machines.
2. Perform inference.
3. Calculate attack scores.
4. Evaluate and plot the results.

The script accepts a starting step from `1` to `5`. Starting at steps 2, 3, or 4 runs that step and the subsequent pipeline steps, which is useful when continuing from existing intermediate results. Step 5 generates histograms and is run separately after the scores have been calculated. Keep the experiment configuration unchanged when resuming a run so that intermediate files are consistent.

## Outputs

The pipeline creates experiment data under `exp/<DATASET>/` and per-model logs under `logs/`. Experiment directories include the saved Tsetlin Machine (`tsetlin_machine.pkl`), membership indicators (`keep.npy`), vote outputs (`logits`), and cached preprocessed data (`data_cache`). The exact files depend on the stages completed.

After evaluation, the repository root contains `eval@fixedFPR.txt` with metrics such as AUC, accuracy, and true-positive rate at the configured false-positive rate. A ROC plot is also written as `exp/<DATASET>/fprtpr.pdf`.

Example evaluation output:

```text
Attack Ours (online)
 AUC 0.5363, Accuracy 0.5258, TPR@1%FPR of 0.0098
Attack Ours (online, fixed variance)
 AUC 0.5492, Accuracy 0.5283, TPR@1%FPR of 0.0380
Attack Ours (offline)
 AUC 0.4992, Accuracy 0.5047, TPR@1%FPR of 0.0102
Attack Ours (offline, fixed variance)
 AUC 0.4999, Accuracy 0.5071, TPR@1%FPR of 0.0204
Attack Global threshold
 AUC 0.5205, Accuracy 0.5239, TPR@1%FPR of 0.0119
```

These values are illustrative output, not expected results for every dataset or configuration.

## Optional features

### Training and test accuracy logs

For additional accuracy information during shadow-model training, enable `print_comparison` in `core/tsetlinMachine.py`:

```python
print_comparison: bool = True
```

This adds training and test accuracy information to the model logs and may increase training time.

### Histograms (experimental)

After step 3 has generated the scores, run:

```bash
./run.sh 5
```

The current script requests 10 samples per class. Histogram files are saved under `exp/<DATASET>/histograms`; adjust the `--num_samples` argument in `run.sh` to change the number of samples.

This feature has been tested with MNIST and Fashion-MNIST only. Other image dimensions may not work, and text datasets are not supported.

## License

This project is released under the [Apache License 2.0](LICENSE). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for notices and licensing information for third-party components.
