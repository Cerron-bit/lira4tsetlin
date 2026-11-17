# lira4tsetlin
This framework measures the exposure of training data whithin a Tsetlin Machine using Membership Inference Attacks.
It helps developers minimize the potential privacy leakage within their models by providing a suitable benchmark.

## Contents
TODO


## Related Works
The framework will conduct a **Likelihood Ratio Attack (LiRA)**, which is a specific kind of Membership Inference Attack based on the [work of Carlini et al](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021).
Since the original attack targets neural networks, this implementation is modified to target Tsetlin Machines based on the [implementation of Ole-Christoffer Granmo](https://github.com/cair/pyTsetlinMachineParallel).

## Installation
There are two possible installation methods provided.
Both assume that your machine supports CUDA and they also might take some time to install, because the pyTsetlinMachineParallel package will be build from source, just like the CUDA libraries, if you use the Nix package manager.

### Nix Package Manager
If you use the Nix package manager or NixOS, you can use the provided flake to automatically setup the entire project.
**This flake was written and tested for NixOS version 26.05.**
Before you install the flake, make sure to have [CUDA support](https://wiki.nixos.org/wiki/CUDA) and [Nix flakes](https://wiki.nixos.org/wiki/Flakes) enabled on your machine.

Once you have confirmed that those features are enabled, navigate into the projects directory and use the following command to begin the installation:

```bash
nix develop
```

The flake will then...
1. configure the `python 3.12` environment using `poetry` and `virtualenv`, 
2. download necessary `CUDA version 12` packages, which are needed by `Tensorflow`, 
3. download relevant python dependencies, 
4. build the `pyTsetlinMachineParallel` package from source 
5. and run some tests to confirm whether the installation was successful.

**Keep in mind that the CUDA libraries and the pyTsetlinMachineParallel module are build from source for your machine.**
Therefore those two steps can take a while to complete.

After the installation, the test outputs should look like this:
```
>>>>>>>>>>>>>>>>>>>>> DETECTING GPU <<<<<<<<<<<<<<<<<<<<<<
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>>>>>>>>>>>>> CHECKING MODIFIED TM-MODULE <<<<<<<<<<<<<<<
Reachable: True
```

### Other Distros & Package Managers
Before proceeding, make sure you have `CUDA version 12` support enabled and `python 3.12`, `poetry` and `virtualenv` installed on your machine.

Once those requirements are fulfilled, create and activate a virtual environment using the following command.

```bash
virtualenv -p python3.12 tsetlin_env
source tsetlin_env/bin/activate
```

Inside this virtual environment, make sure you are in the same directory as the `pyproject.toml` and use `poetry` to install the relevant dependencies.

```bash
poetry install
```

Next you need to build the `pyTsetlinMachineParallel` package from source. 

```bash
cd pyTsetlinMachineParallel
python3 setup.py build
python3 setup.py install
rm -r build pyTsetlinMachineParallel.egg-info
cd ../
```
**OPTIONAL:** This step may not be needed on every system, but if the python module still breaks, you might need to add the compiled library to your LD_LIBRARY_PATH.

```bash
export LD_LIBRARY_PATH="tsetlin_env/lib/python3.12/site-packages/libTM.cpython-312-x86_64-linux-gnu.so:$LD_LIBRARY_PATH"
```

#### Ensuring a correct installation.
You can run these two tests in order to verify your installation.
The first one ensures that `Tensorflow` can detect and use your GPU.

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 2>/dev/null | grep "PhysicalDevice"
```

A correct output should look like this:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

The second test ensures that the compiled library can be used through the `pyTsetlinMachineParallel` module.

```bash
python3 -c "from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D; TM = MultiClassConvolutionalTsetlinMachine2D(number_of_clauses=2000,T=50 * 100,s=5.0,patch_dim=(10, 10),weighted_clauses=True); print('Reachable:', TM.T == 5000)"
```

A correct output should look like this:

```
Reachable: True
```

## Documentation
**Disclaimer:** This documentation assumes you are familiar with the general concept of the [Likelihood Ratio Attack](https://ieeexplore.ieee.org/abstract/document/9833649).

### General Usage
In most cases you just need to interact with two files.
- `run.sh`
- `core/tsetlinMachine.py`

#### Shadow Model Hyperparameters
The hyperparameters of your shadow models can be specified in `core/tsetlinMachine.py` line 32 to 35 and if you use a convolutional Tsetlin Machine, you can set the dimensions of the convolution patch on line 43.
By default the shadow models are trained with the following configuration.

```
number_of_clauses: int = 2000
target_voting: int = 50 * 100
specificity: float = 5.0
weighted: bool = True
patch_dim=(10, 10)
```

#### Configuring LiRA
In `run.sh` you can find all configurable options from line 16 to 35 along with available options.

**1. DATASET:** 
This project uses Keras to load datasets and thus utilizes the same names for datasets as Keras. 
It also already integrates preprocessing for MNIST, Fashion-MNIST and IMDB.

**2. DATASET_SIZE:**
The total number of samples whithin the chosen dataset.
You can find the sizes for the supported datasets commented above this option.

**3. DATASET_CLASSES:**
The number of classes within the chosen dataset.
It is needed during the inference step and in the generation of histograms, which is currently an experimental feature.

**4. DATA_FRACTION:**
This represents the denominator in the fraction of the dataset that is actually used in the attack (e.g. 1/2, 1/4, 1/8).
By default this value is 1, which means the complete dataset is used.

**5. NUM_EXPERIMENTS:**
The number of shadow models that are trained during one run.
A minimum of 16 shadow models is required.

**6. EPOCHS:**
The number of training epochs each shadow model performs.

**7. FPR:**
The fixed False-Positive Rate as a decimal value that should be used during the evaluation at the end.

#### Running LiRA
To run LiRA use the following command:

```bash
./run.sh 1
```

As you can see, you need to specify the step from which the execution should begin.
Step 1 through 4 will be executed automatically after the prior step is completed.
These steps form the complete attack and need to be performed in order.

During the execution, new directories will be created.
In `logs` you can find one log file for every shadow model containing their training and test accuracies.
In `exp/DATASET` you can find one directory for each shadow model, which contain the model itself in a `tsetlin_machine.pkl`, the `keep.npy`, the extracted vote sums in `logits` and the calculated Membership Inference signal scores in `scores`.
There is also the `exp/DATASET/data_cache`, which saves the preprocessed dataset.

Once a run is complete, an `eval@fixedFPR.txt` will be created, which saves the results:

```
Attack Ours (online)
 AUC 0.4922, Accuracy 0.5169, TPR@1%FPR of 0.0216
Attack Ours (online, fixed variance)
 AUC 0.4632, Accuracy 0.5058, TPR@1%FPR of 0.0150
Attack Ours (offline)
 AUC 0.4980, Accuracy 0.5154, TPR@1%FPR of 0.0116
Attack Ours (offline, fixed variance)
 AUC 0.4864, Accuracy 0.5127, TPR@1%FPR of 0.0166
Attack Global threshold
 AUC 0.5295, Accuracy 0.5296, TPR@1%FPR of 0.0166
```

Additionally a `fprtpr.pdf` is created to visualize the resuluts.

TODO add a picture of it here.

### Additional Features
#### Training & Test Accuracy Logs
If you need further insight into the training performance of your shadow models, you can change line 59 within `core/tsetlinMachine.py` to the following:

```python 
print_comparison: bool = True
```

This will result in a longer training time, but logs both training and test accuracies into the same file.

#### Generate Histograms
**Disclaimer: This is an experimental feature.**
The following command will randomly select the provided number of samples from each class and generate a histogram showing the distribution of your shadow models for each of those samples.

```bash
./run 5
```

This step needs to be activated after the score is calculated in step 3 within `run.sh`.
It does not overwrite results of step 4, but instead saving them in `exp/DATASET/histograms`.
The following example shows a histogram generated with 32 shadow models trained on MNIST.

TODO add the histogram here.

**Why is this an experimental feature?**
Generating a histogram has only been tested with MNIST and Fashion-MNIST.
Both datasets contain sample images with the same dimensions (28x28 pixels).
There has never been any test with other images and thus it might be possible that the implementation will break on other datasets.
Text datasets are not supported in this feature.
