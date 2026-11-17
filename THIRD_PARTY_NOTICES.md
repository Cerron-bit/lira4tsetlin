# Third-party notices

This project contains modified code derived from the third-party projects listed below. These notices identify the relevant code components and do not replace the license terms that apply to the rest of this repository. The repository's top-level `LICENSE` file contains the Apache License 2.0 used for this project, and it applies to the original code written for this repository. The MIT-licensed subproject listed below remains under the MIT license and is not re-licensed under Apache 2.0.

## pyTsetlinMachineParallel

- **Upstream project:** [cair/pyTsetlinMachineParallel](https://github.com/cair/pyTsetlinMachineParallel)
- **License:** MIT License
- **Files in this repository:** `pyTsetlinMachineParallel/setup.py`, `pyTsetlinMachineParallel/MANIFEST.in` and `pyTsetlinMachineParallel/pyTsetlinMachineParallel/` (including `tm.py`, `tools.py`, `ConvolutionalTsetlinMachine.c`, `ConvolutionalTsetlinMachine.h`, `MultiClassConvolutionalTsetlinMachine.c`, `MultiClassConvolutionalTsetlinMachine.h`, `Tools.c`, `Tools.h`, `fast_rand.h`, and `makefile`)
- **Status:** This vendored subproject remains under the MIT license. It is not re-licensed under the repository's Apache 2.0 license.
- **Changes:** The vendored package has been modified for this project. The upstream MIT notices are retained in the source files.

Copyright notices found in the upstream package include:

- Copyright (c) 2019 Ole-Christoffer Granmo
- Copyright (c) 2021 Ole-Christoffer Granmo

MIT License

Copyright (c) 2019 Ole-Christoffer Granmo
Copyright (c) 2021 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## TensorFlow Privacy: MI-LiRA 2021 research code

- **Upstream project and source directory:** [tensorflow/privacy/research/mi_lira_2021](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021)
- **License:** Apache License 2.0
- **Files in this repository identified as derived from that code:** `run.sh`, `core/train.py`, `core/inference.py`, `core/score.py`, and `core/plot/evaluation.py`
- **Status:** These files remain subject to the Apache License 2.0 and retain the upstream copyright notices in their headers.
- **Changes:** These files have been modified for this project. Their source headers identify Google LLC's copyright and note modifications by Yannick Fuchs.
- **Upstream NOTICE file:** No `NOTICE` file was present in the referenced `research/mi_lira_2021` directory or the upstream repository root when checked.

The applicable upstream attribution notice is:

```text
Copyright 2021 Google LLC
Licensed under the Apache License, Version 2.0.
```

The upstream project-wide Apache license also contains this notice:

```text
Copyright 2018, The TensorFlow Privacy Authors.
```

The full Apache License 2.0 text is included in the repository's top-level `LICENSE` file. Any modified Apache-licensed files should continue to carry their original copyright and license headers, and should also note material changes as required by Apache License 2.0 section 4(b).
