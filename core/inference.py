# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Yannick Fuchs

# pylint: skip-file
# pyformat: disable

import os
import pickle
import re

import numpy as np
from absl import app, flags
from numpy._core.numerictypes import int32
from numpy._typing import NDArray
from tsetlinMachine import TsetlinMachine

FLAGS = flags.FLAGS


def cache_load(logdir: str) -> TsetlinMachine:
    """
    Load a cached TsetlinMachine-instance back into memory.
    """
    with open(os.path.join(logdir, "tsetlin_machine.pkl"), "rb") as cache_file:
        model: TsetlinMachine = pickle.load(cache_file)
    return model


def main(argv):
    """
    Perform inference of the saved model in order to generate the
    output logits, using a particular set of augmentations.
    """
    del argv

    xs_all = np.load(os.path.join(FLAGS.logdir, "data_cache", "x_train.npy"))[
        : FLAGS.dataset_size
    ]

    # Defines step width inside the dataset for the inference-loop.
    N = 5000

    # Iterates through every experiment.
    for path in sorted(os.listdir(os.path.join(FLAGS.logdir))):
        if re.search(FLAGS.regex, path) is None:
            print("Skipping from regex")
            continue

        logdir = os.path.join(FLAGS.logdir, path)
        model: TsetlinMachine = cache_load(logdir)

        # Creates the directory for the results, if not already present.
        if not os.path.exists(os.path.join(FLAGS.logdir, path, "logits")):
            os.mkdir(os.path.join(FLAGS.logdir, path, "logits"))

        # Use latest epoch, since this is an artefact from the old checkpoint mechanism.
        epoch = FLAGS.epochs
        # Checks if this file has already been processed.
        if os.path.exists(
            os.path.join(FLAGS.logdir, path, "logits", "%010d.npy" % epoch)
        ):
            print("Skipping already generated file", epoch)
            continue

        # Performs the inference.
        stats: list[NDArray[int32]] = []
        print(f"Extracting logits for {path}.")
        for i in range(0, len(xs_all), N):
            # Get raw class votes.
            logits: NDArray[int32] = model.class_scores(xs_all[i : i + N])
            # Save the result.
            stats.extend(logits)

        # Save the inference results.
        np.save(
            os.path.join(FLAGS.logdir, path, "logits", "%010d" % epoch), np.array(stats)
        )


if __name__ == "__main__":
    flags.DEFINE_string("dataset", "cifar10", "Dataset.")
    flags.DEFINE_string(
        "logdir", "experiments/", "Directory where to save the results."
    )
    flags.DEFINE_string("regex", ".*experiment.*", "keep files when matching")
    flags.DEFINE_integer("dataset_size", 50000, "size of dataset.")
    flags.DEFINE_integer("epochs", 30, "number of trained epochs.")
    app.run(main)
