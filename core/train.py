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
import shutil

import numpy as np
from absl import app, flags
from keras.datasets import fashion_mnist, imdb, mnist
from numpy._core.numerictypes import bool_, intp
from numpy._typing import NDArray
from sklearn.feature_selection import SelectKBest, chi2
from tsetlinMachine import TsetlinMachine

FLAGS = flags.FLAGS
# Constants for text-processing.
MAX_NGRAM: int = 2
NUM_WORDS: int = 5000
INDEX_FROM: int = 2


def get_data(seed) -> tuple[dict[str, NDArray], dict[str, NDArray], NDArray]:
    """
    This is the function to generate subsets of the data for training models.

    First, it loads the training and test datasets from Keras with caching to avoid reloading.

    Then, it computes the subset. This works in one of two ways:

    1. If we have a seed, then we just randomly choose examples based on a prng with that seed, keeping FLAGS.pkeep fraction of the data.

    2. Otherwise, if we have an experiment ID, then we do something fancier. If we run each experiment independently then even after a lot of trials there will still probably be some examples that were always included or always excluded. So instead, with experiment IDs, we guarantee that after FLAGS.num_experiments are done, each example is seen exactly half of the time in train, and half of the time not in train.
    """
    cache_dir = os.path.join(FLAGS.logdir, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    x_train_path = os.path.join(cache_dir, "x_train.npy")
    y_train_path = os.path.join(cache_dir, "y_train.npy")
    x_test_path = os.path.join(cache_dir, "x_test.npy")
    y_test_path = os.path.join(cache_dir, "y_test.npy")

    # Load or create training and test data.
    np.random.seed(seed)
    if os.path.exists(x_train_path) and os.path.exists(y_train_path):
        X_train = np.load(x_train_path)
        Y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        Y_test = np.load(y_test_path)
    else:
        print("Loading training dataset from Keras...")
        if "mnist" in FLAGS.dataset:
            # Get unprocessed data.
            if FLAGS.dataset == "mnist":
                (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            elif FLAGS.dataset == "fashion_mnist":
                (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

            # Binarize the data for Tsetlin Machine (0 or 1).
            X_train = np.where(X_train >= 75, 1, 0)
            X_test = np.where(X_test >= 75, 1, 0)

        elif FLAGS.dataset == "imdb":
            (xtrain, ytrain), (xtest, ytest) = imdb.load_data(
                num_words=NUM_WORDS, index_from=INDEX_FROM
            )

            print("Starting preprocessing of imdb dataset.")
            # Retrieve and rearrange the encoding-schema.
            word_to_id: dict[str, int] = imdb.get_word_index()
            word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
            word_to_id["<PAD>"] = 0
            word_to_id["<START>"] = 1
            word_to_id["<UNK>"] = 2
            id_to_word: dict[int, str] = {
                value: key for key, value in word_to_id.items()
            }
            print("Step 1/4 completed.")

            # Ranks how often certain ngrams occour.
            vocabulary: dict[str, int] = {}
            for i in range(ytrain.shape[0]):
                terms: list[str] = []
                for word_id in xtrain[i]:
                    terms.append(id_to_word[word_id])
                for n in range(1, MAX_NGRAM + 1):
                    grams: list[list[str]] = [
                        terms[j : j + n] for j in range(len(terms) - n + 1)
                    ]
                    for gram in grams:
                        phrase = " ".join(gram)
                        if phrase in vocabulary:
                            vocabulary[phrase] += 1
                        else:
                            vocabulary[phrase] = 1

            phrase_bit_nr: dict[str, int] = {}
            bit_nr_phrase: dict[int, str] = {}
            bit_nr: int = 0
            for phrase in vocabulary.keys():
                if vocabulary[phrase] < 10:
                    continue
                phrase_bit_nr[phrase] = bit_nr
                bit_nr_phrase[bit_nr] = phrase
                bit_nr += 1
            print("Step 2/4 completed.")

            # Create binary representation.
            X_train = np.zeros((ytrain.shape[0], len(phrase_bit_nr)), dtype=np.int32)
            Y_train = np.zeros(ytrain.shape[0], dtype=np.int32)
            X_test = np.zeros((ytest.shape[0], len(phrase_bit_nr)), dtype=np.int32)
            Y_test = np.zeros(ytest.shape[0], dtype=np.int32)

            for split in (
                (X_train, Y_train, xtrain, ytrain),
                (X_test, Y_test, xtest, ytest),
            ):
                for i in range(split[3].shape[0]):
                    terms = []
                    for word_id in split[2][i]:
                        terms.append(id_to_word[word_id])
                    for n in range(1, MAX_NGRAM + 1):
                        grams = [terms[j : j + n] for j in range(len(terms) - n + 1)]
                        for gram in grams:
                            phrase: str = " ".join(gram)
                            if phrase in phrase_bit_nr:
                                split[0][i, phrase_bit_nr[phrase]] = 1
                    split[1][i] = split[3][i]
            print("Step 3/4 completed.")

            # Selecting features.
            kbest: SelectKBest = SelectKBest(chi2, k=5000)
            kbest = kbest.fit(X_train, Y_train)
            # create the keep as a boolean mask.
            keep: NDArray[bool_] = kbest.get_support(indices=False)
            X_train = kbest.transform(X_train)
            X_test = kbest.transform(X_test)
            print("Finished preprocessing.")
        else:
            raise ValueError(
                f"Dataset {FLAGS.dataset} not supported. Use 'mnist', 'fashion_mnist', 'imdb' or extend the get_data-method."
            )

        # Reduce if necessary.
        if FLAGS.reduce_dataset_by > 1:
            # Get same number of random samples per class.
            samples_per_class: int = FLAGS.dataset_size // FLAGS.classes
            reduced_sample_ids: NDArray[intp] = np.array([], dtype=intp)
            for c in range(FLAGS.classes):
                all_ids: NDArray[intp] = np.where(Y_train == c)[0]
                chosen_sample_ids: NDArray[intp] = np.random.choice(
                    all_ids, size=samples_per_class, replace=False
                )
                reduced_sample_ids = np.append(reduced_sample_ids, chosen_sample_ids)

            X_train = X_train[reduced_sample_ids]
            Y_train = Y_train[reduced_sample_ids]

        # Save the preprocessed datasets.
        np.save(x_train_path, X_train)
        np.save(y_train_path, Y_train)
        np.save(x_test_path, X_test)
        np.save(y_test_path, Y_test)

    # Determine which training examples to keep.
    if FLAGS.num_experiments is not None:
        keep = np.random.uniform(0, 1, size=(FLAGS.num_experiments, FLAGS.dataset_size))
        order = keep.argsort(0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=FLAGS.dataset_size) <= FLAGS.pkeep

    # Select subset.
    X_train_subset = X_train[keep]
    Y_train_subset = Y_train[keep]

    # wrap data before returning.
    train = {"x": X_train_subset, "y": Y_train_subset}
    test = {"x": X_test, "y": Y_test}

    return train, test, keep


def main(argv):
    del argv

    seed = FLAGS.seed
    if seed is None:
        import time

        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())

    # Create directories and checkpoints.
    if FLAGS.expid is not None:
        logdir = "experiment-%d_%d" % (FLAGS.expid, FLAGS.num_experiments)
    else:
        logdir = "experiment-" + str(seed)
    logdir = os.path.join(FLAGS.logdir, logdir)

    if os.path.exists(os.path.join(logdir, "tsetlin_machine.pkl")):
        print(f"run {FLAGS.expid} already completed.")
        return
    else:
        if os.path.exists(logdir):
            print(f"deleting run {FLAGS.expid} that did not complete.")
            shutil.rmtree(logdir)

    print(f"starting run {FLAGS.expid}.")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Actual training starts here.
    train, test, keep = get_data(seed)
    image_dataset: bool = True
    if FLAGS.dataset == "imdb":
        image_dataset = False
    tm = TsetlinMachine(image_dataset)
    tm.train(FLAGS.epochs, train, test)

    # Save the results.
    np.save(os.path.join(logdir, "keep.npy"), keep)
    with open(os.path.join(logdir, "tsetlin_machine.pkl"), "wb") as file:
        pickle.dump(tm, file)


if __name__ == "__main__":
    flags.DEFINE_string("dataset", "mnist", "Dataset.")
    flags.DEFINE_integer("classes", 10, "number of classes within the dataset.")
    flags.DEFINE_integer("epochs", 30, "Training duration in number of epochs.")
    flags.DEFINE_string(
        "logdir",
        "experiments",
        "Directory where to save checkpoints and tensorboard data.",
    )
    flags.DEFINE_integer(
        "reduce_dataset_by",
        1,
        "Use only the given fraction of the dataset during training.",
    )
    flags.DEFINE_integer("seed", None, "Training seed.")
    flags.DEFINE_float("pkeep", 0.5, "Probability to keep examples.")
    flags.DEFINE_integer("expid", None, "Experiment ID")
    flags.DEFINE_integer("num_experiments", None, "Number of experiments")
    flags.DEFINE_integer("dataset_size", 50000, "number of examples to keep.")
    app.run(main)
