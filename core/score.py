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

import multiprocessing as mp
import os

import numpy as np
from absl import app, flags
from numpy._core.numerictypes import float64, int32, intp
from numpy._typing import NDArray

FLAGS = flags.FLAGS


class ScoringEvaluation:
    """
    A collection of different scoring methods, which is optimized to reuse results of previous calculations.
    """

    def __init__(self, dataset_votes: NDArray[int32]) -> None:
        # Calculate the scores per training sample.
        self.axis = 1
        self.dataset_votes: NDArray[int32] = dataset_votes
        self.sorted_votes: NDArray[int32] = np.sort(
            self.dataset_votes, axis=self.axis, descending=True
        )

    def shifted_top_to_mean_difference(
        self, labels: NDArray[intp], vote_distance: int = 1
    ) -> NDArray[float64]:
        """
        This scoring-method adds the maximum of all vote counts and subtracts the hinge loss from the difference between the highest vote count and the mean of all other vote counts.
        :param labels: An array containing the labels used in training.
        :param vote_distance: The desired minimum difference in votes between the true label and all other labels.
        """
        max: NDArray[int32] = self.max()
        loss: NDArray[int32] = self.hinge_loss(labels, vote_distance)
        difference: NDArray[float64] = self.top_to_mean_difference()
        return difference + max - loss

    def hinge_loss(
        self, labels: NDArray[intp], vote_distance: int = 1
    ) -> NDArray[int32]:
        """
        This scoring-method is a variation of the Hinge Loss-function modified to work with the output of Tsetlin Machines.
        :param labels: An array containing the labels used in training.
        :param vote_distance: The desired minimum difference in votes between the true label and all other labels.
        """
        losses: list[int] = []
        for i in range(len(labels)):
            sample_votes: NDArray[int32] = self.dataset_votes[i]
            true_label: int = labels[i]

            label = 0
            loss = 0
            # Calculation of Hinge loss starts here.
            while label < FLAGS.classes:
                if label != true_label:
                    inner_term: int = (
                        vote_distance + sample_votes[label] - sample_votes[true_label]
                    )
                    loss += max(0, inner_term)
                label += 1
            losses.append(loss)
        return np.array(losses)

    def top_to_mean_difference(self) -> NDArray[float64]:
        """
        This scoring-method returns the difference in votes between the highest class and the mean vote count over all other classes.
        """
        highest_votes: NDArray[int32] = self.sorted_votes[:, 0]
        row_sum: NDArray[int32] = np.sum(self.sorted_votes, axis=self.axis)
        other_mean: NDArray[float64] = (row_sum - highest_votes) / (
            self.sorted_votes.shape[1] - 1
        )
        # The difference needs to be added, since most classes will have negative vote sums.
        difference: NDArray[float64] = highest_votes + other_mean
        return difference

    def max(self) -> NDArray[int32]:
        """
        This scoring-method returns the maximum votes over all classes.
        """
        return np.max(self.dataset_votes, axis=self.axis)


# This is the new load_one-method.
def load_one(base: str):
    """
    This loads some logits and converts them into a scored prediction.
    :param base: Name of root directory for one Tsetlin Machine.
    """
    root = os.path.join(FLAGS.logdir, base, "logits")
    if not os.path.exists(root):
        return None

    if not os.path.exists(os.path.join(FLAGS.logdir, base, "scores")):
        os.mkdir(os.path.join(FLAGS.logdir, base, "scores"))

    for f in os.listdir(root):
        try:
            dataset_votes: NDArray[int32] = np.load(os.path.join(root, f))
        except:
            print("Fail")
            continue

        evaluation = ScoringEvaluation(dataset_votes)
        score: NDArray[float64] = evaluation.shifted_top_to_mean_difference(labels)
        print("Saving results.")
        np.save(
            os.path.join(
                FLAGS.logdir, base, "scores", "shifted_top_to_mean_difference.npy"
            ),
            score,
        )


# Nothing needs to be done here, since this is only the caller for load_one().
def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [x for x in os.listdir(FLAGS.logdir) if "exp" in x])


def main(argv):
    del argv
    global labels
    labels = np.load(os.path.join(FLAGS.logdir, "data_cache", "y_train.npy"))
    load_stats()


if __name__ == "__main__":
    flags.DEFINE_integer("classes", 10, "number of classes within the dataset.")
    flags.DEFINE_string(
        "logdir",
        "experiments",
        "Directory where to save checkpoints and tensorboard data.",
    )
    app.run(main)
