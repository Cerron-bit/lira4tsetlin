# Copyright 2026 Yannick Fuchs
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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags
from matplotlib.axes import Axes
from numpy._core.numerictypes import bool_, float64, int32, intp, uint8
from numpy._typing import NDArray
from pandas import DataFrame

FLAGS = flags.FLAGS


def load_data(
    path: str,
) -> tuple[list[NDArray[bool_]], list[NDArray[float64]]]:
    """
    Loads all scores, which were generated in the previous step into a dictionary and returns it alongside all keeps.
    :param path: the path from where the data will be loaded.
    """
    score_list: list[NDArray[float64]] = []
    keeps: list[NDArray[bool_]] = []
    for root, dirnames, _ in os.walk(path):
        for dir in dirnames:
            if (
                not dir.startswith("experiment")
                or not os.path.exists(os.path.join(root, dir, "scores"))
                or not os.path.exists(os.path.join(root, dir, "keep.npy"))
            ):
                continue
            else:
                # get the keep.
                keeps.append(np.load(os.path.join(root, dir, "keep.npy")))

                score_files: list[str] = sorted(
                    os.listdir(os.path.join(root, dir, "scores"))
                )
                # Tests if the directory is empty.
                if len(score_files) == 0:
                    continue

                score_list.append(
                    np.load(
                        os.path.join(
                            root,
                            dir,
                            "scores",
                            "shifted_top_to_mean_difference.npy",
                        )
                    )
                )

    return keeps, score_list


def get_sample_membership(
    sample_id: int, keeps: list[NDArray[bool_]]
) -> NDArray[bool_]:
    """
    Extracts whether or not the sample was in the training data for every Tsetlin Machine used within the current setup.
    """
    number_of_tms: int = len(keeps)
    sample_memberships: NDArray[bool_] = np.zeros(number_of_tms, dtype=bool)
    for i in range(number_of_tms):
        sample_memberships[i] = keeps[i][sample_id]

    return sample_memberships


def get_sample_scores(
    sample_id: int, all_scores: list[NDArray[float64]]
) -> NDArray[float64]:
    """
    Extracts the scores for a particular sample out of the all given scores from every Tsetlin Machine.
    :param sample_id: the id of the sample you need the scores for.
    :param all_scores: all scores from every Tsetlin Machine within the experiment.
    """
    number_of_tms: int = len(all_scores)
    sample_scores: NDArray[float64] = np.zeros(number_of_tms)

    # extract the scores for this sample.
    tm_score_id: int = 0
    while tm_score_id < number_of_tms:
        tm_scores: NDArray[float64] = all_scores[tm_score_id]
        sample_scores[tm_score_id] = tm_scores[sample_id]
        tm_score_id += 1

    return sample_scores


def get_random_sample_ids(
    samples_per_class: int, Y_train: NDArray[uint8]
) -> dict[int, set[int]]:
    output: dict[int, set[int]] = {}
    for c in range(FLAGS.classes):
        all_ids: NDArray[intp] = np.where(Y_train == c)[0]
        chosen_sample_ids: NDArray[int32] = np.random.choice(
            all_ids, size=samples_per_class, replace=False
        )
        output[c] = set(chosen_sample_ids)

    return output


def plot_class_samples(
    out_dir: str,
    sample_scores: dict[int, NDArray[float64]],
    sample_memberships: dict[int, NDArray[bool_]],
    X_train: NDArray[uint8],
) -> None:
    """
    Plots one histogram per sample within the given class to the specified file.
    :param out_dir: the path of the directory, where you want to save the plots.
    :param sample_scores: a dictionary with the scores from all given samples for this class.
    :param sample_memberships: a dictionary for determining whether a sample was part in training or not.
    :param X_train: the original training dataset.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for sample_id in sample_scores.keys():
        original_img: NDArray[uint8] = X_train[sample_id]
        # Create the figure for plotting.
        number_plots: int = 2
        number_columns: int = 2
        number_rows: int = 1

        fig, axes = plt.subplots(number_rows, number_columns, figsize=(10, 4))
        axes = axes.flatten()

        # Plot the original image.
        axes[0].imshow(original_img, cmap="gray_r")
        axes[0].set_title(f"Sample {sample_id}")
        axes[0].axis("off")

        # seperate the scoring_systems for this sample.
        axes_id: int = 1
        sample_membership: NDArray[bool_] = sample_memberships[sample_id]

        # this is data for the x-axis.
        score_data: NDArray[float64] = sample_scores[sample_id]
        # Divide the scores by their presence in training data for a particular TM.
        formatted_data: DataFrame = pd.DataFrame(
            {
                "Score": score_data,
                "Group": np.where(sample_membership, "Member", "Non-member"),
            }
        )

        # Dynamically calculate the binrange.
        # Padding in votes.
        padding: int = 100
        binrange_start: int = min(score_data)
        binrange_end: int = max(score_data)
        # Test for the minimum and maximum possible votes.
        if binrange_start >= (padding - 5000):
            binrange_start -= padding
        if binrange_end <= (5000 - padding):
            binrange_end += padding

        # Plot the histogram for one sample.
        ax: Axes = axes[axes_id]
        sns.histplot(
            data=formatted_data,
            x="Score",
            hue="Group",
            multiple="layer",
            binrange=(binrange_start, binrange_end),
            binwidth=25,
            ax=ax,
        )
        ax.set_title("Shifted Top to Mean Difference")
        ax.set_xlabel("Score")
        ax.set_ylabel("Number of TMs")
        axes_id += 1

        # Hide the unused axes
        for ax in axes[number_plots:]:
            ax.axis("off")

        # save the resulting plot.
        plot_name: str = f"sample_{sample_id}_scorings.png"
        fig.tight_layout(h_pad=1.0)
        fig.savefig(os.path.join(out_dir, plot_name), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main(argv):
    del argv

    Y_train: NDArray[uint8] = np.load(
        os.path.join(FLAGS.logdir, "data_cache", "y_train.npy")
    )
    X_train: NDArray[uint8] = np.load(
        os.path.join(FLAGS.logdir, "data_cache", "x_train.npy")
    )

    data: tuple[list[NDArray[bool_]], list[NDArray[float64]]] = load_data(FLAGS.logdir)
    keeps: list[NDArray[bool_]] = data[0]
    scores: list[NDArray[float64]] = data[1]

    plot_path: str = os.path.join(FLAGS.logdir, FLAGS.plotdir)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    sample_ids: dict[int, set[int]] = get_random_sample_ids(50, Y_train)

    # Get sample scores and memberships, while preserving the classes and ids.
    class_sample_scores: dict[int, dict[int, NDArray[float64]]] = {}
    class_sample_keeps: dict[int, dict[int, NDArray[bool_]]] = {}
    for c in sample_ids.keys():
        class_sample_scores[c] = {}
        class_sample_keeps[c] = {}
        for sample_id in sample_ids[c]:
            sample_scores: NDArray[float64] = get_sample_scores(sample_id, scores)
            sample_membership: NDArray[bool_] = get_sample_membership(sample_id, keeps)
            class_sample_scores[c][sample_id] = sample_scores
            class_sample_keeps[c][sample_id] = sample_membership

    # plot everything.
    for c in class_sample_scores.keys():
        class_plot_path: str = os.path.join(plot_path, str(c))
        plot_class_samples(
            class_plot_path,
            class_sample_scores[c],
            class_sample_keeps[c],
            X_train,
        )


if __name__ == "__main__":
    flags.DEFINE_string("logdir", "experiments/", "Rootdirectory for the current run.")
    flags.DEFINE_string(
        "plotdir",
        "histograms/",
        "Subdirectory where the plotted histograms will be saved to.",
    )
    flags.DEFINE_integer("classes", 10, "number of classes within the dataset.")
    app.run(main)
