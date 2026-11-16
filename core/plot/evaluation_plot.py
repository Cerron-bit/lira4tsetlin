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

import functools
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from absl import app, flags
from sklearn.metrics import auc, roc_curve

logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
FLAGS = flags.FLAGS


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data(p):
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for root, ds, _ in os.walk(p):
        for f in ds:
            if not f.startswith("experiment"):
                continue
            if not os.path.exists(os.path.join(root, f, "scores")):
                continue
            last_epoch = sorted(os.listdir(os.path.join(root, f, "scores")))
            if len(last_epoch) == 0:
                continue
            scores.append(np.load(os.path.join(root, f, "scores", last_epoch[-1])))
            keep.append(np.load(os.path.join(root, f, "keep.npy")))

    scores = np.array(scores)
    keep = np.array(keep)[:, : scores.shape[1]]

    return scores, keep


def generate_ours(
    keep,
    scores,
    check_keep,
    check_scores,
    in_size=100000,
    out_size=100000,
    fix_variance=False,
):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j])
        dat_out.append(scores[~keep[:, j], j])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score)
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(
    keep,
    scores,
    check_keep,
    check_scores,
    in_size=100000,
    out_size=100000,
    fix_variance=False,
):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j])
        dat_out.append(scores[~keep[:, j], j])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score)
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc)
        answers.extend(ans)

    return prediction, answers


def do_plot(
    fn, keep, scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs
):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(
        keep[:-ntest], scores[:-ntest], keep[-ntest:], scores[-ntest:]
    )

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    # The fixed FPR is set here.
    low = tpr[np.where(fpr < FLAGS.fpr)[0][-1]]

    fixed_fpr = FLAGS.fpr * 100
    if fixed_fpr == int(fixed_fpr):
        fixed_fpr = int(fixed_fpr)

    print(f"Attack {legend} AUC {auc:.4f}, Accuracy {acc:.4f}, TPR@{fixed_fpr}%FPR of {low:.4f}")

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr():

    fig = plt.figure(figsize=(6, 3), layout="constrained")

    do_plot(generate_ours, keep, scores, 1, "Ours (online)\n", metric="auc")

    do_plot(
        functools.partial(generate_ours, fix_variance=True),
        keep,
        scores,
        1,
        "Ours (online, fixed variance)\n",
        metric="auc",
    )

    do_plot(
        functools.partial(generate_ours_offline),
        keep,
        scores,
        1,
        "Ours (offline)\n",
        metric="auc",
    )

    do_plot(
        functools.partial(generate_ours_offline, fix_variance=True),
        keep,
        scores,
        1,
        "Ours (offline, fixed variance)\n",
        metric="auc",
    )

    do_plot(generate_global, keep, scores, 1, "Global threshold\n", metric="auc")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    fig.legend(fontsize=8, loc="outside right center")
    plt.savefig("fprtpr.pdf", format="pdf")


def main(argv):
    del argv

    load_data(FLAGS.logdir)
    fig_fpr_tpr()


if __name__ == "__main__":
    flags.DEFINE_string("logdir", "experiments/", "Rootdirectory for the current run.")
    flags.DEFINE_float("fpr", 0.01, "The fixed FPR value used in the evaluation.")
    app.run(main)
