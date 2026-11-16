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

# Use Keras names for datasets.
# Currently supports
# "mnist",
# "fashion_mnist"
# and "imdb".
DATASET="mnist"
# 60000 for "mnist" and "fashion_mnist".
# 25000 for "imdb".
DATASET_SIZE=60000
# 10 for "mnist" and "fashion_mnist".
# 2 for "imdb".
DATASET_CLASSES=10
# Represents the fraction of the dataset that is used.
# Select 1 for the complete dataset.
DATA_FRACTION=1
NUM_EXPERIMENTS=32
EPOCHS=30
# Fixed FPR as decimal.
FPR=0.01

if [[ $DATA_FRACTION -gt 1 ]]; then
    DATASET_SIZE=$(($DATASET_SIZE / $DATA_FRACTION))
fi

if [[ $# -ne 1 ]]; then
    echo "ERROR: This script needs exactly one argument. Please provide the step from which you want to start the pipeline (1-5).";
elif [[ $1 =~ ^[1-5]$ ]]; then
    case $1 in
        1)
            if [[ ! -d logs/ ]]; then
                mkdir logs/
            fi
            # Train the Tsetlin Machines.
            for ((i=0; i<$NUM_EXPERIMENTS; i++)); do
                python3 -u core/train.py --dataset=$DATASET --classes=$DATASET_CLASSES --epochs=$EPOCHS --num_experiments $NUM_EXPERIMENTS --expid $i --logdir exp/$DATASET --dataset_size=$DATASET_SIZE --reduce_dataset_by=$DATA_FRACTION &>> logs/log_$i
            done
            ;&
        2)
            # Perform inference.
            python3 core/inference.py --epochs=$EPOCHS --dataset=$DATASET --logdir=exp/$DATASET/ --dataset_size=$DATASET_SIZE
            ;&
        3)
            # Calculate scores.
            python3 core/score.py --logdir=exp/$DATASET/ --classes=$DATASET_CLASSES
            ;&
        4)
            # Plot results.
            python3 core/plot/evaluation_plot.py --logdir=exp/$DATASET/ --fpr=$FPR > eval@fixedFPR.txt
            ;;
        5)
            # Plot histograms.
            python3 core/plot/histogram_plot.py --logdir=exp/$DATASET/ --classes=$DATASET_CLASSES
            ;;
    esac
else
    echo "ERROR: The specified step needs to be any number from 1 to 5.";
fi
