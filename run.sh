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
# Denominator in the fraction of the dataset that is used.
# Select 1 for the complete dataset.
DATA_FRACTION=1
# Minimum: 16 models.
# Increase this value only in powers of 2.
NUM_EXPERIMENTS=32
EPOCHS=30
# Fixed FPR as decimal.
FPR=0.01

if [[ $DATA_FRACTION -gt 1 ]]; then
    DATASET_SIZE=$(($DATASET_SIZE / $DATA_FRACTION))
fi

if [[ $# -ne 1 ]]; then
    echo -e "ERROR: This script needs exactly one argument. \nPlease provide the step from which you want to start the pipeline (1-5).";
    exit 1;
elif [[ $NUM_EXPERIMENTS -lt 16 ]]; then
    echo -e "ERROR: The current number of experiments is insufficient.\nPlease use at least 16 shadow models.";
    exit 1;
elif [[ $(($NUM_EXPERIMENTS % 2)) -ne 0 ]]; then
    echo -e "ERROR: The current number of experiments is not a power of 2.\nThis run might break during the inference step.";
    exit 1;
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
            python3 core/plot/evaluation.py --logdir=exp/$DATASET/ --fpr=$FPR > eval@fixedFPR.txt
            ;;
        5)
            # Plot histograms.
            python3 core/plot/histogram.py --num_samples=10 --logdir=exp/$DATASET/ --classes=$DATASET_CLASSES 
            ;;
    esac
else
    echo "ERROR: The specified step needs to be any number from 1 to 5.";
    exit 1;
fi
