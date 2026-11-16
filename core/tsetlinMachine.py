from time import time

from numpy._core.numerictypes import int8, int32
from numpy._typing import NDArray
from pyTsetlinMachineParallel.tm import (
    MultiClassConvolutionalTsetlinMachine2D,
    MultiClassTsetlinMachine,
)


class TsetlinMachine:
    def __init__(self, image_dataset: bool = True):
        """
        Constructor currently supports two types of Tsetlin Machines, one for image classification and one for text.
        :param image_dataset: True if the dataset contains images, False if it contains text.
        """
        number_of_clauses: int = 2000
        target_voting: int = 50 * 100
        specificity: float = 5.0
        weighted: bool = True
        if image_dataset:
            self.core: (
                MultiClassConvolutionalTsetlinMachine2D | MultiClassTsetlinMachine
            ) = MultiClassConvolutionalTsetlinMachine2D(
                number_of_clauses=number_of_clauses,
                T=target_voting,
                s=specificity,
                patch_dim=(10, 10),
                weighted_clauses=weighted,
            )
        else:
            self.core = MultiClassTsetlinMachine(
                number_of_clauses=number_of_clauses,
                T=target_voting,
                s=specificity,
                weighted_clauses=weighted,
            )

    def train(
        self,
        epochs: int,
        train: dict[str, NDArray[int8]],
        test: dict[str, NDArray[int8]],
        print_comparison: bool = True,
    ) -> None:
        """
        Standard Tsetlin Machine training with logging.
        :param epochs: The number of epochs used in training.
        :param train: The training split on which the Tsetlin Machine will be trained on.
        :param test: The test split, which will be used to test the accuracy of the Tsetlin Machine after one epoch.
        :param print_comparison: Whether to measure the accuracy on the training split and print it in the logs too for comparison.
        """
        for i in range(epochs):
            start = time()
            self.core.fit(train["x"], train["y"], epochs=1, incremental=True)
            stop = time()

            result = 100 * (self.core.predict(test["x"]) == test["y"]).mean()

            print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result, stop - start))

            if print_comparison:
                result = 100 * (self.core.predict(train["x"]) == train["y"]).mean()

                print("#-> Accuracy: %.2f%%" % result)

    def class_scores(self, split: NDArray[int8]) -> NDArray[int32]:
        """
        Returns the vote scores of each class for all given samples.
        :param split: The split of samples from which the votes are needed.
        """
        return self.core.get_score(split)[1]
