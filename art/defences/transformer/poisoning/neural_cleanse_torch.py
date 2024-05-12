# Attempt to replace
# /home/tushar/Documents/experiments_defend/defend_exp/lib/python3.10/site-packages/art/defences/transformer/poisoning/neural_cleanse.py

import logging
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import torch
import torch.nn as nn

from art.defences.transformer.transformer import Transformer
from art.estimators.poison_mitigation.neural_cleanse import PyTorchNeuralCleanse
from art.estimators.classification.pytorch import PyTorchClassifier

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class NeuralCleanse(Transformer):
    """
    Implementation of methods in Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.
    Wang et al. (2019).

    | Paper link: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
    """

    params = [
        "loss",
        "steps",
        "init_cost",
        "norm",
        "learning_rate",
        "attack_success_threshold",
        "patience",
        "early_stop",
        "early_stop_threshold",
        "early_stop_patience",
        "cost_multiplier",
        "batch_size",
    ]

    def __init__(self, classifier: "CLASSIFIER_TYPE") -> None:
        """
        Create an instance of the neural cleanse defence.

        :param classifier: A trained classifier.
        """
        super().__init__(classifier=classifier)
        self._is_fitted = False
        self._check_params()

    def __call__(  # type: ignore
            self,
            transformed_classifier: "CLASSIFIER_TYPE",
            loss,
            steps: int = 1000,
            init_cost: float = 1e-3,
            norm: Union[int, float] = 2,
            learning_rate: float = 0.1,
            attack_success_threshold: float = 0.99,
            patience: int = 5,
            early_stop: bool = True,
            early_stop_threshold: float = 0.99,
            early_stop_patience: int = 10,
            cost_multiplier: float = 1.5,
            batch_size: int = 32,
    ) -> PyTorchNeuralCleanse:
        transformed_classifier = PyTorchNeuralCleanse(
            transformed_classifier.model,
            loss=loss,
            input_shape=transformed_classifier.input_shape,
            num_classes=transformed_classifier.nb_classes,
            init_cost=init_cost,
            norm=norm,
            learning_rate=learning_rate,
            attack_success_threshold=attack_success_threshold,
            patience=patience,
            early_stop=early_stop,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            cost_multiplier=cost_multiplier,
            batch_size=batch_size
            # input_shape=transformed_classifier.input_shape,

            # nb_classes = transformed_classifier.nb_classes
        )
        return transformed_classifier

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        raise NotImplementedError

    def _check_params(self) -> None:
        if not isinstance(self.classifier, PyTorchClassifier):
            raise NotImplementedError("Only PyTorch classifiers are supported for this defence.")
