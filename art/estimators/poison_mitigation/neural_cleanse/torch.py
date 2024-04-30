import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.estimators.poison_mitigation.neural_cleanse.neural_cleanse import NeuralCleanseMixin
from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class PyTorchNeuralCleanse(NeuralCleanseMixin, PyTorchClassifier):
    estimator_params = PyTorchClassifier.estimator_params + [
        "steps",
        "init_cost",
        "norm",
        "learning_rate",
        "attack_success_threshold",
        "patience",
        "early_stop",
        "early_stop_threshold",
        "early_stop_patience",
        "cost_multiplier_up",
        "cost_multiplier_down",
        "batch_size",
    ]

    def __init__(
        self,
        model,
        input_shape,
        num_classes,
        channels_first=False,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
        steps=1000,
        init_cost=1e-3,
        norm=2,
        learning_rate=0.1,
        attack_success_threshold=0.99,
        patience=5,
        early_stop=True,
        early_stop_threshold=0.99,
        early_stop_patience=10,
        cost_multiplier=1.5,
        batch_size=32,
        loss = None,
    ):
        super().__init__(
            steps= steps,
            model = model,
            input_shape=input_shape,
            nb_classes=num_classes,
            loss= loss,
            # channels_first,
            # clip_values,
            # preprocessing_defences,
            # postprocessing_defences,
            # preprocessing,
            init_cost=init_cost,
            norm=norm,
            learning_rate=learning_rate,
            attack_success_threshold=attack_success_threshold,
            early_stop=early_stop,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            patience=patience,
            cost_multiplier=cost_multiplier,
            batch_size=batch_size,
        )

        # Randomly initialize mask and pattern
        mask = torch.rand(*input_shape)
        pattern = torch.rand(*input_shape)

        self.mask_tensor_raw = torch.nn.Parameter(mask)
        self.mask_tensor = torch.tanh(self.mask_tensor_raw) / (2 - torch.tensor(torch.finfo(torch.float32).eps)) + 0.5

        self.pattern_tensor_raw = torch.nn.Parameter(pattern)
        self.pattern_tensor = torch.tanh(self.pattern_tensor_raw) / (2 - torch.tensor(torch.finfo(torch.float32).eps)) + 0.5

        # self.optimizer = torch.optim.Adam(
        #     [self.mask_tensor_raw, self.pattern_tensor_raw], lr=learning_rate, betas=(0.5, 0.9)
        #)

    def reset(self):
        super().reset()
        self.mask_tensor_raw.data.zero_()
        self.pattern_tensor_raw.data.zero_()

    def generate_backdoor(
        self, x_val: np.ndarray, y_val: np.ndarray, y_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        self.reset()
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        mask_best = None
        pattern_best = None
        reg_best = float("inf")
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False
        early_stop_counter = 0
        early_stop_reg_best = reg_best
        mini_batch_size = len(dataloader)

        for _ in tqdm(range(self.steps), desc=f"Generating backdoor for class {np.argmax(y_target)}"):
            loss_reg_list = []
            loss_acc_list = []

            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self._device), y_batch.to(self._device)
                loss_ce, loss_reg, loss_combined, loss_acc = self.train(x_batch, y_target.repeat(len(x_batch)))
                loss_reg_list.append(loss_reg.item())
                loss_acc_list.append(loss_acc.item())

            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss_acc = np.mean(loss_acc_list)

            # save best mask/pattern so far
            if avg_loss_acc >= self.attack_success_threshold and avg_loss_reg < reg_best:
                mask_best = self.mask_tensor.detach().cpu().numpy()
                pattern_best = self.pattern_tensor.detach().cpu().numpy()
                reg_best = avg_loss_reg

            # check early stop
            if self.early_stop:
                if reg_best < float("inf"):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    logger.info("Early stop")
                    break

            # cost modification
            if avg_loss_acc >= self.attack_success_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    self.optimizer.param_groups[0]['lr'] = self.learning_rate
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_success_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                self.cost *= self.cost_multiplier_up
                self.optimizer.param_groups[0]['lr'] = self.learning_rate
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                self.cost /= self.cost_multiplier_down
                self.optimizer.param_groups[0]['lr'] = self.learning_rate
                cost_down_flag = True

        if mask_best is None:
            mask_best = self.mask_tensor.detach().cpu().numpy()
            pattern_best = self.pattern_tensor.detach().cpu().numpy()

        if pattern_best is None:
            raise ValueError("Unexpected `None` detected.")

        return mask_best, pattern_best

    def train(self, x: torch.Tensor, y_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()

        reverse_mask_tensor = 1 - self.mask_tensor
        x_adv_tensor = reverse_mask_tensor * x + self.mask_tensor * self.pattern_tensor
        output_tensor = self.model(x_adv_tensor)

        loss_ce = F.cross_entropy(output_tensor, y_target)
        if self.norm == 1:
            loss_reg = torch.sum(torch.abs(self.mask_tensor)) / 3
        elif self.norm == 2:
            loss_reg = torch.sqrt(torch.sum(torch.square(self.mask_tensor)) / 3)

        loss_combined = loss_ce + loss_reg * self.cost
        loss_combined.backward()
        self.optimizer.step()

        # Compute accuracy
        pred_labels = torch.argmax(output_tensor, dim=1)
        correct = pred_labels.eq(y_target).sum().item()
        accuracy = correct / len(y_target)

        return loss_ce, loss_reg, loss_combined, torch.tensor(accuracy)

    def _predict_classifier(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        x = x.astype(ART_NUMPY_DTYPE)
        return PyTorchClassifier._predict_classifier(self, x=x, batch_size=batch_size, training_mode=training_mode, **kwargs)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        x = x.astype(ART_NUMPY_DTYPE)
        return self.fit(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def _get_penultimate_layer_activations(self, x: np.ndarray) -> np.ndarray:

        if self.layer_names is not None:
            penultimate_layer = len(self.layer_names) - 2
        else:
            raise ValueError("No layer names found.")
        return self.get_activations(x, penultimate_layer, batch_size=self.batch_size, framework=False)

    def _prune_neuron_at_index(self, index: int) -> None:

        if self.layer_names is not None:
            layer = self._model.layers[len(self.layer_names) - 2]
        else:
            raise ValueError("No layer names found.")
        weights, biases = layer.get_weights()
        weights[:, index] = np.zeros_like(weights[:, index])
        biases[index] = 0
        layer.set_weights([weights, biases])

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:

        return NeuralCleanseMixin.predict(self, x, batch_size=batch_size, training_mode=training_mode, **kwargs)

    def mitigate(self, x_val: np.ndarray, y_val: np.ndarray, mitigation_types: List[str]) -> None:

        return NeuralCleanseMixin.mitigate(self, x_val, y_val, mitigation_types)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs) -> np.ndarray:
        return self.loss_gradient(x=x, y=y, training_mode=training_mode, **kwargs)

    def class_gradient(
        self, x: np.ndarray, label: Union[int, List[int], None] = None, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
    
        return self.class_gradient(x=x, label=label, training_mode=training_mode, **kwargs)
