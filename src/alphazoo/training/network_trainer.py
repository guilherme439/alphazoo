from __future__ import annotations

import logging
import math
from random import randrange
from typing import Any, Callable

import torch
from torch import Tensor, nn

from ..metrics import MetricsRecorder
from ..networks.model_host import ModelHost
from .replay_buffer import ReplayBuffer

logger = logging.getLogger("alphazoo")

LossFunction = Callable[[Tensor, Tensor], Tensor]


class NetworkTrainer:

    def __init__(self, model_host: ModelHost, optimizer: Any, scheduler: Any) -> None:
        self.model_host = model_host
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = MetricsRecorder()

    def get_metrics(self) -> dict:
        return self.recorder.drain()

    def get_state_dict(self) -> dict:
        return self.model_host.get_state_dict()

    def get_late_heavy_distribution(self, replay_size: int) -> list[float]:
        variation = 0.5
        offset = (1 - variation) / 2
        fraction = variation / replay_size

        probs: list[float] = []
        total = offset
        for _ in range(replay_size):
            total += fraction
            probs.append(total)

        total_sum = sum(probs)
        return [p / total_sum for p in probs]

    def train_with_epochs(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        replay_size: int,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        prog_alpha: float,
        use_progressive_loss: bool,
        learning_epochs: int,
    ) -> list[tuple[float, float, float]]:
        number_of_batches = replay_size // batch_size
        logger.info("\nBatches: " + str(number_of_batches) + " | Batch size: " + str(batch_size))

        total_updates = learning_epochs * number_of_batches
        logger.info("Total updates: " + str(total_updates))

        epoch_losses: list[tuple[float, float, float]] = []

        for e in range(learning_epochs):
            replay_buffer.shuffle()

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_combined_loss = 0.0

            for b in range(number_of_batches):
                start_index = b * batch_size
                next_index = (b + 1) * batch_size

                batch = replay_buffer.get_slice(start_index, next_index)

                value_loss, policy_loss, combined_loss = self._batch_update_weights(
                    batch, policy_loss_function, value_loss_function,
                    normalize_policy, train_iterations, prog_alpha, use_progressive_loss)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
                epoch_combined_loss += combined_loss

            epoch_value_loss /= number_of_batches
            epoch_policy_loss /= number_of_batches
            epoch_combined_loss /= number_of_batches

            epoch_losses.append((epoch_value_loss, epoch_policy_loss, epoch_combined_loss))

            logger.info("Epoch " + str(e + 1) + "/" + str(learning_epochs) + " done.")

        v_losses, p_losses, c_losses = zip(*epoch_losses)
        self.recorder.scalar("train/value_loss", sum(v_losses) / len(v_losses))
        self.recorder.scalar("train/policy_loss", sum(p_losses) / len(p_losses))
        self.recorder.scalar("train/combined_loss", sum(c_losses) / len(c_losses))

        return epoch_losses

    def train_with_samples(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        replay_size: int,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        prog_alpha: float,
        use_progressive_loss: bool,
        num_samples: int,
        late_heavy: bool,
    ) -> tuple[float, float, float]:
        probs: list[float] = []
        if late_heavy:
            probs = self.get_late_heavy_distribution(replay_size)

        average_value_loss = 0.0
        average_policy_loss = 0.0
        average_combined_loss = 0.0

        logger.info("\nTotal updates: " + str(num_samples) + " | Batch size: " + str(batch_size))

        for _ in range(num_samples):
            batch = replay_buffer.get_sample(batch_size, probs)

            value_loss, policy_loss, combined_loss = self._batch_update_weights(
                batch, policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, prog_alpha, use_progressive_loss)

            average_value_loss += value_loss
            average_policy_loss += policy_loss
            average_combined_loss += combined_loss

        average_value_loss /= num_samples
        average_policy_loss /= num_samples
        average_combined_loss /= num_samples

        self.recorder.scalar("train/value_loss", average_value_loss)
        self.recorder.scalar("train/policy_loss", average_policy_loss)
        self.recorder.scalar("train/combined_loss", average_combined_loss)

        return average_value_loss, average_policy_loss, average_combined_loss

    def _batch_update_weights(
        self,
        batch: list[Any],
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        alpha: float,
        use_progressive_loss: bool,
    ) -> tuple[float, float, float]:
        self.optimizer.zero_grad()

        value_loss: Tensor | float = 0.0
        policy_loss: Tensor | float = 0.0
        combined_loss: Tensor | float = 0.0

        if self.model_host.is_recurrent():
            value_loss, policy_loss, combined_loss = self._recurrent_batch_update(
                batch, policy_loss_function, value_loss_function,
                normalize_policy, train_iterations, alpha, use_progressive_loss)
        else:
            value_loss, policy_loss, combined_loss = self._standard_batch_update(
                batch, policy_loss_function, value_loss_function, normalize_policy)

        loss = combined_loss

        loss.backward()  # type: ignore[union-attr]
        self.optimizer.step()
        self.scheduler.step()

        return value_loss.item(), policy_loss.item(), combined_loss.item()  # type: ignore[union-attr]

    def _standard_batch_update(
        self,
        batch: list[Any],
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        states, targets = list(zip(*batch))
        batch_input = torch.cat(states, 0)
        batch_size = len(states)
        outputs = self.model_host.forward(batch_input)
        return self._calculate_loss(
            outputs, targets, batch_size,
            policy_loss_function, value_loss_function, normalize_policy)

    def _recurrent_batch_update(
        self,
        batch: list[Any],
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
        train_iterations: int,
        alpha: float,
        use_progressive_loss: bool,
    ) -> tuple[Tensor | float, Tensor | float, Tensor | float]:
        value_loss: Tensor | float = 0.0
        policy_loss: Tensor | float = 0.0
        combined_loss: Tensor | float = 0.0

        states, targets = list(zip(*batch))
        batch_size = len(states)
        batch_input = torch.cat(states, 0)
        recurrent_iterations = train_iterations

        if use_progressive_loss:
            total_value_loss: Tensor | float = 0.0
            total_policy_loss: Tensor | float = 0.0
            total_combined_loss: Tensor | float = 0.0
            prog_value_loss: Tensor | float = 0.0
            prog_policy_loss: Tensor | float = 0.0
            prog_combined_loss: Tensor | float = 0.0

            if alpha != 1:
                outputs, _ = self.model_host.recurrent_forward(batch_input, recurrent_iterations)
                total_value_loss, total_policy_loss, total_combined_loss = self._calculate_loss(
                    outputs, targets, batch_size,
                    policy_loss_function, value_loss_function, normalize_policy)

            if alpha != 0:
                outputs = self._get_output_for_prog_loss(batch_input, recurrent_iterations)
                prog_value_loss, prog_policy_loss, prog_combined_loss = self._calculate_loss(
                    outputs, targets, batch_size,
                    policy_loss_function, value_loss_function, normalize_policy)

            value_loss = (1 - alpha) * total_value_loss + alpha * prog_value_loss
            policy_loss = (1 - alpha) * total_policy_loss + alpha * prog_policy_loss
            combined_loss = (1 - alpha) * total_combined_loss + alpha * prog_combined_loss
        else:
            outputs, _ = self.model_host.recurrent_forward(batch_input, recurrent_iterations)
            value_loss, policy_loss, combined_loss = self._calculate_loss(
                outputs, targets, batch_size,
                policy_loss_function, value_loss_function, normalize_policy)

        return value_loss, policy_loss, combined_loss

    def _calculate_loss(
        self,
        outputs: tuple[Tensor, Tensor],
        targets: tuple[Any, ...],
        batch_size: int,
        policy_loss_function: LossFunction,
        value_loss_function: LossFunction,
        normalize_policy: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_values, target_policies = list(zip(*targets))
        predicted_policies, predicted_values = outputs
        device = self.model_host.device

        target_policies_t = torch.stack([torch.tensor(p, dtype=torch.float32) for p in target_policies]).to(device)
        target_values_t = torch.tensor(list(target_values), dtype=torch.float32).to(device)

        predicted_policies_flat = predicted_policies.view(batch_size, -1)
        predicted_values_flat = predicted_values.reshape(-1)

        policy_loss: Tensor = policy_loss_function(predicted_policies_flat, target_policies_t)
        value_loss: Tensor = value_loss_function(predicted_values_flat, target_values_t)

        if normalize_policy and len(targets) > 1:
            policy_loss = policy_loss / math.log(len(targets))

        combined_loss = policy_loss + value_loss

        invalid_loss = False
        if torch.any(torch.isnan(value_loss)):
            logger.error("\nValue Loss is nan.")
            invalid_loss = True
        if torch.any(torch.isnan(policy_loss)):
            logger.error("\nPolicy Loss is nan.")
            invalid_loss = True
        if invalid_loss:
            logger.error(predicted_values)
            logger.error(predicted_policies)
            raise Exception("Nan value found when calculating loss.")

        return value_loss, policy_loss, combined_loss

    def _get_output_for_prog_loss(self, inputs: Tensor, max_iters: int) -> tuple[Tensor, Tensor]:
        n = randrange(0, max_iters)
        k = randrange(1, max_iters - n + 1)

        if n > 0:
            _, interim_thought = self.model_host.recurrent_forward(inputs, iters_to_do=n)
            interim_thought = interim_thought.detach()
        else:
            interim_thought = None

        outputs, _ = self.model_host.recurrent_forward(inputs, iters_to_do=k, interim_thought=interim_thought)
        return outputs
