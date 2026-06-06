import logging
import math
from random import randrange
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn

from alphazoo.configs.alphazoo_config import LearningConfig, RecurrentConfig
from alphazoo.internal_utils.common import get_policy_loss_fn, get_value_loss_fn

from ..metrics import MetricsRecorder
from ..networks.model_host import ModelHost
from .replay_buffer import ReplayBuffer

logger = logging.getLogger("alphazoo")

class NetworkTrainer:

    MAX_SAMPLES_BATCH_SIZE_RATIO = 0.05
    MAX_EPOCHS_BATCH_SIZE_RATIO = 0.20

    def __init__(
        self,
        model_host: ModelHost,
        optimizer: Any,
        scheduler: Any,
        replay_buffer: ReplayBuffer,
        config: LearningConfig,
        recurrent_config: Optional[RecurrentConfig] = None
    ) -> None:
        self.model_host = model_host
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.replay_buffer = replay_buffer
        self.config = config
        self.recurrent_config = recurrent_config

        self.policy_loss_function, self.normalize_policy = get_policy_loss_fn(config.policy_loss, config.normalize_ce)
        self.value_loss_function = get_value_loss_fn(config.value_loss)

        self.recorder = MetricsRecorder()

    def get_metrics(self) -> dict:
        return self.recorder.drain()

    def get_model_state_dict(self) -> dict:
        return self.model_host.get_state_dict()
    
    def run_training_step(self) -> None:
        replay_size: int = len(self.replay_buffer)
        if replay_size == 0:
            logger.warning("WARNING: Replay buffer is empty; skipping training step.")
            return

        match(self.config.learning_method):
            case "epochs":
                effective_batch_size = self._capped_batch_size(
                    self.config.epochs.batch_size, replay_size, self.MAX_EPOCHS_BATCH_SIZE_RATIO,
                )
                epochs = self.config.epochs.learning_epochs
                batches_per_epoch = replay_size // effective_batch_size
                total_updates = epochs * batches_per_epoch
                logger.info(
                    f"Total updates: {total_updates} | Batch size: {effective_batch_size} "
                    f"| Epochs: {epochs} | Batches per epoch: {batches_per_epoch}"
                )
                self._train_with_epochs(effective_batch_size, batches_per_epoch, epochs)

            case "samples":
                effective_batch_size = self._capped_batch_size(
                    self.config.samples.batch_size, replay_size, self.MAX_SAMPLES_BATCH_SIZE_RATIO,
                )
                num_samples = self.config.samples.num_samples
                total_drawn_positions = effective_batch_size * num_samples
                if total_drawn_positions > replay_size:
                    logger.warning(
                        f"\nWARNING: Oversampling -> "
                        f"batch_size ({effective_batch_size}) * num_samples ({num_samples}) = {total_drawn_positions} "
                        f"exceeds replay buffer size ({replay_size})"
                    )
                
                logger.info(f"\nTotal updates: {num_samples} | Batch size: {effective_batch_size}")
                self._train_with_samples(effective_batch_size, num_samples)

            case _:
                raise Exception("Bad learning_method config.")
            

    def _capped_batch_size(self, batch_size: int, replay_size: int, ratio: float) -> int:
        """Cap `batch_size` at `ratio` of `replay_size` (minimum 1)."""
        return max(1, min(batch_size, int(ratio * replay_size)))

    def _train_with_epochs(
        self,
        batch_size: int,
        batches_per_epoch: int,
        num_epochs: int
    ) -> list[tuple[float, float, float]]:
        
        epoch_losses: list[tuple[float, float, float]] = []
        for e in range(num_epochs):
            self.replay_buffer.shuffle()

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_combined_loss = 0.0

            for b in range(batches_per_epoch):
                start_index = b * batch_size
                next_index = (b + 1) * batch_size

                batch = self.replay_buffer.get_slice(start_index, next_index)

                value_loss, policy_loss, combined_loss = self._batch_update_weights(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss
                epoch_combined_loss += combined_loss

            epoch_value_loss /= batches_per_epoch
            epoch_policy_loss /= batches_per_epoch
            epoch_combined_loss /= batches_per_epoch

            epoch_losses.append((epoch_value_loss, epoch_policy_loss, epoch_combined_loss))

            logger.info("Epoch " + str(e + 1) + "/" + str(num_epochs) + " done.")

        v_losses, p_losses, c_losses = zip(*epoch_losses)
        self.recorder.scalar("train/value_loss", sum(v_losses) / len(v_losses))
        self.recorder.scalar("train/policy_loss", sum(p_losses) / len(p_losses))
        self.recorder.scalar("train/combined_loss", sum(c_losses) / len(c_losses))

        return epoch_losses

    def _train_with_samples(
        self,
        batch_size: int,
        num_samples: int
    ) -> tuple[float, float, float]:
        
        probs: list[float] = []
        if self.config.samples.late_heavy:
            probs = self.get_late_heavy_distribution()

        average_value_loss = 0.0
        average_policy_loss = 0.0
        average_combined_loss = 0.0

        for _ in range(num_samples):
            batch = self.replay_buffer.get_sample(batch_size, probs)

            value_loss, policy_loss, combined_loss = self._batch_update_weights(batch)

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

    def get_late_heavy_distribution(self) -> list[float]:
        replay_size: int = len(self.replay_buffer)
        
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
    
    def _batch_update_weights(self, batch: list[Any]) -> tuple[float, float, float]:
        self.optimizer.zero_grad()

        value_loss: Tensor | float = 0.0
        policy_loss: Tensor | float = 0.0
        combined_loss: Tensor | float = 0.0

        if self.model_host.is_recurrent():
            value_loss, policy_loss, combined_loss = self._recurrent_batch_update(batch)
        else:
            value_loss, policy_loss, combined_loss = self._standard_batch_update(batch)

        loss = combined_loss

        loss.backward()  # type: ignore[union-attr]
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model_host.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()

        return value_loss.item(), policy_loss.item(), combined_loss.item()  # type: ignore[union-attr]

    def _standard_batch_update(self, batch: list[Any]) -> tuple[Tensor, Tensor, Tensor]:
        states, targets = list(zip(*batch))
        batch_input = torch.cat(states, 0)
        batch_size = len(states)
        outputs = self.model_host.forward(batch_input)
        return self._calculate_loss(outputs, targets, batch_size)

    def _recurrent_batch_update(self, batch: list[Any]) -> tuple[Tensor | float, Tensor | float, Tensor | float]:
        value_loss: Tensor | float = 0.0
        policy_loss: Tensor | float = 0.0
        combined_loss: Tensor | float = 0.0

        states, targets = list(zip(*batch))
        batch_size = len(states)
        batch_input = torch.cat(states, 0)
        recurrent_iterations = self.recurrent_config.train_iterations

        if self.recurrent_config.use_progressive_loss:
            total_value_loss: Tensor | float = 0.0
            total_policy_loss: Tensor | float = 0.0
            total_combined_loss: Tensor | float = 0.0
            prog_value_loss: Tensor | float = 0.0
            prog_policy_loss: Tensor | float = 0.0
            prog_combined_loss: Tensor | float = 0.0
            
            alpha: float = self.recurrent_config.prog_alpha
            if alpha != 1:
                outputs, _ = self.model_host.recurrent_forward(batch_input, recurrent_iterations)
                total_value_loss, total_policy_loss, total_combined_loss = self._calculate_loss(
                    outputs, targets, batch_size)

            if alpha != 0:
                outputs = self._get_output_for_prog_loss(batch_input, recurrent_iterations)
                prog_value_loss, prog_policy_loss, prog_combined_loss = self._calculate_loss(
                    outputs, targets, batch_size)

            value_loss = (1 - alpha) * total_value_loss + alpha * prog_value_loss
            policy_loss = (1 - alpha) * total_policy_loss + alpha * prog_policy_loss
            combined_loss = (1 - alpha) * total_combined_loss + alpha * prog_combined_loss
        else:
            outputs, _ = self.model_host.recurrent_forward(batch_input, recurrent_iterations)
            value_loss, policy_loss, combined_loss = self._calculate_loss(
                outputs, targets, batch_size)

        return value_loss, policy_loss, combined_loss

    def _calculate_loss(
        self,
        outputs: tuple[Tensor, Tensor],
        targets: tuple[Any, ...],
        batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        
        target_values, target_policies = zip(*targets)
        predicted_policies, predicted_values = outputs
        device = self.model_host.device()

        target_policies_t = torch.stack(target_policies).to(device)
        target_values_t = torch.tensor(target_values, dtype=torch.float32).to(device)

        predicted_policies_flat = predicted_policies.view(batch_size, -1)
        predicted_values_flat = predicted_values.reshape(-1)

        policy_loss: Tensor = self.policy_loss_function(predicted_policies_flat, target_policies_t)
        value_loss: Tensor = self.value_loss_function(predicted_values_flat, target_values_t)

        if self.normalize_policy and len(targets) > 1:
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
