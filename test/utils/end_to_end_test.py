"""
Base class for end-to-end training tests.
"""

import torch


class EndToEndTest:

    def assert_run_successful(self, trainer, config) -> None:
        """
        Run a full training loop and assert it made observable progress: it ran
        to the configured final step, self-play filled the replay buffer, at
        least one network weight changed, and public metrics were produced.
        """
        weights_before = {key: value.clone() for key, value in trainer.get_model_state_dict().items()}
        captured_metrics: dict = {}

        def on_step_end(az, step, public) -> None:
            captured_metrics.update(public)

        trainer.train(on_step_end=on_step_end)

        assert trainer.current_step == config.running.training_steps - 1
        assert len(trainer.replay_buffer) > 0, "expected self-play to fill the replay buffer"

        weights_after = trainer.get_model_state_dict()
        assert any(
            not torch.equal(weights_before[key], weights_after[key])
            for key in weights_after
        ), "expected at least one network weight to change during training"

        assert captured_metrics, "expected public metrics to be produced via on_step_end"
