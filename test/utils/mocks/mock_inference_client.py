"""
Mock inference client for search tests.
"""

import torch


class MockInferenceClient:
    """Test double that replaces InferenceClient for search tests."""

    def __init__(self, model, is_recurrent_model=False):
        self.model = model
        self._is_recurrent = is_recurrent_model

    def is_recurrent(self):
        return self._is_recurrent

    def inference(self, state, training=False):
        self.model.eval()
        with torch.no_grad():
            p, v = self.model(state)
        return p, v

    def recurrent_inference(self, state, training, iters_to_do, interim_thought=None):
        p, v = self.inference(state)
        return (p, v), None
