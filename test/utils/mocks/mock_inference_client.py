"""
Mock inference client for search tests.
"""

import torch


class MockInferenceClient:
    """Test double that replaces InferenceClient for search tests."""

    def __init__(self, model):
        self.model = model

    def inference(self, state):
        self.model.eval()
        with torch.no_grad():
            p, v = self.model(state)
        return p, v
