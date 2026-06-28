import json
import os
from typing import Any

import torch


class CheckpointUtils:

    FORMAT_VERSION = 1
    OPTIMIZER_FILE = "optimizer.pt"
    SCHEDULER_FILE = "scheduler.pt"
    REPLAY_BUFFER_FILE = "replay_buffer.pt"
    MODEL_FILE = "model.pt"
    METADATA_FILE = "metadata.json"

    @staticmethod
    def atomic_save(obj: Any, path: str) -> None:
        """
        Serialize ``obj`` with ``torch.save`` to a sibling ``.tmp`` file, then atomically
        rename it into ``path``. A crash mid-write leaves the temp file behind and never
        replaces a previously written ``path``.
        """
        tmp_path = f"{path}.tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    @staticmethod
    def write_metadata(directory: str, iteration: int) -> None:
        """Write the checkpoint's ``metadata.json`` (atomically) as the completeness marker."""
        metadata = {"format_version": CheckpointUtils.FORMAT_VERSION, "iteration": iteration}
        metadata_path = os.path.join(directory, CheckpointUtils.METADATA_FILE)
        tmp_path = f"{metadata_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp_path, metadata_path)

    @staticmethod
    def read_metadata(directory: str) -> dict:
        """Read and validate ``metadata.json``; raise if it is absent or an unknown version."""
        metadata_path = os.path.join(directory, CheckpointUtils.METADATA_FILE)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No '{CheckpointUtils.METADATA_FILE}' in '{directory}'; not a complete alphazoo checkpoint."
            )
        with open(metadata_path) as f:
            metadata = json.load(f)
        version = metadata.get("format_version")
        if version != CheckpointUtils.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format_version {version!r} (expected {CheckpointUtils.FORMAT_VERSION})."
            )
        return metadata

    @staticmethod
    def load_component(directory: str, filename: str, map_location: str) -> Any:
        """Load one checkpoint file via ``torch.load``; raise a clear error if it is missing."""
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint '{directory}' is missing component '{filename}'.")
        return torch.load(file_path, map_location=map_location, weights_only=False)
