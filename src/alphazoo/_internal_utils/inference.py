import logging

import ray
import torch

logger = logging.getLogger("alphazoo")


class InferenceUtils:

    @staticmethod
    def count_compute_gpus() -> int:
        """Number of discrete (non-integrated) CUDA/HIP devices visible to torch.

        Integrated GPUs are excluded: they share their architecture with the CPU
        and cannot run the compute kernels the model relies on.
        """
        if not torch.cuda.is_available():
            return 0
        return sum(
            1 for index in range(torch.cuda.device_count())
            if not torch.cuda.get_device_properties(index).is_integrated
        )

    @staticmethod
    def resolve_backend(backend: str) -> str:
        """Map "auto" to a concrete transport: "rpc" when Ray reports more than
        one live node, else "ipc". Any explicit backend is returned unchanged.
        """
        if backend != "auto":
            return backend

        InferenceUtils.ensure_ray_initialized()
        alive_nodes = sum(1 for node in ray.nodes() if node["Alive"])
        return "rpc" if alive_nodes > 1 else "ipc"

    @staticmethod
    def ensure_ray_initialized() -> None:
        if not ray.is_initialized():
            logger.warning(
                "Ray was not initialized; AlphaZoo expects ray to be initialized before "
                "calling train(). Starting a local single-node Ray."
            )
            ray.init()
