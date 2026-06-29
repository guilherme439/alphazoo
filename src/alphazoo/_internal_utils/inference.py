import torch

from ..inference.iinference_client import IInferenceClient
from .common import CommonUtils


class InferenceUtils:

    @staticmethod
    def get_gpu_names() -> list[str]:
        """Names of the discrete (non-integrated) CUDA/HIP devices visible to torch.

        Integrated GPUs are excluded: they share their architecture with the CPU
        and cannot run the compute kernels the model relies on.
        """
        if not torch.cuda.is_available():
            return []
        return [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
            if not torch.cuda.get_device_properties(index).is_integrated
        ]

    @staticmethod
    def resolve_inference_backend(backend: str) -> str:
        """Map "auto" to a concrete transport: "rpc" when Ray reports more than
        one live node, else "ipc". Any explicit backend is returned unchanged.
        """
        if backend != "auto":
            return backend

        CommonUtils.ensure_ray_initialized()
        return "rpc" if CommonUtils.count_live_nodes() > 1 else "ipc"

    @staticmethod
    def distribute_clients(
        inference_clients: list[IInferenceClient],
        num_gamers: int,
        threads_per_gamer: int,
        num_reanalysers: int,
        threads_per_reanalyser: int,
    ) -> tuple[list[list[IInferenceClient]], list[list[IInferenceClient]]]:
        gamer_clients: list[list[IInferenceClient]] = []
        for i in range(num_gamers):
            start = i * threads_per_gamer
            gamer_clients.append(inference_clients[start : start + threads_per_gamer])

        offset = num_gamers * threads_per_gamer
        reanalyser_clients: list[list[IInferenceClient]] = []
        for i in range(num_reanalysers):
            start = offset + i * threads_per_reanalyser
            reanalyser_clients.append(inference_clients[start : start + threads_per_reanalyser])

        return gamer_clients, reanalyser_clients
