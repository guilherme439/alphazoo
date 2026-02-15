import ray
from ray.runtime_env import RuntimeEnv


def start_ray_local(log_to_driver: bool = False) -> ray.runtime_context.RuntimeContext:
    print("\n\n--------------------------------\n\n")
    return ray.init(log_to_driver=log_to_driver)


