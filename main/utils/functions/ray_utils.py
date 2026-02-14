import ray

from ray.runtime_env import RuntimeEnv


def start_ray_local(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    context = ray.init(log_to_driver=log_to_driver)
    return context

def start_ray_local_cluster(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					pip="./requirements.txt"
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=log_to_driver)
    return context

def start_ray_rnl(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    '''
    env_vars={"CUDA_VISIBLE_DEVICES": "-1",
            "LD_LIBRARY_PATH": "$NIX_LD_LIBRARY_PATH"
            }
    '''

    runtime_env=RuntimeEnv \
					(
					working_dir="/mnt/cirrus/users/5/2/ist189452/TESE/NuZero",
					pip="./requirements.txt",
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=log_to_driver)
    return context