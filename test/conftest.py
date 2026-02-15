import os
import ray
import pytest


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def work_dir(tmp_path):
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)
