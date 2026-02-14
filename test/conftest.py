import os
import sys
import ray
import pytest

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def ray_session():
    # Add test dir to path so Ray workers can find test modules
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    ray.init(
        num_cpus=2,
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {"PYTHONPATH": TEST_DIR},
        },
    )
    yield
    ray.shutdown()


@pytest.fixture
def work_dir(tmp_path):
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)
